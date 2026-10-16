# ExecutionPlan design proposal

Status: **accepted** (2026-10-16). Covers `todo.md` P1. Steps 1–4 are
implemented: every `step_iteration` is planned, validated, and committed as
one `ExecutionPlan`, exposed as `ExecutorIteration::execution_plan`.

## Goal

One executor iteration should be fully described by a validated
`ExecutionPlan` **before** any model executes or any lifecycle, session, or KV
state changes. Execution then commits the plan, and the result shows where
execution diverged from it.

The plan is a runtime IR: it describes what this iteration runs. It does not
describe model graphs, tensor operations, or compiler passes.

## How `step_iteration` works today

Planning and mutation are interleaved inside `Executor::step_iteration`:

1. Select decode candidates with `BatchPolicy` → **execute decodes** (sessions
   mutate, requests may finish and release KV).
2. Select continued-prefill candidates → **execute them**.
3. Compute free slots *from the outcome of steps 1–2*, then loop:
   `Scheduler::select` → tokenize → `AdmissionPolicy::decide` → `BatchPolicy`
   → **`Runtime::start_selected` / `reject`** (lifecycle mutates) →
   `create_session`.
4. Group admitted prompts with `PrefillBatcher` → **execute**.

`IterationPlan` records only selected/deferred `BatchCandidate`s, and it is
assembled as a side effect while execution proceeds. Grouping, tensor versus
individual mode, KV growth, and admission reservations are only visible after
the fact through `ExecutionBatch` and events.

## Responsibilities and boundaries

`ExecutionPlan` **is**:

- plain, cloneable, serializable data for one iteration;
- produced by a read-only *planner* from executor and runtime state plus the
  existing policies (`Scheduler`, `AdmissionPolicy`, `BatchPolicy`,
  `PrefillBatcher`);
- checked by `ExecutionPlan::validate` against the state it was built from;
- the only input the executor commits from.

`ExecutionPlan` **is not**:

- an executor: it holds no sessions, models, or tensors;
- a policy: the decisions come from the policies. Combining those policies is
  P2;
- a measurement: it holds deterministic *estimates*. Timings stay in
  `ola-observe`;
- an owner of lifecycle: only `Runtime` and `Executor` mutate state, and only
  while committing a validated plan.

It lives in `ola-runtime` (`src/plan.rs`). It is runtime-specific, so it does
not belong in `ola-core`.

## Shape

The types are in `src/plan.rs`. In summary:

- `ExecutionPlan { generation, at, lifecycle, batches, deferrals,
  predicted_releases }`;
- `LifecycleAction::{Admit, Reject, Fail}`. `Admit` carries the KV
  reservation; its event comes from the request's first prefill row;
- `PlannedBatch { phase, model_id, mode, rows }` with
  `PlannedPhase::{Prefill, ContinuedPrefill, Decode}`;
- `PlannedRow { request_id, event_index, positions, final_prompt_chunk }`,
  where `positions` is the absolute range processed and therefore also the KV
  growth;
- `PlannedDeferral::{Admission, Budget, Slots}`;
- `PlanningView`, the read-only snapshot (queued, prefilling, decoding, slot
  limit, and model KV capacities) that `validate` checks against.

Padding and `PlanEstimates` are computed from the rows rather than stored, so
they cannot contradict them. A `KernelSelection` field is added when the
kernel track needs it.

`IterationPlan` becomes a projection of this type (`plan.iteration_view()`) so
`ola-observe`, the UI, and benchmarks keep working while they migrate.
`ExecutionBatch` stays as the record of what *actually* ran.
`ExecutorIteration` gains `execution_plan: ExecutionPlan`.

## Lifecycle: plan → validate → commit

```text
state ──► Planner (reads state, consults policies) ──► ExecutionPlan
                                                          │ validate(&executor)
                                                          ▼
                     commit: lifecycle actions → batches in order → events
                                                          │
                                                          ▼
                     ExecutorIteration { execution_plan, execution_batches, events }
```

`validate` rejects a plan, before anything mutates, when:

- **stale:** `generation` differs from the executor's current generation;
- **duplicate:** a request has more than one role (row, reject/fail,
  deferral). The one allowed pair is an `Admit` with its first `Prefill` row,
  and each requires the other;
- **invalid status:** an admitted, rejected, failed, or slot/admission-deferred
  request is not queued; a continued prefill is not prefilling; a decode is
  not decoding;
- **inconsistent:** a batch is empty, an individual batch has several rows, a
  row names the wrong model, positions do not start at the cursor or end past
  the prompt, the final-chunk flag is wrong, a decode row is not exactly its
  next position, or event indices are not exactly `0..n`;
- **bad release:** a predicted release is not a limit-reaching decode row in
  this plan;
- **over capacity:** live minus predicted releases plus admissions exceeds
  `max_active_requests`, or admissions push a model's reservations past its
  KV pool. Models without admissions are not checked, because a permissive
  admission policy may already hold live reservations past capacity.

Two checks remain the admission and batch policies' job: device-wide memory
budgets, and whether budget deferrals are consistent with the token budget.
The plan does not see either configuration.

These rules cover the P1 test list (duplicate, stale, invalid, and
over-capacity plans).

## Problems found in the current code

These are the parts that stop step 1 from being a simple refactor.

### 1. Refilling slots depends on execution outcomes (decided: option C)

Today, slots freed by EOS or `max_new_tokens` during the decode phase are
refilled **in the same iteration**. A plan made before execution cannot know
about EOS. Options:

- **A. Use only boundary slots.** Slots freed during this iteration refill on
  the next one. Simplest and fully deterministic, but every refill is one
  iteration later than today.
- **B. Two plans per iteration** (decode/continue, then admission). Keeps
  today's behaviour, but breaks the definition that one plan explains one
  iteration.
- **C. Predict limit-based completions (chosen).** A decode that will
  reach `max_new_tokens` is known to finish, so the planner counts its slot and
  its reservation as available. Only EOS-freed slots wait an iteration, and
  that wait is recorded as a `Slots` deferral. Deterministic, and it matches
  today's behaviour when requests finish by hitting their limit.

Whichever option is chosen, the change shows up in
`iteration_advances_live_sessions_and_refills_freed_slots` and in TTFT for
EOS-heavy traces. That should be measured, not assumed.

### 2. The scheduler cannot rank without mutation

`Scheduler::select` returns one ID from live state. Admitting several requests
works today only because `start_selected` removes each one from the queue
before the next call. Proposal: add

```rust
fn rank(&mut self, state: &RuntimeState, limit: usize) -> Vec<RequestId>
```

with a default implementation that calls `select` on a cloned queue view.
`FifoScheduler` overrides it with `queued_ids().take(limit)`. The planner
walks the ranked list in order and stops at the first admission deferral,
which keeps today's FIFO semantics.

### 3. Tensor versus individual mode is discovered during execution

`CausalModel::prefill_batch`, `prefill_chunk_batch`, and `decode_batch` return
`None` to mean "unsupported", so the mode is only known after the call.
Proposal: add `CausalModel::batch_capabilities() -> BatchCapabilities`, which
defaults to none and is set by Burn Llama. The plan records the intended
mode. If a model declares support but still returns `None`, the executor falls
back as it does now, and the iteration records the divergence instead of
silently changing mode.

### 4. Rejection and failure currently mutate state mid-plan

Tokenization failure, a missing model, and admission rejection call
`start_selected`/`fail_started`/`reject` while the loop is still running.
These become `LifecycleAction`s that are applied first during commit, in
event order. Tokenization moves into planning. Prompts of deferred requests
are cached by `RequestId` so a deferred request is not re-tokenized every
iteration.

### 5. Policies are `&mut self`

If validation rejects a plan after the planner consulted a stateful policy,
that policy has already advanced. All current policies are stateless, so this
note only documents the constraint. P2 removes it by making policies pure
functions of a snapshot.

## Determinism and events

- `event_index` is assigned while planning, in the same order used today:
  decodes in ready order, then continued prefills, then admissions.
- Commit emits events sorted by `event_index`, so regrouping a batch (for
  example power-of-two bucketing) cannot reorder events. Existing tests
  already assert this.
- Given identical state and configuration, the planner produces a
  byte-identical serialized plan. A test asserts this.

## Report representation

`ola-runtime` and `ola-core` do not depend on serde today; only `ola-bench`
and `ola-burn` do. To keep the runtime free of serialization concerns,
`ola-bench` owns a versioned `PlanReport` with `schema_version`, converted
from `&ExecutionPlan` in the same way as the schema-v2 benchmark reports.
Adding an optional `serde` feature to `ola-core` and `ola-runtime` would also
work if direct derives turn out simpler. The UI gets a short human-readable
`Display` form.

## Implementation steps

Each step keeps `make verify` and the fixture tests green.

1. Add the `plan.rs` types and `validate`, with unit tests for duplicate,
   stale, invalid, and over-capacity plans. No executor changes. **Done.**
2. Add `BatchCapabilities` to `ola-model`, implemented by Burn Llama and the
   test models. **Done.**
3. Build a planner for decodes and continued prefills, which do not depend on
   outcomes, and drive them through plan → validate → commit. Keep the
   current admission path. **Done.** Admissions reuse the same grouping and
   execution helpers.
4. Add `Scheduler::rank` and move admission into the plan using the slot
   option chosen in problem 1. **Done.** Final
   continued-prefill chunks of one-token requests are also predicted
   releases, matching the historical refill behaviour.
5. Record plan-versus-execution divergence (mode fallback, EOS-freed slots).
6. Add the serialized report. Migrate `ola-observe` and the UI from
   `IterationPlan` to `ExecutionPlan` and drop the projection.
7. Add an oracle test that compares token outputs from planned multi-request
   execution against the single-request path on the pinned fixture.

## Out of scope for P1

- Kernel selection. The field exists but stays `None` until the kernel track.
- Time-based cost estimates. Estimates are tokens, padding, invocations, and
  bytes only; calibrated costs belong to P4.
- Combining the policies into one `ExecutionPolicy` (P2).
- Preemption, eviction, and KV release beyond terminal and cancellation
  (P7).
