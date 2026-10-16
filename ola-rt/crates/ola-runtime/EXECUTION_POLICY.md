# ExecutionPolicy design proposal

Status: **accepted** (2026-10-16) with all four recommended decisions.
Covers `todo.md` P2 and builds on `EXECUTION_PLAN.md`. All six steps are implemented.

## Goal

One replaceable `ExecutionPolicy` turns an immutable snapshot of runtime state
into an `ExecutionPlan` plus a record of *why* it decided that. Scheduling,
admission, token budgeting, and prefill grouping are components of the
policy. They are not decisions scattered through `Executor`.

Done when FIFO, memory-aware admission, token budgeting, and prefill grouping
run through that single path with no output or lifecycle regressions. The
check is that the existing executor tests and the fixture oracle
(`planned_multi_request_execution_matches_single_request_oracle`) pass
unchanged.

## Where policy logic lives today

- **Planning is executor code.** `plan_iteration`, `plan_admissions`,
  `plan_decode_batches`, `plan_prefill_batches`, and `assign_batch_modes` are
  private `Executor` methods. They call the four sub-policies and also
  tokenize prompts.
- **The scheduler is owned by `Runtime<S>`**, which is otherwise a lifecycle
  store. `ola-serve` and `ola-bench` replay are generic over `S` only
  because of this; neither schedules anything.
- **Every sub-policy takes `&mut self`.** If a plan fails validation after a
  stateful policy was consulted, that policy has already advanced (P1 problem
  5).
- **Decisions leave no reasons.** The plan records outcomes, but not which
  requests were ranked, what admission saw, which candidates the budget
  rejected, or the cost-aware grouping alternatives. `PrefillBatcher` returns
  only bucket keys, so `CostAwarePrefillDecision` is discarded.
- **`Executor::step()` bypasses planning.** It is the single-operation
  compatibility path, still mutates as it goes, and the UI's local chat
  (`ola-ui/src/local.rs`) uses it.
- **Policies have no stable names.** A report cannot say which policy and
  configuration produced it.

## Proposal

### Snapshot

```rust
pub struct PlanningSnapshot {
    pub at: Timestamp,
    pub generation: u64,
    pub limits: ExecutorLimits,          // max_active_requests, prefill_chunk_size
    pub queued: Vec<QueuedRequest>,      // queue order
    pub prefilling: BTreeMap<RequestId, PrefillView>,
    pub decoding: BTreeMap<RequestId, DecodeView>,
    pub decode_order: Vec<RequestId>,    // round-robin order
    pub prefill_order: Vec<RequestId>,   // round-robin order
    pub models: BTreeMap<ModelId, ModelSnapshot>,
    pub memory: MemoryState,
    pub device: Option<DeviceSnapshot>,          // None until P6 telemetry
    pub measurements: PolicyMeasurements,        // calibration inputs; empty today
}
```

- **Tokenization becomes mechanism, not policy.** The executor tokenizes each
  request once, at submit or lazily before its first planning pass, and caches
  the result until the request starts or is cancelled. Policies see only
  lengths or errors. This also removes the per-iteration re-tokenization of
  deferred requests.
- **`PlanningSnapshot` replaces `PlanningView`.** `ExecutionPlan::validate`
  checks plans against the same snapshot the policy saw. Whether KV capacity
  is enforced is passed separately, because it is the admission policy's
  guarantee, not a property of the state.
- The snapshot is owned rather than borrowed, so it can be stored or
  compared. Live KV reservations are derived from the views
  (`reserved_kv_bytes()`, `model_reserved_kv_bytes()`), and the round-robin
  order of live work is explicit (`decode_order`, `prefill_order`).
- `priority`, `deadline`, and `queued_at` are included now so P3 schedulers
  need no snapshot change.

### Policy trait

```rust
pub trait ExecutionPolicy: Send {
    fn descriptor(&self) -> PolicyDescriptor;
    fn plan(&self, snapshot: &PlanningSnapshot<'_>) -> PolicyDecision;
    /// Called once per committed iteration; the only place a policy may learn.
    fn observe(&mut self, _iteration: &ExecutorIteration) {}
}

pub struct PolicyDecision {
    pub plan: ExecutionPlan,
    pub record: DecisionRecord,
}

pub struct PolicyDescriptor {
    pub name: &'static str,                 // stable, e.g. "fifo-token-budget"
    pub config: Vec<(&'static str, String)>, // e.g. [("max_tokens", "4")]
    pub components: Vec<PolicyDescriptor>,
}
```

- **`plan(&self)`** makes "identical snapshot and configuration gives an
  identical decision" a property of the signature. It also means a rejected
  plan cannot leave a policy half-advanced.
- **`observe(&mut self)`** gives adaptive or fair policies (P3 aging, P4
  calibration) one explicit, post-commit place to update.
- **Mutation is ruled out by types.** Policies receive `&PlanningSnapshot` and
  never `&mut Runtime` or sessions.

### Composition

`ComposedPolicy` implements `ExecutionPolicy` by moving today's planning
algorithm out of `Executor` unchanged. It holds four replaceable components:

| Component | Today | Change |
| --- | --- | --- |
| `Scheduler` | `select(&mut self, &RuntimeState)`, `rank` | `rank(&self, &PlanningSnapshot, limit)`; owned by the policy, not `Runtime` |
| `AdmissionPolicy` | `decide(&mut self, &AdmissionContext)` | `&self`; context built from the snapshot |
| `BatchPolicy` | `select(&mut self, &BatchContext)` | `&self` |
| `PrefillBatcher` | `buckets(&mut self, …)` → keys | renamed `PrefillGrouping`; `group(&self, lengths) -> GroupingDecision` returning keys **and** alternatives and estimates |

Every component also implements `descriptor()`. The default
`ComposedPolicy` (FIFO, unconstrained admission, unbounded batch,
unbucketed grouping) reproduces today's `Executor::default()`.

### Decision record

```rust
pub struct DecisionRecord {
    pub policy: PolicyDescriptor,
    pub ranking: RankingRecord,                 // slots available, limit, ranked IDs
    pub admissions: Vec<AdmissionRecord>,       // context summary + decision per ranked request
    pub selections: Vec<SelectionRecord>,       // phase, candidates, selected, budget used
    pub grouping: Vec<GroupingRecord>,          // model, phase, alternatives, estimates, choice
    pub capability_fallbacks: Vec<CapabilityFallback>, // tensor wanted, hook not declared
}
```

`ExecutorIteration` gains `decision: DecisionRecord`. `PlanReport` moves to
schema version 2 to include it. This closes the two open P1 items about
cost-estimate and policy-reason recording.

A *capability fallback* is a decision made before execution: the policy
wanted a tensor group and the model doesn't declare the hook. A *divergence*
is something observed at execution. They stay separate.

### Executor surface

- `Executor::with_policy(policy: impl ExecutionPolicy)`.
- `ComposedPolicy` builder: `ComposedPolicy::default().with_admission(…)
  .with_batch(…).with_grouping(…).with_scheduler(…)`.
- `Executor::with_admission_policy`, `with_batch_policy`, and
  `with_prefill_batcher` are removed. The call sites to migrate are `ola-ui`
  (1), `ola-bench` (1), the `ola-burn` oracle test (1), and executor tests.
- `Runtime` drops its scheduler generic and becomes a pure lifecycle store.
  `ola-serve`'s `InferenceService<S>` and `Workload::replay<S>` lose `S`.
- `Executor::step()` is retired. The UI's local generation loop switches to
  `step_iteration`, so every state change in the executor goes through a
  validated plan.

## Decisions (all recommendations accepted)

1. **`&self` policies with an `observe` hook** *(recommended)* or keep `&mut
   self`. `&self` makes determinism structural and fixes the half-advanced
   problem. The cost is that stateful policies must update in `observe`.
2. **Move the scheduler out of `Runtime`** *(recommended)* or keep `Runtime<S>`
   and have the policy borrow the runtime's scheduler. The recommended option
   is a breaking change across four crates, but only in type parameters.
3. **Retire `Executor::step()`** *(recommended)* or reimplement it as "plan
   one operation". Retiring it removes the last unplanned mutation path; the
   UI loop changes in about ten lines.
4. **Replace the `with_*` setters with `with_policy(ComposedPolicy…)`**
   *(recommended)* or keep them as shims. Shims need the executor to know
   whether its policy is composed, which is awkward once custom policies
   exist.

## Implementation steps

Each step keeps `make verify` and `make test-fixture` green.

1. **Sub-policies to `&self` plus descriptors.** No behaviour change. **Done.**
2. **Snapshot.** Add `PlanningSnapshot`, `ModelSnapshot`, and the
   prepared-prompt cache; validate against the snapshot; drop
   `PlanningView`. **Done.**
3. **`ExecutionPolicy` and `ComposedPolicy`.** Move the planning code out of
   `Executor`, add `with_policy`, and migrate the `with_*` call sites. **Done.**
   `Executor::step()` was retired in this step rather than step 4, because it
   called the admission policy directly and that policy now lives inside
   `ComposedPolicy`. `Scheduler::rank` reads the snapshot.
4. **Scheduler.** Make `Runtime` non-generic and migrate `ola-serve`,
   `ola-bench`, and `ola-observe`. (`step()` was already retired in step 3.) **Done.**
   `Runtime` and `Executor` are no longer generic, `Scheduler` keeps only
   `rank` and `descriptor`, and the policy-free `Runtime::start_queue_head`
   replaces `schedule_next` for executor-less lifecycle probes.
5. **`DecisionRecord`.** Add the record, `PrefillGrouping` returning
   alternatives, `CostAwarePrefillDecision` preserved, capability fallbacks,
   and `PlanReport` v2. **Done.**
   Deviation: `PrefillBatcher` keeps its name and gains `group(&self, lengths)
   -> GroupingDecision` (buckets plus optional `CostAwarePrefillDecision`)
   instead of being renamed `PrefillGrouping`; the rename would have churned
   `ola-bench` signatures without changing behaviour. Admission records keep
   the complete `AdmissionContext` and decision; selection records keep the
   complete `BatchContext`.
6. **Tests:**
   - determinism: the same snapshot planned twice gives equal decisions;
   - no mutation: an adversarial policy's invalid plan leaves state
     untouched;
   - conflicts: head-of-line deferral against ranking, budget against an
     oversized prefill, memory against predicted release, grouping against a
     missing capability, and admission against the KV check.

   Kernel-selection conflicts are recorded as deferred until the kernel track
   adds a selection to the plan.

   **Done** in `src/composed.rs` tests, planning hand-built snapshots without
   an executor. The kernel-selection conflict remains deferred.

## Out of scope for P2

- New scheduling baselines (priority, deadline, SJF): P3. The snapshot
  already carries the fields they need.
- Populating `DeviceSnapshot` and `PolicyMeasurements`: P6 telemetry and P4
  calibration.
- Kernel selection: the kernel track.
