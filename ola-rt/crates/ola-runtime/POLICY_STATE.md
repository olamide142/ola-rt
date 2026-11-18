# Policy state, observations, and beliefs

Status: **specification of what is implemented** (2026-11-17). Builds on
`EXECUTION_POLICY.md`, `ADAPTIVE_POLICY.md`, `MEMORY_POLICY.md`, and
`ADAPTIVE_OPTIMIZATION_PLAN.md`. Every field below is either traced to a
Rust type or marked as not implemented.

## Decision loop

The decision epoch `n` is one executor iteration (`IterationId`), not a
wall-clock tick: iterations take as long as their work, and timestamps are
supplied by the caller (`Timestamp`, virtual time in simulations).

```text
O_n = ψ(X_n)           PlanningSnapshot built by the executor
B_n                    OutputLengthBelief (optional), frozen for planning
θ_n                    AdaptivePolicy::theta(), installed policy
A_n = π(O_n, B_n, θ_n) ExecutionPolicy::plan(&self, &PlanningSnapshot)
Γ(O_n, A_n; C)         ExecutionPlan::validate(&snapshot, enforce_kv)
                       → on failure: AdaptivePolicy falls back to a safe
                         policy, or step_iteration returns an error and
                         nothing changes
execute A_n            Executor::step_iteration
Y_n                    ExecutorIteration (events, divergences, batches)
B_{n+1}, θ_{n+1}       ExecutionPolicy::observe(&mut self, &Y_n)
```

Three properties hold by construction:

1. **Planning cannot mutate.** `plan` takes `&self` and `&PlanningSnapshot`,
   an owned value with no references back into the runtime. A policy has no
   path to `Runtime`, sessions, or the KV pool during planning.
2. **Learning happens after commit.** `observe` is the only `&mut self`
   method and receives only committed iterations. Beliefs and θ therefore
   change only between epochs.
3. **Hard limits are enforced twice.** `ExecutionPlan::validate` checks the
   plan against the snapshot; the executor still enforces physical KV
   allocation (`PlanDivergence::KvCapacityExhausted`, recovered by
   preemption when the policy `recovers_kv_exhaustion`).

## State `X_n` and observation `ψ`

| Variable | Status | Where it is | Notes |
| --- | --- | --- | --- |
| Queue: ids, model, prompt length, limit, priority, deadline, arrival | observed exactly | `PlanningSnapshot::queued: Vec<QueuedRequest>` | Prompt length after tokenization, or the preparation error. After preemption `max_new_tokens` is the **remaining** limit and the prompt includes generated tokens. |
| Prefix-cache hit per queued request | observed exactly | `QueuedRequest::cached_prefix_tokens`, `cached_prefix_evictable_blocks` | |
| Live prefill progress | observed exactly | `PlanningSnapshot::prefilling: BTreeMap<RequestId, PrefillView>` (`cursor`, `prompt_tokens`) | |
| Live decode progress `g` | observed exactly | `PlanningSnapshot::decoding: BTreeMap<RequestId, DecodeView>` (`generated_tokens`, `position`, `max_new_tokens`) | |
| Round-robin order | observed exactly | `decode_order`, `prefill_order` | |
| KV reservations | observed exactly | `PrefillView/DecodeView::reserved_kv_bytes`, `PlanningSnapshot::reserved_kv_bytes()` | A reservation is a policy commitment, not physical use. |
| Physical KV pool occupancy | observed at the boundary | `ModelSnapshot::kv_pool: Option<KvPoolStatus>` | `None` for models without a shared pool. |
| KV geometry and capacity | observed exactly | `ModelSnapshot::kv_cache: Option<KvCacheProfile>` | |
| Executor limits (slots, chunk) | observed exactly | `PlanningSnapshot::limits: ExecutorLimits` | Policy plans may not raise them. |
| Device memory accounting | observed, configured | `PlanningSnapshot::memory: MemoryState` | Accounting supplied by the caller, not measured device telemetry. `capacity_bytes == 0` means "no device budget". |
| Device telemetry | **not implemented** | `PlanningSnapshot::device: Option<DeviceSnapshot>` is always `None` | Policies must treat it as unknown. |
| Calibrated costs | **not populated at runtime** | `PolicyMeasurements` (empty) | Iteration costs are fitted offline by `ola_bench::fit_cost_model` (OLS) and used only by simulators and replay. |
| Natural output length `L` / remaining output `R` | **hidden** | — | Known only once a request finishes. Estimated by `B_n`, never observed. |
| Future arrivals | **hidden** | — | No arrival model exists. Replay assumes the recent past repeats. |
| Actual iteration latency | observed after execution only | `ExecutorEvent::Token::at`, `ExecutorIteration::at` | Not available during planning. |
| Executor-only state | **not visible** | `Executor` private fields, sessions, KV block tables | Physical allocation, model sessions, and the prepared-prompt cache are executor mechanism. |

`ψ` is `Executor`'s snapshot construction: it copies exactly the rows
above marked observed. It is deterministic given executor state.

## Belief `B_n`

`OutputLengthBelief` (`src/belief.rs`) is the only belief. It describes the
**remaining** output `R = min(L, m) − g`, never the total, for a request
with caller limit `m` and `g` tokens generated.

- **Representation.** A sliding window (default 256) of finished requests
  per class, where class is (model, `⌈log2 m⌉`), plus a pooled window per
  model. A natural stop is an exact sample of `L`; reaching the limit is a
  right-censored sample. The survival function is the Kaplan–Meier
  product-limit estimate, and conditioning on `g` gives
  `P(R > r) = S(g + r) / S(g)`. Mass the window cannot place goes to the
  limit, which is conservative for memory. The query returns a discrete
  distribution (`RemainingOutputDistribution`), with quantiles,
  exceedance, mean, and inverse-CDF sampling.
- **Why this representation.** Output limits truncate observations, so a
  histogram of observed lengths would count limit hits as natural stops and
  underestimate long outputs. Kaplan–Meier handles that censoring exactly
  and needs no distributional assumption. Quantile regression or learned
  predictors would need features the runtime does not have (prompt content
  is not visible to policies).
- **Conditioning.** Only admission-time information: model and output
  limit bucket. Updated as tokens are generated through `g`.
- **Cold start, insufficient samples.** A query needs `min_samples`
  (default 16) finished samples outliving `g`, first in the class, then in
  the model pool; otherwise it returns `BeliefUnavailable` and callers use
  the worst case `m − g`.
- **Invalid input.** `m = 0` or `g ≥ m` returns
  `BeliefUnavailable::InvalidInput`.
- **Preemption.** A re-admitted request keeps its original class, limit,
  and count (`remaining_for`).
- **Completion, failure, cancellation.** Finishing stores a sample; failed
  and rejected requests are forgotten without a sample. Cancellations emit
  no iteration event, so they are forgotten by `forget` or by the bound on
  tracked requests (oldest admission first).
- **Drift.** Only by forgetting old samples. There is no drift test.
- **Bounded cost.** At most `max_classes + models` windows of `window`
  samples and `max_tracked_requests` live entries. A query is linear in the
  window. `BeliefPolicy` clones the belief only when a request finishes.
- **Calibration.** None is claimed. The estimate is empirical on the
  window, and the window may describe a different workload than the next
  request; `regime-shift-study` shows exactly that case.

`B_n` reaches planning only through a frozen `Arc<OutputLengthBelief>`
inside `QuantileAdmission`, rebuilt by `BeliefPolicy::observe`.

## Action `A_n`

`ExecutionPlan`: lifecycle actions (`Admit` with a KV reservation, `Reject`,
`Fail`, `Preempt`), batches of rows with kernels and modes, deferrals, and
predicted releases. Policies choose it; `DecisionRecord` keeps why.

## Constraints `C` and feasibility `Γ`

`ExecutionPlan::validate` enforces: matching generation; one role per
request; admissions paired with first prefills; phase admissibility;
chunk sizes and resumability; kernel requirements; event order; predicted
releases; slot capacity; and, when the policy promises it
(`enforces_model_kv_capacity`), that live plus admitted **reservations**
fit each model's pool.

Validation checks reservations, not physical blocks, and trusts the
reservation an admission reports. Physical safety for policies that
reserve less than the worst case (`WatermarkAdmission`,
`QuantileAdmission`) comes from `ComposedPolicy::with_preemption`, which
checks free blocks against the iteration's growth, preempts before a
shortfall, and defers admissions whose first prefill does not fit
(`AdmissionReason::KvPoolFull`), and from the executor's allocation
recovery. A belief cannot widen any of these checks: requests whose worst
case exceeds the pool are still rejected, and every plan is validated.

## Outcome `Y_n`

`ExecutorIteration`: committed plan, `DecisionRecord`, events (tokens with
timestamps and EOS, preemptions, deferrals, failures), divergences,
execution batches, and post-iteration counts and KV bytes.
`ExecutorIteration::adaptive_observations` derives per-request timing and
workload facts for optimizers.

## Parameters `θ_n` and selection

`AdaptivePolicy` holds θ as `(key, value)` pairs, rebuilds its active
policy through `AdaptivePolicyFactory` after observations, and only
replaces θ when the optimizer returns `Ok` **and** the factory builds it.
A θ that builds but plans infeasibly is not reverted; each such plan is
replaced by the fallback's validated plan.

Selection objective, where implemented:

```text
J(θ) = operating_cost(θ) + switch_cost(θ_prev, θ)
switch_cost(θ_prev, θ) = Σ_k w_k · 1[θ_k ≠ θ_prev,k]    (SwitchingCost)
```

- `MilpCandidateOptimizer::with_switching_cost` adds a binary `z_k` per
  weighted knob with `|x_k − x_prev,k| ≤ M_k z_k`. This is exact for its
  linear operating cost; it agrees with the finite oracle in tests, and
  `last_objective()` reports both terms of the solution it returned.
- `ola_bench::ReplayOptimizer` uses replayed p95 latencies, which are not
  linear in θ, so the MILP cannot express them. The switching term is
  applied in the candidate-selection layer: exhaustive enumeration over
  the grid of `replay_score(θ) + switch_cost(θ_prev, θ)`. It switches only
  if `J(candidate) < replay_score(θ_prev) − margin`, where the margin
  (`min_improvement_percent`) is a separate guard against replay noise.
  `ReplaySelection` records each term exactly as compared. Both default
  to zero switching cost, so earlier results reproduce.

A *switch* is a change in a knob listed in `SwitchingCost::weights`; knobs
not listed (including descriptive θ entries) never count. Units are the
optimizer's objective units (objective microseconds for replay).

## Risk

`kv_overflow_risk` draws seeded scenarios of `R` for the live set from
`B_n` and reports the fraction in which the sum of final KV footprints
exceeds the pool, with a Hoeffding bound on the *sampling* error only.
Summing final footprints overstates peak concurrent demand, so it is
conservative for "will the current live set need preemption"; it ignores
future admissions and error in the belief. It is a reported estimate: no
plan is accepted or rejected because of it.

## Implemented, prototype, and future

| Item | Status |
| --- | --- |
| Snapshot/plan/validate/observe loop | implemented |
| Remaining-output belief with censoring | implemented, uncalibrated |
| Quantile admission with worst-case fallback | implemented, evaluated in virtual time only |
| Switching cost in MILP (exact) and replay selection | implemented |
| `kv_overflow_risk` scenario estimate | prototype: reported, not used for decisions |
| Regime experiment with per-regime optimum | implemented (`regime-shift-study`, simulation) |
| Regime change detection | **not implemented**; replay and belief windows only forget |
| Scenario optimization with recourse / chance constraints | **not implemented**; preemption is the only recourse |
| Arrival model, device telemetry, runtime cost measurements | **not implemented** |
| Approximate execution contracts | out of scope; execution is exact |

## Tests

```sh
cargo test --manifest-path ola-rt/Cargo.toml -p ola-runtime --lib belief admission adaptive
cargo test --manifest-path ola-rt/Cargo.toml -p ola-bench --test belief
cargo test --manifest-path ola-rt/Cargo.toml -p ola-bench --lib online regime
make verify
```
