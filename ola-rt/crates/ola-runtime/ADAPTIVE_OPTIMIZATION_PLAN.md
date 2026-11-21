# Mathematical adaptive-policy optimization plan

Status: **solver-first adaptive optimization path implemented** (finite oracle, MILP adapter, runtime policy installation, resource constraints, stability controls, full objective metrics, and held-out virtual-time study; 2026-10-27). Online replay optimization and corrected study results added 2026-10-31.

The adaptive route is optimization-first. No learned model is required for
the runtime to adapt, and model training is not a prerequisite for any
milestone below.

## Refactor decision

The existing `AdaptivePolicy` foundation should be retained for candidate-plan
validation and safe fallback, deterministic observation-time updates, arrival
and latency observations, policy attribution, and executor-limit ceilings.

It needs one structural refactor before real optimization: the current
`with_update` hook can mutate θ metadata, but the wrapped inner policy remains
unchanged. An optimizer must propose θ, construct the corresponding policy
components, validate the candidate, and install it only at an observation
boundary.

The target boundary is:

```text
PlanningSnapshot -> current policy -> validated ExecutionPlan
                         ^                    |
                         |------ observe ----|
                                  |
                         observations + constraints
                                  |
                         mathematical optimizer
                                  |
                         candidate θ + policy factory
```

The likely API is a boxed policy factory plus an optimizer:

```rust
trait AdaptivePolicyFactory: Send {
    fn build(&self, theta: &PolicyParameters)
        -> Result<Box<dyn ExecutionPolicy>, PolicyError>;
}

trait AdaptiveOptimizer: Send {
    fn propose(
        &mut self,
        current: &PolicyParameters,
        observation: &AdaptiveObservation,
        constraints: &PolicyConstraints,
    ) -> Result<PolicyParameters, OptimizationError>;
}
```

The first refactor now provides `AdaptivePolicyFactory` and stores the active
policy behind `Box<dyn ExecutionPolicy>`. `AdaptiveOptimizer` and
`AdaptiveConstraints` now define the observation-time solver boundary.
`observe` can propose θ, rebuild the active policy from it, and do so without
changing the plan already committed. The next `plan` validates the rebuilt
candidate before it can reach the executor; construction failures keep the
previous policy, while invalid plans use the existing safe fallback (or return
a typed error when no fallback is present). `plan` remains immutable. Typed
integer domains and automatic slot/chunk ceiling enforcement are now in place;
 memory-demand constraints are now available for fixed plus per-parameter byte
 demand; candidate installation is transactional and solver updates can be
 rate-limited. The runtime chooses MILP as its mathematical programming path;
 LP/convex and stochastic extensions remain optional research variants.

## Mathematical formulation

Start with a bounded mixed-integer candidate space. A candidate contains
discrete and continuous policy variables such as integer token budgets,
backfill limits, discrete scheduler/grouping choices, effective slots/chunk
size, and optional continuous objective weights.

For a trace window `W`, minimize:

```text
J(θ) = w_ttft  · p95(TTFT)
     + w_e2e   · p95(completion latency)
     - w_thr   · throughput
     + w_starve · starvation count
     + w_deadline · deadline misses
     + w_fallback · capability/resource fallbacks
```

Subject to:

```text
1 ≤ effective_slots ≤ executor.max_active_requests
1 ≤ chunk_size ≤ executor.prefill_chunk_size, when capped
memory demand(θ, snapshot) ≤ available capacity
candidate plan validates for every evaluated snapshot
seed, policy identity, and workload window remain reproducible
```

Missing metrics are handled explicitly, not as zeroes. Infeasible candidates
are rejected before they can replace the active policy.

## Solver stages

### Stage 1: exact finite candidate optimizer

Implement a dependency-free optimizer over a finite candidate set. Enumerate
valid combinations, evaluate each on deterministic virtual-time windows, and
select the minimum objective with stable tie-breaking. This is mathematically
equivalent to solving a finite integer program and provides the correctness
oracle for later solver-backed implementations.

### Stage 2: MILP interface

Introduce an internal model representation for binary/integer variables,
linear constraints, and a linear objective. Keep the runtime independent of a
specific solver initially; add a `good_lp` adapter using its pure-Rust
`microlp` backend only after the finite optimizer and conformance tests agree
on solutions. This keeps native solver libraries optional while allowing the
runtime to use a maintained MILP modeling API.

### Stage 3: LP/convex optimization

Use LP or projected convex optimization only for continuous knobs represented
by a valid local model. Continuous proposals remain bounded, seeded, and
subject to candidate validation and fallback.

### Stage 4: stochastic optimization

Allow seeded stochastic exploration among feasible candidates or solver
solutions. Randomness is consumed only in `observe`; `plan` stays a pure
function of the snapshot and installed policy. Exploration must be compared
against the exact finite optimizer on seeded traces.

### Online replay optimization

Online, an optimizer only sees the vector it is running, so it cannot score
the others from live metrics. `ola_bench::ReplayOptimizer` scores them the
way the offline study does: it rebuilds the recent completed workload from
adaptive observations (arrival, prompt length and output limit from the
admitting iteration, tokens generated), replays it once per candidate in the
virtual-time simulator, and switches to the best candidate only if it beats
the current vector's own replay by `min_improvement_percent`.

`AdaptivePolicy` passes the optimizer every observation since its previous
call, not just the current iteration's; with an interval of N, the latter
would miss most admissions and completions.

Replays are as good as the simulator's `CostModel`. In the studies below the
"live" executor is the same simulator, so the model is exact; on hardware it
must come from `cost-calibration`.

`ola serve --adaptive REPORT` installs `ola_bench::serving_adaptive_policy`:
the same grid and objective, starting from the largest budget (which behaves
like the default unbounded batching), replaying with the cost model fitted
in `REPORT`. A synthetic-executor test shows it re-plans mid-run without
changing any output; it has not yet been measured on real hardware.

### Results (2026-10-31)

Both studies share a log-scale token-budget grid: 2^4 to 2^10 tokens per
iteration, 25–100% of it prompt work, 28 candidates. The objective is p95
completion plus p95 time to first token, in virtual time.

An earlier grid allowed only 16–32 tokens. Under the default cost model,
iterations are expensive relative to tokens, so larger batches win, and that
grid excluded the best region: `adaptive-policy-study` then reported its
"adaptive" policy at about 2.4× FIFO's p95 latency. With the wider grid it
selects 256 tokens and ties FIFO (p95 completion 42.3 ms against 41.7 ms).

`online-adaptive-policy-study` runs a mixed short/long phase followed by an
overload burst (80 requests, 8 slots). Objective means, 10 held-out seeds:

| Variant | Objective | Notes |
| --- | --- | --- |
| `fifo` | 73.7 ms | unbounded batching |
| `fixed` | 73.7 ms | best budget for the training trace, chosen with hindsight (512 tokens) |
| `online-warm` | 74.5 ms | online, starting from `fixed`'s budget |
| `online` | 80.0 ms | online, starting from the smallest budget (16 tokens) |

So online replay recovers most of the gap from a bad starting point and
leaves a good one essentially alone, but it does not beat the best fixed
budget: on these traces the best budget does not change between phases, so
there is nothing for adaptation to gain. With a 24-request window and a 5%
margin it switched away from the good budget on noise (`online-warm`
81.3 ms on seeds 1–5); the 64-request default was picked on seeds 1–5 and
held on seeds 6–10 (80.6 and 74.6 ms against 73.0 ms fixed). Online
adaptation will only pay where the best configuration moves with the
workload, for example cost models where per-token work dominates, or knobs
whose best value depends on load.

### Switching cost and regime study (2026-11-17)

`ReplayOptimizer` searches its grid by exhaustive enumeration
(`FiniteCandidateOptimizer`), not with the MILP: replayed p95 latencies are
not linear in θ. It now minimizes
`J(θ) = replay_score(θ) + switch_cost(θ_prev, θ)` (`SwitchingCost`, in
objective microseconds per changed knob) and still requires the
`min_improvement_percent` noise margin against the current vector's own
score; `ReplaySelection` records every term as compared. The default cost is
zero, so the results above are unchanged. `MilpCandidateOptimizer` expresses
the same term exactly with big-M indicators. See `POLICY_STATE.md`.

`regime-shift-study` measures the per-regime optimum before testing
adaptation, on KV-reservation headroom instead of the token budget. The
per-regime optima were not distinguishable, and online replay did not beat
the best static headroom; see `REGIME_STUDY.md`.

`SWITCHING_STUDY.md` (2026-11-21) prices the optimizer's measured host
stall into virtual time (`AdaptationCost`, `run_charged_trial`). It finds
the token budget inert at 4 slots and separating between regimes only at
16. Even there, switching at the arrival boundary does not beat the static
budget, because the backlog, which the replay window ignores, outlives the
regime.

## Verification milestones

1. Add typed parameter domains, constraints, and objective components. (Integer
   domains, objective components, slot/chunk ceilings, and optional memory
   constraints are implemented; richer parameter types remain.)
2. Refactor the wrapper around a policy factory and optimizer trait. (The
   factory and optimizer boundaries are now implemented.)
3. Implement the exact finite candidate optimizer. (Implemented.)
4. Add the `good_lp`/`microlp` MILP adapter and compare it with the finite
   oracle on small domains. (Adapter implemented and compared, including
   memory-constrained candidates.)
5. Wire solver parameters into real runtime components. (The adaptive study
   now selects both total and prefill token budgets; `AdaptiveConstraints::from_snapshot`
   exposes live executor and memory facts to callers.)
6. Test candidate rejection, fallback, deterministic ties, and boundary caps.
   (Implemented for solver and factory paths; rejected factory candidates are
   skipped by offline search and cannot partially update runtime θ.)
7. Evaluate fixed baselines versus adaptive candidates on held-out seeded
   virtual-time traces. (Implemented with train/evaluation separation and
   paired scores, offline and online; see Results. No variant yet beats the
   best fixed configuration.)
8. Add LP/convex adapters only when they reproduce the exact optimizer on small
   cases. (Optional: MILP is the selected production route, and LP/convex
   adapters are not required for the runtime path.)
9. Keep learned policies as an optional later experiment, never as a runtime
   requirement.

## Explicit non-goals

- No neural policy or offline training requirement.
- No optimizer-controlled mutation during `plan`.
- No movement of executor slot/chunk ceilings into policy plans.
- No solver dependency before the dependency-free mathematical baseline is
  validated.
