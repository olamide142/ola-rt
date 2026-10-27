# Mathematical adaptive-policy optimization plan

Status: **solver-first adaptive optimization path implemented** (finite oracle, MILP adapter, runtime policy installation, resource constraints, stability controls, full objective metrics, and held-out virtual-time study; 2026-10-27).

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
   paired scores; broader experiment coverage remains.)
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
