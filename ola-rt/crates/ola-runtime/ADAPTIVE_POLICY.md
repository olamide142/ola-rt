# Adaptive policy design note

Status: **foundation implemented; research loop remains backlog** (2026-10-26).
The wrapper, fallback validation, decision attribution, arrival context, and
seeded observation state are implemented. Online tuning and offline evaluation
remain tracked in `todo.md` under *Adaptive execution policy*. Builds on
`EXECUTION_POLICY.md` and `SCHEDULING_BASELINES.md`.

The solver-first implementation plan is in
[`ADAPTIVE_OPTIMIZATION_PLAN.md`](ADAPTIVE_OPTIMIZATION_PLAN.md).

## Goal

Tune policy configuration online, from the workload the runtime is actually
serving, without changing the snapshot → policy → plan structure. An adaptive
policy is an ordinary `ExecutionPolicy` that holds a configuration θ, plans
with it, and updates it from committed iterations.

## The loop

1. **Observe the workload.** Each iteration's `PlanningSnapshot` is the
   current workload state, and the features an optimizer conditions on:
   - queue composition: prompt lengths, output limits, priorities,
     deadlines, and waits (`at − queued_at`);
   - live prefill and decode work;
   - memory and reservations;
   - `pressure()`.
2. **Act with the current configuration.** An `AdaptivePolicy` wrapper holds
   θ and an inner `ComposedPolicy` built from it. θ covers the policy-side
   knobs:
   - token budget and prompt cap;
   - cost-aware grouping overhead;
   - scheduler choice and aging interval;
   - backfill's skip limit;
   - effective slot count and chunk size, at or below the executor's limits.

   `plan(&self, &snapshot)` plans with the inner policy. Rebuilding the inner
   policy belongs in `observe`, not in every `plan` call.
3. **Measure and update.** After each committed iteration,
   `observe(&mut self, &iteration)` receives the plan, the `DecisionRecord`,
   the actual `execution_batches`, `divergences`, and events. The policy
   accumulates a reward over a window, for example p95 time to first token
   plus a throughput penalty, and updates θ. Candidate methods: a bandit over
   discrete configurations, SPSA or stochastic gradient over continuous
   knobs, or Bayesian optimization between windows.
4. **Changes land only at iteration boundaries.** `plan` takes `&self` and
   only `observe` mutates, so θ never changes in the middle of planning.

## Properties that already help

- **Safety, with a fallback.** `ExecutionPlan::validate` rejects any plan a
  bad θ produces before anything changes state; since `dad4593` this
  includes partial prompt rows on non-resumable models and rows over the
  chunk size. A rejected plan, however, makes `step_iteration` return an
  error and the executor does not advance; in the P3 harness that aborts the
  trial. The wrapper must therefore validate its own candidate plan
  (`validate` is public and the wrapper has the snapshot) and fall back to a
  known-safe configuration when it fails. Validation guards correctness,
  not quality, so exploration still needs a bounded region of θ.
- **Reproducibility.** `plan(&self)` must not consume randomness. A seeded
  RNG that advances only in `observe(&mut self)` is repeatable given the
  seed and the iteration history; so is deriving randomness from the seed
  and `snapshot.generation`.
- **Attribution, if the wrapper records itself.** `DecisionRecord.policy` is
  whatever descriptor the planning policy writes. Delegating to an inner
  `ComposedPolicy` records the inner settings (which include tuned values
  such as `max_tokens`) but not the wrapper or its learning state, while
  `Executor::policy_descriptor()` reports the wrapper. The wrapper should
  write its own descriptor, carrying θ and an update counter, into the
  record so `PlanReport` shows exactly which θ produced each iteration.
- **Offline tuning first.** `run_scheduling_trial` accepts any
  `ExecutionPolicy` and the executor calls `observe` on every committed
  iteration, so an optimizer can be pretrained and evaluated in reproducible
  virtual time against the FIFO, priority, SJF, EDF, and backfill baselines
  before running online.

## Remaining gaps

1. **Measurement and device inputs are empty.** `PolicyMeasurements` waits
   for P4 calibration and `DeviceSnapshot` for P6 telemetry, so cost- and
   device-aware tuning is not yet possible.
2. **Kernels and model placement are not in the plan.** The kernel track and
   P8 must add them before they can be tuned.

## Backlog

See `todo.md`, *Adaptive execution policy*.
