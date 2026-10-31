# OlaRT Optimization Roadmap

Last reviewed: 2026-10-31

This roadmap aligns OlaRT with `Formalizing Inference Execution Policies: A Unified Control Framework for Adaptive Generative Model Serving`.

The repository implements a strong deterministic and adaptive foundation, but not yet a full stochastic-programming controller.

## Status legend

- `[x]` implemented and tested.
- `[~]` partially implemented or simulation-only.
- `[ ]` planned research or engineering work.

## 1. Implemented IEP foundation

- [x] Separate logical model semantics from physical execution plans.
- [x] Use immutable state snapshot -> policy decision -> validated plan -> execution -> observation.
- [x] Compose scheduler, admission, batching, prefill, memory, preemption, kernel, and fallback policies.
- [x] Preserve deterministic decisions for identical snapshots and configuration.
- [x] Record policy identity, inputs, alternatives, estimates, outcomes, and reasons.
- [x] Validate plans before lifecycle, model, or KV mutation.
- [x] Keep authoritative runtime state outside policy code.
- [x] Provide portable correctness paths for optimized execution.
- [x] Expose CLI, UI, JSON reports, JSONL traces, and HTTP request traces.

## 2. Current execution capabilities

- [x] FIFO, priority, deadline-aware, shortest-job, and adaptive scheduling.
- [x] Token-budget and prefill/decode mixing policies.
- [x] Chunked and tensor-batched prefill.
- [x] KV reservation, pressure handling, preemption, recompute recovery, and prefix caching.
- [x] Llama and Mistral architecture implementations.
- [x] Portable and CubeCL paged attention with numerical comparison.
- [x] CPU, WGPU, and CUDA build paths.
- [x] Runtime driver, request handles, cancellation, backpressure, streaming, retention, and graceful shutdown.
- [x] Per-token timing, device telemetry hooks, kernel identity, and plan correlation IDs.

## 3. Current distributed capabilities

- [x] In-process multi-worker coordination and deterministic simulation.
- [x] TCP workers, gateway routing, heartbeats, leases, readiness, and health.
- [x] Adaptive routing using load, KV headroom, prefix affinity, and calibrated network cost.
- [x] Model placement and capacity-constrained batch routing.
- [x] Request migration with source commit/abort and target acknowledgement.
- [x] Architecture-owned Llama/Mistral KV snapshot transfer.
- [x] Binary attachment frames for large session snapshots.
- [x] Prefill, decode, and mixed worker roles.
- [x] Disconnect recovery, stale-token suppression, and cancellation.
- [x] Gateway text streaming and usage accounting.
- [~] Automatic role scheduling between prefill and decode workers.
- [~] Broader live workload comparison for adaptive cluster routing.
- [ ] Pipeline parallelism across workers; research-only D6.

## 4. Formal IEP state model

The paper defines state across requests, queues, KV, hardware, topology, telemetry, plans, workload beliefs, and service obligations. OlaRT currently has these fields in several snapshots rather than one unified formal state.

- [ ] Define a first-class `IEPState` composed from runtime and cluster state.
- [ ] Include request lifecycle, queue, phase, KV, device, topology, plan, telemetry, SLO, and policy-history state in one documented schema.
- [ ] Add explicit history and finite-memory controller state.
- [ ] Add output-length beliefs for active requests rather than only observed generated-token counts.
- [ ] Add workload-regime and distribution-shift indicators.
- [ ] Define which state fields each policy observes and which remain hidden.

## 5. Action space and execution contracts

- [x] Admission, routing, batching, scheduling, chunking, preemption, KV allocation, migration, kernel selection, and fallback actions.
- [~] Model placement and phase-disaggregation actions.
- [ ] Make execution actions a typed, inspectable action vector.
- [ ] Define a first-class feasibility predicate covering memory, compute, bandwidth, topology, service, and semantic constraints.
- [ ] Define execution contracts for bitwise, numerical, distributional, and task-level equivalence.
- [ ] Record contract level and allowed quality/performance tradeoff in every decision.
- [ ] Add explicit switching costs for migration, cache invalidation, recompilation, graph recapture, and role changes.

## 6. Uncertainty and stochastic programming

Current experiments use seeded traces, held-out evaluation, deterministic MILP-style candidate selection, and feedback adaptation. They do not yet solve a formal stochastic program.

- [ ] Define random variables for arrivals, output lengths, execution time, cache hits, failures, and network conditions.
- [ ] Add a `BeliefState` for partially observed output-length and workload uncertainty.
- [ ] Define reproducible scenario sets with probabilities.
- [ ] Add two-stage decisions: here-and-now admission/placement followed by scenario-dependent recourse.
- [ ] Enforce non-anticipativity so decisions cannot use future information.
- [ ] Add expected-cost, worst-case, robust, and risk-adjusted objectives.
- [ ] Add chance constraints for memory safety and SLO attainment probabilities.
- [ ] Add CVaR or another explicit tail-risk objective.
- [ ] Add infeasibility and uncertainty fallback policies.
- [ ] Compare stochastic policies against deterministic, robust, and oracle baselines.

## 7. Optimization methods

- [x] Finite candidate search and deterministic MILP-backed placement/routing.
- [x] Transactional adaptive policy installation with safe fallback.
- [x] Replay-based optimization over observed workload windows.
- [~] Online adaptive optimization on synthetic or replayed traces.
- [ ] Scenario-based stochastic MILP.
- [ ] Receding-horizon model-predictive control.
- [ ] Queueing-based policy optimization.
- [ ] Robust optimization under bounded model error.
- [ ] Bandit or learned policy experiments behind the same policy interface.
- [ ] Pareto-frontier selection instead of only weighted-sum objectives.

## 8. Performance models and control cost

- [x] Calibrate fixed, per-token, per-call, KV, and network costs where supported.
- [~] Use calibrated costs in more policy decisions and live routing.
- [ ] Define a common state/action -> metric prediction interface.
- [ ] Track prediction error and confidence intervals per operating region.
- [ ] Detect out-of-distribution states and fall back conservatively.
- [ ] Measure telemetry, state construction, optimization, validation, plan transition, and switching overhead.
- [ ] Include switching cost in objective evaluation.
- [ ] Add controller latency and resource overhead to experiment reports.

## 9. Multi-timescale policy hierarchy

- [x] Iteration-scale scheduling, batching, memory, and kernel decisions.
- [x] Request-scale admission, cancellation, migration, and routing decisions.
- [~] Cluster-scale placement, worker roles, and link-aware routing.
- [ ] Define deployment-scale model residency, device allocation, and scaling.
- [ ] Define ownership and coordination boundaries between policy levels.
- [ ] Add hierarchical handoff rules and conflict resolution.
- [ ] Add switching budgets and hysteresis to prevent policy thrashing.

## 10. Evaluation program

- [x] Compare fixed policies on seeded virtual-time workloads.
- [x] Evaluate adaptive policies on held-out traces.
- [x] Compare distributed routing with simulated link costs.
- [~] Evaluate adaptive routing on live loopback worker/gateway workloads.
- [ ] Add static-versus-adaptive comparisons across multiple workload regimes.
- [ ] Add uncertainty robustness and distribution-shift experiments.
- [ ] Add hindsight oracle gap and cumulative regret metrics.
- [ ] Add Pareto curves for latency, throughput, memory, fairness, energy, cost, and SLO violations.
- [ ] Measure controller overhead separately from model execution.
- [ ] Verify output, numerical, and execution-contract equivalence for every optimization variant.
- [ ] Persist decision records, scenario definitions, beliefs, constraints, and objective values in experiment reports.

## 11. Later engine and research work

- [ ] Pipeline parallelism across workers.
- [ ] Tensor parallelism across devices or hosts.
- [ ] Quantized weights and KV caches.
- [ ] Speculative decoding.
- [ ] KV swapping to host memory.
- [ ] Heterogeneous CPU/GPU execution.
- [ ] Learned cost models.
- [ ] Model-independent graph/compiler IR.
- [ ] Multimodal and vision-language execution.
- [ ] Autoscaling and elastic resource allocation.

## 12. Immediate implementation order

1. Define `IEPState`, observation boundaries, and the state-to-policy mapping.
2. Add `BeliefState` for output lengths, arrivals, execution time, and cache behavior.
3. Add scenario generation and reproducible scenario reports.
4. Implement a two-stage stochastic admission/routing prototype with recourse.
5. Add chance or CVaR constraints for KV safety and latency SLOs.
6. Add switching costs and controller-overhead measurements.
7. Evaluate deterministic, robust, stochastic, adaptive, and hindsight-oracle policies on the same workloads.
8. Extend the live loopback study to measure adaptive cluster improvement.
9. Revisit automatic prefill/decode role scheduling.
10. Keep pipeline parallelism separate as D6 research.

## Design rules

- Keep logical model semantics separate from physical execution choices.
- Make constraints explicit and validate plans before mutation.
- Keep policies pure during planning; learning occurs only after observation.
- Preserve deterministic replay and seeded experiments.
- Record every decision, alternative, estimate, constraint, and outcome.
- Treat uncertainty and switching cost as first-class optimization inputs.
- Keep safe fallback policies available whenever prediction or optimization fails.
- Require output or numerical equivalence unless an execution contract explicitly permits a quality tradeoff.
- Measure adaptation benefit against controller overhead.

## References

- `Formalizing Inference Execution Policies: A Unified Control Framework for Adaptive Generative Model Serving`
- `ola-rt/crates/ola-runtime/EXECUTION_POLICY.md`
- `ola-rt/crates/ola-runtime/ADAPTIVE_OPTIMIZATION_PLAN.md`
- `ola-rt/crates/ola-cluster/README.md`
- `ola-rt/crates/DISTRIBUTED.md`
- `ola-rt/TESTING_GUIDE.md`
