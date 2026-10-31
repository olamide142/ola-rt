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

## 13. Suggested additions (review of 2026-10-31)

Items not covered above, each tied to the paper section, research question
(RQ) or hypothesis (H) it serves. The first group is what §43 Results needs
before any of sections 4–7 pays off; most of it runs in virtual time on the
laptop.

### Evidence the paper needs first

- [ ] **SLO-constrained goodput metric.** H2 and RQ3 are stated in goodput,
  but no study reports it. Add per-request TTFT/TPOT targets to `TraceRequest`
  and report goodput and SLO attainment as `Metric`s (`ola-bench/src/experiment/metric.rs`)
  and as an `AdaptiveObjectiveMetrics` objective.
- [ ] **Workloads where the optimum moves** (§37.5–37.9): step shift in
  arrival rate, prompt-distribution shift, output-distribution shift, burst,
  and hardware degradation (scale the `CostModel` mid-trace), as
  `SchedulingTrace` constructors next to `phased_trace` (`ola-bench/src/trace.rs`).
  The online study so far found no adaptation benefit because the best fixed
  budget never changes on its traces; H1 has to be shown before H2 can be.
- [ ] **Precondition check for H1.** A study that sweeps the static grid per
  regime and reports whether the per-regime argmin differs (with intervals).
  Run it before any adaptive-versus-static comparison.
- [ ] **Missing baselines** (§38) as variants of `online-adaptive-policy-study`:
  B3 per-regime oracle static policy (regime known, best fixed budget per
  regime), B4 rule-based threshold controller (queue depth or prompt length
  → budget). B4 separates "adapting at all" from "optimizing well".
- [ ] **Cost-model error injection** (§39.5, RQ5, H5): a seeded
  `PerturbedCostModel` wrapper with T̂ = T(1 + ε), ε ~ N(0, σ²) or bounded
  adversarial, given to the optimizer while the simulator keeps the true model.
- [ ] **Ranking accuracy, not just absolute error** (H5): report Kendall's τ
  between the optimizer's predicted candidate order and the order measured by
  replay with the true model. H5 claims ordering is enough; measure ordering.
- [ ] **Oracle output lengths for the predictor ablation** (§39.6, H7): the
  simulator knows each request's true length, so "oracle" and "mean" predictors
  are cheap; "interval" needs the `BeliefState` from section 6.
- [ ] **`--cost CALIBRATION.json` on virtual-time studies** (RQ10): load a
  `CostFit` the way `ola serve --adaptive` does, so each GPU's fitted costs
  drive the same study. First result already in the lab: the cost-aware grouper
  picks PowerOfTwo at the CPU ratio (≈64 tokens/call) and Unbucketed at the
  GPU ratio (≈207). Note that the lab notebook labels this "H3", which is not
  the paper's H3; align lab labels with the paper's H1–H7.

### Runtime pieces the formalism assumes

- [ ] **Online model update** (§27 step 7): refit `CostFit` from executed
  iterations (recursive least squares over calls and tokens) and feed it to
  `ReplayOptimizer`. Today the cost model is fixed after offline calibration.
  Also fix the CPU fit's misspecification (negative fixed cost).
- [ ] **Switching cost in the objective** (§15, H6, RQ6): replace
  `min_improvement_percent` hysteresis with λ_sw·C_switch(aₙ₋₁, a) in
  `ReplayOptimizer`, and run λ_sw = 0 versus > 0 while sweeping how often the
  workload shifts. The existing `switches()` counter is the thrashing metric.
- [ ] **Confidence predicate** (§35): fall back to π_safe when the replay window
  is too small or recent prediction error exceeds a threshold, and record why.
- [ ] **Action ablation harness** (§39.2, RQ4): freeze any `ComposedPolicy`
  component (scheduler, admission, batch, grouping, preemption, kernels) at a
  fixed choice, so a study can run batching only, then + chunking, then + KV,
  then + kernel.
- [ ] **State ablation harness** (§39.1): mask `PlanningSnapshot` fields a
  policy may read (queue, KV, hardware, timing) and measure the loss.
- [ ] **Expressiveness suite** (§25, RQ1): express Orca, vLLM, Sarathi-Serve,
  FastServe-style preemption, DistServe and Llumnix as named OlaRT policy
  configurations, with tests asserting their characteristic decisions. This
  would be the paper's "existing techniques are restricted policy classes"
  table, backed by code.
- [ ] **Mechanism registry listing** (§31.2): `ola mechanisms` prints every
  scheduler, admission, batch, grouping, preemption and kernel alternative
  with its capabilities, from the same registry the policies select from.
- [ ] **Declarative policy spec** (§32): a TOML or JSON policy file (state
  read, action domains, constraints, objective, timescale, fallback) compiled
  into an `AdaptivePolicy`'s domains and objective, starting with the
  token-budget policy.
- [ ] **Precision as an action** (§8.9, §9): fp16/bf16 execution in `ola-burn`
  (weights are cast to f32 today), with a distributional-equivalence contract
  checked against f32 outputs. Also needed to run 7B models (about 29 GiB in
  f32).

### Hardware and generality (RQ8, RQ10)

- [ ] **Hardware identity in reports**: GPU model, memory, driver, CUDA
  version, compute capability, CPU, host. `RunMetadata.device` is only a label
  today, so reports from different GPUs cannot be told apart.
- [ ] **Run the CUDA path once.** It compiles but has never executed (no NVRTC
  on the laptop); section 2's "CUDA build paths" is accurate only for building.
  Add CUDA to `paged-attention-kernel` too (WGPU and CPU only today).
- [ ] **Vast.ai bootstrap**: a Dockerfile on `nvidia/cuda:*-devel` with a
  prebuilt `ola --features cuda`, a suite script (smoke test, calibration,
  generation, grouping and policy studies with the fitted cost), and
  `lab/VAST.md`. Test the image locally with `nvidia-container-toolkit` first.

### Paper consistency

- [ ] Update §31 of the paper from "proposed runtime" to an implementation
  status table (what exists, what is simulation-only, what is missing).

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
