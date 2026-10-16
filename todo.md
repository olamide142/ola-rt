# OlaRT Framework TODO

Framework-path backlog after the initial paged-attention and CubeCL kernel work.

Last reviewed: 2026-10-16

## P0: Stabilize current work

- [x] Separate framework, benchmark, Burn, UI, and kernel changes into coherent commits.
- [x] Decide whether `.codex/` should be tracked or ignored. (Ignored.)
- [x] Run `make verify`.
- [x] Run WGPU checks and available CUDA feature compilation. (Compile only; live WGPU blocked by the NVIDIA driver.)
- [x] Run tiny-Llama scalar/conformance checks.
- [x] Update `progress_report.md` with the current revision and verification evidence.
- [x] Resolve the missing `IMPLEMENTATION.md` reference.
- [x] Update the stale Phase 0 checklist in `README.md`.

## P1: Execution-plan IR

Keep this runtime-focused rather than starting with a full model/compiler IR.
Design: `ola-rt/crates/ola-runtime/EXECUTION_PLAN.md` (slot refill: option C).

- [x] Define `ExecutionPlan` responsibilities and boundaries.
- [x] Represent selected requests and deterministic ordering.
- [x] Represent prefill, continued-prefill, and decode phases.
- [x] Represent tensor groups and individual fallbacks.
- [x] Represent real input and padding positions.
- [x] Represent request-local KV growth and release actions. (Growth as row positions, reservations on `Admit`, predicted releases; unpredicted releases in divergences.)
- [ ] Represent model, backend/device, and kernel selections. (Model done; device and kernel wait for the kernel track.)
- [x] Record token, memory, padding, and invocation-cost estimates. (Grouping cost estimates arrive with the P2 decision record.)
- [x] Record deferrals and policy reasons. (Policy reasons arrive with the P2 decision record.)
- [x] Validate plans before lifecycle, session, or KV mutation.
- [x] Preserve deterministic events when execution is regrouped.
- [x] Add a stable serialized/report representation.
- [x] Test duplicate, stale, invalid, and over-capacity plans.
- [x] Compare planned execution against the single-request correctness path.

Done when one iteration can be explained by a complete, validated plan before model execution begins. **Met 2026-10-16;** the three open items above carry into P2 and the kernel track.


## P2: Composable execution policy

Design: `ola-rt/crates/ola-runtime/EXECUTION_POLICY.md`.

- [x] Define an immutable workload, lifecycle, model, KV, memory, device, and measurement snapshot. (Device and measurement fields exist but stay empty until P6/P4.)
- [x] Define `ExecutionPolicy` as runtime state to `ExecutionPlan`.
- [x] Compose or adapt `Scheduler`, `AdmissionPolicy`, `BatchPolicy`, and `PrefillBatcher`.
- [x] Keep individual sub-policies independently replaceable.
- [x] Make decisions deterministic for identical state and configuration.
- [x] Record decision inputs, alternatives, estimates, and reasons.
- [x] Add capability checks and explicit fallbacks.
- [x] Prevent policies from mutating authoritative runtime state.
- [x] Give every reportable policy a stable name and configuration.
- [ ] Test scheduler, admission, batching, memory, and kernel-selection conflicts. (All but kernel selection done; that waits for the kernel track.)

Done when existing FIFO, memory admission, token budgeting, and grouping run through the unified policy path without output or lifecycle regressions. **Met 2026-10-16.**

## P3: Scheduler and policy baselines

Design: `ola-rt/crates/ola-runtime/SCHEDULING_BASELINES.md`.

- [ ] Preserve FIFO as the correctness baseline.
- [x] Add priority scheduling.
- [x] Add deadline-aware scheduling.
- [x] Add estimated-size or shortest-job-first scheduling.
- [x] Add a memory/cache-aware baseline. (Memory-aware admission with backfill.)
- [x] Add configurable prefill/decode mixing. (`MixedTokenBudgetPolicy`; adaptive budget policy exists, comparison pending.)
- [ ] Compare fixed and adaptive token budgets.
- [ ] Define starvation and fairness measurements.
- [ ] Test cancellation, failure, and memory-pressure isolation.
- [ ] Add mixed short/long prompt traces.
- [ ] Add prefill/decode competition traces.
- [ ] Add overload and impossible-request traces.
- [ ] Compare queue latency, TTFT, completion latency, throughput, memory, fairness, and deferrals.

Done when FIFO, priority, estimated-cost, and deadline-aware policies share one reproducible comparison protocol.

## P4: Framework-grade experiments and reports

- [ ] Define generic experiment identity, configuration, workload, protocol, sample, summary, and metadata types.
- [ ] Retain raw measurements alongside derived summaries.
- [ ] Record model, device, revision, build, policy, kernel, warm-up, repetitions, and workload identity.
- [ ] Generalize versioned persistence beyond prefill reports.
- [ ] Keep report creation non-overwriting by default.
- [ ] Validate schema versions and sample consistency.
- [ ] Compare reports across models, devices, policies, kernels, and revisions.
- [ ] Report regressions and uncertainty rather than relying on one run.
- [ ] Expand calibration to multiple shapes and confidence intervals.
- [ ] Make UI benchmark and experiment catalogs data-driven.
- [ ] Run the same named experiments through the CLI and UI.
- [ ] Separate simulation-time metrics from host performance.
- [ ] Label operational probes versus publication-quality experiments.

Done when a new policy or kernel comparison requires neither bespoke UI code nor a one-off report schema.

## P5: Runtime driver and serving

- [ ] Define an asynchronous runtime/executor driver.
- [ ] Add request handles for status, cancellation, tokens, and final results.
- [ ] Stream tokens without polling retained executor results.
- [ ] Propagate client cancellation into session and KV cleanup.
- [ ] Define bounded queues and backpressure.
- [ ] Define result retention, eviction, and persistence.
- [ ] Translate a process monotonic clock into OlaRT timestamps.
- [ ] Expose health, readiness, and loaded-model state.
- [ ] Add a minimal HTTP generation endpoint.
- [ ] Add streaming HTTP responses.
- [ ] Add an OpenAI-compatible boundary after the native API stabilizes.
- [ ] Keep transport, runtime, model, admission, and execution errors distinct.
- [ ] Add graceful shutdown and in-flight request handling.
- [ ] Test streaming, cancellation, overload, failure, and shutdown.

Done when a client can submit, observe, stream, cancel, and retrieve a request without accessing `Executor` directly.

## P6: Observability closure

- [ ] Retain per-token timestamps.
- [ ] Report real TPOT distributions instead of a terminal-span average.
- [ ] Record scheduler, admission, batching, memory, and kernel choices.
- [ ] Record kernel and model-operation latency with documented boundaries.
- [ ] Add automatic GPU memory telemetry.
- [ ] Add GPU utilization and device-activity telemetry.
- [ ] Report model residency and load/unload events.
- [ ] Add prefix-cache metrics when prefix caching exists.
- [ ] Add stable request, iteration, and plan correlation IDs.
- [ ] Add structured tracing for CLI, server, and experiments.
- [ ] Keep observation read-only with respect to policy and runtime state.

Done when a request can be traced from arrival through every decision, execution batch, kernel, token, and terminal result.

## P7: Memory and cache policy

- [ ] Reclaim unused worst-case reservations as requests approach completion.
- [ ] Define cache eviction policy and observable reasons.
- [ ] Add safe request preemption.
- [ ] Evaluate swapping after eviction and preemption semantics stabilize.
- [ ] Add prefix detection and prefix-cache reuse.
- [ ] Track prefix ownership, lifetime, accounting, and isolation.
- [ ] Define recovery when a planned batch cannot obtain its KV allocation.
- [ ] Add fragmentation-aware admission or placement experiments.
- [ ] Test cache pressure, preemption, cancellation, and recovery.
- [ ] Preserve deterministic policies for workload replay.

Done when memory pressure can produce an explicit measured action other than indefinite deferral or permanent rejection.

## P8: Model and runtime usability

- [ ] Add probabilistic sampling with deterministic seeded tests.
- [ ] Support indexed and sharded SafeTensors checkpoints.
- [ ] Support selected tied-embedding configurations.
- [ ] Support selected RoPE-scaling configurations.
- [ ] Decide support for projection biases and activation variants.
- [ ] Add a second architecture through the architecture registry.
- [ ] Define model load, unload, placement, and residency operations.
- [ ] Account for model weights, workspace, and non-KV memory automatically.
- [ ] Load configuration for models, devices, limits, policies, and experiments.
- [ ] Add a stable high-level runtime builder after policy boundaries settle.

## Kernel follow-up track

Route kernel choices through framework plan, policy, capability, and benchmark boundaries rather than embedding them directly in Llama.

- [ ] Finish and validate the CubeCL paged-attention extension.
- [ ] Define kernel capabilities and selection constraints.
- [ ] Add provider selection and portable fallback.
- [ ] Compare every optimized implementation with the portable reference.
- [ ] Fuse page traversal, score reduction, and value accumulation.
- [ ] Optimize request-local tensor-batched decode attention.
- [ ] Investigate ragged multi-request decode attention.
- [ ] Investigate ragged multi-position prefill chunks.
- [ ] Benchmark page-size, occupancy, and fragmentation trade-offs.
- [ ] Compare optimized paths against the single-request oracle.
- [ ] Record kernel identity, target, configuration, and fallback in reports.
- [ ] Require warm-ups, repetitions, distributions, workload identity, and numerical conformance before making performance claims.

## Later inference-engine work

- [ ] Quantized weights and KV caches
- [ ] Speculative decoding
- [ ] Multi-GPU placement
- [ ] Tensor and pipeline parallelism
- [ ] Distributed request routing
- [ ] Distributed KV-cache management
- [ ] Heterogeneous CPU/GPU execution
- [ ] VLM and multimodal execution
- [ ] Vision and OlaDetection workloads
- [ ] Learned cost models and policies
- [ ] Model-independent graph/model IR and compiler passes

## Working rules

- [ ] Keep authoritative lifecycle mutation inside the runtime.
- [ ] Make policies consume state and return decisions.
- [ ] Validate plans before model or memory work.
- [ ] Keep sessions, block tables, cancellation, and results request-local.
- [ ] Preserve a portable correctness path for every optimized path.
- [ ] Require output or numerical equivalence in comparisons.
- [ ] Keep backend tensors out of the public runtime API unless research evidence requires changing that boundary.
- [ ] Record limitations and negative results.
- [ ] Update this TODO and `progress_report.md` whenever milestones change.

## Milestone order

1. Stabilize and checkpoint the current tree.
2. Introduce the small execution-plan IR.
3. Route existing behavior through a composable execution policy.
4. Add comparable scheduler and policy baselines.
5. Generalize experiments, reports, and the workbench catalog.
6. Complete observability needed to explain policy outcomes.
7. Add the asynchronous streaming service boundary.
8. Add memory and cache policies.
9. Use the framework to select and evaluate optimized kernels.
10. Expand model support, then move toward distributed and multimodal research.
