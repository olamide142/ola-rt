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
- [x] Represent model selections. (Backend/device and kernel selections moved to the kernel track.)
- [x] Record token, memory, padding, and invocation-cost estimates. (Grouping cost estimates arrive with the P2 decision record.)
- [x] Record deferrals and policy reasons. (Policy reasons arrive with the P2 decision record.)
- [x] Validate plans before lifecycle, session, or KV mutation.
- [x] Preserve deterministic events when execution is regrouped.
- [x] Add a stable serialized/report representation.
- [x] Test duplicate, stale, invalid, and over-capacity plans.
- [x] Compare planned execution against the single-request correctness path.

Done when one iteration can be explained by a complete, validated plan before model execution begins. **Met 2026-10-16.**

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
- [x] Test scheduler, admission, batching, and memory conflicts. (Kernel-selection conflicts moved to the kernel track.)

Done when existing FIFO, memory admission, token budgeting, and grouping run through the unified policy path without output or lifecycle regressions. **Met 2026-10-16.**

## P3: Scheduler and policy baselines

Design: `ola-rt/crates/ola-runtime/SCHEDULING_BASELINES.md`.

- [x] Preserve FIFO as the correctness baseline.
- [x] Add priority scheduling.
- [x] Add deadline-aware scheduling.
- [x] Add estimated-size or shortest-job-first scheduling.
- [x] Add a memory/cache-aware baseline. (Memory-aware admission with backfill.)
- [x] Add configurable prefill/decode mixing. (`MixedTokenBudgetPolicy`.)
- [x] Compare fixed and adaptive token budgets.
- [x] Define starvation and fairness measurements.
- [x] Test cancellation, failure, and memory-pressure isolation.
- [x] Add mixed short/long prompt traces.
- [x] Add prefill/decode competition traces.
- [x] Add overload and impossible-request traces.
- [x] Compare queue latency, TTFT, completion latency, throughput, memory, fairness, and deferrals.

Done when FIFO, priority, estimated-cost, and deadline-aware policies share one reproducible comparison protocol. **Met 2026-10-16.**

## P4: Framework-grade experiments and reports

Design: `ola-rt/crates/ola-bench/EXPERIMENTS.md`.

- [x] Define generic experiment identity, configuration, workload, protocol, sample, summary, and metadata types.
- [x] Retain raw measurements alongside derived summaries.
- [x] Record model, device, revision, build, policy, warm-up, repetitions, and workload identity. (Reports have a kernel field; populating it is in the kernel track.)
- [x] Generalize versioned persistence beyond prefill reports.
- [x] Keep report creation non-overwriting by default.
- [x] Validate schema versions and sample consistency.
- [x] Compare reports across models, devices, policies, kernels, and revisions.
- [x] Report regressions and uncertainty rather than relying on one run.
- [x] Expand calibration to multiple shapes and confidence intervals.
- [x] Make UI benchmark and experiment catalogs data-driven.
- [x] Run the same named experiments through the CLI and UI.
- [x] Separate simulation-time metrics from host performance.
- [x] Label operational probes versus publication-quality experiments.

Done when a new policy or kernel comparison requires neither bespoke UI code nor a one-off report schema. **Met 2026-10-16.**

## Adaptive execution policy

Optional research track, outside the milestone sequence. Online tuning of
policy configuration. Design note:
`ola-rt/crates/ola-runtime/ADAPTIVE_POLICY.md`. Depends on P4 calibration for
cost-aware knobs.

- [ ] Add an `AdaptivePolicy` wrapper holding configuration θ and an inner `ComposedPolicy`.
- [ ] Validate candidate plans inside the wrapper and fall back to a known-safe configuration.
- [ ] Record the wrapper's descriptor, θ, and update counter in every `DecisionRecord`.
- [ ] Give `observe` arrival times (or the planning snapshot) so rewards can use TTFT and completion latency.
- [ ] Decide whether slot count and chunk size stay executor ceilings or move into the plan.
- [ ] Keep randomness seeded and consumed only in `observe`.
- [ ] Pretrain and evaluate offline in the P3 virtual-time harness against the baselines.

Done when an adaptive policy improves a chosen reward over the best fixed baseline on held-out seeded traces without validation failures.

## P5: Runtime driver and serving

Design: `ola-rt/crates/ola-serve/SERVING.md`.

- [x] Define an asynchronous runtime/executor driver.
- [x] Add request handles for status, cancellation, tokens, and final results.
- [x] Stream tokens without polling retained executor results.
- [x] Propagate client cancellation into session and KV cleanup.
- [x] Define bounded queues and backpressure.
- [x] Define result retention and eviction.
- [x] Translate a process monotonic clock into OlaRT timestamps.
- [x] Expose process health.
- [x] Expose readiness to accept work.
- [x] Expose loaded-model state.
- [x] Add a minimal HTTP generation endpoint.
- [x] Add streaming HTTP responses.
- [x] Add an OpenAI-compatible boundary after the native API stabilizes.
- [x] Keep transport, runtime, model, admission, and execution errors distinct.
- [x] Add graceful shutdown and in-flight request handling.
- [x] Test streaming, cancellation, overload, failure, and shutdown.

Done when a client can submit, observe, stream, cancel, and retrieve a request without accessing `Executor` directly. **Met 2026-10-16.**

## P6: Observability closure

Design: `ola-rt/crates/ola-observe/OBSERVABILITY.md`.

- [x] Retain per-token timestamps. (Stamped when the producing invocation finishes.)
- [x] Report real TPOT distributions instead of a terminal-span average.
- [x] Record scheduler, admission, batching, and memory choices. (Per request, in traces.)
- [x] Record model-operation latency with documented boundaries.
- [x] Add automatic GPU memory telemetry. (CubeCL allocator; validated on CubeCL's CPU runtime, GPU unobserved here.)
- [x] Add GPU utilization and device-activity telemetry. (CubeCL utilization counters and a portable busy fraction.)
- [x] Add stable request, iteration, and plan correlation IDs.
- [x] Add structured tracing for CLI, server, and experiments.
- [x] Keep observation read-only with respect to policy and runtime state.

Kernel, prefix-cache, and model-residency instrumentation belong to the milestones that create those capabilities (kernel track, P7, P8).

Done when a request can be traced from arrival through every decision, execution batch, token, and terminal result, with per-token timestamps and correlation IDs, and the traces are covered by tests. **Met 2026-10-17.**

## P7: Memory and cache policy

Design: `ola-rt/crates/ola-runtime/MEMORY_POLICY.md`.

- [x] Reclaim unused worst-case reservations as requests approach completion. (By not reserving them: watermark admission with preemption.)
- [x] Define cache eviction policy and observable reasons. (LRU of unreferenced cached blocks, only when an allocation needs space; counted.)
- [x] Add safe request preemption. (Recompute; outputs identical to uninterrupted runs on the synthetic model and tiny Llama.)
- [ ] Evaluate swapping after eviction and preemption semantics stabilize.
- [x] Add prefix detection and prefix-cache reuse.
- [x] Add prefix-cache hit, reuse, and eviction metrics.
- [x] Track prefix ownership, lifetime, accounting, and isolation. (Reference counts, LRU lifetime, verified matches, full immutable blocks; reservations stay conservative.)
- [x] Define recovery when a planned batch cannot obtain its KV allocation. (Requeue as a preemption unless the request can never fit.)
- [ ] Add fragmentation-aware admission or placement experiments.
- [ ] Test cache pressure, preemption, cancellation, and recovery.
- [ ] Preserve deterministic policies for workload replay.

Done when, under a replayed cache-pressure trace, the runtime reclaims, evicts, or preempts with recorded reasons instead of deferring indefinitely or rejecting; prefix reuse produces identical outputs with measured savings; and the pressure, preemption, cancellation, and recovery tests are deterministic.

## P8: Model and runtime usability

- [ ] Add probabilistic sampling with deterministic seeded tests.
- [ ] Support indexed and sharded SafeTensors checkpoints.
- [ ] Support selected tied-embedding configurations.
- [ ] Support selected RoPE-scaling configurations.
- [ ] Decide support for projection biases and activation variants.
- [ ] Add a second architecture through the architecture registry.
- [ ] Define model load, unload, placement, and residency operations.
- [ ] Report model residency and load/unload events.
- [ ] Account for model weights, workspace, and non-KV memory automatically.
- [ ] Load configuration for models, devices, limits, policies, and experiments.
- [ ] Add a stable high-level runtime builder after policy boundaries settle.

Done when a second architecture and a sharded checkpoint load through the registry and pass conformance tests, seeded sampling is reproducible, and a runtime can be built from a configuration file.

## Kernel follow-up track

Route kernel choices through framework plan, policy, capability, and benchmark boundaries rather than embedding them directly in Llama.

- [ ] Finish and validate the CubeCL paged-attention extension.
- [ ] Define kernel capabilities and selection constraints.
- [ ] Represent backend/device and kernel selections in `ExecutionPlan`.
- [ ] Test kernel-selection conflicts with scheduling, admission, batching, and memory policy.
- [ ] Add provider selection and portable fallback.
- [ ] Compare every optimized implementation with the portable reference.
- [ ] Fuse page traversal, score reduction, and value accumulation.
- [ ] Optimize request-local tensor-batched decode attention.
- [ ] Investigate ragged multi-request decode attention.
- [ ] Investigate ragged multi-position prefill chunks.
- [ ] Benchmark page-size, occupancy, and fragmentation trade-offs.
- [ ] Compare optimized paths against the single-request oracle.
- [ ] Record kernel identity, target, configuration, and fallback in reports.
- [ ] Record kernel choices and kernel latency in observability traces.
- [ ] Require warm-ups, repetitions, distributions, workload identity, and numerical conformance before making performance claims.

Done when policy selects an optimized paged-attention kernel with a portable fallback, it matches the reference within tolerance, and a study-grade report measures its effect.

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
- [ ] Persistent request results
- [ ] Learned cost models and policies
- [ ] Model-independent graph/model IR and compiler passes

## Working rules

- Keep authoritative lifecycle mutation inside the runtime.
- Make policies consume state and return decisions.
- Validate plans before model or memory work.
- Keep sessions, block tables, cancellation, and results request-local.
- Preserve a portable correctness path for every optimized path.
- Require output or numerical equivalence in comparisons.
- Keep backend tensors out of the public runtime API unless research evidence requires changing that boundary.
- Record limitations and negative results.
- Update this TODO and `progress_report.md` with every committed step, not only when milestones change.

## Milestone order

Section numbers are stable identifiers; this list is the order of work.

1. P0: Stabilize and checkpoint the current tree.
2. P1: Introduce the small execution-plan IR.
3. P2: Route existing behavior through a composable execution policy.
4. P3: Add comparable scheduler and policy baselines.
5. P4: Generalize experiments, reports, and the workbench catalog.
6. P5: Add the asynchronous streaming service boundary.
7. P6: Complete observability needed to explain policy outcomes.
8. P7: Add memory and cache policies.
9. Kernel track: Use the framework to select and evaluate optimized kernels.
10. P8: Expand model support, then move toward distributed and multimodal research.

The adaptive execution policy is an optional research track. It became
available after P4 and blocks no milestone above.
