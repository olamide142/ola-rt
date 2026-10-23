# Kernel routing and optimization design

Status: **accepted** (2026-10-20). Covers `todo.md` Kernel follow-up track.

## Goal

Policy selects an optimized paged-attention kernel with a portable fallback,
the chosen kernel matches the reference within tolerance, and a study-grade
report measures its latency, throughput, and memory effects across workloads
and page sizes.

Kernel decisions must route through framework plan, policy, capability, and
benchmark boundaries rather than remaining hardcoded inside individual model
implementations.

## What existed

- **A narrow kernel contract exists.** `ola-kernel` defines
  `PagedAttentionKernel<B>`, `PagedAttentionConfig`, `PagedKvPage`, and
  `PagedAttentionInput`. Shape validation and GQA checks happen at the input
  boundary.
- **A portable reference exists.** `PortablePagedAttention` computes online
  softmax across logical KV pages using Burn tensor operations without
  materializing sequence-sized temporary tensors. It serves as the numerical
  baseline.
- **An experimental CubeCL implementation exists.** `CubePagedAttention` in
  `ola-kernel/src/cubecl_extension.rs` provides WGPU and CUDA execution
  paths via Burn's CubeCL/Fusion backend extension. However:
  1. It concatenates page tensors (`BurnTensor::cat`) into a contiguous
     tensor before dispatch to work around CubeCL 0.11 sequence limits.
  2. The kernel recomputes query-key scores for every output dimension.
  3. `ola-kernel/src/cubecl_paged_attention.rs` is a tracked but unwired
     alternative attempting per-page bindings.
- **Kernel selection is embedded in Llama.** In `ola-burn/src/llama/model.rs`,
  decode attention checks compile-time `#[cfg(feature = "wgpu")]` and
  `#[cfg(feature = "cuda")]` flags and hardcodes `CubePagedAttention::new`,
  falling back to `PortablePagedAttention`. The runtime planner and execution
  policy have no visibility or control over this selection.
- **Plans and traces omit kernel choices.** `ExecutionPlan` in `ola-runtime`
  records `model_id` and `BatchExecutionMode` (Tensor vs. Individual), but not
  the target device or kernel selection. `TraceEvent::Invocation` records
  phase, mode, tokens, and padding, but no kernel descriptor or kernel latency.
- **Reports reserve a kernel field.** `ola-bench::Variant` has an optional
  `kernel: Option<String>` field and report comparison checks it, but all
  built-in benchmarks set it to `None`.

## Architectural decisions

1. **Kernel capabilities and selection constraints.**
   Define `KernelCapability` reflecting supported execution targets (`Portable`,
   `Cpu`, `Wgpu`, `Cuda`), data types (`f32`, `f16`, `bf16`), head dimensions,
   GQA ratios, and maximum page limits. Kernel providers register or declare
   their capabilities so the runtime can evaluate feasibility deterministically.

2. **Route kernel selection through `ExecutionPlan` and `ExecutionPolicy`.**
   - Extend `PlannedBatch` in `ExecutionPlan` to carry `KernelSelection`
     (consisting of the selected `KernelDescriptor` or an explicit portable
     fallback indicator) for decode and resumable-prefill work that actually
     invokes paged attention. Dense full-prompt prefill carries no such choice.
   - `ExecutionPolicy` evaluates the snapshot (device, model, batch shape,
     dtype) and selects the appropriate kernel implementation.
   - If an optimized kernel is unavailable, unsupported, or disabled by policy,
     the plan explicitly falls back to `PortablePagedAttention` with a
     documented reason.

3. **Decouple model execution from concrete kernel instantiation.**
   Models in `ola-burn` must not instantiate kernels directly through compile-time
   feature gates. Instead, model execution consumes the planned kernel or queries
   a kernel provider configured on the model or execution context. The portable
   Burn implementation remains available unconditionally as the fallback and oracle.

4. **Eliminate device-side page packing in CubeCL paged attention.**
   Refactor `CubePagedAttention` to eliminate `BurnTensor::cat` on every
   step. KV pages should be indexed either through a device-resident page table
   buffer or structured batch bindings. Fuse the score reduction (online softmax)
   and value accumulation into a shared-memory or register-tiled kernel pass.

5. **Multi-request and multi-position operations.**
   - Decode: Move from request-local page loops to batched decode attention
     where multiple sequences in a tensor batch share a single kernel launch.
   - Resumable prefill: Investigate ragged multi-position chunk execution rather
     than stepping one offset at a time.

6. **Observability and report integration.**
   - Update `TraceEvent::Invocation` in `ola-observe` to record the selected
     `KernelDescriptor` and kernel launch latency.
   - Populate `Variant::kernel` in `ola-bench` and record kernel target and
     conformance metrics in `ExperimentReport` (`ola-experiment/1`).

7. **Empirical benchmarking and validation standard.**
   - Compare every optimized implementation against `PortablePagedAttention`
     and the scalar reference across single/multi-page boundaries, partial final
     pages, and GQA configurations.
   - Measure warm-up, kernel compilation overhead, and steady-state latency
     separately.
   - Benchmark page-size tradeoffs (4, 8, 16, 32, 64 tokens) with real kernel
     dispatch costs, completing the empirical analysis initiated in P7.

## Implementation steps

1. **Kernel capabilities and provider selection.** **Complete 2026-10-20.**
   Define `KernelCapability`, `KernelSelection`, and a `KernelRegistry` /
   provider abstraction in `ola-kernel`. Capability queries now match target,
   dtype, head dimension, GQA ratio, and logical page count. Provider priority
   and stable-name tie-breaking make selection independent of registration
   order; incompatible optimized providers produce a typed portable-fallback
   reason. Backend-neutral telemetry types moved from `ola-observe` to
   `ola-core`, removing the dependency cycle that would otherwise prevent
   `ola-runtime` from consuming kernel selections.
2. **Execution-plan IR and policy representation.** **Complete 2026-10-21.**
   Applicable `PlannedBatch` values now carry exact `KernelRequirements` and
   the selected provider. Models publish a `PagedAttentionProfile`; policy
   derives page count only after scheduling, admission, batching, and grouping
   settle decode or resumable-prefill rows. Dense full-prompt prefill remains
   unannotated. Plan validation recomputes requirements, `PlanReport` v6 records
   target/dtype/configuration/provider/fallback details, and conflict tests show
   deferred or budget-excluded work cannot distort selection.
3. **Decouple `ola-burn` model execution.** **Complete 2026-10-22.**
   `KernelSelection` retains the selected provider factory behind its stable
   identity. The runtime passes it through `ModelExecutionContext` for decode
   and resumable prefill; Llama consumes it in individual and tensor-batched
   paths, while direct session calls use the portable correctness provider.
   Models that do not implement delegation reject a supplied choice instead of
   silently ignoring it. Compiled CubeCL providers are registered by the
   default composed policy, and a counting-provider integration test proves the
   planned implementation reaches tensor decode.
4. **Observability and benchmarking closure.**
   Emit kernel identity and kernel timing in `ola-observe::TraceEvent::Invocation`.
   Populate `Variant::kernel` in `ola-bench` experiment runs and report
   comparisons.
5. **Optimize CubeCL paged attention.**
   Eliminate pre-dispatch tensor concatenation in `CubePagedAttention`. Implement
   fused page traversal, online softmax reduction, and value accumulation.
   Clean up or remove obsolete unwired prototypes.
6. **Conformance and study-grade evaluation.**
   Implement conformance tests across diverse page counts and head configurations.
   Execute an empirical kernel comparison benchmark in `ola-bench` evaluating
   portable versus CubeCL execution across page sizes and batch sizes.

## Out of scope

Quantized KV caches, FlashAttention-3 specific hardware intrinsics, and
distributed multi-node KV routing (deferred to later inference-engine work).
