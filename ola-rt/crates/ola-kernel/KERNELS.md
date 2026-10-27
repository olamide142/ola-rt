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
- **CubeCL has a fused paged traversal.** `CubePagedAttention` in
  `ola-kernel/src/cubecl_extension.rs` receives one physical K/V pool and a
  device-resident logical-to-physical block table. One CubeCL launch computes
  scores, online softmax, and value accumulation while traversing the block
  table. It does not concatenate the active sequence on each call. Each output
  dimension currently recomputes each score; a future tiled kernel can share
  score work across dimensions.
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
   Complete: Llama stores each layer in flattened physical K/V allocations,
   uploads a validated device block table when a sequence's allocation changes,
   and CubeCL traverses it in one fused dispatch. The portable implementation
   still consumes logical page views. Score work is currently repeated per
   output dimension; sharing it across dimensions remains a kernel optimization.

5. **Multi-request and multi-position operations.**
   - Decode: **Request-local tensor batching complete 2026-10-24.** The
     provider boundary accepts independent inputs; CubeCL stacks queries and
     consumes a padded device block table in one launch, while portable
     providers retain the independent-request fallback.
   - Decode: Ragged logical lengths are validated for one physical KV geometry.
     Each physical KV pool carries an explicit identity, so homogeneous
     same-pool requests retain the fused launch while incompatible or
     cross-pool groups fall back to independent launches safely.
   - Resumable prefill: Active requests at each offset now share the batched
     one-position dispatch, including unequal chunk lengths; runtime/model
     conformance covers unequal absolute cursors. Fusing multiple positions
     into one launch remains a separate contract because each query position
     has a different causal valid length.

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
4. **Observability and benchmarking closure.** **Complete 2026-10-23.**
   Actual `ExecutionBatch` records now retain the provider descriptor, call
   count, and aggregate synchronous host time inside the selected kernel.
   `ola-observe::TraceEvent::Invocation` emits that identity, target, count,
   and duration; model-backed `ola-bench` variants populate `Variant::kernel`
   from provider names observed across the run. The host metric includes device
   work only when a provider synchronizes before returning, so later studies
   must use backend device events before calling it device-kernel latency.
5. **Optimize CubeCL paged attention.**
   **First fused implementation complete 2026-10-23.** Eliminate active-sequence
   concatenation and fuse physical page traversal, score calculation, online
   softmax, and value accumulation into a single CubeCL dispatch. Retain the
   device-resident flattened K/V pool and block-table layout. The old per-page
   binding prototype was removed after a compile probe confirmed CubeCL 0.11
   cannot use `Sequence<Tensor<F>>` launch arguments and its primitive
   conversions bypassed Burn Fusion. Further work can share score computation
   across output dimensions and benchmark the memory/performance tradeoff.
6. **Conformance and study-grade evaluation.**
   Implement conformance tests across diverse page counts and head configurations.
   Single-request and ragged batched CubeCL outputs are now checked against the
   portable single-request oracle. The versioned `ola-bench` study evaluates
   portable versus CubeCL execution across page sizes, batch sizes, partial
   final pages, and configurable physical-pool slack; accelerator occupancy
   still requires profiler-backed measurement.

## Out of scope

Quantized KV caches, FlashAttention-3 specific hardware intrinsics, and
distributed multi-node KV routing (deferred to later inference-engine work).
