# ola-kernel

`ola-kernel` defines the boundary between model execution and replaceable
tensor kernels. It starts with one operation: single-sequence paged decode
attention. The portable implementation uses Burn tensor operations and remains
the correctness baseline. The opt-in GPU implementation uses CubeCL, integrated
through Burn's CubeCL/Fusion backend extension. This keeps the kernel
implementation in Rust and lets CubeCL target WGPU and CUDA runtimes without
maintaining separate WGSL and CUDA kernels.

This crate deliberately does not own scheduling, model weights, tokenization,
KV allocation, or hardware policy. The runtime chooses work and the model
projects tensors; a kernel receives already prepared query and populated KV
pages and returns an attention context.

## Current boundary

```text
ola-runtime        chooses requests and batches
      |
      v
ola-burn           owns Llama weights, projections, RoPE, and KV pages
      |
      v
ola-kernel         validates and executes a replaceable tensor operation
      |
      v
CubeCL kernel      WGPU/CUDA execution; Burn tensor fallback remains available
```

The public building blocks are:

- `KernelDescriptor`, `KernelOperation`, and `KernelTarget` for stable
  diagnostic and benchmark metadata;
- `KernelRequirements` and `KernelCapability` for backend-neutral target,
  dtype, head-dimension, GQA-ratio, and page-count constraints;
- `PagedAttentionProvider`, `KernelRegistry`, and `KernelSelection` for
  deterministic provider choice and typed portable-fallback reasons;
- `PagedAttentionConfig` for the shape invariants shared by every
  implementation;
- `PagedKvPage` and `PagedAttentionInput` for an explicit operator boundary;
- `PagedAttentionKernel` for interchangeable implementations;
- `PortablePagedAttention` as the initial reference implementation;
- `KernelError` for configuration, input, and implementation failures.

The registry always contains the portable provider. Feature-gated CubeCL
implementations publish f32 target capabilities and a higher preference; equal
priorities are ordered by stable provider name, so registration order cannot
change a plan. Runtime plans now retain the exact requirements, selected
provider, and portable-fallback diagnostics. The Llama decoder still selects
its implementation locally until execution delegation is complete.

## Invariants

The initial operation is intentionally narrow:

- the query represents one sequence and one decode token;
- query heads are divisible by key/value heads;
- every KV page has batch size one and at least one populated token;
- the final page may be shorter than earlier pages;
- pages are supplied in logical sequence order;
- the output is an attention context, before the model's output projection;
- the portable implementation uses an online softmax and never concatenates
  all pages into a sequence-sized temporary.

The crate validates structural tensor shapes. The caller remains responsible
for logical ordering, populated-prefix correctness, device placement, and
ensuring that query and pages belong to the same model request.

## Adding an optimized kernel

The initial CubeCL implementation computes single-token attention over
concatenated page tensors, invoked through Burn's CubeCL/Fusion extension. It
returns metadata identifying the CubeCL/WGPU or CubeCL/CUDA target and honors
the same input/output contract. This is a correctness-oriented first kernel;
page packing and per-output score recomputation are likely performance costs.
Pin CubeCL to the version used by the workspace's Burn release because its
public API can change between minor versions.

Before an optimized implementation is selected by the model path:

1. compare its outputs with `PortablePagedAttention` across page boundaries,
   partial final pages, grouped-query attention, and long contexts;
2. run the pinned Llama conformance fixture;
3. benchmark warm and steady-state execution separately;
4. record backend, device, dtype, dimensions, page size, batch shape, and
   software revision;
5. keep the portable path available as a correctness oracle.

## Next integration step

Let the Llama execution context resolve the plan's `KernelSelection` instead of
instantiating a CubeCL kernel behind compile-time feature checks. Tune the
existing CubeCL kernel against the portable path after that boundary is
observable. It
currently concatenates page tensors before dispatch and recomputes each
query/key score for every output dimension, so optimization should focus on
device-side page layout and reducing redundant arithmetic.

The current `Vec` of page tensors is a correctness-friendly interface, but the
GPU implementation must be checked for launch and data-layout costs. If one
kernel cannot consume the pages efficiently through Burn's primitive boundary,
refine the input into a device-resident paged-cache view before optimizing the
kernel itself. Record launch count, compilation/warm-up time, and steady-state
latency separately.

## Initial validation

The CubeCL CPU and WGPU implementations are compared against
`PortablePagedAttention` on two pages with GQA, including a one-token final
page. WGPU validation also compares a 256-token case. Run the checks with:

```text
cargo test -p ola-kernel --features cubecl-cpu cubecl_cpu_matches_portable_across_pages_and_grouped_heads
cargo test -p ola-kernel --features wgpu cubecl_wgpu_matches_portable_across_pages_and_grouped_heads
```

On the Intel UHD 630, the two-stage score/value kernel measured 4.85 ms/run
across two pages and 4.37 ms/run for one logical page tensor. Portable attention
measured 16.38 ms and 11.12 ms, respectively, in two warmed 8-run repetitions.
The roughly 0.48 ms two-page difference is an estimate of page-boundary and
packing overhead, not a pure `cat` measurement. Timings include synchronization
and output readback; these are microbenchmark results, not a stable throughput
claim or a substitute for full decode benchmarks. The initial one-work-item
kernel measured 10.68 ms/run for the two-page case in earlier runs, but system
load differed, so treat that as directional rather than a controlled before/
after comparison. Reproduce the current comparison with:

```text
cargo test -p ola-kernel --features wgpu benchmark_cubecl_wgpu_against_portable_attention -- --ignored --nocapture
```

No NVIDIA timing is available in this environment. The exposed WGPU adapter is
Intel UHD 630; CUDA execution remains unverified here.

After CubeCL paged attention is measured, the same pattern can be evaluated for
prefill attention, fused QKV plus RoPE and cache writes, RMSNorm, SwiGLU, and
quantized matrix multiplication. Keep these as separate operation contracts.

## Versioned references

The workspace uses Burn 0.22.0, which resolves CubeCL 0.11.0. Keep direct
CubeCL integration aligned with those versions. The implementation references
are the [Burn custom CubeCL kernel guide](https://burn.dev/books/burn/advanced/backend-extension/custom-cubecl-kernel.html)
and the [CubeCL project](https://github.com/tracel-ai/cubecl). Recheck the
versioned examples when upgrading either dependency.
