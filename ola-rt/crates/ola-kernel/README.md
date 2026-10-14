# ola-kernel

`ola-kernel` defines the boundary between model execution and replaceable
tensor kernels. It starts with one operation: single-sequence paged decode
attention. The existing implementation uses portable Burn tensor operations
and remains the correctness baseline for later CubeCL, CUDA, CPU, or other
hardware-specific implementations.

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
Burn backend       CPU, WGPU, CUDA, or a future implementation
```

The public building blocks are:

- `KernelDescriptor`, `KernelOperation`, and `KernelTarget` for stable
  diagnostic and benchmark metadata;
- `PagedAttentionConfig` for the shape invariants shared by every
  implementation;
- `PagedKvPage` and `PagedAttentionInput` for an explicit operator boundary;
- `PagedAttentionKernel` for interchangeable implementations;
- `PortablePagedAttention` as the initial reference implementation;
- `KernelError` for configuration, input, and implementation failures.

The Llama decoder currently selects `PortablePagedAttention` directly. This is
intentional for the first slice: it proves the crate boundary and moves the
reference computation without adding a speculative global registry or device
policy.

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

An optimized implementation should implement `PagedAttentionKernel<B>` for its
Burn backend, return a descriptor that identifies its real hardware target,
and honor the same input/output contract. Keep unsupported shape detection
explicit; it should eventually feed selection and portable fallback rather
than silently changing semantics.

Before an optimized implementation is selected by the model path:

1. compare its outputs with `PortablePagedAttention` across page boundaries,
   partial final pages, grouped-query attention, and long contexts;
2. run the pinned Llama conformance fixture;
3. benchmark warm and steady-state execution separately;
4. record backend, device, dtype, dimensions, page size, batch shape, and
   software revision;
5. keep the portable path available as a correctness oracle.

## Next integration step

The next small milestone is provider selection at the `ola-burn` execution
boundary. Keep the selected provider outside the checkpoint-bearing Burn
module, then pass it into decode calls or hold it in the session/executor. Add
capability checks and portable fallback only when a second implementation
exists; until then, a registry would add indirection without testing a real
selection decision.

After paged attention has two implementations, the same pattern can be
evaluated for prefill attention, fused QKV plus RoPE and cache writes, RMSNorm,
SwiGLU, and quantized matrix multiplication. Do not force those operations into
one large model-specific trait.
