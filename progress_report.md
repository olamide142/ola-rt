# OlaRT Progress Report

**Last updated:** 2026-10-07 09:55 BST (Europe/London)  
**Source revision:** `29a0043` on `master`  
**Repository state at update:** synchronized with `origin/master`

This is the authoritative, timestamped snapshot of OlaRT's implementation
progress. The root `README.md` describes the project vision, `IMPLEMENTATION.md`
describes the long-term build plan, and `ola-rt/README.md` documents stable
implementation semantics. This report answers the narrower question: **what
works today, what evidence supports it, and what should be built next?**

## Executive summary

OlaRT has moved beyond repository scaffolding into its first complete inference
vertical slice. It can load a narrowly supported Hugging Face Llama checkpoint,
tokenize a prompt, execute prefill, perform cached greedy decoding, report
request-local KV-cache storage, and expose that workflow through a Ratatui
terminal application.

The runtime foundation is also operational: requests have validated lifecycle
transitions, scheduling is replaceable with FIFO as the baseline, deterministic
workloads can be replayed, and a backend-neutral KV block manager can model
capacity, allocation, freeing, and internal fragmentation.

The most important distinction in the current system is:

- **model-level KV caching works** using contiguous Burn tensors;
- **logical block allocation works** in the runtime;
- the two are not yet connected by physical paged tensor storage.

That connection is the next major systems milestone.

## Current delivery status

| Area | Status | Current capability |
| --- | --- | --- |
| Workspace architecture | Working | Seven crates with explicit dependency boundaries |
| Core request model | Working | IDs, timestamps, multimodal inputs, generation controls, validation |
| Model selection | Working | Catalog, model definitions, architecture registry, factories |
| Runtime lifecycle | Working | Queue, active set, terminal states, timing, model residency checks |
| Scheduling | Baseline working | Replaceable scheduler trait with deterministic FIFO |
| Workloads and metrics | Foundation working | Fixed trace generation, replay, nearest-rank latency summaries |
| Burn integration | Working | Backend-neutral adapter plus CPU `NdArray<f32>` execution |
| Llama loading | Narrow path working | Local config, tokenizer, single SafeTensors file, strict validation |
| Llama execution | Working | Embedding, GQA attention, RoPE, SwiGLU, RMSNorm, LM head |
| Generation sessions | Working | Prefill, cached one-token decode, greedy selection, EOS/limit checks |
| Contiguous KV cache | Working | Per-layer compact K/V tensors and exact element-byte accounting |
| KV block manager | Metadata working | Fixed blocks, logical tables, atomic growth, free/reuse, fragmentation |
| Serving | Foundation working | Transport-neutral request/service boundary |
| Terminal UI | Working foundation | Overview, chat, benchmarks, experiments, background local adapter |
| Local UI integrations | Working | Llama chat, generation probe, KV pressure benchmark, FIFO experiment |
| GPU execution | Not implemented | Hardware is available, but OlaRT still uses the CPU backend |
| Paged tensor storage | Not implemented | Block metadata exists; attention still reads contiguous tensors |
| Continuous batching | Not implemented | Sessions execute independently, one request at a time |
| Production server | Not implemented | No async executor, streaming HTTP, or OpenAI-compatible API yet |

## Implemented system

### `ola-core`

- Stable model, architecture, and request identifiers.
- Monotonic microsecond timestamps suitable for real or simulated time.
- Text, token, and image input representations.
- Generation configuration, priority, deadlines, and experiment attributes.
- Model descriptors and capability validation.
- Minimal backend/device ownership abstraction without tensor leakage.

### `ola-model`

- Separates model identity, architecture identity, and artifact source.
- Generic architecture registry and factory selection.
- Shared tokenizer and model weights with isolated request sessions.
- Backend-neutral `prefill`, `decode`, and session-metrics contracts.
- Explicit distinction between catalog availability and runtime residency.

### `ola-runtime`

- Validated `Queued → Running → Completed/Failed/Cancelled` transitions.
- Authoritative queue, active-set, request record, and timing state.
- Read-only scheduler view with runtime-validated decisions.
- FIFO baseline behind a replaceable scheduler trait.
- Memory categories for weights, activations, KV cache, and workspace.
- Deterministic fixed-size KV block manager with:
  - logical sequence and physical block identifiers;
  - atomic token growth;
  - deterministic lowest-block allocation;
  - block-table lookup;
  - freeing and reuse;
  - occupancy and internal-fragmentation metrics;
  - explicit capacity and configuration errors.

### `ola-burn`

- Strict Hugging Face Llama configuration parser.
- Hugging Face tokenizer loading.
- Single-file SafeTensors loading with RMSNorm key remapping.
- Explicit rejection of unsupported architecture features.
- Correctness-first Llama decoder implementing:
  - token embeddings;
  - pre-normalized decoder blocks;
  - grouped-query causal self-attention;
  - split-half Llama rotary embeddings;
  - SwiGLU feed-forward layers;
  - final RMSNorm and vocabulary projection.
- Request-local contiguous KV cache:
  - compact, unexpanded KV-head storage;
  - prompt population during prefill;
  - absolute RoPE position offsets during decode;
  - one-token decode without full-sequence recomputation;
  - exact cache-element byte reporting.
- Original full-sequence forward path retained as a numerical oracle.

### `ola-bench`

- Deterministic timestamped workload representation.
- Fixed workload generator with explicit request shapes.
- Replay into any compatible runtime scheduler.
- Nearest-rank latency summaries in microseconds.

### `ola-serve`

- Transport-independent generation request boundary.
- Runtime submission, model discovery, and health foundation.
- HTTP framework and asynchronous execution intentionally deferred.

### `ola-ui`

- Ratatui application shell with Overview, Chat, Benchmarks, and Experiments.
- Pure, independently testable navigation and input state.
- Unicode-safe prompt editing and compact-terminal fallback.
- Typed snapshot/command adapter contract.
- Background local worker so model work does not block terminal input.
- Optional local Llama loading using `--model` or `OLA_MODEL_DIR`.
- Real cached chat generation.
- Llama generation latency/throughput probe.
- Real KV block-pressure benchmark.
- Real fixed-workload FIFO experiment.
- Completed-job KV usage is reported as an observation, not falsely retained as
  live allocated memory.

## Correctness and verification evidence

At the source revision recorded above:

- `cargo test --workspace`: **39 passed**, **2 ignored external-fixture tests**.
- Pinned Llama scalar-oracle conformance test: **passed manually**.
- Pinned UI-to-Llama background integration test: **passed manually**.
- Cached decode logits match full-sequence logits at every tested step.
- KV allocator tests cover deterministic mapping, atomic out-of-capacity
  failure, freeing/reuse, invalid configuration, and fragmentation metrics.
- Ratatui screens render through `TestBackend`, including the small-terminal
  fallback.
- Interactive PTY launch and terminal restoration were verified.
- `cargo clippy --workspace --all-targets -- -D warnings`: **passed**.
- `cargo fmt --all -- --check`: **passed**.
- `cargo doc --workspace --no-deps`: **passed**.

The external fixture and its hashes are documented in
`ola-rt/crates/ola-burn/CONFORMANCE.md`. The independent scalar oracle is
`ola-rt/tools/llama_scalar_reference.py`.

## What can be run now

From the `ola-rt/` implementation workspace:

```text
# Full normal verification
cargo test --workspace

# Load a local artifact and generate tokens
cargo run -p ola-burn --example run_llama -- \
  /path/to/llama-artifact "Hello" 8

# Open the UI with runtime/KV/FIFO probes
cargo run -p ola-ui

# Open the UI with local Llama chat
cargo run -p ola-ui -- --model /path/to/llama-artifact
```

## Known limitations

- The executable Llama path uses CPU `NdArray<f32>`; CUDA is not wired in.
- The cache grows with `Tensor::cat`, so each append copies prior contiguous
  cache contents.
- Runtime block tables do not yet address physical Burn tensor pages.
- There is no cache eviction, preemption, swapping, prefix reuse, or
  quantization.
- Generation is batch size one and greedy only.
- There is no runtime executor coordinating queued requests with model sessions.
- Runtime memory accounting is not yet updated automatically from live model
  allocations.
- Checkpoint loading accepts one SafeTensors file only.
- RoPE scaling, tied embeddings, projection biases, and non-SiLU Llama variants
  are rejected.
- The serving layer has no streaming transport or HTTP implementation.
- Benchmark probes are foundations, not publication-quality performance claims.

## Recommended next milestones

1. **Physical paged KV storage**
   - define the backend-facing page storage contract;
   - allocate per-layer K/V pages using runtime block tables;
   - remove contiguous `Tensor::cat` growth;
   - compare every cached logit with the existing oracle.

2. **Runtime executor loop**
   - associate admitted requests with loaded model sessions;
   - drive prefill and decode from scheduler decisions;
   - propagate live KV occupancy into `RuntimeState::memory`;
   - expose live request progress to `ola-ui`.

3. **GPU execution path**
   - add a deliberate Burn accelerator feature/backend;
   - validate numerical conformance before benchmarking;
   - record device, dtype, context, warm-up, and compilation conditions.

4. **Multi-request execution**
   - add continuous/iteration-level batching;
   - introduce memory-aware admission and scheduling baselines;
   - measure TTFT, inter-token latency, throughput, and fragmentation.

5. **Serving and experiment infrastructure**
   - add streaming output and an HTTP boundary;
   - persist structured benchmark configurations and results;
   - make UI benchmark and experiment catalogs data-driven.

## Update protocol

Update this file whenever a meaningful capability is merged. Each update should:

1. replace the timestamp, source revision, and repository state;
2. move capabilities between implemented, partial, and deferred honestly;
3. record the exact verification evidence run for that revision;
4. update known limitations rather than silently removing them;
5. reorder the next milestones when the engineering boundary changes;
6. add a short entry to the history below.

## History

- **2026-10-07 09:55 BST — `29a0043`:** Created the dedicated report after
  merging cached Llama decoding, the KV block manager, pinned conformance work,
  and the Ratatui workbench with local integrations.
