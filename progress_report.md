# OlaRT Progress Report

**Last updated:** 2026-10-11 20:01 BST (Europe/London)

**Source revision:** working tree based on GPU milestone `27841d9` on `master`

**Repository state at update:** GPU milestone merged and pushed; multi-request
iteration execution implemented locally and awaiting its full verification pass

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
transitions, scheduling is replaceable with FIFO as the baseline, and a
synchronous executor now drives real model sessions through prefill and
single-token decode while maintaining lifecycle and live KV-memory state.

The most important distinction in the current system is now:

- **logical block allocation works** in `ola-runtime`;
- **physical per-layer K/V pages use those block tables** in `ola-burn`;
- **decode attention consumes pages directly** using an online softmax without
  reconstructing sequence-sized K/V tensors.

The portable WGPU path now runs the same model and cache implementation on the
development GTX 1650. CPU and WGPU select identical fixture tokens, and sampled
logits agree within the accelerator tolerance. Native CUDA support compiles,
but local execution is blocked by the missing CUDA toolkit/NVRTC installation.

The multi-request control-plane milestone is now underway. A configurable
executor can keep several sessions live, advance each existing session once at
an iteration boundary, refill released slots, isolate cancellation and results,
and aggregate their KV usage. It deliberately executes each model operation
individually so the next tensor-batching implementation has a deterministic
correctness oracle.

## Current delivery status

| Area | Status | Current capability |
| --- | --- | --- |
| Workspace architecture | Working | Seven crates with explicit dependency boundaries |
| Core request model | Working | IDs, timestamps, multimodal inputs, generation controls, validation |
| Model selection | Working | Catalog, model definitions, architecture registry, factories |
| Runtime lifecycle | Working | Queue, active set, terminal states, timing, model residency checks |
| Runtime executor | Multi-request baseline working | Configurable live sessions, single-operation steps, iteration steps, stable interleaving, results, failure/cancellation cleanup |
| Live memory accounting | Baseline working | Active session KV bytes propagated into `RuntimeState::memory` |
| Scheduling | Baseline working | Replaceable scheduler trait with deterministic FIFO |
| Workloads and metrics | Foundation working | Fixed trace generation, replay, nearest-rank latency summaries |
| Burn integration | Working | Backend-neutral adapter with CPU, opt-in WGPU, and opt-in native CUDA features |
| Llama loading | Narrow path working | Local config, tokenizer, single SafeTensors file, strict validation |
| Llama execution | Working | Embedding, GQA attention, RoPE, SwiGLU, RMSNorm, LM head |
| Generation sessions | Working | Prefill, cached one-token decode, greedy selection, EOS/limit checks |
| Paged KV cache | Working foundation | Per-layer physical pages addressed through runtime block tables |
| KV block manager | Working | Fixed blocks, logical tables, atomic growth, free/reuse, fragmentation |
| Serving | Foundation working | Transport-neutral request/service boundary |
| Terminal UI | Working foundation | Overview, chat, benchmarks, experiments, background local adapter |
| Local UI integrations | Working | Llama chat and generation run through the executor; KV and FIFO probes remain available |
| GPU execution | Working WGPU path | Llama generation runs through WGPU on the GTX 1650; native CUDA compiles but needs toolkit/NVRTC locally |
| Page-native decode attention | Working reference | Stable online softmax consumes K/V pages directly |
| Fused accelerator attention | Not implemented | Page traversal currently uses portable Burn tensor operations |
| Continuous batching | Control-plane foundation | Iterations advance multiple isolated sessions and continuously refill slots; tensor operations are not yet batched |
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
- Synchronous multi-request executor with:
  - runtime metadata plus attached executable models;
  - configurable concurrent session limits;
  - scheduler-driven admission into available slots;
  - one prefill or decode operation per observable step;
  - iteration steps that advance each existing session once before refilling;
  - stable round-robin request ordering and explicit prefill/decode events;
  - token, completion, and failure events;
  - retained token/text results and terminal metrics;
  - automatic lifecycle transitions;
  - aggregate live KV-byte propagation and per-request cleanup;
  - monotonic step-time validation.

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
- Shared, configurable KV page pool:
  - fixed-size physical per-layer K/V pages;
  - logical-to-physical mapping supplied by `ola-runtime::KvBlockManager`;
  - compact, unexpanded KV-head storage;
  - atomic capacity failure before cache mutation;
  - deterministic release and physical-block reuse;
  - page-aware reserved-byte reporting, including internal fragmentation;
  - prompt writes spanning page boundaries;
  - absolute RoPE position offsets during one-token decode.
- Page-native single-token attention:
  - consumes logical pages in block-table order;
  - never concatenates K/V pages into a sequence-sized tensor;
  - combines per-page scores with a numerically stable online softmax;
  - preserves grouped-query head semantics;
  - matches full-sequence logits across multiple page sizes.
- Original full-sequence forward path retained as a numerical oracle.
- Feature-gated accelerator selection:
  - portable WGPU execution through the platform graphics stack;
  - native CUDA compilation through Burn/CubeCL;
  - backend-generic token selection across differing integer element types;
  - CPU/WGPU fixture-logit comparison with an explicit tolerance.

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
- Explicit CPU/WGPU/CUDA selection using `--backend` or `OLA_BACKEND`.
- Real cached chat generation through the runtime executor.
- Llama generation latency/throughput probe.
- Real KV block-pressure benchmark.
- Real fixed-workload FIFO experiment.
- Completed-job KV usage is reported as an observation, not falsely retained as
  live allocated memory.

## Correctness and verification evidence

At the working tree recorded above:

- `cargo test --workspace`: **50 passed**, **2 ignored external-fixture tests**.
- `OLA_LLAMA_CONFORMANCE_DIR=... cargo test --workspace -- --ignored`:
  **2 passed** against the pinned artifact.
- Pinned Llama scalar-oracle conformance test: **passed manually**.
- Pinned UI-to-Llama background integration test: **passed manually**.
- WGPU UI-to-executor generation integration test: **passed manually**.
- WGPU fixture logits versus CPU fixture logits: **passed manually** with a
  `2e-4` absolute tolerance over the selected sample.
- CPU and WGPU generation selected the same fixture tokens: **passed
  manually** (`[1789, 19053]` after prompt tokens `[1, 15043]`).
- The WGPU example completed on the development NVIDIA GeForce GTX 1650.
- Native CUDA feature compilation: **passed**; native execution correctly
  reports the missing local CUDA toolkit/NVRTC prerequisite.
- Cached decode logits match full-sequence logits at every tested step.
- Paged cache tests cover cross-page reconstruction, non-contiguous mappings,
  atomic capacity failure, release, and physical-block reuse.
- Executor tests cover prefill/decode progression, retained output, live memory
  accounting, execution failure, cancellation, regressive timestamps,
  multi-session iteration order, slot refill, and isolated cancellation.
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

# Run the same path on a WGPU-compatible GPU
cargo run -p ola-burn --features wgpu --example run_llama -- \
  --backend wgpu /path/to/llama-artifact "Hello" 8

# Open the UI with runtime/KV/FIFO probes
cargo run -p ola-ui

# Open the UI with local Llama chat
cargo run -p ola-ui -- --model /path/to/llama-artifact

# Open the UI with WGPU-backed local Llama chat
cargo run -p ola-ui --features wgpu -- \
  --backend wgpu --model /path/to/llama-artifact
```

## Known limitations

- Page-native attention is a correctness-first composition of portable Burn
  operations, not a fused CUDA/WGPU kernel; performance must be measured before
  making speed claims.
- Native CUDA is wired and compiles, but this machine cannot execute it until a
  compatible CUDA toolkit providing NVRTC is installed. WGPU is the currently
  validated local GPU path.
- Physical page tensors are materialized lazily and retained at the pool's
  high-water mark for deterministic reuse.
- There is no cache eviction, preemption, swapping, prefix reuse, or
  quantization.
- Generation is batch size one and greedy only.
- The synchronous executor interleaves multiple sessions but invokes their
  model operations individually; it does not yet form tensor batches.
- Admission is bounded by request count, not predicted KV demand, token budget,
  or device capacity.
- Executor results are retained in memory and there is no result eviction or
  persistence policy yet.
- Checkpoint loading accepts one SafeTensors file only.
- RoPE scaling, tied embeddings, projection biases, and non-SiLU Llama variants
  are rejected.
- The serving layer has no streaming transport or HTTP implementation.
- Benchmark probes are foundations, not publication-quality performance claims.

## Recommended next milestones

1. **Admission and iteration policy**
   - introduce memory-aware admission and token-budget baselines;
   - separate batch-composition decisions from their execution mechanism;
   - measure TTFT, inter-token latency, throughput, and fragmentation.

2. **Continuous tensor batching**
   - batch compatible prefill and decode work on the selected backend;
   - keep per-request block tables, lifecycle, cancellation, and outputs
     isolated;
   - compare throughput and latency against the single-request oracle.

3. **Paged-attention optimization**
   - benchmark page size and fragmentation tradeoffs;
   - fuse page traversal, score reduction, and value accumulation where the
     selected accelerator backend permits;
   - retain per-step numerical comparison with the full-sequence oracle.

4. **Serving and experiment infrastructure**
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

- **2026-10-11 20:01 BST — working tree on `27841d9`:** Began the
  multi-request phase with configurable live sessions, deterministic
  iteration-level execution, slot refill, aggregate KV accounting, and
  isolated cancellation/results.
- **2026-10-11 19:46 BST — GPU milestone `27841d9`:** Added opt-in WGPU
  and CUDA backends to the example and terminal workbench, validated WGPU
  fixture generation and CPU-logit agreement on the GTX 1650, and documented
  the native CUDA toolkit requirement.
- **2026-10-07 22:00 BST — working tree on `29a0043`:** Replaced contiguous
  decode-time K/V gathering with direct page traversal and a stable online
  softmax; verified cached logits across 1-, 2-, 4-, and 16-token pages.
- **2026-10-07 21:35 BST — working tree on `29a0043`:** Added the synchronous
  executor, executable-model attachment, incremental events and results, live
  KV accounting, cleanup semantics, and routed local UI Llama generation
  through the new execution path.
- **2026-10-07 15:35 BST — working tree on `29a0043`:** Connected the runtime
  block allocator to configurable, shared Burn K/V page storage; added
  cross-page, reuse, capacity, accounting, and cached-logit correctness tests.
- **2026-10-07 09:55 BST — `29a0043`:** Created the dedicated report after
  merging cached Llama decoding, the KV block manager, pinned conformance work,
  and the Ratatui workbench with local integrations.
