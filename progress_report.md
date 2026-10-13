# OlaRT Progress Report

**Last updated:** 2026-10-13 09:46 BST (Europe/London)

**Source revision:** working tree based on admission milestone `540dee2` on
`master`

**Repository state at update:** admission milestone merged and pushed;
batch-composition/token-budget policy, root developer Makefile, observability,
variable-length tensor-batched Llama prefill/decode, and prompt-length
bucketing with padding measurements implemented locally; not yet committed

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

The multi-request control-plane milestone is merged. A configurable executor
can keep several sessions live, advance each existing session once at an
iteration boundary, refill released slots, isolate cancellation and results,
and aggregate their KV usage. The individual execution path remains available
as a deterministic correctness oracle and fallback when a model does not
implement tensor batching.

Memory-aware admission is merged: Llama publishes its real physical KV
geometry, and the strict baseline makes page-rounded worst-case reservations
before session creation. Temporary model-pool or device pressure defers work
without changing lifecycle state; impossible requests are rejected terminally
instead of blocking FIFO indefinitely.

The current working tree adds explicit batch composition at the iteration
boundary. An unbounded compatibility policy preserves existing behavior, while
a FIFO token-budget baseline limits input positions processed per iteration.
Every iteration returns a deterministic plan of selected and deferred prefill
and decode operations. Oversized head prefills run alone to guarantee progress.
A root Makefile now exposes the maintained build, test, lint, docs, UI,
conformance, and GPU workflows through `make help`.

The working tree also introduces `ola-observe`, a read-only consumer of
executor iterations and runtime lifecycle records. It derives documented queue,
TTFT, prefill, service, end-to-end, decode-span, and aggregate inter-token
latencies; records iteration work and memory snapshots; exposes interval token
rates; and retains cumulative counters without reading a wall clock.

Compatible prefill and decode operations are now grouped by loaded model and
offered to optional model-level tensor paths. Burn Llama right-pads
variable-length prompts and batches embeddings, projections, normalization,
MLP, LM head, and token selection. Only real prompt prefixes enter each
request's cache. Decode retains request-local RoPE positions and page traversal;
both phases retain isolated block tables, lifecycle, and results. KV growth is
atomic across each complete batch. Every iteration records whether operations
used a real tensor batch or individual fallback.

Prefill grouping is now independently replaceable. The compatibility strategy
keeps one same-model group, while the power-of-two strategy separates distant
prompt lengths to limit right-padding waste without changing scheduler-visible
event order. Execution records and `ola-observe` report real input positions,
padding positions, dense-input rate, and input-token efficiency so policy
comparisons can measure the compute tradeoff rather than infer it.

## Current delivery status

| Area | Status | Current capability |
| --- | --- | --- |
| Workspace architecture | Working | Eight crates with explicit dependency boundaries |
| Core request model | Working | IDs, timestamps, multimodal inputs, generation controls, validation |
| Model selection | Working | Catalog, model definitions, architecture registry, factories |
| Runtime lifecycle | Working | Queue, active set, terminal states, timing, model residency checks |
| Runtime executor | Tensor batching working | Configurable sessions, iteration planning, model-grouped tensor prefill/decode, fallback execution, stable events/results |
| Live memory accounting | Baseline working | Active session KV bytes propagated into `RuntimeState::memory` |
| Scheduling | Baseline working | Replaceable scheduler trait with deterministic FIFO |
| Admission | Memory-aware baseline | Typed admit/defer/reject decisions using page-rounded model-pool and optional device budgets |
| Batch composition | Token-budget and bucketing baseline | Replaceable selection policy, auditable plans, FIFO token budget, optional power-of-two prefill buckets |
| Observability | Foundation working | Request latency, iteration work/rates, padding efficiency, terminal measurements, cumulative counters |
| Workloads and metrics | Foundation working | Fixed trace generation, replay, nearest-rank distribution summaries |
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
| Continuous batching | Tensor foundation working | Compatible same-model variable-length prefill and decode are tensorized; decode page traversal remains request-local |
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
- Backend-neutral individual and optional batched prefill/decode contracts plus
  session metrics.
- Backend-neutral physical KV resource profiles for pre-session planning.
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
  - conservative live KV reservations distinct from observed usage;
  - replaceable admission policy after scheduler selection;
  - strict page-rounded model-pool and optional device-budget enforcement;
  - typed temporary deferral and permanent rejection outcomes;
  - replaceable iteration-level batch-composition policy;
  - explicit prefill/decode candidates with input-token costs;
  - auditable selected/deferred iteration plans and batch-deferral events;
  - FIFO token-budget enforcement with oversized-prefill progress;
  - model-compatible grouping of selected prefill and decode work;
  - replaceable prefill-length grouping with unbucketed compatibility and
    power-of-two strategies;
  - optional phase-specific tensor-batch hooks with transparent individual
    fallback;
  - explicit tensor/individual execution records with request-row order, real
    input positions, and padding positions;
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
  - backend-neutral publication of the exact physical pool geometry used by
    runtime admission.
- Atomic multi-sequence KV growth for tensor decode.
- Atomic request-specific KV growth for variable-length tensor prefill.
- Tensor-batched variable-length prefill:
  - right-padded dense transformer computation with causal isolation;
  - per-row final real-token selection;
  - cache writes restricted to each request's real prompt prefix;
  - output, cache metrics, and subsequent decode agreement with independent
    sessions.
- Tensor-batched cached decode across compatible live sessions:
  - batched embeddings, Q/K/V and output projections, norms, MLP, LM head,
    argmax, and host token read;
  - request-local rotary positions and physical-page attention;
  - output and metrics agreement with independent decode at unequal positions;
  - executor-to-Llama integration coverage proving the tensor path is used.
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

### `ola-observe`

- Read-only observation of executor iterations and retained runtime state.
- Explicit microsecond formulas for queue latency, TTFT, prefill latency,
  service latency, end-to-end latency, and decode span.
- Aggregate mean inter-token latency for terminal requests with at least two
  observed output tokens.
- Selected prefill/decode work, input-token cost, policy deferrals, emitted
  tokens, terminal outcomes, live request counts, and KV snapshots per
  iteration.
- Executed real input positions, artificial right-padding positions,
  dense-input rate, and input-token efficiency.
- Input/output token rates over non-zero caller-supplied observation intervals.
- Cumulative counters and retained terminal measurements in stable request-ID
  order.
- Timestamp-regression and invalid-request diagnostics without wall-clock reads.

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

- `cargo test --workspace`: **73 unit tests and 4 documentation tests passed**;
  **2 external-fixture tests ignored** by default.
- `OLA_LLAMA_CONFORMANCE_DIR=... cargo test --workspace -- --ignored`:
  **2 passed** against the pinned artifact.
- Pinned Llama scalar-oracle conformance test: **passed manually**.
- Pinned UI-to-Llama background integration test: **passed manually**.
- WGPU UI-to-executor generation integration test: **passed manually**.
- Strict-admission CPU UI-to-executor generation integration test: **passed
  manually** against the pinned fixture.
- Strict-admission WGPU UI-to-executor generation integration test: **passed
  manually** on the GTX 1650 against the pinned fixture.
- WGPU fixture logits versus CPU fixture logits: **passed manually** with a
  `2e-4` absolute tolerance over the selected sample.
- CPU and WGPU generation selected the same fixture tokens: **passed
  manually** (`[1789, 19053]` after prompt tokens `[1, 15043]`).
- The WGPU example completed on the development NVIDIA GeForce GTX 1650.
- Native CUDA feature compilation: **passed**; native execution correctly
  reports the missing local CUDA toolkit/NVRTC prerequisite.
- Current WGPU and CUDA feature compilation: **passed**. A fresh live WGPU run
  was not completed because `nvidia-smi` could not communicate with the NVIDIA
  driver during this update; the successful GTX 1650 results above are retained
  historical evidence rather than represented as a new run.
- Cached decode logits match full-sequence logits at every tested step.
- Paged cache tests cover cross-page reconstruction, non-contiguous mappings,
  atomic capacity failure, release, and physical-block reuse.
- Executor tests cover prefill/decode progression, retained output, live memory
  accounting, execution failure, cancellation, regressive timestamps,
  multi-session iteration order, slot refill, isolated cancellation, token
  budget deferral, auditable plans, and oversized-prefill progress.
- Batch-policy tests cover FIFO prefix selection across phases, invalid zero
  budgets, and the deliberate oversized-prefill escape hatch.
- Tensor-batching tests cover atomic fixed- and variable-length multi-sequence
  KV reservation, prefill output/cache/subsequent-decode agreement at unequal
  prompt lengths, decode agreement at unequal positions, executor grouping,
  transparent fallback, and execution-mode records.
- Prefill-bucketing tests cover power-of-two boundaries, padding accounting,
  nearby-length grouping, distant-length splitting, and FIFO event restoration.
- Observation tests cover every documented latency boundary, interval rates,
  dense padding efficiency, cumulative counts, regressive timestamps,
  single-token inter-token handling, and failure atomicity.
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

From the repository root, the same maintained workflows are discoverable with
`make help`; `make verify` runs formatting, checks, tests, Clippy, and docs.
Model commands accept `MODEL=/path/to/llama-artifact`, with optional `PROMPT`
and `TOKENS` variables.

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
- Variable-length prefill uses right padding to the longest prompt in its
  execution group. Power-of-two bucketing is available but not yet selected
  adaptively; chunked prefill is not implemented. Generation remains greedy.
- Decode batches still traverse each request's physical pages independently;
  page attention is not yet a fused ragged multi-request kernel.
- Admission reserves worst-case KV demand; it does not yet predict early EOS or
  reclaim part of a reservation as a sequence approaches completion.
- Device-wide admission is enforced only when a caller supplies a capacity and
  non-KV accounting; automatic GPU-memory telemetry is not wired in yet.
- Inter-token latency currently uses the terminal decode span divided by token
  gaps; individual token timestamps are not retained yet.
- Executor results are retained in memory and there is no result eviction or
  persistence policy yet.
- Checkpoint loading accepts one SafeTensors file only.
- RoPE scaling, tied embeddings, projection biases, and non-SiLU Llama variants
  are rejected.
- The serving layer has no streaming transport or HTTP implementation.
- Benchmark probes are foundations, not publication-quality performance claims.

## Recommended next milestones

1. **Optimize continuous tensor batching**
   - benchmark unbucketed versus power-of-two prompt grouping;
   - add chunked prefill or an adaptive padding/throughput strategy;
   - optimize the request-local portion of tensor-batched paged attention;
   - keep per-request block tables, lifecycle, cancellation, and outputs
     isolated;
   - compare throughput and latency against the single-request oracle.

2. **Measurement and policy experiments**
   - feed the new observation records into reproducible benchmark summaries;
   - compare token budgets and prefill/decode mixes under fixed traces;
   - expose repeatable configurations and results through the workbench.

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

- **2026-10-13 09:46 BST — working tree on `540dee2`:** Added replaceable
  prefill-length grouping, power-of-two buckets, per-invocation padding
  accounting, observer efficiency/rate measurements, and deterministic
  grouping/event-order coverage.
- **2026-10-13 08:02 BST — working tree on `540dee2`:** Added optional
  variable-length batched prefill, right-padded Burn Llama execution with
  real-prefix cache writes, atomic per-request KV reservation, executor
  grouping/fallback records, and independent-session oracle coverage.
- **2026-10-12 18:55 BST — working tree on `540dee2`:** Added optional
  model-level batched decode, executor grouping and tensor/fallback records,
  atomic multi-sequence KV growth, real Burn Llama tensor decode, observation
  counters, numerical oracle comparison, and executor integration coverage.
- **2026-10-12 17:40 BST — working tree on `540dee2`:** Added the
  `ola-observe` crate with documented lifecycle latency formulas, iteration
  measurements and rates, cumulative counters, terminal request capture, and
  deterministic tests.
- **2026-10-12 16:40 BST — working tree on `540dee2`:** Added replaceable
  batch composition, auditable selected/deferred iteration plans, a FIFO
  input-token budget with oversized-prefill progress, and a documented root
  Makefile for verification, UI, fixture, and GPU workflows.
- **2026-10-12 13:03 BST — working tree on `5827879`:** Verified strict
  admission through both CPU and GTX 1650 WGPU UI-to-executor generation paths;
  the pinned external fixture hashes matched the conformance record.
- **2026-10-11 21:31 BST — working tree on `5827879`:** Added physical KV
  resource profiles and a replaceable strict admission baseline with
  page-rounded reservations, model/device capacity checks, typed defer/reject
  outcomes, and FIFO-safe lifecycle behavior.
- **2026-10-11 20:01 BST — multi-request milestone `5827879`:** Added
  configurable live sessions, deterministic iteration-level execution, slot
  refill, aggregate KV accounting, and isolated cancellation/results.
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
