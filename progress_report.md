# OlaRT Progress Report

**Last updated:** 2026-10-16 22:40 BST (Europe/London)

**Source revision:** working tree based on `748ebfb` on `master`

**Repository state at update:** P1 (execution-plan IR), P2 (composable
execution policy), P3 (scheduler and policy baselines), P4 (experiment
framework), and P5 (runtime driver and serving) are complete. Every
benchmark and experiment is a registered experiment with a versioned,
interval-carrying report; `ola serve` serves a model over a native HTTP API
and an OpenAI-compatible completions endpoint, with streaming,
cancellation, backpressure, health, and draining shutdown, and its outputs
match direct execution on tiny Llama. P6 (observability) is in progress:
tokens and lifecycle events are now stamped when the model call producing
them finishes, which corrected one P3 finding, and inter-token latency is
reported as a real per-gap distribution. Current
branch commits are not pushed

This is the authoritative, timestamped snapshot of OlaRT's implementation
progress. The root `README.md` describes the project vision, the local and
untracked `IMPLEMENTATION.md` describes the long-term build plan, and
`ola-rt/README.md` documents stable implementation semantics. This report
answers the narrower question: **what works today, what evidence supports it,
and what should be built next?**

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
TTFT, prefill, service, end-to-end, and decode-span latencies and per-gap
inter-token distributions; records iteration work and memory snapshots; exposes interval token
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

The benchmark layer now compares the same prompt-length trace under unbucketed
and power-of-two grouping. It reports padding saved, input efficiency, and the
invocation-count tradeoff without presenting deterministic shape arithmetic as
elapsed performance. The comparison is runnable from the terminal workbench's
Experiments screen without loading a model.

The working tree adds the next measurement layer: identical tokenized requests
are executed against a loaded model under both grouping strategies. The runner
rejects output divergence and reports real executor invocations, padding,
emitted-token throughput, prefill-iteration and total host wall time, and peak
observed/reserved KV bytes. A repeated protocol discards explicit warm-up
pairs, retains raw measured pairs, alternates which strategy runs first, and
derives nearest-rank latency summaries. The UI exposes four measured pairs
after one warm-up pair. Documentation explicitly distinguishes
prefill-iteration wall time from TTFT and treats results as operational probes
rather than publication-quality performance claims.

Benchmark results can now cross process boundaries. A versioned JSON schema
records the full tokenized workload, repetition protocol, model/device/build
metadata, paired execution order, every raw measurement, exact aggregate
token/time inputs, latency summaries, and KV peaks. File creation refuses to
overwrite existing results, and readers reject unknown schema versions or
inconsistent trial indexes. UI persistence is opt-in through
`--results-dir`/`OLA_BENCH_RESULTS_DIR`, so ordinary interactive use remains
side-effect free.

The runtime now has its first workload-aware grouping policy.
`CostAwarePrefillBatcher` evaluates the complete same-model prefill set and
chooses either one wide tensor invocation or next-power-of-two buckets. Its
explicit token-equivalent invocation overhead balances fixed launch cost
against dense padding work; both candidate estimates and the selected strategy
are auditable. Decisions contain no timing reads, preserve lifecycle order, and
can be calibrated independently per model/backend. A model-independent UI
experiment demonstrates that the same policy selects bucketing for a diverse
trace and one wide batch for a clustered trace.

The adaptive cost can now be derived from execution instead of chosen only by
hand. The benchmark fits unbucketed and power-of-two p50 prefill measurements to
`time = dense_tokens * a + invocations * b`, rejects invalid/non-positive fits,
and uses the rounded `b/a` token-equivalent overhead in a fresh three-strategy
comparison. Unbucketed, power-of-two, and adaptive order rotates across three
measured trials, outputs must remain identical, and schema-v2 reports retain
the full order, calibrated overhead, and every adaptive sample. The complete
flow passed against the real CPU tiny-Llama fixture.

The latest working-tree addition exercises chunking under mixed arrivals and
decode competition. A repeated fairness sweep compares complete prefill with
selected chunk sizes while limiting active requests, rotates strategy order,
discards warm-ups, validates output equality, and retains each request's TTFT
and completion host latency alongside scheduler iterations. The UI probe runs
four differently sized prompts with two active slots, chunk sizes 2/4/8, one
warm-up, and three measured runs. Per-request p50/p95 summaries are operational
host observations, not a wall-clock fairness guarantee or publication-quality
performance claim.

## Current delivery status

| Area | Status | Current capability |
| --- | --- | --- |
| Workspace architecture | Working | Ten crates with explicit dependency boundaries |
| Core request model | Working | IDs, timestamps, multimodal inputs, generation controls, validation |
| Model selection | Working | Catalog, model definitions, architecture registry, factories |
| Runtime lifecycle | Working | Queue, active set, terminal states, timing, model residency checks |
| Runtime executor | Tensor batching working | Configurable sessions, iteration planning, model-grouped tensor prefill/decode, fallback execution, stable events/results |
| Live memory accounting | Baseline working | Active session KV bytes propagated into `RuntimeState::memory` |
| Scheduling | Baselines working | FIFO, priority, earliest-deadline-first, and shortest-job-first (optional aging) ranking from the snapshot; admission backfill; prompt-capped and queue-adaptive token budgets |
| Scheduling experiments | Working | Seeded traces, deterministic synthetic model, virtual-time trials with perturbations, and `compare_schedulers` (latency percentiles, throughput, KV, deferrals, slowdown and Jain fairness, starvation, deadline misses, per-class breakdowns) |
| Admission | Memory-aware baseline | Typed admit/defer/reject decisions using page-rounded model-pool and optional device budgets |
| Execution policy | Working | `ExecutionPolicy` plans from an immutable snapshot through `&self`; `ComposedPolicy` combines replaceable scheduler, admission, batch, and grouping components with stable descriptors and full decision records |
| Execution planning | Working | Each iteration is a validated `ExecutionPlan` (admissions, invocations, deferrals, predicted releases) committed atomically; divergences recorded; versioned `PlanReport` JSON |
| Batch composition | Adaptive baseline working | Auditable token budget plus unbucketed, power-of-two, and cost-aware workload grouping |
| Observability | Foundation working | Request latency, iteration work/rates, padding efficiency, terminal measurements, cumulative counters |
| Workloads and metrics | Calibrated execution probe | Fixed replay, cost calibration, rotating three-way trials, repeated mixed-workload chunk fairness, per-request TTFT/completion latency distributions, raw samples, summaries, schema-v2 JSON |
| Experiment framework | Working | `ExperimentRegistry` of nine experiments (probes and studies); `ola-experiment/1` reports with typed metrics, raw samples, recomputed summaries, seeded bootstrap intervals, and full identity; report comparison with interval-separated verdicts; least-squares cost calibration feeding virtual time; legacy v2 import |
| Experiment CLI | Working | `ola experiment list/run/compare` with CI exit codes; reports identical to the workbench's |
| Kernel abstraction | First vertical slice | Validated paged-attention contract, stable metadata, and portable Burn reference used by Llama decode |
| Burn integration | Working | Backend-neutral adapter with CPU, opt-in WGPU, and opt-in native CUDA features |
| Llama loading | Narrow path working | Local config, tokenizer, single SafeTensors file, strict validation |
| Llama execution | Working | Embedding, GQA attention, RoPE, SwiGLU, RMSNorm, LM head |
| Generation sessions | Working | Prefill, cached one-token decode, greedy selection, EOS/limit checks |
| Paged KV cache | Working foundation | Per-layer physical pages addressed through runtime block tables |
| KV block manager | Working | Fixed blocks, logical tables, atomic growth, free/reuse, fragmentation |
| Serving | Driver working | A dedicated thread owns the executor; async or blocking handles submit, stream text deltas, cancel (including on drop), and report status; bounded in-flight admission and finished-request retention; lock-free health and readiness; draining or immediate shutdown |
| Terminal UI | Working foundation | Overview, chat, benchmarks, experiments, background local adapter |
| Local UI integrations | Working | Llama chat/generation, KV/FIFO probes, fixed/adaptive shape experiments, real-model prefill benchmarks, and repeated mixed-workload chunk fairness sweep |
| GPU execution | Working WGPU path | Llama generation runs through WGPU on the GTX 1650; native CUDA compiles but needs toolkit/NVRTC locally |
| Page-native decode attention | Working reference | Stable online softmax consumes K/V pages directly |
| Fused accelerator attention | Not implemented | Page traversal currently uses portable Burn tensor operations |
| Continuous batching | Tensor foundation working | Compatible same-model variable-length prefill and decode are tensorized; decode page traversal remains request-local |
| Production server | Native API working | `ola serve`: health, readiness, models, JSON or server-sent-event generation, request status and cancellation, disconnect-cancels, distinct error statuses, OpenAI-compatible `/v1/completions` (greedy); drain on Ctrl-C; no chat completions, auth, or TLS yet |

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
- Explicit resumable-prefill capability negotiation and typed pending/token
  segment outcomes.
- Optional tensor-batched chunk contract carrying per-row final-segment flags.
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
  - bounded resumable prefill with request-local cursors and progress events;
  - shared token-budget composition for decode and continued prefill work;
  - cancellation-safe partial-session cleanup and architecture fallback to
    complete-prompt execution;
  - model-grouped tensor execution for resumable chunks with unequal cursors
    and segment lengths;
  - model-compatible grouping of selected prefill and decode work;
  - replaceable prefill-length grouping with unbucketed compatibility and
    power-of-two strategies;
  - complete same-model workload grouping hooks for policies that require more
    than one prompt length at a time;
  - cost-aware adaptive selection balancing dense positions against an explicit
    token-equivalent invocation overhead, with auditable candidate estimates;
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
- Atomic per-segment reservation for resumable prefill, with accepted-token
  state unchanged on capacity failure.
- Request-local chunk attention over the populated page prefix, with final
  token and subsequent decode agreement against complete-prompt execution.
- Atomic variable-length chunk reservation and per-offset active-row tensor
  batching across sessions at unequal absolute positions.
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

### `ola-kernel`

- Fine-grained `PagedAttentionKernel<B>` contract without scheduler, model, or
  allocation ownership.
- Validated single-sequence decode inputs with explicit GQA dimensions and
  populated logical KV pages.
- Stable operation, implementation, and hardware-target metadata for future
  diagnostics and benchmark reports.
- Portable Burn online-softmax implementation extracted from the Llama model
  and retained as the numerical baseline.
- Direct use by Llama cached decode and resumable-prefill attention.
- Documented handoff for adding a second implementation, capability checks,
  provider selection, fallback, conformance, and benchmarking.

### `ola-bench`

- Deterministic timestamped workload representation.
- Fixed workload generator with explicit request shapes.
- Replay into any compatible runtime scheduler.
- Nearest-rank latency summaries in microseconds.
- Deterministic same-model prefill-shape summaries with invocation counts, real
  inputs, padding inputs, and efficiency.
- Side-by-side unbucketed versus power-of-two comparison with explicit padding
  savings and no implied wall-clock claim.
- Execution-backed comparison against one loaded model with output-equivalence
  validation, invocation/padding counts, host wall time, output throughput, and
  peak observed/reserved KV bytes.
- Warm-up-aware repeated comparisons with alternating paired order, retained
  raw samples, nearest-rank prefill/total latency summaries, and aggregate
  throughput.
- Two-term invocation-cost calibration with explicit validity checks and a
  fresh order-rotated three-way unbucketed/power-of-two/adaptive benchmark.
- Complete-versus-chunked execution comparison with output equivalence,
  segment invocation/iteration counts, timing, and KV peaks.
- Repeated mixed-workload fairness sweep comparing complete prefill with
  configured chunk sizes under an active-request limit, rotating strategy
  order, discarding warm-ups, retaining per-request TTFT/completion host latency
  and scheduler iterations, and deriving nearest-rank distributions.
- Versioned JSON report schema with exact protocol, metadata, raw paired
  or three-way samples, integer microsecond timing, full execution order,
  adaptive calibration, derived summaries, and non-overwriting persistence.
- Deterministic cost-aware policy analysis exposing candidate estimates,
  selected strategy, and resulting invocation/padding shape.

- Experiment framework (`experiment/`, see `EXPERIMENTS.md`): the
  `Experiment` trait and `ExperimentRegistry` with descriptors (category,
  probe or study, clock, requirements, typed parameters); `ExperimentReport`
  (`ola-experiment/1`) with unit/clock/direction per metric, raw samples,
  recompute-on-read validation, seeded bootstrap 95% intervals, build and
  workload identity, and non-overwriting persistence; `compare_reports`;
  conversion of legacy prefill v2 reports.
- Least-squares iteration-cost calibration over a batched/split shape grid
  with bootstrap intervals, convertible to a virtual-time `CostModel`.

### `ola-cli`

- `llama-generation` reports TTFT and inter-token p50/p90/max in host time
  (fixture: TTFT 11.4 ms, gaps ~10 ms, consistent with 243.5 ms elapsed).
- `ola serve --model DIR`: loads a Llama artifact with memory-aware
  admission and serves the HTTP API, draining on Ctrl-C (`make serve`).
- `ola` binary: `experiment list`, `experiment run` (model, backend,
  parameter overrides, results directory, JSON), and `experiment compare`
  (exit 1 on a regression, 2 on an error). Wrapped by `make experiments` and
  `make experiment ID=…`.

### `ola-observe`

- Read-only observation of executor iterations and retained runtime state.
- Explicit microsecond formulas for queue latency, TTFT, prefill latency,
  service latency, end-to-end latency, and decode span.
- Inter-token gap distributions (`TokenGaps`: mean, p50/p90/p99, max) for
  terminal requests with at least two observed tokens.
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

- `Driver`: one thread owns the `Executor`, blocks when idle, and backs off
  after iterations without progress (design in `SERVING.md`).
- `RuntimeHandle`, `Submission`, `RequestHandle`, and `Reply` work under any
  async runtime or blocking on plain threads (tokio `sync` channels only).
- Streams of `Token` text deltas (via `ola_model::IncrementalDecoder`) end in
  exactly one `Finished`, `Failed`, or `Cancelled` event; results move out of
  the executor.
- Cancellation, including dropping an unfinished handle or submission,
  releases sessions and KV; `detach` opts out.
- Client-side bounded in-flight admission (`Overloaded`) and distinct
  `ServeError` kinds.
- Finished requests leave the executor; their final status stays queryable
  in a bounded table (default 1,024).
- `health()` reads state, load, and KV gauges without waiting for the
  driver; `ServiceState` covers `Ready`, `NoModels`, `Draining`, and
  `Stopped` (including after a panic); `models()` reports executability.
- `drain(grace)` finishes accepted work until a deadline on the driver's
  clock; `shutdown()` stops at once; both return the executor.
- `MonotonicClock` and `ManualClock` (`ola-core`) stamp arrivals and steps.
- `http` (axum): `/health`, `/ready`, `/v1/models`, `/v1/generate` (JSON or
  server-sent events), and `GET`/`DELETE /v1/requests/{id}`; malformed
  bodies are transport errors; each serving error has its own status and
  stable kind; closing a connection cancels, streaming or not.
- `POST /v1/completions` (OpenAI-compatible, streamed or not) refuses
  anything greedy decoding cannot honour instead of ignoring it.

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
- Loaded-model prefill-strategy benchmark with execution timing, padding,
  invocation, throughput, peak-KV, warm-up, calibrated adaptive cost, and
  rotating three-way p50 results.
- Loaded-model chunked-prefill benchmark comparing complete execution with
  three-token resumable segments and verifying identical generated output.
- Repeated mixed-workload chunk fairness probe with four unequal prompts, two
  active slots, chunk sizes 2/4/8, per-request first-token/completion latency
  and iteration summaries, and output-equivalence checks.
- Opt-in report persistence using `--results-dir` or
  `OLA_BENCH_RESULTS_DIR`, with optional device/build identity overrides.
- Real KV block-pressure benchmark.
- Real fixed-workload FIFO experiment.
- Model-independent prefill-bucketing experiment showing padding, invocation,
  and efficiency tradeoffs.
- Model-independent adaptive-prefill experiment comparing clustered and diverse
  prompt traces under an explicit invocation-cost assumption.
- Completed-job KV usage is reported as an observation, not falsely retained as
  live allocated memory.

## Correctness and verification evidence

At the working tree recorded above:

- `make verify`: **passed** (formatting, workspace checks, 253 tests, Clippy
  with warnings denied, and documentation) with zero compiler warnings; 10
  fixture tests ignored by default.
- The CLI and the workbench produce identical reports (apart from recording
  time) for prefill bucketing and the five-seed scheduler study; `ola
  experiment compare` flags a doubled FIFO p95 as a regression with exit 1
  and an identical study as clean with exit 0.
- The driver tests use a scripted model whose calls wait on a test gate,
  making cancellation and shutdown races deterministic. The driver and
  lifecycle suites (15 tests) passed 1,760 runs, 960 of them under CPU
  contention, after stress runs exposed and fixed health gauges lagging
  terminal events. The HTTP suite (8 tests over real sockets) passed 390
  runs, 240 under contention.
- A live `ola serve` run on the tiny-Llama fixture answered health, models,
  JSON and streamed generation (deltas equal to the final text), and
  drained cleanly on Ctrl-C.
- `make test-fixture` against the pinned artifact: **10 passed**. The five
  serving tests drive `ola-serve` over real sockets and compare every output
  with direct execution: JSON, streamed, and OpenAI outputs agree; four
  concurrent clients each get their single-request output; HTTP
  cancellation releases the KV cache; overload and an out-of-vocabulary
  token fail only their own requests; draining finishes a running stream
  while refusing new work (25 of 25 repeated runs passed). The others: the
  model experiments' conformance, scalar-oracle
  logits within `2e-5`, the UI integration, planned multi-request execution
  matching single-request output, and every scheduling baseline reproducing
  single-request outputs on tiny Llama with at least three distinct
  admission orders.
- Virtual-time scheduling comparisons (seed 3, 120 requests) are exactly
  reproducible and pin measured effects: priority cuts urgent p95
  completion from 159.4 ms to 23.6 ms; SJF speeds short requests and slows
  long ones and, after the P6 timing correction, raises slowdown fairness
  (0.55 vs FIFO's 0.32); backfill lowers p50 completion and starvation when the KV pool
  binds; EDF reduces deadline misses only under moderate contention.
- Isolation: cancellation (running and queued), injected failures, and
  impossible requests leave other requests' outputs unchanged and leak no
  slot or KV under every scheduler.
- A live WGPU run was not repeated in this update.

Historical evidence from earlier revisions:

- `OLA_LLAMA_CONFORMANCE_DIR=... cargo test --workspace -- --ignored`:
  **2 passed** against the pinned artifact.
- Pinned Llama scalar-oracle conformance test: **passed manually**.
- Pinned UI-to-Llama background integration test: **passed manually**.
- WGPU UI-to-executor generation integration test: **passed manually**.
- Strict-admission CPU UI-to-executor generation integration test: **passed
  manually** against the pinned fixture.
- Repeated execution-backed prefill comparison through the CPU UI adapter:
  **passed manually** against the pinned fixture, including one discarded
  calibration warm-up, four calibration pairs, one three-way warm-up, three
  order-rotated measured trials, output-equivalence validation, calibrated cost
  rendering, schema-v2 JSON persistence, and validated adaptive readback.
- Complete-versus-chunked Llama prefill through the CPU UI adapter: **passed
  manually** against the pinned fixture with a three-token segment limit,
  three resumable calls for 7/5-token prompts, tensor execution for the first
  two calls, and identical output.
- Mixed-workload chunk fairness sweep through the CPU tiny-Llama UI fixture:
  **passed manually** with four unequal prompts, two active slots, chunk sizes
  2/4/8, per-request first-token/completion iteration reporting, and identical
  generated outputs.
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
- Adaptive-prefill tests cover candidate cost arithmetic, break-even tie
  behavior, workload-level assignment, real executor grouping, and UI routing.
- Benchmark/UI tests cover invalid traces, reproducible bucketing shape
  comparison, execution-backed grouping and output equivalence, and workbench
  availability without a loaded model.
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

# List, run, and compare registered experiments
cargo run -p ola-cli -- experiment list
cargo run -p ola-cli -- experiment run scheduler-comparison --results-dir results
cargo run -p ola-cli -- experiment compare base.json candidate.json

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
  execution group. Cost-aware grouping can choose power-of-two buckets.
  Resumable Llama chunks currently advance one prompt position at a time. Each
  offset tensor-batches its active rows, but there is not yet a fused
  multi-position ragged chunk kernel. Generation remains greedy.
- Decode batches still traverse each request's physical pages independently;
  page attention is not yet a fused ragged multi-request kernel.
- Admission reserves worst-case KV demand; it does not yet predict early EOS or
  reclaim part of a reservation as a sequence approaches completion.
- Device-wide admission is enforced only when a caller supplies a capacity and
  non-KV accounting; automatic GPU-memory telemetry is not wired in yet.
- Slots freed by EOS or failure are refilled on the next iteration because
  plans are built before execution; only limit-reaching completions are reused
  in the same iteration. The TTFT effect on EOS-heavy traces is unmeasured.
- Execution plans record the model but not backend/device or kernel choices,
  and do not yet carry the cost-aware grouping policy's estimates or reasons.
- Batch timing measures from just before a model call to after its token IDs
  return to the host; work a backend leaves queued beyond that boundary is
  not attributed.
- Finished-request status is kept in memory for a bounded number of
  requests and never persisted.
- Checkpoint loading accepts one SafeTensors file only.
- RoPE scaling, tied embeddings, projection biases, and non-SiLU Llama variants
  are rejected.
- The HTTP server has no chat completions, authentication, TLS, or HTTP/2,
  and serves one model per process. OpenAI compatibility covers greedy
  `/v1/completions` only.
- Benchmark probes are foundations, not publication-quality performance claims.
- Scheduling comparisons default to an illustrative cost model and a
  synthetic model. `cost-calibration` can now supply a measured cost model,
  but the scheduling conclusions have not yet been re-run against one.
- The in-executor adaptive prefill policy still calibrates from two points;
  the least-squares fit is available to experiments but not yet wired into
  the default policy.
- Reports have a kernel-identity field, but no experiment sets it until the
  kernel track exists.
- Report comparison treats overlapping intervals as inconclusive; it does
  not yet run a paired test, so small real changes need more repetitions.

## Recommended next milestones

1. **Optimize continuous tensor batching**
   - fuse multi-position ragged chunk execution where accelerator backends
     permit it;
   - benchmark chunk-size/fairness tradeoffs under mixed decode and long-prompt
     traces;
   - wire the multi-shape calibration into the adaptive policy before
     considering it as the default;
   - optimize the request-local portion of tensor-batched paged attention;
   - keep per-request block tables, lifecycle, cancellation, and outputs
     isolated;
   - compare throughput and latency against the single-request oracle.

2. **Measurement and policy experiments**
   - re-run the scheduler and budget studies against a measured cost model;
   - gate regressions in CI with `ola experiment compare`.

3. **Paged-attention optimization**
   - benchmark page size and fragmentation tradeoffs;
   - fuse page traversal, score reduction, and value accumulation where the
     selected accelerator backend permits;
   - retain per-step numerical comparison with the full-sequence oracle.

4. **Observability (P6, next)**
   - retain per-token timestamps and report real TPOT distributions;
   - add correlation IDs and structured tracing across CLI, server, and
     experiments, so a served request can be traced through every decision
     and batch.

## Update protocol

Update this file whenever a meaningful capability is merged. Each update should:

1. replace the timestamp, source revision, and repository state;
2. move capabilities between implemented, partial, and deferred honestly;
3. record the exact verification evidence run for that revision;
4. update known limitations rather than silently removing them;
5. reorder the next milestones when the engineering boundary changes;
6. add a short entry to the history below.

## History

- **2026-10-16 22:40 BST — working tree on `748ebfb`:** P6 step 2: `TokenGaps` per-gap
  inter-token distributions replace the span average in `ola-observe`;
  `llama-generation` reports TTFT and inter-token percentiles on a real
  clock; HTTP results return per-token times and the gap summary. Fixed a
  race in the serving tests' gate.

- **2026-10-16 22:31 BST — working tree on `ff89d24`:** P6 step 1: `Executor::with_clock`
  times every model invocation and stamps tokens, first token, completion,
  and failure when the producing invocation finishes; results keep per-token
  timestamps; the simulator stamps at virtual batch finish. Negative result:
  the old iteration-start stamping had understated isolated times and
  reversed the P3 SJF fairness finding (SJF raises fairness: 0.55 vs 0.32).

- **2026-10-16 22:00 BST — working tree on `e172ea5`:** Completed P5: OpenAI-compatible
  `/v1/completions` with streaming, `finish_reason` on results, and
  end-to-end serving tests on tiny Llama that compare every HTTP output
  with direct execution.

- **2026-10-16 21:52 BST — working tree on `dac3969`:** P5 step 4: native HTTP API
  (axum) with server-sent events, distinct error statuses, and
  disconnect-cancels for streaming and JSON requests; `ola serve` and
  `make serve`.

- **2026-10-16 20:56 BST — working tree on `bdeade4`:** P5 step 3: bounded
  finished-request retention, lock-free health and readiness with
  `ServiceState`, per-model executability, panic-safe stopped state, and
  draining shutdown with a deadline on the driver's clock.

- **2026-10-16 20:50 BST — working tree on `c63196b`:** P5 steps 1–2: clocks, result and
  record retention controls, incremental detokenization, and the driver
  thread with streaming, cancellable request handles and bounded admission.
  Reconciled `todo.md`: kernel, residency, and prefix-cache instrumentation
  moved to the milestones that create them; the milestone order now matches
  the work order; bundled items split; every milestone has a testable exit
  criterion; working rules are plain bullets.

- **2026-10-16 17:47 BST — working tree on `f2fd10d`:** Completed P4: the
  `ola-experiment/1` report with typed metrics, raw samples, recomputed
  summaries and full identity; seeded bootstrap intervals and
  `compare_reports`; the experiment registry with all nine benchmarks and
  experiments ported; legacy prefill v2 import; least-squares cost
  calibration feeding virtual time; the workbench driven by the registry;
  and the `ola` CLI with a CLI/UI report-parity test.

- **2026-10-16 16:24 BST — working tree on `896b9c9`:** Completed P3:
  priority, earliest-deadline-first, and shortest-job-first schedulers with
  optional aging; admission backfill; prompt-capped and queue-adaptive
  token budgets with `BatchPressure`; a deterministic synthetic model,
  seeded traces, and virtual-time trials with perturbations;
  `compare_schedulers` with fairness and starvation metrics; isolation and
  real-model equivalence tests; and a UI scheduler-comparison experiment.

- **2026-10-16 14:46 BST — working tree on `36d4af2`:** Completed P2:
  immutable, named sub-policies; `PlanningSnapshot` with a one-time
  prompt-tokenization cache; `ExecutionPolicy` and `ComposedPolicy`;
  scheduling removed from `Runtime`; `Executor::step()` retired;
  `DecisionRecord` with ranking, admission inputs, batch selections,
  grouping alternatives, and capability fallbacks; `PlanReport` schema v2;
  determinism and conflict tests.

- **2026-10-16 13:21 BST — working tree on `a88bf75`:** Completed the P1
  execution-plan IR: plan types and validation, declared model batch
  capabilities, planned live work and admissions (slots reused in the same
  iteration only for limit-reaching completions; EOS-freed slots refill next
  iteration), plan-versus-execution divergences, `PlanReport` JSON, removal of
  `IterationPlan`, and a fixture oracle test for planned multi-request output.

- **2026-10-16 12:28 BST — working tree on `15a191c`:** Migrated the default
  CPU backend from deprecated Burn NdArray to Burn Flex, switched token readback
  to dtype-casting `try_to_vec_as` so `i32` and `i64` argmax backends agree,
  replaced deprecated tensor readbacks, fixed a Rust 1.95 Clippy lint, updated
  a stale UI fixture assertion, untracked `.codex/`, and refreshed P0
  documentation. Full verification, fixture conformance, and WGPU/CUDA
  compilation passed.

- **2026-10-14 11:22 BST — working tree on `9403ab3`:** Added the initial
  `ola-kernel` crate, paged-attention metadata and validated input contract,
  portable Burn reference implementation, direct Llama integration, focused
  tests, extension guidance, and full-workspace verification.

- **2026-10-14 15:24 BST — working tree on `9403ab3`:** Added tensor-batched
  resumable Llama chunks across unequal request cursors and segment lengths,
  atomically reserved KV rows, mixed-length chunk benchmarking, and UI output
  reporting tensor versus individual calls. CPU fixture output equivalence,
  full verification, and WGPU/CUDA feature compilation passed.

- **2026-10-14 11:13 BST — working tree on `9403ab3`:** Added bounded
  resumable Llama prefill, scheduler-visible prompt progress, shared
  prefill/decode token budgeting, cancellation-safe partial sessions, atomic
  segment reservation, complete-versus-chunked benchmarking, and a UI entry.

- **2026-10-14 10:47 BST — working tree on `9403ab3`:** Added measured
  invocation-cost calibration, fresh rotating three-way execution trials,
  adaptive output-equivalence checks, schema-v2 adaptive report persistence,
  UI summaries, and real-fixture validation.

- **2026-10-13 20:18 BST — working tree on `9403ab3`:** Added complete-workload
  prefill grouping and an auditable cost-aware policy that balances dense
  padding against token-equivalent invocation overhead, with executor, analysis,
  documentation, and UI experiment coverage.

- **2026-10-13 19:57 BST — working tree on `9403ab3`:** Added schema-versioned
  JSON benchmark reports with exact protocol/metadata, raw paired samples,
  integer timings, summaries, create-new persistence, UI/Makefile configuration,
  and real-fixture write/read validation.

- **2026-10-13 19:04 BST — working tree on `9403ab3`:** Upgraded the loaded-model
  prefill comparison to warm-up-aware alternating paired trials with retained
  raw samples, nearest-rank latency distributions, aggregate throughput, UI
  percentile reporting, and real-fixture validation.

- **2026-10-13 14:42 BST — working tree on `9403ab3`:** Added a reusable
  execution-backed prefill-strategy comparison with output-equivalence checks,
  timing, throughput, invocation, padding, and peak-KV metrics; exposed it as a
  model-dependent UI benchmark and documented measurement limits.

- **2026-10-13 14:09 BST — working tree on `c788f1a`:** Added deterministic
  unbucketed-versus-power-of-two prefill comparison, padding/invocation/
  efficiency summaries, input validation, documentation, and a runnable UI
  experiment.
- **2026-10-13 09:46 BST — `c788f1a`:** Committed the continuous-batching,
  Makefile, observability, variable-length prefill/decode, and prompt-bucketing
  milestone.
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
