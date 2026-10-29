# OlaRT Progress Report

**Last updated:** 2026-10-24 14:18 BST (Europe/London)

**Source revision:** working tree based on `3363e15` on `master`

**Repository state at update:** P1 (execution-plan IR), P2 (composable
execution policy), P3 (scheduler and policy baselines), P4 (experiment
framework), and P5 (runtime driver and serving) are complete. Every
benchmark and experiment is a registered experiment with a versioned,
interval-carrying report; `ola serve` serves a model over a native HTTP API
and an OpenAI-compatible completions endpoint, with streaming,
cancellation, backpressure, health, and draining shutdown, and its outputs
match direct execution on tiny Llama. P6 (observability) is complete:
tokens and lifecycle events are stamped when the model call producing them
finishes (which corrected one P3 finding), inter-token latency is a real
per-gap distribution, iterations carry correlation IDs, every request can
be traced from arrival to outcome over HTTP or JSON Lines, and health
reports device memory, utilization, and time spent in model calls. P7
(memory and cache policy) is complete: models report KV-pool occupancy
to the planner, pool exhaustion is a typed failure, and live requests can be
preempted and resumed by recompute with output identical to uninterrupted
runs. Watermark admission with preemption cut completion p95 by 47% and
raised throughput 58% on an early-stopping workload in virtual time, with no
measurable cost where requests run to their limit. A block-level prefix
cache reuses shared prompt prefixes with output identical to recomputation.
Swapping was evaluated and deferred in favour of recompute, a page-size
sweep found 8–16-token pages the best trade-off between waste and reuse,
and replay under pressure, preemption, eviction, and cancellation is
deterministic. Current kernel-track work adds a backend-neutral capability
contract and deterministic provider registry. It selects optimized providers
by declared constraints and stable priority, or records a typed reason for
portable fallback. Model artifacts can now publish target/dtype/attention/page
profiles; policy resolves a provider only after other policies settle exact
rows, and validated execution plans plus schema-v6 plan reports retain the
requirements, provider, and fallback diagnostics. Runtime execution now passes
that exact provider into individual and tensor-batched Llama decode/resumable
prefill calls. Actual execution records and request traces now add provider
identity, target, call count, and clock-gated synchronous host duration;
model-backed experiment variants retain the canonical provider set they
actually invoked. Kernel-track steps 1–4 are committed through `5cd3da2`.
The current working tree completes the first fused CubeCL paged-attention
implementation: Llama stores each layer in flattened physical K/V tensors,
passes a validated device block table, and CubeCL fuses page traversal, score
calculation, online softmax, and value accumulation into one dispatch. The old
per-page tensor-sequence prototype was removed after a compile probe showed
CubeCL 0.11 cannot represent it as a launch argument. Tensor-batched Llama
decode now routes through a batch kernel boundary; CubeCL stacks request-local
queries and block tables into one dispatch while portable providers preserve
the independent-request fallback. The same dispatch is conformance-tested with
ragged logical block counts and per-request token lengths; cross-pool grouping
remains deferred.

This is the authoritative, timestamped snapshot of OlaRT's implementation
progress. The root `README.md` describes the project vision, the local and
untracked `IMPLEMENTATION.md` describes the long-term build plan, and
`ola-rt/README.md` documents stable implementation semantics. This report
answers the narrower question: **what works today, what evidence supports it,
and what should be built next?**

## Executive summary

OlaRT has moved beyond repository scaffolding into its first complete inference
vertical slice. It can load a narrowly supported Hugging Face Llama checkpoint,
tokenize a prompt, execute prefill, perform cached greedy or seeded
temperature/top-p decoding, report
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
| Execution policy | Kernel-aware | `ExecutionPolicy` plans from an immutable snapshot through `&self`; `ComposedPolicy` combines scheduler, admission, batch, grouping, and kernel-registry decisions with stable descriptors and full decision records |
| Execution planning | Kernel-aware | Each validated `ExecutionPlan` records admissions, invocations, kernel requirements/selections, deferrals, and predicted releases before atomic commit; divergences recorded; schema-v6 `PlanReport` JSON |
| Batch composition | Adaptive baseline working | Auditable token budget plus unbucketed, power-of-two, and cost-aware workload grouping |
| Observability | Kernel-aware | Model-call and synchronous kernel host timing, executed provider identity/call counts, per-token timestamps and inter-token distributions, iteration correlation IDs, per-request traces, device memory/utilization, and busy fraction; shown not to change execution |
| Workloads and metrics | Calibrated execution probe | Fixed replay, cost calibration, rotating three-way trials, repeated mixed-workload chunk fairness, per-request TTFT/completion latency distributions, raw samples, summaries, schema-v2 JSON |
| Experiment framework | Working | `ExperimentRegistry` of twelve experiments (probes and studies); `ola-experiment/1` reports with typed metrics, raw samples, recomputed summaries, seeded bootstrap intervals, and full identity; report comparison with interval-separated verdicts; least-squares cost calibration feeding virtual time; legacy v2 import |
| Experiment CLI | Working | `ola experiment list/run/compare` with CI exit codes; reports identical to the workbench's |
| Kernel abstraction | Observable planned execution | Validated paged-attention contract, model profiles, deterministic provider registry, typed portable fallback, exact-provider delegation, execution/trace identity and synchronous host timing, and experiment variant identity |
| Burn integration | Working | Backend-neutral adapter with CPU, opt-in WGPU, and opt-in native CUDA features |
| Llama loading | Narrow path working | Local config, tokenizer, single SafeTensors file, strict validation |
| Llama execution | Working | Embedding, GQA attention, RoPE, SwiGLU, RMSNorm, LM head |
| Generation sessions | Working | Prefill, cached one-token decode, greedy or seeded temperature/top-p selection, EOS/limit checks |
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

- Fine-grained `PagedAttentionKernel` contract without scheduler, model, or
  allocation ownership.
- Validated single-sequence decode inputs with explicit GQA dimensions and
  populated logical KV pages.
- Stable operation, implementation, and hardware-target metadata for future
  diagnostics and benchmark reports.
- Portable Burn online-softmax implementation extracted from the Llama model
  and retained as the numerical baseline.
- Direct use by Llama cached decode and resumable-prefill attention.
- Backend-neutral kernel requirements covering target, dtype, attention head
  width, GQA ratio, and logical page count.
- Validated provider capability declarations with deterministic priority and
  stable-name tie-breaking independent of registration order.
- An unconditional portable provider and typed rejection records explaining
  fallback when no optimized provider satisfies every constraint.
- Feature-gated CubeCL providers for CPU, WGPU, and CUDA selection targets.

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
- `ola experiment run --trace FILE` for traceable experiments
  (`llama-generation`, `scheduler-comparison`), one labelled run per
  executor.
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
- `Tracer`: per-request JSON Lines traces of arrival, ranking, admission,
  backfill, batch selection, invocations, tokens, and outcome, keyed by
  request and iteration ID.
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
- Request traces: kept per request (bounded by retention) and served at
  `GET /v1/requests/{id}/trace`; `ola serve --trace FILE` streams them as
  JSON Lines; a failing sink is reported in health.
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

- CubeCL fused paged-attention implementation: 19 kernel tests passed with the
  CPU feature, including an optimized-versus-portable comparison over two
  logical pages, a partial final page, grouped query heads, and reversed
  physical block mapping. The WGPU numerical conformance test passed on the
  local integrated GPU. CUDA and WGPU workspace all-target feature checks both
  passed. The 27 local `ola-burn` tests and its README doctest passed (3 tests
  requiring the pinned external model fixture remain ignored).
- CubeCL optimization probe: directly compiling the old per-page prototype
  failed because CubeCL 0.11 cannot use `Sequence<Tensor<F>>` launch arguments;
  the prototype also bypassed Burn Fusion conversion. It was removed, the
  limitation was recorded, and CubeCL CPU feature compilation passes again.
- Kernel-track step 4 validation: 247 tests passed across kernel, model,
  runtime, observation, Burn, and benchmark crates (3 external-fixture tests
  ignored), including exact-provider call aggregation, trace JSON round-trip,
  and report comparison on kernel identity. WGPU and CUDA all-target feature
  compilation, workspace checks, Clippy with warnings denied, and
  documentation passed. Clockless execution disables wall-time sampling so
  replay records retain deterministic provider/call data.
- Kernel-track step 3 validation: the focused kernel/model/runtime/Burn suites
  passed (158 tests, with 3 ignored fixture tests); all 15 CubeCL CPU kernel
  tests passed, including numerical conformance and compiled-provider
  selection; WGPU and CUDA feature compilation passed; formatting, workspace
  checks, Clippy with warnings denied, and documentation passed. A final
  workspace-test rerun completed at the process level, but its terminal adapter
  did not return an exit status, so it is not counted as additional evidence.
- `make verify`: **passed** (formatting, workspace checks, 289 tests, Clippy
  with warnings denied, and documentation) with zero compiler warnings; 13
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
- `make test-fixture` against the pinned artifact: **13 passed**, including
  prefix-cache reuse matching recomputation with the real tokenizer,
  preempted requests on a tight pool matching their solo output, including
  a served request traced from arrival to completion with each token
  carrying its invocation's finish time. The five
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
- CubeCL 0.11 cannot launch a sequence of tensor page bindings. Removing
  pre-dispatch concatenation therefore requires a device-resident flattened K/V
  allocation plus page-offset metadata, not the removed per-page prototype.
- The fused CubeCL kernel currently recomputes each query-key score for every
  output dimension. The flattened Burn KV pool also needs study-grade latency
  and memory profiling before any performance benefit is claimed.
- Native CUDA is wired and compiles, but this machine cannot execute it until a
  compatible CUDA toolkit providing NVRTC is installed. WGPU is the currently
  validated local GPU path.
- Physical page tensors are materialized lazily and retained at the pool's
  high-water mark for deterministic reuse.
- Prefix caching is opt-in per model pool; worst-case reservations count a
  shared prefix once per request (conservative). There is no swapping or
  quantization.
- Variable-length prefill uses right padding to the longest prompt in its
  execution group. Cost-aware grouping can choose power-of-two buckets.
  Resumable Llama chunks currently advance one prompt position at a time. Each
  offset tensor-batches its active rows, but there is not yet a fused
  multi-position ragged chunk kernel. Generation supports greedy and seeded
  temperature/top-p selection.
- Decode batches still traverse each request's physical pages independently;
  page attention is not yet a fused ragged multi-request kernel.
- Admission reserves worst-case KV demand; it does not yet predict early EOS or
  reclaim part of a reservation as a sequence approaches completion.
- Preemption and watermark admission are opt-in
  (`ComposedPolicy::with_preemption`, `WatermarkAdmission`); the default
  stays conservative worst-case reservation. The `kv-pressure` gains are
  virtual-time results on the synthetic model and need confirming on a real
  model under load. Without preemption, running out of KV blocks still fails
  the request.
- Preemption recomputes: a resumed request re-prefills its prompt and output
  (15.7% more input tokens in `kv-pressure`). Swapping KV to host memory was
  evaluated and deferred; its estimated advantage on large models needs a
  GPU to measure.
- The page-size sweep is a virtual-time result on one shared-prefix
  workload; the best page size on a real model also depends on kernel
  efficiency per page.
- Device-wide admission is enforced only when a caller supplies a capacity and
  non-KV accounting. GPU memory is now reported for health, but admission
  does not yet consume it.
- CubeCL device telemetry was validated on CubeCL's CPU runtime and compiles
  for WGPU; this machine's NVIDIA driver is unusable, so no GPU memory or
  utilization reading has been observed.
- Slots freed by EOS or failure are refilled on the next iteration because
  plans are built before execution; only limit-reaching completions are reused
  in the same iteration. The TTFT effect on EOS-heavy traces is unmeasured.
- Invocation traces carry the selected provider, target, call count, and
  synchronous host duration inside `PagedAttentionKernel::execute`. That is
  not device-event timing: asynchronous backend work is included only when the
  provider synchronizes before returning.
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
- Traces of simulated runs carry iteration-start times: the simulator's
  executor has no clock, and only its summaries use the corrected virtual
  finish times. Cancellation perturbations in simulations are not traced.
- Scheduling comparisons default to an illustrative cost model and a
  synthetic model. `cost-calibration` can now supply a measured cost model,
  but the scheduling conclusions have not yet been re-run against one.
- The in-executor adaptive prefill policy still calibrates from two points;
  the least-squares fit is available to experiments but not yet wired into
  the default policy.
- Model-backed experiment variants populate kernel identity from providers that
  were actually called. Full-prefill-only calibration correctly remains
  unlabelled because it does not invoke paged decode attention.
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

3. **Paged-attention optimization (Kernel track, next)**
   - re-run the page-size sweep with real kernel cost per page;
   - fuse page traversal, score reduction, and value accumulation where the
     selected accelerator backend permits;
   - retain per-step numerical comparison with the full-sequence oracle.

4. **Model and runtime usability (P8, after the Kernel track)**
   - broaden model and checkpoint support;
   - confirm the P7 virtual-time results (watermark admission, prefix
     reuse, page size) on a real model under load.

## Update protocol

Update this file whenever a meaningful capability is merged. Each update should:

1. replace the timestamp, source revision, and repository state;
2. move capabilities between implemented, partial, and deferred honestly;
3. record the exact verification evidence run for that revision;
4. update known limitations rather than silently removing them;
5. reorder the next milestones when the engineering boundary changes;
6. add a short entry to the history below.

## History

- **2026-10-26 — working tree:** Documented the solver-first adaptive route in
  `ADAPTIVE_OPTIMIZATION_PLAN.md`. The plan retains the safety/observation
  foundation, refactors policy replacement behind a factory, starts with an
  exact finite candidate optimizer, and defers MILP/LP/convex adapters until
  they match the dependency-free oracle.

- **2026-10-26 — working tree:** Added a read-only `AdaptivePolicy::theta`
  accessor and removed unnecessary descriptor initialization from construction.
  Runtime tests passed (121 passed).

- **2026-10-26 — working tree:** Updated the offline adaptive comparison to
  exercise a real observation hook that mutates a bounded θ field at iteration
  boundaries. The deterministic comparison test passed.

- **2026-10-26 — working tree:** Added regression coverage proving stale or
  otherwise invalid adaptive candidate plans are replaced by the safe fallback
  before returning from `plan`. Adaptive runtime tests passed (7 passed).

- **2026-10-26 — working tree:** Made adaptive plan validation intrinsic to
  `AdaptivePolicy`: candidate and fallback plans are validated against the
  planning snapshot and their KV-capacity guarantees before delegation returns.
  Runtime tests passed (120 passed).

- **2026-10-26 — working tree:** Added deterministic offline evaluation
  coverage for `AdaptivePolicy` through the existing virtual-time scheduling
  comparison harness. Adaptive and FIFO runs produce reproducible summaries;
  `cargo test -p ola-bench --lib` passed (68 passed).

- **2026-10-26 — working tree:** Added an observation-only adaptive update
  hook. It receives committed iteration timing observations and may update θ
  for the next planning boundary; planning remains read-only. Runtime tests
  passed (120 passed).

- **2026-10-26 — working tree:** Added a direct adaptive reward-observation
  helper that derives per-request TTFT and completion latency from committed
  events and arrival context. Runtime tests passed (119 passed).

- **2026-10-26 — working tree:** Added `AdaptiveRequestObservation` derived
  from committed iteration events and decision-record arrival timestamps. It
  exposes per-request first-token and completion timestamps plus TTFT and
  completion latency in microseconds. Runtime tests passed (119 passed).

- **2026-10-26 — working tree:** Closed the adaptive limits boundary: active
  slots and resumable-prefill chunk size remain executor hard ceilings, while
  adaptive policies may choose lower effective values through their plans.
  Updated `ADAPTIVE_POLICY.md` to distinguish implemented foundation from the
  remaining research loop.

- **2026-10-26 — working tree:** Added deterministic observation-time
  randomness to `AdaptivePolicy`. Seeds and random state are exposed in the
  policy descriptor, while state advances only from committed `observe` calls.
  Runtime tests passed (118 passed).

- **2026-10-26 — working tree:** Added planning-visible request arrival times
  to `DecisionRecord`, covering queued, prefilling, and decoding requests.
  Adaptive observers can now derive waiting, TTFT, and completion rewards from
  the committed iteration without changing executor construction. Runtime and
  benchmark tests passed (117 and 67 passed).

- **2026-10-26 — working tree:** Added adaptive candidate validation and typed
  rejection handling. A validator can reject a candidate before commit, routing
  it to the safe fallback; candidate policy errors use the same path. Runtime
  tests passed (117 passed).

- **2026-10-26 — working tree:** Extended `AdaptivePolicy` with an optional
  safe fallback. Candidate policy errors now fall back to the configured policy
  while retaining adaptive identity in the decision record; both policies are
  observed and safety capabilities are combined conservatively. Runtime tests
  passed (116 passed).

- **2026-10-26 — working tree:** Added the initial `AdaptivePolicy` wrapper.
  It delegates planning to an inner policy while recording sorted θ parameters
  and the committed observation counter in every decision descriptor. Runtime
  tests passed (115 passed).

- **2026-10-26 — working tree:** Wired `ExperimentRunConfig` into the CLI via
  `ola experiment run <id> --config FILE`; explicit `--set` overrides take
  precedence and mismatched experiment IDs are rejected. CLI integration tests
  passed (7 passed).

- **2026-10-26 — working tree:** Added serializable `ExperimentRunConfig`
  selection with JSON loading and registry execution, including malformed JSON
  and parameter-override coverage. `cargo test -p ola-bench --lib` passed (67
  passed).

- **2026-10-26 — working tree:** Added `ExecutorBuilder::from_json` as the
  direct runtime configuration-file boundary, with invalid-JSON rejection
  coverage. `cargo test -p ola-runtime --lib` passed (114 passed).

- **2026-10-26 — working tree on `a685411`:** Extended `ExecutorConfig` with
  serializable batch-policy selection for unbounded, FIFO token-budget, and
  mixed token-budget batching. Invalid zero or inconsistent budgets fail before
  executor construction with a typed configuration error. Added policy
  selection coverage; `cargo test -p ola-runtime --lib` passed (112 passed).

- **2026-10-26 — working tree on `a685411`:** Extended `ExecutorConfig` with
  an optional device label. Configured executors now expose a planning
  `DeviceSnapshot` with that label and memory capacity, while live backend
  telemetry remains separately injectable. Added builder/snapshot coverage;
  `cargo test -p ola-runtime --lib` passed (112 passed).

- **2026-10-26 — working tree on `a685411`:** Made backend-neutral model
  definitions serializable. Core model IDs, architecture IDs, descriptors,
  kinds, and modalities now support Serde, while `ModelSource` and
  `ModelDefinition` round-trip local and Hugging Face artifact references.
  Added a JSON round-trip regression test; `cargo test -p ola-core --lib`
  passed (4 passed), `cargo test -p ola-model --lib` passed (11 passed), and
  `cargo test -p ola-runtime --lib` passed (112 passed).

- **2026-10-26 — working tree on `a685411`:** Added `ExecutorBuilder` as the
  stable high-level construction surface. It composes `ExecutorConfig` with a
  dynamic execution policy and validates limits through the existing executor
  errors, while model loading remains explicit. Builder construction and
  policy selection are covered; `cargo test -p ola-runtime --lib` passed (112
  passed).

- **2026-10-26 — working tree on `a685411`:** Added `ExecutorConfig`, a
  serializable configuration boundary for active-request limits, resumable
  prefill chunk size, and device-memory capacity. `Executor::from_config`
  applies those values while leaving model artifacts and dynamic policy
  objects explicit runtime dependencies. Added JSON parsing and construction
  tests; `cargo test -p ola-runtime --lib` passed (111 passed).

- **2026-10-26 — working tree on `a685411`:** Added explicit backend-neutral
  model placement metadata with `Host`, named `Device`, and `Unspecified`
  states. Placement changes are rejected while requests use the model and do
  not claim to migrate backend buffers; resident placement can be queried
  alongside lifecycle events. Runtime placement regression coverage passes with
  the 109-test `ola-runtime` suite.

- **2026-10-26 — working tree on `a685411`:** Added the backend-neutral
  `ModelMemoryProfile` boundary for resident weights and persistent workspace.
  Executors now add declared bytes on model registration/attachment and
  release them on unload; models without a profile remain explicitly unknown.
  Resource accounting is covered by the lifecycle test; `cargo test -p
  ola-model --lib` passed (10 passed) and `cargo test -p ola-runtime --lib`
  passed (109 passed).

- **2026-10-26 — working tree on `a685411`:** Exposed model residency and
  lifecycle observability from the executor. Callers can enumerate resident
  executable IDs and read ordered loaded/unloaded events, while planning
  snapshots continue to distinguish attached and detached metadata. Lifecycle
  tests cover load, blocked unload, successful unload, and reattach;
  `cargo test -p ola-runtime --lib` passed (109 passed).

- **2026-10-26 — working tree on `a685411`:** Added explicit executable model
  unload/reload lifecycle operations to the runtime executor. Unload refuses
  models referenced by queued, active, or prefilling requests, retains model
  metadata for a later reattach, exposes stable loaded-model IDs, and reports
  typed not-loaded and busy errors. Added lifecycle regression coverage;
  `cargo test -p ola-runtime --lib` passed (109 passed).

- **2026-10-26 — working tree on `a685411`:** Added configured attention and
  MLP projection-bias support for Llama. Linear modules now preserve the
  checkpoint-declared bias shape, while activation handling remains explicit:
  only gated `silu` is supported and other activation names are rejected until
  a checkpoint-backed conformance case defines their semantics. Added a
  module-construction regression test; `cargo test -p ola-burn --lib` passed
  (34 passed, 3 ignored) and `cargo test -p ola-runtime --lib` passed (108
  passed).

- **2026-10-26 — working tree on `a685411`:** Added selected linear RoPE
  scaling for Llama. Configuration parsing validates the scaling object and
  factor, rotary positions are divided by that factor across full, cached,
  resumable, and batched attention paths, and unsupported scaling types remain
  rejected. Added config and nonzero-position regression tests; `cargo test
  -p ola-burn --lib` passed (33 passed, 3 ignored).

- **2026-10-26 — working tree on `a685411`:** Added selected tied-embedding
  support for Llama. The output projection now reuses the input embedding
  matrix, the non-persistent tie flag preserves the checkpoint hierarchy, and
  SafeTensors loading permits only the expected absent `lm_head.weight` while
  still rejecting all other missing tensors. Added a projection-equivalence
  regression test; `cargo test -p ola-burn --lib` passed (31 passed, 3
  ignored).

- **2026-10-26 — working tree on `a685411`:** Added Llama probabilistic
  sampling with session-owned SplitMix64 state. Temperature zero remains
  greedy; positive temperature applies nucleus `top_p` filtering and weighted
  sampling. Single- and tensor-batched prefill/decode paths use the same
  selector, and seeded sessions reproduce token sequences in focused tests.

- **2026-10-26 — working tree on `a685411`:** Added indexed SafeTensors
  loading for `model.safetensors.index.json` and
  `pytorch_model.safetensors.index.json`. The loader validates referenced
  shards, deduplicates their paths, merges lazy tensor snapshots with
  duplicate-name detection, and applies the complete checkpoint once; the
  existing single-file path remains unchanged.

- **2026-10-26 — working tree on `a685411`:** Extended CubeCL ragged decode
  batching to partition mixed inputs into homogeneous physical-pool/config
  launch groups. Compatible requests still share one padded-table dispatch,
  while incompatible or unsupported requests retain request-local execution;
  outputs are restored to caller order. This advances provider-side launch
  grouping without claiming the remaining scheduler-side grouping work.

- **2026-10-24 — working tree on `3363e15`:** Routed tensor-batched resumable
  prefill attention through the same batched provider operation used by decode.
  Unequal prompt chunks still execute at their active offsets and retain output
  equivalence; a true multi-position ragged attention contract remains future
  work. Focused unequal-position chunk tests pass.

- **2026-10-24 — working tree on `3363e15`:** Extended kernel telemetry
  measurement across the new batch boundary. Measured providers now record one
  synchronous call and duration per batched dispatch, including portable
  fallback execution inside that operation; the provider-routing test was
  updated to distinguish dispatch count from inner request count.

- **2026-10-24 — working tree on `3363e15`:** Closed the optimized-path oracle
  comparison for single and ragged batched CubeCL attention. Each optimized
  result is checked against portable single-request execution; performance
  measurement is still intentionally deferred until warm-up and repetition
  reporting is wired.

- **2026-10-24 — working tree on `3363e15`:** Extended the ignored WGPU kernel
  microbenchmark with warm-up/repetition timing for one-page, two-page, and
  two-request batched attention. Measurements include output synchronization and
  readback and are labeled as host-call latency; publication-grade report
  integration remains outstanding.

- **2026-10-24 — working tree on `3363e15`:** Added a regression test for
  batched provider telemetry: one `execute_batch` operation records one
  dispatch even when the portable fallback processes multiple request inputs.
  This keeps report call counts aligned with actual provider launches.

- **2026-10-25 — working tree on `3363e15`:** Added the optional
  `paged-attention-kernel` `ola-experiment/1` host study under the
  `ola-bench/cubecl-cpu` feature. It records warm-ups, repetitions, workload
  parameters, portable and CubeCL kernel identities, latency distributions,
  batch/page dimensions, and maximum absolute oracle error. Default and
  CubeCL-feature registry tests pass; accelerator report execution remains
  pending.

- **2026-10-25 — working tree on `3363e15`:** Integrated the kernel study into
  the benchmark registry behind `ola-bench/cubecl-cpu`. Default builds remain
  lightweight; feature-enabled runs validate the report protocol and execute
  65 benchmark tests, including the registered paged-attention study.

- **2026-10-25 — working tree on `3363e15`:** Added the opt-in
  `ola-bench/wgpu` feature to run the same versioned paged-attention study on
  WGPU. The local adapter test completed with one page, one request, one
  warm-up, and five measured repetitions; report protocol and oracle validation
  passed.

- **2026-10-25 — working tree on `3363e15`:** Extended the kernel study
  workload to 23 populated tokens across configurable page sizes, including a
  partial final page. Reports now retain logical page count and padding-token
  waste beside latency, batch size, and oracle error. The CPU registry study and
  the local WGPU integration study both pass.

- **2026-10-26 — working tree on `3363e15`:** Made CubeCL batch execution
  safe for incompatible request groups. Different attention shapes or physical
  KV geometries now use independent provider launches instead of failing after
  batch preparation; homogeneous groups retain the fused dispatch. Conformance
  covers the fallback path, while explicit cross-pool launch grouping remains.

- **2026-10-26 — working tree on `3363e15`:** Added explicit physical KV-pool
  identity to `PagedKvBlocks`. Llama cache views from one pool share that
  identity, while independently created pools are prevented from entering the
  fused batch path accidentally; incompatible and cross-pool groups fall back
  to independent launches. Focused CubeCL, Llama, and WGPU benchmark checks
  pass.

- **2026-10-26 — working tree on `3363e15`:** Audited ragged resumable
  prefill. Unequal chunk lengths and unequal absolute cursors already execute
  through active-offset batching and match complete independent prefill, so
  the runtime/model item is complete. A separate follow-up remains for a
  causal multi-position kernel launch with per-query valid-length metadata.

- **2026-10-27 — working tree on `3363e15`:** Extended the versioned
  paged-attention study with configurable unused physical KV blocks. Reports
  now include physical block count, unused blocks, and pool-fragmentation ratio
  alongside page padding and latency. The study deliberately does not infer
  accelerator occupancy from host timing; profiler-backed occupancy remains.

- **2026-10-27 — working tree on `6ef1f7a`:** Closed the offline adaptive
  evaluation TODO. `adaptive-policy-study` trains θ on seed 0, evaluates the
  fixed candidate against FIFO on held-out seeded virtual-time traces, and
  records reproducible p95 latency, throughput, and starvation samples. The
  feature-enabled registry test passed.

- **2026-10-27 — working tree on `ec0dac0`:** Closed the second-architecture
  TODO. The architecture registry now has tested Burn Llama and Mistral
  factories; Mistral includes configuration, model, tokenizer, paged KV cache,
  batching, cached-prefix, and preemption paths with focused session/factory
  coverage.

- **2026-10-27 — working tree on `cb89dad`:** Closed the experiment
  configuration portion of the remaining P5 configuration item. JSON
  `ExperimentRunConfig` already selects a registered experiment and applies
  parameter overrides with validation; adaptive θ/domain/factory construction
  remains explicit code by design until its safe serialization boundary is
  defined.

- **2026-10-27 — working tree on `d32eeb9`:** Added the adaptive policy data
  configuration boundary. `AdaptivePolicyConfig` serializes θ, seed,
  optimization interval, and runtime constraints, then resolves parameter names
  explicitly before building an executable policy. Duplicate and unknown names
  are rejected; factories, optimizers, validators, and hooks remain code-only.
  Two focused round-trip/validation tests pass.

- **2026-10-27 — working tree on `09c6f71`:** Closed the ragged multi-request
  decode investigation. CubeCL now partitions mixed batches into homogeneous
  same-pool launches, handles incompatible and cross-pool inputs safely, and
  restores the caller's request order. Focused CPU conformance covers ragged
  logical lengths, incompatible geometry, and same-shaped independent pools.

- **2026-10-28 — working tree on `dd32b65`:** Fused ragged multi-position
  resumable prefill at the model layer. Llama and Mistral now process each
  transformer layer's valid chunk positions together, write contiguous K/V
  ranges, and issue one grouped paged-attention batch with a causal prefix per
  position. Full Llama (15) and Mistral (13) session suites pass, including
  unequal chunks, cached prefixes, sliding windows, and preemption.

- **2026-10-28 — working tree on `4d12980`:** Closed the page-size and KV
  fragmentation benchmark portion. The CPU study regression now exercises
  nonzero physical-pool slack and validates reported physical-block,
  unused-block, and fragmentation metrics; WGPU retains the same workload
  path. Accelerator occupancy is intentionally split into a profiler-only
  follow-up.

- **2026-10-24 14:18 BST — working tree on `3363e15`:** Extended batched
  CubeCL conformance to ragged requests with different logical block counts and
  token lengths sharing one physical KV geometry. Per-request token counts and
  padded block-table rows are honored; incompatible physical pools remain a
  launch-grouping follow-up. Focused kernel and Llama batch tests pass.

- **2026-10-24 — working tree on `3363e15`:** Added the tensor-batched decode
  kernel boundary. CubeCL now executes multiple request-local paged-attention
  queries in one dispatch using a padded device block table and per-request
  token counts; Llama routes tensor-batched decode through it. Portable
  execution remains the correctness fallback. Kernel and Burn tests pass,
  including batched CubeCL conformance and unequal-position Llama decode.

- **2026-10-23 18:13 BST — working tree on `3363e15`:** Completed the first
  fused CubeCL paged-attention path. Llama now uses flattened per-layer K/V
  tensors and validated device block tables; CubeCL traverses non-contiguous
  physical pages and combines score calculation, online softmax, and value
  accumulation in one dispatch. CPU and WGPU conformance, `ola-burn` tests,
  and CUDA/WGPU workspace checks pass. Score calculation is still repeated per
  output dimension, and no performance claim is made pending measurement.

- **2026-10-23 13:08 BST — working tree on `5cd3da2`:** Probed and rejected the
  unwired per-page CubeCL prototype: `Sequence<Tensor<F>>` is not a valid
  CubeCL 0.11 launch argument and its primitive conversions bypass Burn Fusion.
  Removed the obsolete file, documented the flattened device-buffer/page-offset
  direction, and revalidated CubeCL CPU feature compilation.

- **2026-10-23 12:46 BST — working tree on `c5c22da`:** Completed kernel-track
  step 4: measured provider wrappers aggregate synchronous call count and host
  duration per actual execution batch; request traces emit provider identity,
  target, calls, and optional nanoseconds; clockless replay keeps timing absent
  and deterministic; model-backed experiment variants retain providers that
  were actually invoked, and report comparison surfaces kernel changes. All
  247 affected-crate tests pass (3 fixture tests ignored), and WGPU/CUDA
  all-target feature compilation, workspace checks, Clippy, and documentation
  pass.

- **2026-10-23 12:11 BST — working tree on `14aabe6`:** Completed kernel-track
  step 3: exact provider handles survive planning, `ModelExecutionContext`
  routes them through individual and tensor-batched decode/resumable prefill,
  Llama no longer chooses CubeCL through compile-time branches, compiled
  providers populate the default policy registry, and a counting-provider
  integration test proves the planned implementation executes. Default core
  tests pass, all 15 CubeCL CPU kernel tests pass, and WGPU/CUDA targets compile.

- **2026-10-22 12:05 BST — working tree on `b78bb5e`:** Completed kernel-track
  step 2: model paged-attention profiles, per-decode/resumable-prefill
  requirements and provider selections in `ExecutionPlan`, post-policy
  capability matching, validation against rows and model facts, schema-v6 plan
  report fields, and conflicts proving budget-excluded and memory-deferred work
  cannot affect selection.
  All 108 runtime tests, 74 benchmark tests, and the relevant core, model,
  kernel, observation, and Burn tests pass.

- **2026-10-20 12:02 BST — working tree on `6f24463`:** Accepted the kernel
  routing design and completed kernel-track step 1: backend-neutral invocation
  requirements, validated provider capabilities, deterministic registry
  selection, typed portable-fallback reasons, portable and CubeCL provider
  adapters, and removal of the kernel/observation/runtime dependency cycle by
  moving backend-neutral telemetry to `ola-core`. Core, kernel, and observation
  tests pass; all 13 CubeCL CPU kernel tests pass, including numerical
  conformance and live allocator telemetry.

- **2026-10-18 10:00 BST — working tree on `2248703`:** P7 step 5 closes P7:
  swapping evaluated with a cost model and deferred (recompute costs 15.7%
  more input tokens in `kv-pressure`); the `kv-page-size` study
  (fragmentation 1.1% at 4-token pages to 19.9% at 64; reuse and p95 hold
  through 32 and degrade at 64); replay determinism under pressure,
  preemption, prefix eviction, and cancellation.

- **2026-10-17 21:31 BST — working tree on `e278b3d`:** P7 step 4: block-level prefix
  cache (reference counts, verified chained hashes, LRU eviction), planned
  prefix attachment, shared-prefix traces, and the `prefix-cache` study
  (85% fewer prefill tokens with a 256-token shared prefix, 45% with 32);
  outputs identical to recomputation on the synthetic model, a random
  Llama, and the fixture. Fixed a latent Llama bug where a prefill chunk
  spanning several blocks panicked.

- **2026-10-17 20:10 BST — working tree on `754c6e8`:** P7 step 3: `WatermarkAdmission`,
  preempt-instead-of-fail recovery for allocations the planner did not
  predict (only requests that could never fit fail), the `early_stopping`
  trace, and the `kv-pressure` study: on early-stopping requests watermark
  admission with preemption cut completion p95 473 → 250 ms and raised
  throughput 58% with separated intervals; on full-length workloads it was
  neutral.

- **2026-10-17 18:27 BST — working tree on `3bbbe0d`:** P7 step 2: preemption by
  recompute (plan action, runtime transition, resume without repeated
  tokens, `PreemptionPolicy` with newest-first and lowest-priority-first,
  pool-demand planning, physical admission check when preemption is on),
  with output equivalence shown on the synthetic model, a random-weight
  Llama, and the tiny-Llama fixture; the scheduler study is unchanged.

- **2026-10-17 14:41 BST — working tree on `a93f220`:** P7 step 1: `KvPoolStatus` from
  models into the planning snapshot, typed KV-capacity exhaustion recorded
  as a plan divergence (`PlanReport` schema 4), and a real block pool in
  the synthetic model; the scheduler study's samples are unchanged.

- **2026-10-17 14:14 BST — working tree on `81513d9`:** Completed P6: `DeviceTelemetry`
  with CubeCL memory and utilization (validated on CubeCL's CPU runtime,
  WGPU compiles), busy time and busy fraction, telemetry in `/health` and
  `ola serve`, read-only equivalence tests, and an end-to-end trace test on
  the fixture.

- **2026-10-17 13:05 BST — working tree on `c807fea`:** P6 step 4: traces kept by the
  driver and served over HTTP, `--trace` JSON Lines sinks for `ola serve`
  and traceable experiments, and a simulator observation hook.

- **2026-10-17 12:31 BST — working tree on `898789c`:** P6 step 3: `IterationId`
  correlation IDs on iterations, plans, and `PlanReport` (schema 3), and
  the `ola-observe` trace builder with JSON Lines output.

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
