# ola-ui

`ola-ui` is the terminal workbench for OlaRT. It provides a Ratatui application
shell that can grow with the runtime without coupling terminal rendering to
model execution, scheduling, or experiment code.

The initial shell contains four screens:

- **Overview** — runtime connectivity, request counts, memory, and activity;
- **Chat** — a transcript and Unicode-safe prompt editor;
- **Benchmarks** — discoverable runs, progress, and compact results;
- **Experiments** — research runs exposed through the same interaction model.

## Integration boundary

The UI consumes coherent `WorkspaceSnapshot` values and emits commands through
`UiBackend`. An adapter can invoke OlaRT in-process, communicate with a server,
or enqueue work on background threads. Long-running calls should return quickly
and expose progress through later snapshots so terminal input remains
responsive.

`LocalBackend` is the first concrete adapter. It runs work on a background
thread and currently connects:

- a local Hugging Face Llama artifact through `ola-burn`;
- cached greedy chat generation and a generation microbenchmark;
- an execution-backed unbucketed versus power-of-two Llama prefill benchmark,
  followed by calibrated, rotating unbucketed/power-of-two/adaptive trials with
  timing, throughput, padding, invocation, and peak-KV measurements;
- a loaded-model chunked-prefill benchmark comparing complete 7/5-token
  prompts with three-token resumable segments and validating identical output,
  plus a mixed-workload chunk-size sweep with per-request first-token and
  completion iterations and host latency p50/p95 from one warm-up and three
  order-rotated measured runs;
- the real `KvBlockManager` through a deterministic pressure benchmark;
- `FixedWorkload` replay through the FIFO runtime as an experiment;
- unbucketed versus power-of-two prefill shape comparison, including padding
  and invocation tradeoffs.
- a model-independent adaptive-prefill experiment showing how an explicit
  token-equivalent invocation cost changes grouping decisions by workload.

The bundled `DisconnectedBackend` remains available as a minimal adapter
template for remote or alternative integrations.

## Run

From the implementation workspace:

```text
cargo run -p ola-ui
```

The runtime and non-model probes work immediately. Enable local Llama chat by
supplying an artifact in the supported Hugging Face layout:

```text
cargo run -p ola-ui -- --model /path/to/llama-artifact
```

`OLA_MODEL_DIR` may provide the path instead. Model loading and launched work
stay off the terminal thread; the UI refreshes their state once per second.

The benchmark and experiment lists come from `ola-bench`'s
`ExperimentRegistry`; each entry is labelled as a probe (not for publication)
or a study, with its clock (virtual, host, or untimed). Report persistence is
opt-in. Pass `--results-dir <directory>` or set `OLA_BENCH_RESULTS_DIR`; every
completed run then writes an `ola-experiment/1` JSON report without
overwriting existing files:

```text
cargo run -p ola-ui -- \
  --model /path/to/llama-artifact \
  --results-dir /path/to/benchmark-results
```

Reports record the selected backend label by default; set
`OLA_BENCH_DEVICE_LABEL` for a more specific hardware identifier. The build
revision is captured at compile time from git, or from `OLA_REVISION` when set
during the build.

CPU is the default compute backend. To run model chat and its generation
benchmarks through WGPU:

```text
cargo run -p ola-ui --features wgpu -- \
  --backend wgpu --model /path/to/llama-artifact
```

The equivalent environment selection is `OLA_BACKEND=wgpu`. `cuda` is also a
supported feature/backend pair, but native CUDA execution requires a local CUDA
toolkit/NVRTC installation in addition to the NVIDIA driver.

Navigation uses `1`–`4` or `Tab`, arrow keys select table entries, `Enter`
opens or runs the selected item, `r` refreshes, and `q` exits. On the Chat
screen, `Enter` or `i` starts editing; `Enter` submits and `Esc` cancels editing.

The prefill-strategy benchmark uses four fixed token prompts with lengths 2, 3,
5, and 7 and generates at most two tokens per request. It first measures four
unbucketed/power-of-two pairs after one warm-up pair and calibrates the adaptive
invocation cost from their p50 prefill times. A fresh comparison then discards
one three-strategy warm-up and retains three measured trials, rotating strategy
order. The result shows p50 prefill-iteration time, aggregate throughput, shape
costs, and peak KV usage for unbucketed/power-of-two/adaptive execution.
Prefill time covers the full executor iteration containing prefill, not
per-request TTFT. Results remain operational rather than publication-quality
because host load and environment-level repetitions are not controlled.

The chunked-prefill benchmark displays complete versus segmented invocation
count, iteration count, accumulated prefill time, and peak KV usage. The
current Llama implementation tensor-batches active rows at each segment
offset, including unequal request cursors, while retaining an individual
single-request fallback. The mixed-workload sweep reports per-request TTFT and
completion latency distributions; they are local host observations, not
wall-clock fairness guarantees.

Ratatui's fullscreen runner owns raw mode, alternate-screen setup, restoration,
and panic cleanup. Rendering is independently testable with `TestBackend`.
