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
- the real `KvBlockManager` through a deterministic pressure benchmark;
- `FixedWorkload` replay through the FIFO runtime as an experiment;
- unbucketed versus power-of-two prefill shape comparison, including padding
  and invocation tradeoffs.

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

CPU is the default compute backend. To run model chat and its generation
benchmark through WGPU:

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

Ratatui's fullscreen runner owns raw mode, alternate-screen setup, restoration,
and panic cleanup. Rendering is independently testable with `TestBackend`.
