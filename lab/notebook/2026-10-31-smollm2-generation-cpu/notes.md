# SmolLM2 generation on CPU

- **Status:** done (characterization)
- **Track:** A runtime
- **Research question:** learning; baseline for RQ2 measurements
- **Experiment:** `llama-generation` (parameters in `config.json`)
- **Started:** 2026-10-31 at revision `563d530`
- **Entry:** `lab/notebook/2026-10-31-smollm2-generation-cpu`

> A first measurement of a real model; no prediction was recorded before
> the run.

## Question

How fast does OlaRT generate with a real 135M-parameter Llama-architecture
model on this machine's CPU?

## Protocol

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (bf16 checkpoint, run in
  f32; 30 layers, hidden size 576, 9 query and 3 KV heads), in
  `~/models/smollm2-135m-instruct`, fetched with `lab/lab fetch`.
- Machine: 12 CPU threads, 15 GiB RAM, CPU backend, release build.
- Prompt "The capital of France is", 64 greedy tokens, one run (a probe).

## Results

64 tokens; TTFT 239.2 ms; inter-token p50 186.9 ms, max 204.5 ms;
5.31 tokens/s; 3.5 MiB KV cache.

## Interpretation

About 187 ms per decoded token is far slower than the arithmetic implies:
135M parameters is roughly 0.27 GFLOP per token, a few milliseconds on 12
threads even at modest efficiency. Per-token overhead dominates, not
computation; the calibration entry
(`notebook/2026-10-31-smollm2-calibration-cpu/notes.md`) puts it at about
222 ms per model call.

## What I learned

- bf16 checkpoints did not load at first; loaders now cast to f32
  (commit `563d530`).
- One probe run says nothing about variance; the calibration study is the
  measurement to cite.

## Follow-ups

- Profile one decode step (`perf record`) to find where the per-call time
  goes: tensor allocation, kernel dispatch, KV-cache bookkeeping, or
  sampling.
- Repeat on WGPU (`2026-10-31-smollm2-generation-wgpu`).
