# SmolLM2 generation on WGPU

- **Status:** done (characterization)
- **Track:** G kernels
- **Research question:** learning; H3 baseline
- **Experiment:** `llama-generation` (parameters in `config.json`)
- **Started:** 2026-10-31 at revision `563d530`
- **Entry:** `lab/notebook/2026-10-31-smollm2-generation-wgpu`

> A first measurement; no prediction was recorded before the runs.

## Question

How does generation on the GTX 1650 through WGPU compare with the CPU
(`notebook/2026-10-31-smollm2-generation-cpu/notes.md`)?

## Protocol

Same model, prompt, and 64 greedy tokens as the CPU entry; `--backend wgpu`
on a GTX 1650 (4 GB, Turing, Vulkan), release build. Two runs, in separate
processes.

## Results

| Run | TTFT | Inter-token p50 | Inter-token max | Throughput |
| --- | ---: | ---: | ---: | ---: |
| 001 | 3.16 s | 56.4 ms | 449.6 ms | 8.98 tok/s |
| 002 | 0.80 s | 57.3 ms | 125.7 ms | 14.38 tok/s |
| CPU | 0.24 s | 186.9 ms | 204.5 ms | 5.31 tok/s |

## Interpretation

- Steady-state decode is about 3.3× faster than on the CPU (57 ms against
  187 ms per token), and stable across runs.
- Run 001's TTFT and worst token include compiling GPU shaders on first
  use. Run 002, a fresh process, was much faster, so compiled shaders are
  cached across processes. Every WGPU measurement must discard a first,
  cold run, and a fresh machine (such as a rented GPU) starts cold.
- Even warm, TTFT is above the CPU's: the first call still pays setup the
  CPU does not.
- 57 ms per token for a 135M model is still overhead-dominated (see the
  calibration entry: 85 ms per call).

## Follow-ups

- Make TTFT and throughput a study (repetitions and warm-ups), not a probe.
- Locate the per-call cost on the GPU: kernel launches per layer, host
  synchronization, or tensor allocation.
