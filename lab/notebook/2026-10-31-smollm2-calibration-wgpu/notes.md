# SmolLM2 cost calibration on WGPU

- **Status:** done (characterization)
- **Track:** B batching
- **Research question:** RQ3, H3 (does the best policy depend on the hardware?)
- **Experiment:** `cost-calibration` (parameters in `config.json`)
- **Started:** 2026-10-31 at revision `563d530`
- **Entry:** `lab/notebook/2026-10-31-smollm2-calibration-wgpu`

> A first measurement; no prediction was recorded before the run.

## Question

How do per-call and per-token costs on the GTX 1650 compare with the CPU
(`notebook/2026-10-31-smollm2-calibration-cpu/notes.md`), and does their
ratio, which drives batching decisions, change?

## Protocol

Same grid and protocol as the CPU entry (widths 1, 2, 4 × lengths 4, 8, 16;
1 warm-up, 5 sweeps), `--backend wgpu`, run after the generation entry had
warmed the shader cache.

## Results

| Term | CPU | WGPU | Simulator default |
| --- | ---: | ---: | ---: |
| fixed per iteration | −34.7 ms (misfit) | 7.79 ms [7.03–8.50] | 0.2 ms |
| per dense token | 3.49 ms | 0.412 ms [0.400–0.426] | 0.004 ms |
| per call | 222.1 ms | 85.4 ms [85.2–85.7] | 0.3 ms |
| residual RMS | 36.0 ms | 7.5 ms | |
| **call ≈ tokens** | **64** | **207** | **75** |

## Interpretation

- The GPU cuts per-token cost 8.5× but per-call cost only 2.6×, so a call
  is worth about 207 tokens on the GPU against 64 on the CPU. The GPU
  should therefore favour fewer, larger calls far more strongly.
- The linear cost model fits the GPU well (positive fixed term, small
  residual), unlike the CPU.
- First evidence for H3: the parameter that decides batching policy
  differs by about 3× between two backends of the same machine. The
  grouping entries (`2026-10-31-prefill-grouping-overhead-64` and `-207`)
  show it changes a real decision.

## Follow-ups

- Run the virtual-time policy studies with each fitted cost model and see
  whether their selected policies differ (needs a cost-model parameter on
  those studies).
- Repeat on other GPUs (Vast.ai) to see how the ratio moves with
  architecture.
