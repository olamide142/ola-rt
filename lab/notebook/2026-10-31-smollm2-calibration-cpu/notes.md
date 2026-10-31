# SmolLM2 cost calibration on CPU

- **Status:** done (characterization)
- **Track:** B batching
- **Research question:** RQ3 (cost models for adaptive optimization), H3
- **Experiment:** `cost-calibration` (parameters in `config.json`)
- **Started:** 2026-10-31 at revision `563d530`
- **Entry:** `lab/notebook/2026-10-31-smollm2-calibration-cpu`

> A first measurement; no prediction was recorded before the run.

## Question

What do per-iteration, per-call, and per-token costs look like for a real
model on this CPU, and how do they compare with the simulator's default
cost model?

## Protocol

SmolLM2-135M-Instruct on the CPU backend, release build; default grid
(batch widths 1, 2, 4 × prompt lengths 4, 8, 16), 1 warm-up and 5 measured
sweeps, one least-squares fit per sweep.

## Results

| Term | Fit | 95% interval |
| --- | ---: | --- |
| fixed per iteration | −34.7 ms | −37.3 to −31.1 |
| per dense token | 3.49 ms | 3.42 to 3.57 |
| per call | 222.1 ms | 218.2 to 226.5 |
| residual RMS | 36.0 ms | |

One call costs as much as about 64 tokens.

## Interpretation

- Per-call overhead dominates, consistent with the 187 ms per decoded
  token in the generation entry: decode is one call per token, so it is
  overhead-bound on this CPU.
- The call-to-token ratio (about 64) is close to the simulator's default
  `CostModel` (300 µs / 4 µs = 75), although absolute times are roughly
  800× larger. Policy choices depend on such ratios, so the default-model
  conclusions (large token budgets win) may well hold on this CPU; that is
  now testable.
- The negative fixed term means the linear model
  `fixed + per_token·tokens + per_call·calls` is misspecified here (a
  physical fixed cost cannot be negative), and the residual is large
  (36 ms). `CostFit::cost_model` clamps it to zero, which `ola serve
  --adaptive` would use.

## What I learned

The fitted cost model is only as good as its functional form; check the
sign of every coefficient and the residual before using a fit.

## Follow-ups

- Test whether the default-model conclusions hold here: run
  `adaptive-policy-study` style selection with this fitted model (needs a
  way to pass a cost model to the virtual-time studies).
- Try a model with batch-dependent per-call cost, and widen the grid
  (`widths`, `lengths`) to see where the linear form breaks.
- Calibrate on WGPU and compare the ratio (H3: does the best policy depend
  on the hardware?).
