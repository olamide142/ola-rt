# Regime shift study

Status: **run once at default parameters, plus a 10-seed check**
(2026-11-17, uncommitted working tree on `ef578f2`). Virtual-time
simulation: the real `Executor`, policies, and plan validation drive the
synthetic model, and a `CostModel` assigns time. No hardware result is
implied.

## Why this study exists

`online-adaptive-policy-study` (`ADAPTIVE_OPTIMIZATION_PLAN.md`, lab entry
`2026-10-31-online-replay-vs-fixed`) found that online replay of the token
budget did not beat the best fixed budget on a mixed-then-burst trace:
73.7 ms for `fixed` against 74.5 ms for `online-warm`, means over 10
held-out seeds of p95 completion plus p95 TTFT. The recorded explanation
is that the best budget does not change between phases, so there was
nothing to adapt to. That was inferred from `fixed` equalling FIFO; the
per-phase optimum was never measured. The result stands as a negative
result and is not evidence for adaptation.

This study measures the per-regime optimum **first**, for a knob whose
best value plausibly depends on the workload: how much KV an admitted
request reserves beyond its prompt.

## Design (`regime-shift-study`)

- **Regimes.** `early_stopping` (16–64-token prompts, 128–256-token
  limits, three quarters stop after 8–64 tokens), then `run_to_limit` (the
  same shape, nobody stops early). 60 requests each, 1 ms mean gap, 24
  slots, a 2,048-token pool that binds, the default cost model.
- **Actions.** Watermark headroom 16, 32, 64, 128, or 256 tokens, or the
  worst case, all with newest-first preemption so an over-committed pool
  recovers by recompute and switching to the worst case mid-run is safe.
  `baseline` is the repository default: worst case, no preemption.
- **Controllers.** Every fixed grid point; `oracle-static`, the fixed
  action with the best mean objective on the **evaluated traces
  themselves**, an upper bound for any static choice that no online
  controller can know; `replay` (`ReplayOptimizer` over the grid, from the
  worst case, every 40 iterations, last 32 completed requests, 5% margin);
  `replay-switching` (the same plus a 5,000 µs switching cost per change,
  chosen before the run and not tuned); and `belief-quantile`
  (`BeliefPolicy` with `QuantileAdmission` at the 90th percentile, 32-request
  window).
- **Per-regime optimum.** Every controller also runs each regime alone
  (`objective_early_alone`, `objective_limit_alone`). The summary says the
  preferred action changes only if each regime's best fixed action beats
  the other's on paired per-seed differences whose 95% bootstrap interval
  excludes zero.
- **Metrics.** Objective (p95 completion + p95 TTFT), p95s, throughput,
  peak KV, preemptions, failures, unfinished requests, configuration
  switches, and `replays` (simulated replays per run, the controller's
  deterministic overhead; host time is not measured).

## Results

Default parameters (5 seeds). Objectives in ms, mean [95% bootstrap CI]:

| Variant | Shifted | Early alone | Limit alone | Throughput tok/s | Preemptions | Switches | Replays |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 1782 [1696, 1868] | 416 | 1366 | 13,514 | 0 | 0 | 0 |
| `fixed-h16` | 1103 [1024, 1171] | 187 [152, 216] | 921 [883, 958] | 19,122 | 194.6 | 0 | 0 |
| `fixed-h32` | 1071 [987, 1141] | 205 [158, 260] | 926 [903, 945] | 19,627 | 122.8 | 0 | 0 |
| `fixed-h64` | 1101 [1025, 1161] | 226 [179, 268] | 914 [894, 932] | 19,471 | 44.0 | 0 | 0 |
| `fixed-h128` | 1302 | 307 | 1042 | 17,314 | 1.2 | 0 | 0 |
| `fixed-h256`, `fixed-worst-case` | 1782 | 416 | 1366 | 13,514 | 0 | 0 | 0 |
| `replay` | 1160 [1073, 1240] | 231 | 1009 | 18,655 | 148.8 | 4.4 | 238 |
| `replay-switching` | 1184 [1087, 1253] | 239 | 1014 | 18,611 | 155.0 | 3.2 | 240 |
| `belief-quantile` | 1615 [1526, 1717] | 300 | 1366 | 14,570 | 5.6 | 0 | 0 |
| `oracle-static` (= `fixed-h32`) | 1071 | 205 | 926 | 19,627 | 122.8 | 0 | 0 |

No variant failed or left work unfinished. A 256-token headroom equals the
worst case here because no limit exceeds 256.

Ten seeds (1–10, including the five above): shifted objective
`oracle-static` (`fixed-h32`) 1136 ms, `replay` 1222, `replay-switching`
1251, `belief-quantile` 1758, `baseline` 1868; best fixed alone
`fixed-h16` (240 ms) on `early_stopping` and `fixed-h32` (936 ms) on
`run_to_limit`, again with overlapping paired intervals.

## Interpretation

- **The preferred action does not detectably change.** The per-regime
  argmins differ (h16 vs h64 at 5 seeds, h16 vs h32 at 10), but in both
  regimes 16–64 tokens of headroom form a plateau whose differences are
  within noise. What does differ sharply is the worst case, which is far
  worse in *both* regimes. Even when every request runs to its limit,
  over-committing and preempting beats reserving the worst case on this
  pool and cost model.
- **So adaptation again has nothing to gain**, and the measurements show
  it: online replay is 8% worse than the hindsight static choice at 5
  seeds (1160 vs 1071 ms) and 7.6% at 10 (1222 vs 1136 ms), paying for exploration and for starting at the worst
  case. This repeats the earlier load-burst result on a different knob; it
  does not support the hypothesis that adaptation helps.
- **The switching cost reduced switching but not latency** (4.4 → 3.2
  switches per run, objective within noise). It behaves as designed; there
  was no switching penalty in the simulator for it to save.
- **The belief policy is conservative, not wrong.** It learns from the
  window that run-to-limit requests use their limit and reserves the worst
  case, matching `baseline` exactly on that regime. The 90th percentile is
  too cautious here, because over-commitment wins even when the belief is
  accurate. Reservation quantiles are not the right target when preemption
  is cheap; this is a property of the cost model, not a calibration
  result.

## Threats to validity

The replay model and the "live" executor are the same simulator, so
replays are exact here and will not be on hardware. Preemption costs only
recomputed prompt tokens under the default `CostModel`; a model where
recompute is expensive could move the optimum toward the worst case in the
run-to-limit regime. Five and ten seeds of one trace family. The switching
cost and belief quantile were set once, not tuned.

## What would show adaptation helping

A regime pair whose per-regime optima separate on paired intervals, for
example a cost model with expensive recompute (so run-to-limit prefers the
worst case) combined with early stopping (which prefers small headroom).
That is a follow-up, to be designed before it is run.

## Running it

```sh
make experiment ID=regime-shift-study
make experiment ID=regime-shift-study SET="seeds=10"
cargo test --manifest-path ola-rt/Cargo.toml -p ola-bench --lib regime
```
