# Switching cost and regime change for the token budget

Status: **run once** (2026-11-21, branch `switching-regime` from `95a18e5`).
Virtual time under the RTX 4090 cost model
(`OLA_COST=10745,30227,482,97,13047,24568`, load scale 414×). No new
hardware result. The optimizer's stall comes from the second GPU round's
E4 logs.

## Questions

1. Does the best fixed token budget, the knob `ola serve --adaptive` tunes,
   change between workload regimes at all?
2. Does adapting across a regime change pay once the optimizer's host
   stall is charged, which earlier virtual-time studies treated as free?

## Decision cost, measured

`ola serve --adaptive` runs `ReplayOptimizer::propose` inside `observe` on
the executor thread, so each replaying call stalls the next iteration. A
least-squares fit to the 50 logged calls of the E4 runs (machine 26487)
gives **6,476 µs + 59.4 µs × replays × window requests**: about 120 ms for
a full 30-replay, 64-request call, roughly half a decode iteration.
`AdaptationCost::RTX_4090_E4` holds this fit. `run_charged_trial` adds the
stall to the virtual clock after the iteration that made the call, so
arrivals during it queue and every request in flight waits. The cost of
the switch itself (for example, first use of new kernel shapes) has not
been measured, so it is zero here.

## Regimes (`budget_regimes`)

The sweep runs each of the 28 grid points as a fixed policy. It covers 4
trace families (`mixed_short_long`, `prefill_decode_competition`,
`priority_classes`, `early_stopping`) at 4 loads (nominal gaps of 4,000,
2,000, 1,000 and 50 µs, scaled), with 120 requests each. The objective is
p95 completion plus p95 TTFT. `lab/replay/regimes.py` counts a regime pair
as changing its preferred budget only if each regime's best beats the
other's best in that regime, on paired per-seed differences whose 95%
bootstrap interval excludes zero.

| Slots | Seeds | Regimes where the largest budget is best or within noise | Separating pairs |
| ---: | ---: | ---: | --- |
| 4 (`ola serve` default) | 20 | 16 of 16 | none |
| 16 | 10 | 15 of 16 | `prefill_decode_competition` overload vs burst |
| 32 | 10 | 16 of 16 | none |

At 4 slots the knob is inert. With at most 4 requests of up to about 256
prompt tokens, a budget of 1024 never binds, and smaller ones only hurt.
This explains why E4 on hardware switched once in six runs.

The one separating pair is at 16 slots. Under sustained overload, 512
tokens with half for prompts beats unbounded by 3.7 s [2.9, 4.5]. In a
burst, unbounded beats 512/50 by 8.0 s [7.1, 9.0]. The headroom over the
best single budget is 1.8 s per regime run.

## Back to back (`regime_switching`)

The same pair runs as one trace of 120 overload requests and 120 burst
requests, in both orders, over 10 seeds at 16 slots. All 240 requests
completed in every run. Objectives are means in seconds over the whole
trace; Δ is against `fixed-1024/100` with a paired 95% bootstrap interval.

| Controller | overload→burst | Δ | burst→overload | Δ | Calls | Stall (s) |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `fixed-1024/100` | 169.4 | — | 204.3 | — | 0 | 0 |
| `fixed-512/50` | 182.4 | +13.0 [+12.2, +13.8] | 209.6 | +5.3 [+2.8, +7.2] | 0 | 0 |
| `oracle-switch` | 169.6 | +0.2 [−0.1, +0.5] | 209.6 | +5.3 [+2.8, +7.2] | 0 | 0 |
| `adaptive-free` (interval 40) | 170.6 | +1.2 [+0.0, +2.7] | 204.3 | +0.0 | 8 | 0 |
| `adaptive-charged` (40) | 171.7 | +2.3 [+1.1, +4.0] | 205.2 | +1.0 [+0.9, +1.0] | 8 | 0.67 |
| `adaptive-charged-i10` | 175.0 | +5.7 [+4.2, +7.7] | 208.0 | +3.7 [+3.6, +3.7] | 32–34 | 2.62 |
| `adaptive-charged-i160` | 170.2 | +0.8 [+0.2, +1.8] | 204.5 | +0.2 [+0.2, +0.2] | 2 | 0.18–0.24 |

`oracle-switch` installs each phase's best budget when that phase's first
request arrives.

## Interpretation

- **A regime defined by arrivals is not the regime the executor is in.**
  Switching at the arrival boundary, with perfect knowledge and no cost,
  does not beat the static unbounded budget, and in burst→overload it is
  worse. The first phase's backlog outlives the phase: overload requests
  are still running about 80 s after the last one arrives. So the best
  budget at any moment depends on the queue the executor holds, not on
  the recent arrival process. The 1.8 s isolated headroom does not
  survive the transition.
- **`ReplayOptimizer` is blind to exactly that state.** `window_trace`
  replays only *completed* requests from an empty executor. Requests that
  are queued or running when it decides are not in the replay, so it scores
  candidates against the past rather than against the system it is about
  to schedule. A regime-aware controller has to replay from the current
  state.
- **Deciding costs latency even with nothing to gain.** A charged stall
  raises the objective by about 1.4–2.2× its total, because it delays
  every request in flight. Re-planning every 10 iterations costs 3.7–5.7 s
  (2–3%), every 40 costs 1.0–2.3 s, and every 160 costs 0.2–0.8 s. The
  free simulator hid all of this.
- **`SwitchingCost` was the wrong lever.** It penalizes *changing* the
  vector, but the measured cost is *deciding*, paid on every replaying
  call whether or not the vector changes. The levers are how often to
  decide (interval or event trigger), how much each decision replays
  (grid size × window), and whether it runs on the executor thread at all.
  `AdaptationCost::latency_switching_cost` converts a measured switch stall
  into objective units for when one exists.

## Threats to validity

The live executor and the replays are the same simulator. The stall fit
comes from 50 calls on one host, with windows of 8–64 requests. The
per-switch cost is unmeasured. There is one trace family, one regime pair
and 10 seeds. Burst and overload are defined by scaled gaps, not by
production traces.

## Next

1. **Replay from the current state.** Seed each replay with the queued and
   running requests and their progress, plus a projection of recent
   arrivals, so the optimizer evaluates candidates from where the
   executor is (receding horizon).
2. **Event-triggered decisions.** Propose when the executor's state moves
   (queue depth, ratio of prefill backlog to live decodes) rather than
   every N iterations. This cuts the decision cost that the interval sweep
   shows dominates.
3. **Take the optimizer off the executor thread.** Propose asynchronously
   and install at a later boundary, trading the stall for staleness. This
   is a runtime change in `AdaptivePolicy`.
4. **Measure the switch cost on hardware.** Alternate two budgets on a
   fixed schedule and compare the first iteration after each switch with
   steady state.

## Running it

```sh
OLA_COST=10745,30227,482,97,13047,24568 SEEDS=10 MAX_ACTIVE=16 \
  cargo run --release -p ola-bench --example budget_regimes -- regimes16.csv
python3 lab/replay/regimes.py regimes16.csv
OLA_COST=10745,30227,482,97,13047,24568 \
  cargo run --release -p ola-bench --example regime_switching -- switching.csv
```
