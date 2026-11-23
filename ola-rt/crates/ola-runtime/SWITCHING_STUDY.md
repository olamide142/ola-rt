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

## Step 1: replaying from the live state (2026-11-23)

`ReplayHorizon::FromState` seeds each replay with the queued and running
requests, followed by projected arrivals (see the variant's docs for the
approximations). It is the same study and seeds, with three more
controllers; `lab/replay/switching.py` prints the table.

| Controller | overload→burst Δ (s) | Switches | burst→overload Δ (s) | Switches | Stall (s) |
| --- | --- | ---: | --- | ---: | ---: |
| `adaptive-free` (past) | +1.20 [+0.00, +2.74] | 0.6 | +0.00 | 0.1 | 0 |
| `state-free` | +0.00 [+0.00, +0.00] | 0.0 | +0.05 [+0.00, +0.12] | 0.3 | 0 |
| `adaptive-charged` (past) | +2.31 [+1.05, +4.02] | 0.6 | +0.95 [+0.93, +0.96] | 0.1 | 0.66–0.67 |
| `state-charged` | +3.06 [+2.92, +3.21] | 0.0 | +3.48 [+3.42, +3.54] | 0.3 | 1.65–1.96 |
| `state-charged-i160` | +0.85 [+0.73, +1.00] | 0.0 | +0.67 [+0.67, +0.67] | 0.0 | 0.40–0.53 |

- **Seeding from state removes the wrong switches.** The past replay moved
  away from the best budget after overload→burst and lost 1.2 s. The
  state replay stays, matching `fixed-1024/100`.
- **It cannot win here, because there is nothing to win.** `oracle-switch`
  does not beat the fixed budget on this pair either, so the success test
  for step 1 (beat `fixed-1024/100` with the stall charged) is not reachable
  on this trace pair. A scenario where switching can pay is needed before
  it can be passed.
- **Each decision costs about 2.5× more.** The replayed trace includes the
  backlog, so the modeled stall (proportional to replayed requests) rises.
  That makes the trigger (step 2) and asynchronous replay (step 6) more
  important.

## Step 2: an HMM regime trigger (2026-11-23)

`RegimeTrigger` puts a two-state `GaussianHmm` (`regime_filter.rs`) in
front of the replay. Its emissions are 8-gap arrival windows (log mean gap
and coefficient of variation), and it is fitted by Baum–Welch on held-out
seeds 101–130. The optimizer is called every iteration but replays only
when the most probable regime changes with at least 90% posterior.

### Detection (`regime_detection`, 20 seeds)

| Change | Detector | Delay (arrivals) | Alarms with no change, per run | Predicted / actual remaining windows |
| --- | --- | ---: | ---: | --- |
| 20× rate (overload ↔ burst) | HMM | 8.0 | 0 | 28.0 / 13.0 |
| 20× rate | Page–Hinkley | 8.0 | 0 | — |
| 2× rate (moderate ↔ overload) | HMM | 10.0–10.4 | 0.9 (moderate only) | 19.7–21.5 / 12.7–12.8 |
| 2× rate | Page–Hinkley | 35.6–38.8 | 0 | — |

- A 20× change is caught within one window by either detector.
- On a 2× change the HMM is about 3.5× faster, but raises about one false
  alarm per run on moderate-only traces; Page–Hinkley raises none. The 90%
  confidence threshold sets this trade-off.
- **The duration estimate is about twice too long.** Each training trace
  holds one change, so the fit learns a stay probability of about 0.96,
  and a geometric duration cannot represent fixed-length phases. A
  semi-Markov model, or training traces with realistic regime lengths, is
  needed before `1/(1−p)` can drive the switch rule.

### Closed loop (`regime_switching`, same seeds)

| Controller | overload→burst Δ (s) | burst→overload Δ (s) | Replays | Stall (s) |
| --- | --- | --- | ---: | ---: |
| `adaptive-charged` (past, every 40) | +2.31 [+1.05, +4.02] | +0.95 [+0.93, +0.96] | 8 | 0.66–0.67 |
| `past-hmm-charged` | +0.20 [+0.18, +0.22] | +0.04 [+0.04, +0.04] | about 2 | 0.02–0.10 |
| `state-charged` (every 40) | +3.06 [+2.92, +3.21] | +3.48 [+3.42, +3.54] | 8 | 1.65–1.96 |
| `state-hmm-charged` | +0.79 [+0.77, +0.81] | +0.65 [+0.64, +0.66] | about 2 | 0.33–0.39 |

Both HMM-triggered controllers check on all 321–341 iterations but replay
about twice: once on the first confident regime and once at the change.
That cuts the cost of adapting by 4–12× against the 40-iteration interval,
and neither switches wrongly. Neither beats the fixed budget, because on
this pair nothing can: `oracle-switch` does not either. The cheapest
configuration of the full design, `state-hmm-charged`, costs 0.65–0.79 s
(0.3–0.5%) to run where adapting has nothing to gain.

## Other knobs, and whether separation survives back to back (2026-11-23)

`policy_regimes` sweeps seven named policies with memory-aware admission:
FIFO, shortest-job-first by prompt (`sjf-prompt`) and by prompt plus output
limit (`sjf-total`), FIFO with backfill, two token budgets, and `sjf-total`
with a budget. It covers the same traces and loads with 10 seeds, and
`regimes.py --across` also pairs different workload mixes at one load.

Unlike the token budget, **the scheduler separates regimes even at 4
slots**:

- `prefill_decode_competition` light prefers `sjf-total`, by 4.4 s
  [2.1, 7.1] over FIFO; moderate load and overload prefer FIFO, by 13.9 s
  and 20.4 s.
- At 16 slots, a burst prefers `sjf-total` and overload prefers
  `fifo-512/50`, by 10.9 s and 15.4 s.
- Mix changes separate too: in a burst, `mixed_short_long` prefers
  `sjf-total` and `prefill_decode_competition` prefers FIFO.

`scenario_switching` runs five of these pairs back to back, in both orders
(120 requests per phase, 10 seeds). It compares each phase's best policy
held throughout with `oracle-switch`, which installs the second phase's
best at its first arrival, free of cost.

| Slots | Order (best policy) | Best static | `oracle-switch` Δ (s) [95% CI] |
| ---: | --- | --- | --- |
| 4 | pdc light (`sjf-total`) → pdc moderate (`fifo`) | `fifo` | +0.67 [+0.38, +1.06] |
| 4 | pdc moderate → pdc light | `fifo` | +0.64 [+0.29, +0.99] |
| 4 | pdc light → pdc overload | `fifo` | +0.09 [+0.00, +0.18] |
| 4 | pdc overload → pdc light | `fifo` | +147.37 [+143.65, +151.05] |
| 4 | mixed burst (`sjf-total`) → pdc burst (`fifo`) | `fifo` | −0.09 [−0.22, +0.04] |
| 4 | pdc burst → mixed burst | `fifo` | +10.41 [+9.40, +11.49] |
| 4 | early light (`fifo`) → pdc light (`sjf-total`) | `fifo` | +96.69 [+83.17, +109.75] |
| 4 | pdc light → early light | `fifo` | +0.16 [+0.00, +0.47] |
| 16 | pdc burst (`sjf-total`) → pdc overload (`fifo-512/50`) | `fifo-512/50` | +0.01 [−0.06, +0.07] |
| 16 | pdc overload → pdc burst | `fifo-512/50` | +14.75 [+12.29, +17.89] |

**In no case does the perfect switch beat the best static policy.**
Switching *into* shortest-job-first while the previous regime's backlog is
still queued is catastrophic, up to +147 s, because SJF starves the
backlog's long jobs and the whole-trace p95 is set by them. Every
separation above was measured from an empty executor, and none of it
transfers to a regime entered with a backlog.

What this means for the design:

- **A regime's best policy depends on the state it is entered in**, not
  only on the workload. Separation has to be tested *conditional on the
  inherited backlog*, which a state-seeded evaluator does and an
  isolated-regime sweep cannot.
- **The phases are too short.** At 120 requests, the transient after a
  change is a large share of each phase. Switching can only pay when
  regimes last long compared with the drain time, which needs traces with
  realistic regime lengths: production traces such as ServeGen's, not
  synthetic pairs.
- **The objective may hide the gain.** A whole-trace p95 is set by the worst
  phase; per-window or SLO-attainment objectives would credit a switch that
  helps the later regime.

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
