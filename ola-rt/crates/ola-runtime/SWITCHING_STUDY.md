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

## A production day: ServeGen (2026-11-23)

`lab/replay/servegen.py` builds traces from ServeGen's per-client workload
fits (Alibaba Cloud Model Studio, NSDI '26), fetched over HTTPS from commit
`d70c8b2` with nothing cloned or cached. It uses the four busiest
`language/m-large` clients over one day, compressed 6× (one trace hour is 10
minutes), at rate scale 0.0000824: about 1,280 requests per seed, 10 seeds.
That rate puts the heaviest regime near 80% of the simulated RTX 4090's
decode capacity (about 35 tokens/s at any slot count, because each decode
row costs about 25 ms). The first attempt, about 5× heavier, never drained
the night and was discarded.

The day has three regimes, each lasting hours:

| Regime | Requests per seed | Median input | Median output |
| --- | ---: | ---: | ---: |
| night 00–08 | 307 | ~75 | ~440 |
| day 08–18 | 623 | ~26 | ~55–60 |
| evening 18–24 | 353 | ~575–640 | ~15–46 |

`servegen_regimes` scores eight named policies per regime and over the
day (mean of p95 completion + p95 TTFT, in seconds). `oracle-switch`
installs each regime's best policy at that regime's first arrival.

**At 4 slots (the `ola serve` default)**, FIFO is best at night and
`sjf-total` is best by day and in the evening. Against the best single
policy (`sjf-total`):

| | night | day | evening | whole day |
| --- | --- | --- | --- | --- |
| `oracle-switch` − `sjf-total` (s) | **−10.46 [−11.86, −8.79]** | +0.00 [+0.00, +0.01] | +0.00 [+0.00, +0.00] | +0.09 [−0.44, +0.65] |

> **Corrected (2026-11-25):** the reference here is the best policy by
> pooled whole-day p95, `sjf-total`. FIFO is within 0.4 s of it on that
> score and 4.1 s better per hour, and against FIFO the night offers
> nothing. See "Scoring per hour, and the right baseline" below.

This is the first case where a switch pays after a real regime transition.
The night's p95 falls by about 8% at no cost in the other regimes, because
the regimes last far longer than the backlog takes to drain. **The
whole-day p95 does not show it.** Pooling every request into one p95 hides
a per-regime gain, which is why the objective has to be scored per regime
or per window (as SLOs are) for adaptation to be credited.

**At 16 slots** the scheduling policies converge (FIFO, both SJF variants
and backfill tie in every regime), and the oracle gains nothing: +0.43 s
[+0.14, +0.75] over the day against `fifo-1024/25`. Token budgets that cap
prompt work (`fifo-512/50`, `fifo-128/100`) are much worse everywhere.

Run it:

```sh
python3 lab/replay/servegen.py --model m-large --hours 24 --clients 4 \
    --rate-scale 0.0000824 --time-scale 6 --seeds 1,2,3,4,5,6,7,8,9,10 \
    --out 'sg-{seed}.jsonl'
OLA_COST=10745,30227,482,97,13047,24568 MAX_ACTIVE=4 \
  cargo run --release -p ola-bench --example servegen_regimes -- sg-*.jsonl
```

## A real controller on the ServeGen day (2026-11-23)

> **Corrected (2026-11-25):** the deltas in this section are against
> `sjf-total`. Against FIFO, the better baseline, see "Scoring per hour,
> and the right baseline" below.

`servegen_adaptive` replaces the oracle with online controllers that choose
between FIFO and `sjf-total`. It runs at 4 slots on the same 10 traces and
charges the RTX 4090 stall. Every controller starts at `sjf-total`. The
HMMs have 3 states and are fitted on 10 held-out seeds (11–20).

| controller | night − `sjf-total` (s) | day | evening | whole day | replays | stall (s) |
| --- | --- | --- | --- | --- | ---: | ---: |
| `oracle-switch` | −10.46 [−11.86, −8.79] | +0.00 | +0.00 | +0.09 [−0.44, +0.65] | – | – |
| `replay-past` (every 40 iterations) | −8.00 [−9.12, −6.58] | +0.67 [+0.25, +1.25] | +0.28 [+0.15, +0.42] | +0.82 [+0.03, +1.67] | 2,431 | 50.9 |
| `replay-state` (every 40 iterations) | −9.08 [−10.61, −7.47] | +0.62 [+0.27, +1.10] | +0.71 [+0.28, +1.23] | +1.03 [+0.37, +1.69] | 2,431 | 52.1 |
| `hmm-past` (arrival trigger) | −2.67 [−5.67, +0.00] | +0.08 | +0.24 | +0.04 [−0.27, +0.42] | 8.2 | 0.16 |
| `hmm-state` (arrival trigger) | −2.05 [−4.62, +0.00] | +0.09 | +0.70 [+0.25, +1.21] | +0.20 [−0.04, +0.53] | 8.2 | 0.17 |
| `sizes-state` (size trigger) | −1.08 [−3.25, +0.00] | +0.03 | +0.17 | +0.22 [−0.00, +0.59] | 3.4 | 0.06 |
| **`table-sizes`** | **−10.89 [−12.48, −9.08]** | +0.22 [−0.15, +0.73] | +0.00 | −0.02 [−0.71, +0.70] | 0 | 0 |
| `table-arrivals` | −10.90 [−12.49, −9.08] | +0.46 [+0.07, +1.01] | +1.18 [+0.81, +1.56] | +0.36 [−0.24, +0.97] | 0 | 0 |

The `-free` variants (stall uncharged) are identical to the charged ones,
to two decimals: once a trigger gates the replays, the stall no longer
matters.

**Periodic replay finds the night and gives it back elsewhere.**
Replaying every 40 iterations recovers 77–87% of the night gain, but
switching wrongly by day and in the evening makes it significantly worse
over the whole day. `replay-state` switches about 41 times a day.

**Triggered replay is cheap but chooses badly.** The 3-state arrival HMM
replays about 8 times a day, but the night, day and evening barely differ
in arrival process (the fitted means overlap). The size HMM over windows
of 8 admitted requests (`RegimeFeatures::Sizes`: log mean prompt, log mean
output limit) recovers the day's structure without labels:
- night: prompt ≈ 150, output ≈ 370;
- day: ≈ 120 / 80;
- evening: ≈ 630 / 40.

It replays about 3 times a day. Detection is no longer the problem; the
decision is. One replay of a 64-request window at the start of a regime
does not show FIFO's advantage, which builds up in the p95 over hours of
long jobs. So the triggered controllers mostly keep `sjf-total`.

**Regimes recur, so learn the map once.** `table-sizes` replays nothing.
- **Learning the table (offline):** the held-out seeds are decoded with
  the size HMM, and FIFO and `sjf-total` are scored on each decoded
  state's requests. No true labels are used.
- **Using it (online):** the controller installs a state's policy when that
  state's posterior reaches 90%.

The learned table is night → FIFO, day and evening → `sjf-total`, the same
as the oracle's. It captures the whole night gain at no cost, gives up a
non-significant 0.22 s by day (detection lag at the transition), and loses
nothing in the evening. The arrival HMM's table is FIFO in every state, so
`table-arrivals` is plain FIFO.

At 16 slots, where the policies converge, both tables learn FIFO
everywhere and match `sjf-total` within 0.01 s: the table does no harm
where there is nothing to gain.

This is the "recurring states carry a cached best θ" item of
`REGIME_DESIGN.md`. On this workload it is the whole controller, and the
replay is unnecessary.

Caveats specific to this result:
- **Output lengths are oracle knowledge.** In these traces `max_new_tokens`
  is each request's sampled output length. So `sjf-total` and the size
  features both know true output lengths. In production `max_tokens` is a
  client-set cap, so both would be weaker. The next section tests this.
- **The table comes from the same deployment.** It is learned on other
  days drawn from the same distribution. A shifted workload needs a refit,
  or a replay to confirm the table entry.
- **The objective is per regime.** The whole-day pooled p95 still shows no
  difference.

```sh
python3 lab/replay/servegen.py --model m-large --hours 24 --clients 4 \
    --rate-scale 0.0000824 --time-scale 6 --seeds 11,12,13,14,15,16,17,18,19,20 \
    --out 'fit-{seed}.jsonl'
OLA_COST=10745,30227,482,97,13047,24568 \
  cargo run --release -p ola-bench --example servegen_adaptive -- \
  sg-{1..10}.jsonl --fit fit-{11..20}.jsonl
```

All controllers take about 45 minutes on 12 cores. `ADAPTIVE=none` runs
only the static policies, the oracle and the tables, in under a minute.

### With output caps instead of true lengths (2026-11-25)

`servegen.py --limit` keeps the same requests (same seeds and draws) but
sends each with a cap as `max_new_tokens` and stops it at its true length
(`eos_after`):
- `fixed:1024`: every request is capped at 1,024, a server default;
- `pow2`: the cap is the next power of two at or above the true length, a
  client that sizes its caps roughly.

`servegen_adaptive` now picks the oracle's per-regime policies and the
reference (the best single policy, which controllers start at) from the
static runs on the same traces. With exact lengths it reproduces the table
above exactly. Results at 4 slots, 10 seeds, tables fitted on seeds 11–20
of the same cap mode, Δ against the best single policy:

| caps | best single | oracle night | `table-sizes` night | `table-sizes` day | `table-sizes` evening | learned table (day, night, evening states) |
| --- | --- | --- | --- | --- | --- | --- |
| exact | `sjf-total` | −10.46 [−11.86, −8.79] | −10.89 [−12.48, −9.08] | +0.22 [−0.15, +0.73] | +0.00 | sjf-total, fifo, sjf-total |
| `pow2` | `sjf-total` | −8.37 [−10.44, −6.28] | −8.72 [−10.83, −6.53] | +0.25 [−0.14, +0.76] | +0.00 | sjf-total, fifo, sjf-total |
| `fixed:1024` | `fifo` | +0.40 [+0.04, +0.98] | +0.00 | +0.00 | −0.33 [−0.77, +0.13] | fifo, fifo, sjf-total |

- **Rough caps keep about 80% of the gain.** With power-of-two caps,
  `sjf-total` still ranks jobs roughly by length. The night still prefers
  FIFO by 8.4 s, and the size HMM still finds the regimes (each state's
  scores are within 0.7 s of the exact case). The learned table is
  unchanged and again matches the oracle.
- **A uniform cap removes the headroom.** With every cap at 1,024,
  `sjf-total` ranks by prompt alone. FIFO becomes the best single policy.
  By day and in the evening, `sjf-total` is better by under 0.5 s, and the
  CIs include zero. The oracle gains nothing at night, and is 0.40 s worse
  there because the night's leftover backlog runs under `sjf-total` after
  the switch. `table-sizes` learns FIFO for night and day, so it does no
  harm.

So the result depends on clients sending informative output limits, not on
exact lengths. Where every request carries the same default cap, there is
nothing to adapt for in this policy pair. A length predictor (for example,
per-client history) would be the way to win it back.

```sh
python3 lab/replay/servegen.py --model m-large --hours 24 --clients 4 \
    --rate-scale 0.0000824 --time-scale 6 --limit pow2 \
    --seeds $(seq -s, 1 20) --out 'pow2-{seed}.jsonl'
OLA_COST=10745,30227,482,97,13047,24568 ADAPTIVE=none \
  cargo run --release -p ola-bench --example servegen_adaptive -- \
  pow2-{1..10}.jsonl --fit pow2-{11..20}.jsonl
```

## Scoring per hour, and the right baseline (2026-11-25)

A pooled p95 over 14,400 s is dominated by the busiest stretch. SLOs are
usually judged per window, so `servegen_adaptive` now also scores each
trace hour (about 53 requests) and averages the 24 hourly objectives. The
best single policy, which controllers start at and are compared with, is
chosen on that hourly mean.

**The baseline changes.** By hourly mean FIFO beats `sjf-total` by 4.1 s
[2.9, 5.6]. On the pooled day p95 they were within 0.4 s, not
significantly different. The earlier sections compared against
`sjf-total`, which made the night look like the gain. Against FIFO the
night offers nothing (FIFO is the night's best). What remains is the day
and the evening, where `sjf-total` is better by 0.46 s and 1.18 s.

All controllers at 4 slots, exact lengths, 10 seeds, stall charged.
Δ against FIFO (s) [95% CI]:

| controller | night | day | evening | whole day | hourly mean | replays | stall (s) |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| `sjf-total` | +10.86 | −0.46 | −1.18 | −0.43 [−0.99, +0.12] | +4.12 [+2.89, +5.63] | – | – |
| `oracle-switch` | +0.40 [+0.05, +1.00] | −0.46 | −1.18 | −0.34 [−0.59, −0.05] | −0.17 [−0.44, +0.16] | – | – |
| `replay-past` | +1.43 | +0.16 | −0.90 | +0.39 [+0.03, +0.75] | +1.39 [+0.38, +2.88] | 2,432 | 50.9 |
| `replay-state` | +1.46 | +0.16 | −0.51 | +0.65 [+0.24, +1.04] | +0.26 [+0.08, +0.46] | 2,432 | 52.1 |
| `hmm-past` | +0.00 | −0.07 | −0.58 | −0.16 [−0.39, +0.04] | −0.12 [−0.21, −0.05] | 8.2 | 0.16 |
| `hmm-state` | +1.08 | −0.13 | −0.07 | +0.07 [−0.03, +0.23] | +0.49 [−0.06, +1.55] | 8.2 | 0.17 |
| `sizes-past` | +0.00 | −0.05 | −0.27 | −0.16 [−0.38, +0.00] | −0.12 [−0.24, −0.03] | 3.4 | 0.06 |
| `sizes-state` | +1.08 | −0.08 | −0.29 | +0.06 [+0.00, +0.17] | +0.45 [−0.06, +1.43] | 3.6 | 0.07 |
| `table-arrivals` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **`table-sizes`** | +0.00 | **−0.24 [−0.40, −0.09]** | **−1.18 [−1.56, −0.81]** | **−0.42 [−0.65, −0.20]** | **−0.38 [−0.46, −0.29]** | 0 | 0 |

- **The honest gain is small.** Adapting the scheduler on this day is worth
  about 0.7% on the hourly mean and 0.6% on the whole day. That is about
  6% in the evening, at no cost.
- **`table-sizes` beats the oracle.** The oracle switches to `sjf-total`
  at the day's first arrival, so the night's leftover long jobs wait
  behind short ones (+0.40 s at night). The table's detection lags the
  boundary, and that lag lets the night backlog drain under FIFO first.
  Instant switching is not optimal when the backlog carries over.
- **Periodic replay is worse than not adapting** on every whole-day score.
  It switches into `sjf-total` at night.
- **Triggered replay is roughly neutral.** Past-window replays gain a
  little (−0.12 s hourly). State-seeded replays sometimes pick `sjf-total`
  at night, which widens their CIs.
- **Rough output limits change nothing.** With power-of-two limits,
  `table-sizes` gains −0.37 s [−0.46, −0.29] hourly against FIFO. With a
  uniform 1,024 limit the gain is −0.09 s [−0.18, −0.00], essentially nil.

Scoring per window should be the default for this study. The pooled p95
picked the wrong baseline, and that made a 0.4 s gain per hour look like a
10.5 s gain at night.

## Where is the headroom? A sweep of ServeGen workloads (2026-11-25)

`servegen_headroom` screens a workload before any controller is built. It
runs the 8 named policies and scores every trace hour. The **hindsight
bound** takes each hour's best policy, and the **headroom** is the best
single policy's hourly mean minus that bound. The bound ignores switching
and carried-over backlog, and picking each hour's best of 8 noisy runs
rewards luck, so it is an optimistic upper bound.

`servegen.py --peak-output-rate` sets the rate scale so the busiest hour
asks for a target decode rate. That puts different workloads at the same
load. The sweep covered:
- 4 workloads: language m-large, m-mid and m-small, and reasoning
  deepseek-r1;
- 4 and 16 clients;
- peaks of 18, 24 and 30 tokens/s (about 51%, 69% and 86% of the
  simulated decode capacity);
- 4 slots, 10 seeds each.

| workload | clients | peak 18 | peak 24 | peak 30 |
| --- | ---: | ---: | ---: | ---: |
| m-large | 4 | 1.6% | 2.0% | 1.4% |
| m-large | 16 | 3.8% | 4.0% | 2.9% |
| m-mid | 4 | 0.7% | 1.1% | 4.1% |
| m-mid | 16 | 1.3% | 2.1% | **6.3%** |
| m-small | 4 | 0.8% | 2.6% | 5.4% |
| m-small | 16 | 2.1% | 3.0% | 5.8% |
| deepseek-r1 | 4 | 0.0% | 0.0% | 1.6% |
| deepseek-r1 | 16 | 0.1% | 0.2% | 1.8% |

Headroom grows with load and with the number of clients mixed together.
deepseek-r1's outputs are clipped at `--max-output` 1,024, which leaves few
requests, so its row says little about reasoning workloads.

**Real controllers on the best candidates** (16 clients, peak 30; 10
held-out seeds for fitting), plus the original m-large workload. Δ in the
hourly mean against FIFO, which is the best single policy in all three:

| controller | m-large, 4 clients | m-mid, 16 clients | m-small, 16 clients |
| --- | --- | --- | --- |
| hindsight bound | −1.06 (2.0%) | −8.35 (6.3%) | −3.36 (5.8%) |
| `oracle-switch` (night, day and evening bins, chosen in-sample) | −0.17 [−0.44, +0.16] | −3.36 [−4.08, −2.68] | +0.29 [−2.02, +3.14] |
| `clock-table` (best policy per hour of day, learned held-out) | **−0.31 [−0.38, −0.24]** | +3.36 [−2.77, +10.43] | +0.41 [−1.15, +2.21] |
| `table-sizes` (size HMM, table learned per hour) | −0.19 [−0.27, −0.10] | +0.79 [−3.32, +8.02] | +1.26 [+0.32, +2.54] |
| `table-arrivals` | 0 (FIFO throughout) | 0 (FIFO throughout) | +0.13 [−0.30, +0.70] |

**Tables must be learned on the evaluation objective.** Learning a table
with one pooled p95 per state picked `sjf-total` everywhere on m-mid. That
is 35.8 s worse per hour than FIFO, because shortest-job-first starves
the hours where long requests dominate. `learn_table` now scores each
state hour by hour. On m-large this changes the size table from (day
`sjf-total`, night FIFO, evening `sjf-total`) to (day FIFO, night FIFO,
evening `sjf-total`), and its gain from −0.38 s to −0.19 s per hour. The
earlier −0.38 s came partly from a table chosen on a different score.

**Conclusion.** On ServeGen at 4 slots, switching between scheduling
policies is worth at most about 0.6% on an hourly score with a controller
that can actually be built (`clock-table` on m-large). The 2–6% hindsight
bounds are not captured. Per-hour preferences are noisy, and switching
mid-backlog costs more than the per-hour differences, so held-out tables
do not transfer on m-mid and m-small. The one sizeable gain, the oracle on
m-mid (2.5%), picks its bins in-sample, and no held-out controller
reproduces it.

Further work on regime switching for the scheduler is unlikely to pay. A
knob with larger per-regime effects is needed: admission and slot count
under KV pressure, prefill chunking, or routing across replicas (the
policy agent's area).

```sh
lab/replay/servegen.py --model m-mid --clients 16 --peak-output-rate 30 \
    --time-scale 6 --seeds $(seq -s, 1 20) --out 'm-mid-{seed}.jsonl'
OLA_COST=10745,30227,482,97,13047,24568 HEADER=1 LABEL="m-mid, 16 clients" \
  cargo run --release -p ola-bench --example servegen_headroom -- m-mid-{1..10}.jsonl
OLA_COST=10745,30227,482,97,13047,24568 ADAPTIVE=none \
  cargo run --release -p ola-bench --example servegen_adaptive -- \
  m-mid-{1..10}.jsonl --fit m-mid-{11..20}.jsonl
```

## Other knobs: KV admission and the token budget (2026-11-25)

The adaptive machinery tunes any integer grid, not only the scheduler.
`servegen_headroom` now takes `GRID`:
- `admission`: FIFO with newest-first preemption, reserving a watermark of
  16 to 256 tokens beyond the prompt, or the worst case. This is the grid
  of `regime-shift-study`.
- `budget`: FIFO under every point of `TOKEN_BUDGET_DOMAINS` (the grid
  `ola serve --adaptive` tunes), plus no budget.

It also takes `KV_TOKENS` (pool size) and `PREFILL_CHUNK`. Workloads are m-mid
and m-small with 16 clients at peak 30, which are the scheduler sweep's
largest bounds, with 10 seeds.

**Admission** runs on traces whose output limit is a fixed 1,024
(`--limit fixed:1024`), so a worst-case reservation overstates what most
requests need. With exact limits the worst case is exact and there is
nothing to tune.

| workload | slots | KV tokens | worst case | best watermark | best single | headroom |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| m-mid | 16 | 8,192 | 139.6 | 107.4 | h64 | 1.48 [1.07, 1.98] (1.4%) |
| m-mid | 16 | 16,384 | 101.0 | 96.1 | h128 | 0.49 [0.38, 0.62] (0.5%) |
| m-mid | 4 | 8,192 | 146.4 | 134.3 | h16 | 0.17 [0.09, 0.25] (0.1%) |
| m-small | 16 | 8,192 | 55.7 | 48.8 | h64 | 1.06 [0.89, 1.23] (2.2%) |
| m-small | 16 | 16,384 | 47.6 | 46.5 | h64 | 0.52 [0.40, 0.64] (1.1%) |
| m-small | 4 | 8,192 | 59.4 | 57.8 | h32 | 0.17 [0.10, 0.26] (0.3%) |

Hourly means are in seconds. With a pool that never binds, every action
scores the same (m-mid at 16 slots: 95.95 s).

Under KV pressure, the choice of action matters a lot: the worst case is
3–30% worse than a watermark. But all the watermarks from 16 to 128 tokens
score within 1% of each other, so one fixed watermark wins nearly every
hour. Switching headroom is at most 2.2%, and that is a hindsight bound
over noisy hourly picks. This is a static tuning result: replacing the
worst case with any modest watermark (plus preemption) recovers almost
all of it. It does not call for adaptation.

**Token budget.** Without prefill chunking every budget is far worse than
no budget. On m-mid, the scheduler grid's `fifo-1024/25` scores 418 s and
`fifo-128/100` scores 2,054 s, against 133 s for FIFO. A prompt larger
than the budget waits until nothing else runs. So the budget points in
the scheduler sweep were never real contenders. With `PREFILL_CHUNK=256`:

| workload | chunk | unbudgeted | best budget | headroom |
| --- | ---: | ---: | --- | ---: |
| m-mid | 256 | 139.8 | b1024/100 (139.75) | 0.03 [0.02, 0.03] (0.0%) |
| m-mid | 64 | 168.5 | b1024/100 (168.5) | 0.04 [0.02, 0.07] (0.0%) |
| m-small | 256 | 59.4 | b1024/100 (59.4) | 0.04 [0.03, 0.05] (0.1%) |
| m-small | 64 | 68.3 | b256/100 (68.3) | 0.07 [0.05, 0.09] (0.1%) |

Chunking itself costs on these traces: m-mid scores 133.0 s unchunked,
139.8 s at a 256-token chunk and 168.5 s at 64; m-small scores 57.5, 59.4
and 68.3 s. Once prompts are chunked, the budget hardly matters. No point
in the grid moves the hourly mean by more than 0.1%, and at most 12
hours in 240 are won by more than 1%.

**Conclusion.** None of the three knobs (scheduler, KV admission, token
budget) has switching headroom on ServeGen worth a controller. The
largest bounds are 6.3% for the scheduler and 2.2% for admission, and no
held-out controller captures them. Two static changes do pay:
- a watermark instead of worst-case reservation under KV pressure (3–30%);
- leaving the token budget off (or above the largest prompt) when prompts
  are long.

```sh
lab/replay/servegen.py --model m-mid --clients 16 --peak-output-rate 30 \
    --time-scale 6 --limit fixed:1024 --seeds $(seq -s, 1 10) --out 'm-mid-cap-{seed}.jsonl'
OLA_COST=10745,30227,482,97,13047,24568 GRID=admission MAX_ACTIVE=16 KV_TOKENS=8192 \
  cargo run --release -p ola-bench --example servegen_headroom -- m-mid-cap-{1..10}.jsonl
OLA_COST=10745,30227,482,97,13047,24568 GRID=budget PREFILL_CHUNK=256 \
  cargo run --release -p ola-bench --example servegen_headroom -- m-mid-{1..10}.jsonl
```

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
