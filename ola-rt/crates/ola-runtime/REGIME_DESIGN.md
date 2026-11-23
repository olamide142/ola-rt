# Regime-aware adaptation: design

Status: **proposal** (2026-11-23), being implemented in order. Builds on
`SWITCHING_STUDY.md` (evidence) and `ADAPTIVE_POLICY.md` (mechanism).

## Definitions

- **Regime**, *relative to a knob*: a sustained condition under which one
  value of that knob stays best. Two conditions are different regimes only
  if each one's best value beats the other's on paired intervals, the test
  in `lab/replay/regimes.py`. A knob whose best value never changes has no
  regimes. At 4 slots this is true of the token budget.
- **Workload** features describe what arrives: rate, burstiness, prompt and
  output lengths.
- **Operating-condition** features describe what the executor holds: queue
  depth, the prefill backlog against live decodes, KV occupancy, TTFT and
  inter-token latency.
- **Regime change**: a persistent shift in either kind of feature that may
  change the best value. A spike inside a regime is not one.

## What the evidence requires

1. **The operating condition must drive policy, not only safety.**
   Switching the token budget exactly at the workload boundary, at no cost,
   did not beat a fixed budget, and was 5.3 s worse in one order. The
   previous regime's backlog outlived its arrivals by about 80 s.
2. **Evaluation must start from the live state.**
   `ReplayOptimizer::window_trace` replays only completed requests from an
   empty executor. Under load, that window describes arrivals from about
   one queueing delay ago.
3. **Deciding is the cost to control.** The measured stall is 6.5 ms plus
   59 µs per replayed request per replay. It cost 0.2–5.7 s depending on
   the interval, and it is paid on every replaying call, whether or not the
   call switches. `SwitchingCost` penalises only switches.
4. **Adapting must outlast the drain.** A switch pays only if
   `expected remaining duration × gain per second` exceeds
   `decision cost + switch cost + backlog drain time`. A regime shorter than
   its drain time is not worth adapting to, however clearly it is detected.

## Pipeline

```
ExecutorIteration (+ queued sizes) ─► features ─► HMM regime filter ─┐
                                                                   │ trigger (when)
                    live state + projected arrivals ─► replay of each candidate (what)
                                                                   │
                         switch if gain over expected remaining duration > total cost
fast path, unchanged: executor limits, memory-aware admission, fallback policy
```

- **The detector decides *when*, never *what*.** It replaces the fixed
  40-iteration interval, so a quiet system makes no optimizer calls.
  Multi-axis regime labels (for example `arrival=bursty, phase=prefill-heavy`)
  are logged for debugging and the paper; nothing maps a label to a policy.
- **The evaluator decides *what*.** It is `ReplayOptimizer`, seeded with the
  queued and running requests and their progress, followed by arrivals
  projected from the recent rate and size mix: a receding horizon from the
  present, not a replay of the past.
- **Two speeds.** Fast safety reactions (slots, KV, admission, fallback) stay
  in the executor and are untouched. Adaptation is slow and is gated by the
  rule in item 4.

## Features and what they need

| Feature | Source | Change needed |
|---|---|---|
| Arrival rate, inter-arrival CV | `decision.arrival_times`, seen at arrival | none |
| Queue depth, active requests | `queued_requests`, `active_requests` | none |
| Executed prefill vs decode work | `execution_batches` | none |
| KV occupancy, reservations | `kv_cache_bytes`, `reserved_kv_bytes` | none |
| TTFT, inter-token latency | token events | none |
| Prompt and output size of **queued** requests | snapshot only | report them in the decision record (`ola-runtime`; coordinate) |
| SLO pressure | `deadline_missed` is hard-coded `false` | fill it in (`ola-runtime`; coordinate) |

Outcome features (TTFT, inter-token latency) are not regime evidence on
their own. Round 2 showed server processes that stay in a slow decode mode
for their whole life, which looks like a load change to an outcome-only
detector.

## Detector: a hidden Markov model over the workload

The hidden state is the **workload regime**: the arrival process, which
the executor does not cause. Each feature window (log mean inter-arrival
gap, inter-arrival CV, and later the length mix) is a Gaussian emission.
Online forward filtering gives a posterior over regimes after every
window, at O(K²) per step, which is cheap enough for `observe`.

- **Stickiness replaces hand-tuned hysteresis.** High self-transition
  probabilities make the posterior slow to flip on a spike.
- **Expected remaining duration.** With self-transition probability `p`,
  a state is expected to last `1/(1−p)` more windows. That is the duration
  the switch rule in item 4 needs.
- **Recurring states carry a cached best θ.** The replay refreshes it
  rather than searching from scratch on every trigger.
- **The operating condition is observed, not hidden.** Queue depth and
  backlog depend on the policy, so as emissions they would make the model
  partly detect its own actions. They go straight to the evaluator instead.
- **Fitting.** Baum–Welch on training traces, then merge states whose best
  θ does not differ on `regimes.py`'s test: unsupervised states need not be
  policy states.
- **Baseline.** Page–Hinkley on the log inter-arrival gap. The HMM has to
  beat it on detection delay and false alarms, or justify itself by
  providing the duration estimate.

Sizes are seen only at admission, so until queued sizes are reported the
fast signal is the arrival process alone.

## Order of work

Each step is measured in virtual time with `run_charged_trial` before the
next one starts.

1. **State-seeded replay** (`ola-bench`). Seed the replay with the live set
   the optimizer already holds in `seen`. Success: on `regime_switching`,
   the adaptive policy beats `fixed-1024/100` with the stall charged, which
   today's policy does not.
2. **HMM trigger** (`ola-bench`, through `AdaptivePolicy`'s existing
   hooks: the optimizer is called every iteration and replays only when
   the regime posterior changes). Success: fewer calls and a lower charged stall than interval 40,
   at equal or better objective; detection delay and false alarms measured
   on `followed_by` traces.
3. **Gain-over-horizon switch rule** (`ola-bench`). It replaces the
   relative 5% margin with item 4's absolute test.
4. **Runtime observability** (`ola-runtime`, coordinated with the policy
   agent). Report queued request sizes and real deadline misses.
5. **Hardware**. Measure the per-switch cost on the GPU by alternating two
   budgets, then repeat E4 at 16 slots on a warm server.
6. **Optional: asynchronous optimizer** (`AdaptivePolicy`, coordinated).
   Only if the trigger alone leaves the stall material.

## Open questions

- Which knob is worth adapting? The token budget has regimes only near 16
  slots, on one trace family. Slots, prefill chunk size and KV headroom with
  expensive recompute are the candidates; run `budget_regimes`-style sweeps
  on them before building anything for them.
- How far ahead should the projected arrivals reach? Is the drain time of
  the current backlog a natural horizon?
- Should the cluster-level adaptation in `ola-cluster/src/adaptive.rs` share
  the same detector and cost rule? That is for the policy agent to decide.
