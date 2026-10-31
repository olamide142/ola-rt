# Online replay vs fixed budgets

- **Status:** done
- **Track:** E adaptive
- **Research question:** RQ3, hypothesis H1
- **Experiment:** `online-adaptive-policy-study` (parameters in `config.json`)
- **Started:** 2026-10-31 at revision `09b75b4+dirty`; cited run 002 at `923d7dd`
- **Entry:** `lab/notebook/2026-10-31-online-replay-vs-fixed`

> **Worked example, written retrospectively.** The question and
> hypothesis below were set before the study existed, but the prediction
> section was written after the results were known; a real entry writes it
> first. Use this entry as a model of the format, not of the discipline.

## Question

Does re-optimizing the token budget online, by replaying recent work in
virtual time, beat the best budget chosen offline when the workload shifts
from mixed short/long requests to an overload burst?

## Hypothesis

H1 (thesis): adaptive scheduling reduces tail latency under workload
shifts. Here: online replay ends with a lower p95 completion + p95 TTFT
objective than the best fixed budget, because each phase should prefer a
different budget.

## Prediction

Online at least 5% below `fixed`. The hypothesis fails if online is no
better than `fixed`, or if `fixed` and FIFO are already equal (then no
budget beats unbounded batching, and the phases cannot prefer different
budgets).

## Protocol

- 10 held-out seeds (1–10) of the `phased` trace: 40 mixed short/long
  requests at a quarter of the burst rate, then 40 overload requests at a
  2 ms mean gap; 8 executor slots; default `CostModel`; virtual time.
- `fixed`: the best of 28 budgets (16–1024 tokens, 25–100% prompt share)
  for training seed 0, chosen with hindsight.
- `online`: starts at 16 tokens; every 40 iterations replays the last 64
  completed requests under each candidate and switches on a 5% gain.
  `online-warm`: the same, starting from `fixed`'s budget.
- The 64-request window and 5% margin were chosen by looking at seeds 1–5
  (a 24-request window drifted off good budgets); seeds 6–10 were not used
  to choose anything.

## Results

Run 002 (run 001 identical; `lab/lab compare` shows +0.0% on every metric).
Objective is p95 completion + p95 TTFT, mean over seeds 1–10:

| Variant | Objective | p95 completion | Starved |
| --- | ---: | ---: | ---: |
| `fifo` | 73.7 ms | 41.0 ms | 4.6 |
| `fixed` (512 tokens, 384 prompt) | 73.7 ms | 41.0 ms | 4.6 |
| `online-warm` | 74.5 ms | 41.5 ms | 4.8 |
| `online` | 80.0 ms | 44.2 ms | 5.5 |

On seeds 6–10 alone: `online` 80.6 ms, `online-warm` 74.6 ms, `fixed`
73.0 ms.

## Interpretation

The hypothesis is not supported, and the test could not have supported it:
`fixed` equals FIFO, so no token budget beats unbounded batching on this
trace under this cost model, and both phases prefer the same (largest)
budget. There was nothing for adaptation to find.

What the result does show: online replay recovers most of the gap from a
bad start (80.0 ms from the grid's smallest budget, which it left; the
smallest budget's own latency was not measured here) and
leaves a good budget nearly alone (74.5 ms against 73.7 ms). Its residual
cost is the time spent at poor budgets before the first proposals.

Threats to validity: the default `CostModel` makes iterations expensive
relative to tokens, which favours large batches everywhere; the "live"
executor is the same simulator the optimizer replays, so the replay model
is exact here and will not be on hardware; 10 seeds of one trace family.

## What I learned

- Check that the effect can exist before testing it: the first thing to
  establish is that the per-phase optimum differs.
- A search space that excluded large budgets earlier made the offline
  study's "adaptive" policy 2.4× worse than FIFO; always check the
  baseline's behaviour is a candidate.
- `AdaptivePolicy` used to show the optimizer only one iteration in 40;
  an online optimizer is only as good as the observations it is handed.

## Follow-ups

- Curriculum E3: a trace or cost model where the best budget changes
  between phases, verified by selecting per phase first.
- Curriculum E5: perturb the cost model the optimizer replays with.
- Curriculum E6: measure `ola serve --adaptive` on real hardware with a
  calibrated cost model.
