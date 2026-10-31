# Prefill grouping at a call cost of 64 tokens

- **Status:** done
- **Track:** B batching
- **Research question:** H3
- **Experiment:** `adaptive-prefill-cost` (parameters in `config.json`)
- **Started:** 2026-10-31 at revision `563d530`
- **Entry:** `lab/notebook/2026-10-31-prefill-grouping-overhead-64`

## Question

Does the cost-aware prefill grouper choose differently at the call-to-token
ratio measured on the CPU (64 tokens) than at the other backend's ratio
(207; see `2026-10-31-prefill-grouping-overhead-207`)?

## Prediction

Written after the calibrations, before this run: a higher call cost makes
merging prompts into fewer calls worth more padding, so at 207 the grouper
should pick fewer calls than at 64 for the diverse prompt set.

## Protocol

Default prompt sets: diverse (5, 7, 9, 63 tokens) and clustered (5, 6, 7,
8); `invocation_overhead_tokens=64`. Untimed: the estimated cost is
padded tokens plus calls × 64.

## Results

- Diverse: **power-of-two** chosen (unbucketed 316, power-of-two 278); 2 padding tokens, 3 calls.
- Clustered: unbucketed (96 either way); 6 padding tokens, 1 call.

(The run's printed summary reads "cost A → B"; A is the unbucketed cost
and B the power-of-two cost. The summary now says so.)

## Interpretation

The prediction held: for the same prompts, the CPU's ratio picks three
power-of-two calls with almost no padding, while the GPU's ratio picks one
call with 168 padding tokens. The same policy code, given each backend's
measured costs, makes opposite decisions. This is an estimated-cost
result, not a timing; the next step is to time both choices on each
backend.

## Follow-ups

- Time both groupings with `llama-prefill-strategies` on each backend and
  check that the cheaper estimate is the faster run.
- Find the crossover ratio for these prompts (between 64 and 207).
