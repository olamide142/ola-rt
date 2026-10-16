# ola-bench

`ola-bench` provides deterministic experimental inputs and small, explicit
measurement utilities.

`Workload` is an ordered trace of timestamped `RequestSpec` values. Replaying a
trace only admits requests; an experiment controls scheduling and simulated or
real execution independently. `FixedWorkload` is the first baseline generator
and intentionally has no randomness. Later stochastic generators must accept
and report seeds and distribution parameters.

Latency percentiles use the nearest-rank method: percentile `p` selects rank
`ceil(p * N / 100)` from sorted samples. Durations are reported in microseconds,
matching OlaRT's monotonic timestamp resolution.

## Prefill grouping comparison

`compare_prefill_batching` runs one same-model prompt-length trace through the
unbucketed and power-of-two grouping rules. It reports invocation count, tensor
versus individual invocations, real prompt positions, padding positions, and
input-token efficiency. The calculation is deterministic compute-shape
analysis; it intentionally makes no elapsed-time or throughput claim.

```rust
use ola_bench::compare_prefill_batching;

let comparison = compare_prefill_batching(&[5, 9, 7])?;
assert_eq!(comparison.unbucketed.padding_tokens, 6);
assert_eq!(comparison.power_of_two.padding_tokens, 2);
assert_eq!(comparison.padding_tokens_saved(), 4);

# Ok::<(), ola_bench::WorkloadError>(())
```

This crate is for reusable benchmark machinery. Individual hypotheses,
configurations, raw data, plots, and interpretations belong in `experiments/`.

`analyze_cost_aware_prefill` applies the runtime's workload-aware grouping
policy to a prompt-length trace. It reports both candidate cost estimates, the
selected strategy, and the selected strategy's actual invocation/padding shape.
The configured invocation overhead is expressed in token-equivalent units;
this remains deterministic shape analysis and makes no elapsed-time claim.

## Execution-backed comparison

`benchmark_prefill_execution` complements shape analysis by executing the same
already-tokenized requests twice against one loaded `CausalLanguageModel`: once
with unbucketed prefill and once with power-of-two buckets. It verifies that
both runs return identical token sequences and reports:

- complete executor iterations and emitted output tokens;
- tensor and individual prefill invocation counts;
- real and padded prefill positions;
- host wall time for iterations containing prefill and for the complete run;
- output-token throughput and peak observed/reserved KV bytes.

The prefill duration surrounds the complete synchronous executor iteration, so
it includes runtime overhead and is not per-request time to first token.

`benchmark_prefill_trials` adds an explicit protocol around that primitive. It
discards caller-selected warm-up pairs, retains caller-selected measured pairs,
and alternates which strategy runs first. Raw samples remain paired by vector
index, while nearest-rank summaries report prefill-iteration and total wall-time
distributions. Aggregate output throughput and peak KV measurements are also
available per strategy. Alternating order reduces a systematic first-run bias;
it does not control host load or replace repeated runs across environments.

`calibrate_prefill_invocation_overhead` fits the two p50 measurements to the
documented model `time = dense_tokens * a + invocations * b` and returns `b/a`
as a rounded token-equivalent invocation cost. Calibration is rejected when
the strategies do not trade dense work for invocations, samples change shape,
timings are zero, or the observations imply non-positive coefficients.

`benchmark_three_way_prefill_trials` then performs a fresh comparison of
unbucketed, power-of-two, and cost-aware grouping. Strategy order rotates across
trials so each strategy occupies each position once per three measured trials.
All three executions must retain identical generated token sequences.

`benchmark_chunked_prefill_execution` compares complete-prompt execution with
a caller-selected maximum segment size. It verifies identical outputs and
reports the additional prefill invocations and executor iterations alongside
timing, token work, and KV peaks. Smaller segments intentionally trade total
work/launch overhead for scheduler interruption points; the probe does not
claim that chunking improves throughput.
Multi-request probes also report how many segment invocations used the tensor
hook versus the individual fallback.

`benchmark_prefill_fairness` runs one mixed set of prompt lengths with a bounded
live-session count and sweeps one or more chunk sizes. It records each request's
first-token and completion scheduler iteration, host TTFT and completion
latency, total iterations, and pending segment events. The repeated
`benchmark_prefill_fairness_trials` protocol discards warm-ups, rotates strategy
order, retains raw measured runs, and derives per-request nearest-rank latency
summaries. Every strategy must preserve output tokens. These are operational
host measurements, not a wall-clock fairness guarantee or publication-quality
performance claim.

## JSON reports

`PrefillBenchmarkReport` converts a completed repeated comparison into a
versioned, self-contained persistence schema. Version 2 records:

- the complete tokenized prompts and generation limit;
- warm-up and measured trial counts;
- model, device, software revision, and Unix-microsecond timestamp metadata;
- the complete strategy order in every measured trial;
- the calibrated invocation overhead and adaptive samples when present;
- every raw invocation, padding, timing, token, and KV-memory sample;
- nearest-rank latency summaries and exact aggregate token/time totals.

Durations are stored as integer microseconds. Aggregate throughput is preserved
as exact token and elapsed-microsecond inputs instead of a lossy JSON float.
`write_json_new` refuses to replace an existing path, while `read_json` and
`from_json` reject unsupported schema versions and structurally inconsistent
trial indexes.

`PlanReport` (schema version 2) serializes one executor iteration's validated
`ExecutionPlan`, its divergences, and the policy's decision record (ranking,
every admission's inputs and outcome, every batch-policy call, grouping
alternatives and cost estimates, and capability fallbacks) with stable
identifiers: integer request
IDs, string model IDs, snake_case phases and modes, positions as start/end
pairs, and admission reasons as their display text. Build one with
`PlanReport::from_iteration`; `from_json` rejects unknown schema versions.
