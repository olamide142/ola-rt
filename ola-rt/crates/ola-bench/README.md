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
