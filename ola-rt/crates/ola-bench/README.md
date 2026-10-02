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

This crate is for reusable benchmark machinery. Individual hypotheses,
configurations, raw data, plots, and interpretations belong in `experiments/`.
