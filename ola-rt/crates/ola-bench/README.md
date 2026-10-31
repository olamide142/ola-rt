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

The feature-gated `paged-attention-kernel` study compares portable and CubeCL
attention across page sizes, batches, partial final pages, and configurable
unused physical KV blocks (`pool_slack_blocks`). It reports physical-pool
fragmentation explicitly; accelerator occupancy comes from the separate
profiler-backed `paged-attention-occupancy` study below.

## Accelerator occupancy

The feature-gated `paged-attention-occupancy` study measures CubeCL paged
attention over the cross product of `page_sizes`, `batch_sizes`, and
`sequence_lengths` (one variant per shape, labelled
`page=P,batch=B,seq=S`), with configurable `query_heads`, `kv_heads`, and
`head_dim`. Every sample records host latency, oracle error against the
portable kernel, page/padding counts, and the kernel's work-item count.

Occupancy is never inferred from host time. It comes from an Nsight Compute
capture of an identical run, attached with `profile=<csv>`. The study then
adds, per measured repetition:

| Metric | Nsight counter | Domain |
| --- | --- | --- |
| `gpu_duration_micros` | `gpu__time_duration.sum` | device |
| `achieved_occupancy` | `sm__warps_active.avg.pct_of_peak_sustained_active` | untimed ratio |
| `theoretical_occupancy` | `sm__maximum_warps_per_active_cycle_pct` | untimed ratio |
| `sm_throughput` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | untimed ratio |
| `memory_throughput` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | untimed ratio |
| `registers_per_thread` | `launch__registers_per_thread` | untimed count |

Two-pass workflow on an NVIDIA machine (CUDA toolkit with NVRTC and Nsight
Compute installed, and profiling permission: root or
`NVreg_RestrictProfilingToAdminUsers=0`):

```sh
cargo build --release -p ola-cli --features cuda
P="--set page_sizes=8,16,32 --set batch_sizes=1,8 --set sequence_lengths=64,256"

# 1. Capture. Kernel replay serializes and perturbs latency, so this run's
#    own report is discarded.
ncu --kernel-name regex:paged_attention \
    --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__grid_size,launch__block_size \
    --csv --page raw --log-file occupancy.csv \
    target/release/ola experiment run paged-attention-occupancy $P

# 2. Measure latency unprofiled with the same parameters, attaching the capture.
target/release/ola experiment run paged-attention-occupancy $P \
    --set profile=occupancy.csv --results-dir results
```

`nsight_metrics()` returns the exact `--metrics` list. If the generated CUDA
kernel name differs on a toolchain, change both `--kernel-name` and the
study's `profile_kernel` filter. Each shape launches
one oracle-check dispatch, `warmups`, then `repetitions`, in page → batch →
sequence order, and each homogeneous batch is exactly one `paged_attention`
launch, so launches map onto samples by position. The capture is rejected
before any dispatch if its launch count differs or a launch's
`grid × block` thread count does not cover that shape's work items, which
catches a capture made with different parameters. Percentages are stored as
0–1 ratios and durations are normalized to microseconds from the unit row.
The capture's digest, device name, and launch count go into
`details.profiler` and the workload digest, so reports with different
captures never compare as the same workload.

Hardware limitations:

- Nsight Compute counters exist only for NVIDIA GPUs through the `cuda`
  feature. A WGPU run on an NVIDIA card goes through Vulkan, which `ncu`
  does not profile; WGPU and `cubecl-cpu` builds produce latency-only
  reports (summary ends with `no profile attached`).
- Turing (the development GTX 1650) and newer support the counters above;
  older architectures need `nvprof` and are not parsed.
- Without the `profile` parameter the study is still a valid host-time study;
  with it, occupancy reflects replayed launches, not the timed ones.
- The development laptop's NVIDIA driver is currently unavailable
  (`nvidia-smi` cannot reach it), so no profiled report has been recorded yet.

## Experiment framework

Every benchmark and experiment is an `Experiment` in `ExperimentRegistry`
(see `EXPERIMENTS.md`). A run produces an `ExperimentReport`
(`ola-experiment/1`):

- every metric declares its unit, clock (virtual, host, or untimed), and
  which direction is better;
- raw samples are kept beside summaries, and summaries are recomputed and
  checked on every read;
- each summary carries a seeded bootstrap 95% interval;
- identity covers experiment and payload version, probe or study, model,
  device, build (git revision, profile, target, features), workload digest,
  protocol, and each variant's policy and kernel;
- `write_new` never overwrites, and `read_compatible` also converts legacy
  prefill schema-v2 reports.

`compare_reports` lists identity differences and classifies each metric as
improved or regressed only when intervals separate and the change exceeds a
threshold; otherwise inconclusive. Probes are labelled not for publication;
studies must meet minimum repetitions (3, or 5 plus a warm-up in host time).

The `paged-attention-kernel` study is available with
`--features cubecl-cpu` for CPU CubeCL validation or `--features wgpu` for the
local WGPU device. It records portable and optimized latency distributions,
page/batch workload parameters, and maximum absolute oracle error. These
feature-enabled runs are the required entry point for kernel performance
claims; the default `ola-bench` build does not compile accelerator backends.
The default workload uses 23 populated tokens, so page sizes that do not divide
that length also report logical page count and padding-token waste.
`cost-calibration` fits `fixed + per_token·dense + per_call·calls` by least
squares over a shape grid, and `CostFit::from_report` turns a saved fit into
a `CostModel` for virtual-time comparisons. The terminal workbench and the
`ola` CLI (`ola-cli`) are both thin frontends over the registry.

## Memory pressure

The synthetic model allocates from a real `KvBlockManager` pool, so
simulations exercise allocation failure, preemption, and resume. Output
tokens are at least 1000 and prompt tokens below it, so a resumed request
(prompt plus earlier output) continues exactly as a real model would.
`SchedulingTrace::early_stopping` models callers that set a generous output
limit most requests never reach.

`synthetic_artifact_with_prefix_cache` (or `SimulationConfig::prefix_cache`)
enables the pool's prefix cache, and `SchedulingTrace::shared_prefix` builds
prompts that share a system prompt and end in tokens unique to each request.
The `prefix-cache` study compares the pool with and without the cache.

The `kv-pressure` study compares worst-case reservation
(`MemoryAwareAdmission`) with `WatermarkAdmission` plus newest-first
preemption over seeded traces in virtual time, reporting latency,
throughput, peak KV, preemptions, failures, unfinished requests, and input
tokens (which include recompute after preemption).

The `kv-page-size` study sweeps tokens per page over a shared-prefix trace,
reporting internal fragmentation without the prefix cache (from
`SchedulingSummary::kv_internal_fragmentation`) and prefix reuse and latency
with it.

## Scheduling comparisons

Scheduling policies are compared in virtual time, so every number is exactly
reproducible and independent of the host:

- `SchedulingTrace` generates seeded traces (`mixed_short_long`,
  `prefill_decode_competition`, `priority_classes`, `deadlines`, `overload`,
  `impossible`) using integer-only arithmetic.
- `run_scheduling_trial` replays a trace through a real `Executor` and
  execution policy with a deterministic `SyntheticModel`, advancing a virtual
  clock by a `CostModel` over each iteration's actual invocations.
  `run_perturbed_trial` additionally cancels requests or injects invalid ones
  mid-run.
- `compare_schedulers` summarizes each policy against a shared baseline, each
  request's time when run alone: queue, TTFT, and completion percentiles;
  throughput; peak KV; deferrals and backfill skips; slowdown with Jain's
  fairness index; starvation (waits over 10× isolated time); deadline misses;
  and per-priority and size classes.

Effects only appear under contention: with short queues, ranking rarely
matters. Tests pin measured regimes (priority protects urgent requests, SJF
trades long-request latency for short, backfill helps when the KV pool binds,
EDF helps under moderate contention but not overload). The terminal
workbench's *Scheduler comparison* experiment runs FIFO, SJF, and priority
on two seeded traces.

## Adaptive policy studies

`adaptive-policy-study` and `online-adaptive-policy-study` tune a
`MixedTokenBudgetPolicy` over a log-scale grid (2^4 to 2^10 tokens per
iteration, 25–100% prompt work) to minimize p95 completion plus p95 time to
first token, in virtual time.

- `adaptive-policy-study` picks the budget offline: it replays a training
  trace (seed 0) under every candidate, then compares the winner with FIFO
  on held-out seeds.
- `online-adaptive-policy-study` uses `ReplayOptimizer` inside an
  `AdaptivePolicy`: every `optimization_interval` iterations it replays the
  last `window` completed requests under each candidate and switches when
  one beats the current budget by `min_improvement_percent`. It reports
  `online` (from the smallest budget), `online-warm` (from the offline
  choice), `fixed` (the offline choice), and `fifo`, on traces whose
  workload shifts from mixed short/long requests to an overload burst.

Results and their limits are in `../ola-runtime/ADAPTIVE_OPTIMIZATION_PLAN.md`.
Replays use the simulator's `CostModel`, so on hardware the optimizer is only
as good as a calibrated one.

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
