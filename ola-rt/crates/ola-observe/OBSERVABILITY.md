# Observability design

Status: **accepted** (2026-10-16) with all recommended decisions.
Covers `todo.md` P6. All five steps are implemented; P6 is complete.

## Goal

A request can be traced from arrival through every decision, execution
batch, token, and terminal result, with per-token timestamps and correlation
IDs, and the traces are covered by tests. That is P6's completion criterion.

## What existed

- **Lifecycle times were stamped at the iteration's planning time.** The
  first token and completion carried the time the iteration *started*, so
  TTFT and completion omitted the producing iteration's own model time. The
  P3 simulator inherited the same bias.
- **No model-operation timing.** `ExecutionBatch` recorded work, not time.
- **Inter-token latency was an average:** decode span ÷ token gaps.
- **`ola-observe` had no consumers** outside its own tests.
- **GPU memory is reachable, utilization is not.** CubeCL reports allocator
  usage through `burn-cubecl`; utilization would need NVML, and this
  machine's NVIDIA driver is not currently usable, so neither GPU path can
  be validated here.

## Decisions

1. **Times mark when results exist.** `Executor::with_clock` reads the clock
   around every model invocation. Each `ExecutionBatch` records
   `started_at` (just before the call) and `finished_at` (after the selected
   token IDs are back on the host, so device synchronization is included).
   Token events, first token, completion, and failure take the producing
   invocation's `finished_at`. Without a clock every time is the planning
   time, which keeps replays deterministic. The driver installs its clock.
   The simulator stamps tokens at the virtual finish of their batch.
2. **Per-token timestamps and real TPOT.** Results keep one timestamp per
   token; `ola-observe` reports per-gap distributions per request and pooled
   across requests, replacing the averaged inter-token latency.
3. **Correlation IDs.** An `IterationId` on each iteration and its plan;
   batches are addressed by iteration and index; request IDs already exist.
4. **Typed traces.** `ola_observe::trace` assembles one record stream per
   request (arrival; ranking, admission, and batch decisions with reasons;
   each batch with phase, mode, and timing; each token; the terminal
   outcome) read-only from iterations, serialized as JSON Lines. Exposed by
   `ola serve --trace FILE`, `GET /v1/requests/{id}/trace`, and
   `ola experiment run --trace FILE` for executor-driven experiments. The
   `tracing` crate is not used: tests must assert on traces.
5. **Device telemetry.** A `DeviceTelemetry` trait: CubeCL allocator memory
   and utilization, none on the CPU backend, and a portable busy fraction
   from measured batch intervals, reported by `/health`. (Revised in step
   5: CubeCL 0.11 reads utilization itself, through NVML on NVIDIA cards,
   so it is not deferred.)
6. **Read-only.** Tests show tracing and telemetry leave plans, outputs, and
   lifecycle unchanged.

## Steps

1. **Event-time correctness.** **Done.** `Executor::with_clock`;
   `ExecutionBatch::{started_at, finished_at, latency_micros}`;
   `ExecutorEvent::Token::at`; `ExecutionResult::token_times` and
   `GenerationOutput::token_times`; the driver installs its clock; and
   `CostModel::batch_finish_offsets` lets the simulator stamp tokens at
   their batch's virtual finish. Tests use a clock that advances on every
   read, and a driver test releases each gated model call 100 µs after the
   previous one and checks every token time.

   *Negative result.* Correct stamping reversed one P3 finding. Isolated
   times had been understated by one iteration, and were zero for
   one-token requests, which then dropped out of slowdown. SJF had appeared
   to lower slowdown fairness; it raises it (overload trace, seed 3: SJF
   0.55 vs FIFO 0.32; five-seed study: FIFO 0.47 → 0.41, SJF 0.46 → 0.49).
   Latencies moved 0.5–1.5%. `SCHEDULING_BASELINES.md` records the
   correction; the other P3 findings still hold.
2. **Inter-token distributions.** **Done.** `ola_observe::TokenGaps`
   (count, mean, nearest-rank p50/p90/p99, max; `pooled` never bridges
   requests) replaces the averaged inter-token latency in
   `RequestMeasurement`; the observer collects token times from events and
   drops them once a request is summarized. `llama-generation` (payload 2)
   now runs on a real clock and reports TTFT and inter-token p50/p90/max; on
   the tiny-Llama fixture, TTFT 11.4 ms plus 23 gaps of ~10 ms accounts for
   the 243.5 ms elapsed. Native HTTP results add `token_us` and
   `inter_token`. An HTTP test with a 1 ms stall among 100 µs gaps checks
   every field: the stall shows at p99 and max, not p50. Writing it exposed
   a race in the test gate (a just-permitted caller still counted as
   blocked); after fixing it the serving suites passed 990 runs.
3. **Correlation IDs and traces.** **Done.** `IterationId`, numbered from 1
   by the executor and stamped on both `ExecutorIteration` and its
   `ExecutionPlan` before validation, so a policy cannot forge it; a
   rejected step consumes none. `PlanReport` schema 3 records it.
   Invocations are addressed by iteration and index. `ola_observe::trace`
   builds per-request records and writes JSON Lines. A real executor run
   traces queueing behind a full slot, admission into a predicted release,
   completion, and cancellation before any iteration, with times and
   iteration IDs checked, and the JSON Lines round trip exact.
4. **Traces in the server and experiments.** **Done.**
   `DriverConfig::trace` keeps traces bounded by the retention table;
   `Driver::spawn_with_trace_sink` streams them; a failing sink is dropped
   and reported in health. `GET /v1/requests/{id}/trace` (404
   `tracing_disabled` when off). `ola serve --trace FILE` and
   `ola experiment run --trace FILE`; `Experiment::traceable` marks
   `llama-generation` and `scheduler-comparison`, whose simulator gained an
   observation hook (`run_observed_trial`, `compare_schedulers_observed`).
   Records carry a `run` label so one file holds many executor runs. Live
   on tiny Llama, a trace shows 0.4 ms from admission to the first call,
   a 15 ms prefill, and 12.5 ms decodes; the server's endpoint and sink
   held the same 13 records.
5. **Telemetry, read-only guarantees, end-to-end trace.** **Done.**
   `ola_observe::DeviceTelemetry` (memory, utilization) and
   `ola_kernel::CubeclTelemetry` on CubeCL's runtime-erased `Device`, with
   `wgpu_telemetry` and `cuda_telemetry`. The driver samples telemetry on
   its own thread after every iteration and publishes it, with cumulative
   invocation time, in `Health` and `/health`; `ola serve` attaches it on
   WGPU and CUDA builds. `IterationMeasurement` gained `invocation_micros`
   and `busy_fraction`, which pairs each interval with the previous
   iteration's calls. Evidence: on CubeCL's CPU runtime a 1 MiB allocation
   shows in the reported memory and utilization reads in range; the WGPU
   build compiles, but this machine's NVIDIA driver is unusable, so no GPU
   reading has been observed. Read-only: a workload run plain and with the
   tracer and observer on every iteration produces identical iterations
   (plans, decisions, events, invocations) and results, and the scheduler
   study reports identical samples with and without `--trace`. On the
   fixture, a served request's trace runs from arrival to completion, each
   token carrying the finish time of the invocation before it, together
   equal to the response's token times.

## Out of scope

NVML utilization, distributed tracing (OpenTelemetry), and kernel,
prefix-cache, and residency instrumentation, which belong to the kernel
track, P7, and P8.
