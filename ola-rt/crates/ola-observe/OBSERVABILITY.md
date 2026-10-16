# Observability design

Status: **accepted** (2026-10-16) with all recommended decisions.
Covers `todo.md` P6. Steps 1–2 are implemented.

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
   (compile-checked only here), none on CPU, and a portable busy fraction
   from measured batch intervals, reported by `/health`. NVML utilization is
   deferred.
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
3. Iteration and batch correlation IDs; the per-request trace builder and
   JSON Lines output.
4. Traces in the server and in executor-driven experiments.
5. Device telemetry and busy fraction; read-only tests; an end-to-end trace
   test on the fixture; docs; close P6.

## Out of scope

NVML utilization, distributed tracing (OpenTelemetry), and kernel,
prefix-cache, and residency instrumentation, which belong to the kernel
track, P7, and P8.
