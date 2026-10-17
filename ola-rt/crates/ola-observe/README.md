# ola-observe

`ola-observe` converts OlaRT lifecycle state and executor iterations into
backend-neutral measurements. It reads runtime state but never mutates it or
selects work. This keeps instrumentation outside the correctness-critical
lifecycle path and lets servers, benchmarks, the terminal workbench, and future
optimization policies consume the same definitions.

All stored durations use integer microseconds because `ola_core::Timestamp`
uses a caller-defined monotonic microsecond epoch. No wall clock is read here.

## Request latency definitions

For a terminal request:

| Measurement | Formula |
| --- | --- |
| Queue latency | `started_at - queued_at` |
| Time to first token (TTFT) | `first_token_at - queued_at` |
| Prefill latency | `first_token_at - started_at` |
| Service latency | `completed_at - started_at` |
| End-to-end latency | `completed_at - queued_at` |
| Decode span | `completed_at - first_token_at` |
| Inter-token latency (TPOT) | distribution of `token[i] - token[i-1]` over observed token times |

Fields whose lifecycle boundary was never reached remain `None`. For example,
a request rejected while queued has end-to-end latency but no queue, prefill,
or service latency.

Token times come from each `ExecutorEvent::Token`'s `at`: the time the model
invocation that produced the token finished when the executor has a clock,
otherwise the iteration's planning time. `TokenGaps` summarizes the gaps
between consecutive tokens (count, mean, nearest-rank p50/p90/p99, max), so
the first token contributes no gap and a stall shows in the tail instead of
being averaged across the decode span. It is `None` with fewer than two
observed tokens; an observer attached mid-generation summarizes only the
gaps it saw. `TokenGaps::pooled` combines requests without bridging them. The
observer keeps token times only until a request is summarized.

## Iteration measurements

`ExecutionObserver::observe_iteration` records:

- selected prefill/decode rows and input-token cost from the iteration's
  `ExecutionPlan` estimates;
- batch-policy (budget) deferrals from the plan;
- the number of plan divergences observed while committing it;
- actual tensor-batch and individual-invocation counts from execution records;
- meaningful model-input positions, right-padding positions, and input-token
  efficiency from execution records;
- emitted tokens, completed requests, failures, rejections, and admission
  deferrals from executor events;
- live request counts and KV usage from the iteration snapshot;
- elapsed microseconds since the previously observed iteration.

Selected-input, dense-input, and output rates are only returned for a non-zero
observed interval. Dense input includes padding, while input-token efficiency is
`real / (real + padding)`. The first sample therefore has no rate.
`ObservationTotals` accumulates counts but does not pretend that caller-supplied
simulated time is wall-clock performance.

```rust,no_run
use ola_observe::ExecutionObserver;

let mut observer = ExecutionObserver::default();
# let iteration: ola_runtime::ExecutorIteration = unimplemented!();
# let state: ola_runtime::RuntimeState = unimplemented!();
let measurement = observer.observe_iteration(&iteration, &state)?;
println!("selected {} input tokens", measurement.scheduled_input_tokens);
println!("padding positions: {}", measurement.padding_input_tokens);

# Ok::<(), ola_observe::ObservationError>(())
```

## Request traces

`Tracer` turns executor iterations into one record stream per request,
reading the iteration and the runtime state after it and changing neither.
Each `TraceRecord` carries the request ID, the iteration ID (shared with the
iteration's `ExecutionPlan` and plan reports), the time the fact became
true, and one `TraceEvent`:

| Event | When | Time |
| --- | --- | --- |
| `arrived` | first seen (model, limit, priority, deadline) | queued |
| `ranked` | ranked by the scheduler (rank of n) | planning |
| `admission` | `admitted` (reserved KV, budget fit), `deferred` or `rejected` (reason), `no_slot`, `preparation_failed` | planning |
| `backfill_skipped` | later work admitted past it | planning |
| `selection` | the batch policy selected or deferred its next operation (stage, token cost) | planning |
| `invocation` | it ran in a model call (index, phase, tensor or individual, rows, tokens, padding, finish) | call start |
| `prefill_progress` | a prompt segment ran | call finish |
| `token` | a token was produced (index, ID, phase) | call finish |
| `completed`, `failed`, `rejected`, `cancelled` | terminal | terminal |

Within an iteration, decisions come first, then each invocation in
execution order followed by what it produced. Cancellations happen between
iterations: report them with `Tracer::observe_cancellation`. Feed the
tracer before terminal runtime records are forgotten. `trace::write_jsonl`
and `trace::read_jsonl` use JSON Lines, one record per line.

## Device activity and telemetry

Every `ExecutionBatch` records when its model invocation started and
finished (with an executor clock). `IterationMeasurement::invocation_micros`
sums an iteration's invocation latency, and `busy_fraction` is the share of
the interval since the previous iteration started that the previous
iteration's invocations occupied: a backend-independent proxy for device
activity. `ObservationTotals::invocation_micros` accumulates it.

`DeviceTelemetry` is the backend-neutral interface for device readings:
`memory` (bytes in use and reserved by the allocator) and `busy_percent`
(utilization from the device's own counter). `ola-kernel` implements it for
CubeCL devices. Readings may wait for the device, so drivers sample them on
the thread that runs execution, never on request paths.
