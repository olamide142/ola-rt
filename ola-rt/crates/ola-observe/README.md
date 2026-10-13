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
| Mean inter-token latency | decode span divided by `output_tokens - 1` |

Fields whose lifecycle boundary was never reached remain `None`. For example,
a request rejected while queued has end-to-end latency but no queue, prefill,
or service latency. Mean inter-token latency is an aggregate span, not a record
of each token gap, and is unavailable when fewer than two output tokens were
observed.

## Iteration measurements

`ExecutionObserver::observe_iteration` records:

- selected prefill/decode operations and input-token cost from `IterationPlan`;
- batch-policy deferrals from the plan;
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
