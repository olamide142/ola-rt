# ola-serve

`ola-serve` is the application-facing boundary around `ola-runtime`: clients
submit, observe, stream, cancel, and retrieve requests without touching
`Executor`. The design is in `SERVING.md`.

A `Driver` starts one thread that owns the executor. Model work blocks a CPU
or GPU, so it never runs on an async runtime. Clients hold a cloneable
`RuntimeHandle`:

```text
let driver = Driver::spawn(executor, MonotonicClock::start(), DriverConfig::default())?;
let mut request = driver.handle().submit(spec).await?;   // or .wait() on a thread
while let Some(event) = request.next().await {           // or blocking_next()
    // Token { text, .. } … then Finished, Failed, or Cancelled
}
```

- **Arrival time** is stamped from the driver's clock when `submit` is
  called, so time spent waiting behind a running iteration counts toward
  latency.
- **Streaming.** Each request gets `Token` events whose `text` deltas
  concatenate to the final text (characters split across tokens are held
  back), then exactly one terminal event. Results move out of the executor
  rather than being retained in it.
- **Cancellation.** `cancel` ends a request and releases its session and KV
  cache. Dropping an unfinished `RequestHandle`, or an unresolved
  `Submission`, cancels too, so a client disconnect cleans up; `detach` opts
  out.
- **Backpressure.** At most `max_in_flight` requests may be unfinished;
  further submissions fail at once with `ServeError::Overloaded`. Streaming
  never blocks the driver: a request emits at most `max_new_tokens` + 1
  events.
- **Retention.** A finished request's record leaves the executor and its
  final status stays queryable until `retain_finished` (default 1,024)
  newer requests have finished; after that it is unknown.
- **Health.** `RuntimeHandle::health` reads state, load, KV gauges, and an
  iteration counter without waiting for the driver, so it answers during a
  long iteration. `ServiceState` is `Ready`, `NoModels` (submissions fail
  with `NotReady`), `Draining`, or `Stopped`; a driver thread that panics
  reports `Stopped`. `models` lists registered models and whether each is
  executable.
- **Shutdown.** `Driver::drain(grace)` refuses new work and lets accepted
  requests finish until the grace period ends on the driver's clock.
  `Driver::shutdown`, or dropping the driver, stops at once. Either way
  unfinished requests end with `ServeError::ShuttingDown` and the executor
  is returned.

`Reply` and `Submission` are futures that work under any async runtime, and
also offer a blocking `wait` for plain threads. The crate depends only on
tokio's `sync` channels.
