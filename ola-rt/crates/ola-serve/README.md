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
also offer a blocking `wait` for plain threads. The driver itself uses only
tokio's `sync` channels.

## HTTP

`ola_serve::http::router` (axum) exposes the driver; `ola serve --model DIR`
or `make serve MODEL=…` runs it.

| Method and path | Result |
| --- | --- |
| `GET /health` | Always 200 while the process runs: state, load, KV, iterations |
| `GET /ready` | 200 when accepting work, otherwise 503 with the same body |
| `GET /v1/models` | Registered models, whether each can execute, KV geometry |
| `POST /v1/generate` | The result as JSON, or server-sent events with `"stream": true` |
| `GET /v1/requests/{id}` | A live or recently finished request's status and timing |
| `DELETE /v1/requests/{id}` | Cancel; 204, also for a request that already ended |

```text
POST /v1/generate
{"model": "llama", "prompt": "Hello", "max_new_tokens": 16,
 "priority": 0, "deadline_ms": 2000, "stream": true, "attributes": {}}
```

Send exactly one of `prompt` and `tokens` (token IDs). Unknown fields are
rejected, including sampling controls: generation is greedy. A stream sends
`start` (`{"id"}`), one `token` per token (`token_id`, `text` delta,
`phase`, `generated_tokens`), then `done` with the full result or `error`.
Closing the connection cancels the request, whether streaming or waiting for
JSON. Timings are microseconds since the server started.

Errors are `{"error": {"kind", "message"}}`:

| Status | Kind |
| --- | --- |
| 400 | `bad_request` (malformed body, never reaches the runtime), `invalid_request` |
| 404 | `unknown_model`, `unknown_request` |
| 409 | `cancelled` |
| 422 | `rejected` (admission: can never fit) |
| 429 | `overloaded`, with `Retry-After: 1` |
| 500 | `execution_failed`, `internal` |
| 503 | `not_ready`, `shutting_down`, with `Retry-After: 1` |
