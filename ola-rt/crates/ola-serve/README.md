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
- **Tracing.** With `DriverConfig::trace`, the driver keeps each request's
  trace (see `ola-observe`) while it is live and for as long as its final
  status is retained; `RuntimeHandle::trace` returns it.
  `Driver::spawn_with_trace_sink` also writes every record as JSON Lines,
  flushing after each iteration. A failing sink is dropped and reported as
  `Health::trace_write_failed`; requests are unaffected.
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
| `GET /v1/requests/{id}/trace` | The request's trace records (`ola_observe::TraceRecord`) |
| `POST /v1/completions` | OpenAI-compatible completions, JSON or streamed |

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

A finished result includes `finish_reason` (`end_of_sequence` or `length`),
`token_us` (when each token was produced, stamped as its model call
finished), and `inter_token` (`gaps`, `mean_us`, `p50_us`, `p90_us`,
`p99_us`, `max_us` over the gaps between consecutive tokens; absent with
fewer than two tokens).

Errors are `{"error": {"kind", "message"}}`:

| Status | Kind |
| --- | --- |
| 400 | `bad_request` (malformed body, never reaches the runtime), `invalid_request` |
| 404 | `unknown_model`, `unknown_request`, `tracing_disabled` |
| 409 | `cancelled` |
| 422 | `rejected` (admission: can never fit) |
| 429 | `overloaded`, with `Retry-After: 1` |
| 500 | `execution_failed`, `internal` |
| 503 | `not_ready`, `shutting_down`, with `Retry-After: 1` |

### OpenAI-compatible completions

`POST /v1/completions` accepts `model`, `prompt` (one string or one token
array), `max_tokens` (default 16), `stream`, and `user`, and answers with a
`text_completion` object (`id` `cmpl-<request id>`, one choice,
`finish_reason` `stop` or `length`, and `usage`). Streaming sends one chunk
per non-empty text delta, a final chunk with `finish_reason` and `usage`,
then `data: [DONE]`.

Generation is greedy, so requests that would change the output are refused
with 400 instead of being silently ignored: `temperature` other than 0,
`top_p` other than 1, `n` or `best_of` other than 1, `logprobs`, `echo`,
`suffix`, `stop`, `logit_bias`, non-zero penalties, batched prompts, and
unknown fields. `seed` is accepted and has no effect. Errors use OpenAI's
shape, `{"error": {"message", "type", "param", "code"}}`, with `type` set to
the native kind. Chat completions need prompt templates and are not offered.

## Cluster roles

`ola serve` also runs the two roles of a distributed deployment
(`../DISTRIBUTED.md`):

```text
ola serve --role worker  --model DIR --name worker-a --addr 127.0.0.1:7001
ola serve --role worker  --model DIR --name worker-b --addr 127.0.0.1:7002
ola serve --role gateway --addr 127.0.0.1:8080 \
          --worker 127.0.0.1:7001 --worker 127.0.0.1:7002
```

A **worker** (`cluster::serve_worker`) wraps one executor and waits for one
coordinator connection. It heartbeats and reports its load every 100 ms,
runs executor iterations whenever it has work, and uses its start time as
its epoch, so a restarted worker replaces its old lease. Start workers
before the gateway; the gateway connects to each listed address at startup.

The **gateway** (`cluster::ClusterGateway`) owns the coordinator and serves
`GET /health`, `GET /ready`, `POST /v1/generate`, `POST /v1/completions`, and
`DELETE /v1/requests/{id}`. It routes with `AdaptiveClusterPolicy` by
default (`ClusterGateway::connect_with` takes another policy and a
`ClusterGatewayConfig`).

- `/health` reports connected `workers` and `ready_workers`; `/ready` is 503
  until at least one worker is ready.
- A request waits in the coordinator while every worker hosting its model is
  busy or draining, and fails if no worker hosts the model.
- A worker silent for the lease (2 s by default, checked every 50 ms) or
  whose connection drops is lost. Its requests restart on another worker by
  recompute; clients already streaming see no repeated tokens, since the
  gateway's routes decode greedily.
- A client that disconnects has its request cancelled on the worker.
- Cancelling an unknown request is 404.

The gateway's request bodies match the single-process routes in shape but
are narrower: `/v1/generate` takes `model`, exactly one of `prompt` or
`tokens`, `max_new_tokens`, `priority`, and `stream`; `/v1/completions`
takes `model`, a `prompt` (one string or one token array), `max_tokens`, and
`stream`. Errors are `{"error": message}`. Workers decode their own tokens,
so `/v1/generate` stream events carry `token_id`, `text`, and
`generated_tokens`, and completion streaming sends one chunk per non-empty
text delta, a final chunk with `finish_reason` (`length` at the token
limit, else `stop`) and `usage`, then `data: [DONE]`.
