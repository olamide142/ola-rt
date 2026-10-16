# Runtime driver and serving design

Status: **accepted** (2026-10-16) with all recommended decisions.
Covers `todo.md` P5. Steps 1–2 are implemented.

## Goal

A client can submit, observe, stream, cancel, and retrieve a request without
touching `Executor`. This is P5's completion criterion.

## What exists

- **`Executor` is `Send` but has no owner loop.** Its policy, models, and
  sessions are all `Send`; the terminal workbench drives it by hand on a
  worker thread over `std::sync::mpsc`.
- **`ola-serve` is a thin facade.** `InferenceService` hands out
  `&mut Runtime` and reports a constant `Ready`. Nothing depends on it.
- **Everything is retained forever.** The executor keeps every result and the
  runtime keeps every request record.
- **Text arrives only at the end.** The tokenizer is reachable through
  `CausalLanguageModel::tokenizer`, but nothing decodes as tokens arrive.

## Decisions

1. **One driver thread owns the executor.** Model work blocks a CPU or GPU,
   so it never runs on an async runtime. Clients hold a cloneable
   `RuntimeHandle` and talk to the thread over channels. When there is no
   work the thread blocks on its command channel instead of spinning. The
   driver lives in `ola-serve` and uses only tokio's `sync` channels, which
   work from plain threads and from any async runtime, so `ola-runtime`
   stays synchronous.
2. **Request handles.** `submit` returns a `RequestHandle` with `id`,
   `status`, `cancel`, an event stream (`Token { id, text_delta, phase }`,
   then exactly one terminal `Finished`, `Failed`, or `Cancelled`), and an
   awaitable or blocking `result`. Results move out of the executor with
   `Executor::take_result`. Dropping an unfinished handle cancels the
   request, which is how a client disconnect becomes session and KV
   cleanup; `detach` opts out.
3. **Backpressure.** Admission is a bounded count of unfinished requests;
   past it, `submit` fails at once with `Overloaded`. Streaming never blocks
   the model loop: a request emits at most `max_new_tokens` + 1 events, so
   its buffer is bounded by construction.
4. **Retention.** Terminal records stay in a bounded most-recently-used
   table (default 1,024) so status lookups by ID keep working; older ones
   are dropped from the driver and from runtime state with
   `Executor::forget`. Persistence is out of scope for P5.
5. **Clock.** `ola_core::Clock` converts a process `Instant` epoch into
   `Timestamp`s (`MonotonicClock`); `ManualClock` makes driver tests
   deterministic.
6. **Errors stay distinct.** `ServeError`: `Invalid` (400), `UnknownModel`
   (404), `Overloaded` (429), `Rejected(AdmissionReason)` (422),
   `Execution` (500), `ShuttingDown` and `NotReady` (503), and `Transport`
   for the HTTP layer only.
7. **Shutdown.** `Drain { deadline }` stops accepting work, lets queued and
   active requests finish until the deadline, cancels the rest, and joins
   the thread; `Abort` cancels everything at once. Every open handle
   receives a terminal event either way.
8. **HTTP (axum).** `GET /health`, `GET /ready`, `GET /v1/models`,
   `POST /v1/generate` (JSON, or server-sent events with `stream: true`;
   a dropped connection cancels), `GET` and `DELETE /v1/requests/{id}`, and
   finally an OpenAI-compatible `POST /v1/completions`. Chat completions
   need prompt templates and belong to P8. `ola serve --model DIR` starts it.

## Steps

1. **Foundations.** **Done.** `Clock`, `MonotonicClock`, and `ManualClock`
   in `ola-core`; `RequestStatus::is_terminal` and `Runtime::forget_terminal`;
   `Executor::take_result` and `Executor::forget`; and
   `ola_model::IncrementalDecoder`, which decodes a short window twice and
   emits the difference, holding back text that ends in an incomplete
   character.
2. **Driver and handles.** **Done.** `Driver`, `RuntimeHandle`,
   `Submission`, `RequestHandle`, `Reply`, and `RequestEvent`; capacity is
   checked on the client side so `Overloaded` never waits for the driver.
   The driver blocks on its command channel when idle and backs off briefly
   after an iteration that made no progress. Listeners that disappear are
   detected before commands are drained, so a `Detach` sent just before a
   handle is dropped still wins. A failed plan ends every request with
   `Internal` instead of retrying forever. Tests use a scripted model whose
   calls wait on a gate, which makes cancellation and shutdown races
   deterministic; they passed 1,300 consecutive runs, 800 of them under CPU
   contention.
3. Retention table, health and readiness, draining shutdown, and `NotReady`.
4. Native HTTP endpoints with server-sent events and disconnect-cancels;
   `ola serve` and `make serve`.
5. `/v1/completions`; end-to-end streaming, cancellation, overload, failure,
   and shutdown tests, including the tiny-Llama fixture; docs.

## Out of scope

Result persistence, authentication, TLS, multiple executor threads or
devices, chat templates, and sampling beyond greedy decoding.
