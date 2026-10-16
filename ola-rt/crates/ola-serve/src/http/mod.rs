//! Native HTTP transport over a [`RuntimeHandle`].
//!
//! HTTP concepts stop here: handlers translate JSON into a `RequestSpec`,
//! call the handle, and map each [`ServeError`] to its own status. Malformed
//! bodies are transport errors and never reach the driver.

mod openai;

use std::{collections::BTreeMap, convert::Infallible, future::Future, io};

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::{Stream, StreamExt, stream};
use ola_core::{GenerationConfig, Input, Priority, RequestId, RequestSpec};
use ola_observe::TokenGaps;
use ola_runtime::{ExecutionPhase, RequestStatus, RequestTiming};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::{
    FinishReason, GenerationOutput, Health, RequestEvent, RequestHandle, RequestSnapshot,
    RuntimeHandle, ServeError, ServiceState,
};

/// Builds the router for every native endpoint.
///
/// | Method and path | Purpose |
/// | --- | --- |
/// | `GET /health` | Liveness and load; always 200 while the process runs |
/// | `GET /ready` | 200 when accepting work, otherwise 503 |
/// | `GET /v1/models` | Registered models and whether each can execute |
/// | `POST /v1/generate` | Generate; JSON, or server-sent events with `stream` |
/// | `GET /v1/requests/{id}` | A live or recently finished request's state |
/// | `DELETE /v1/requests/{id}` | Cancel a request |
/// | `POST /v1/completions` | OpenAI-compatible completions (greedy only) |
pub fn router(runtime: RuntimeHandle) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/generate", post(generate))
        .route("/v1/requests/{id}", get(status).delete(cancel))
        .route("/v1/completions", post(openai::completions))
        .with_state(runtime)
}

/// Serves [`router`] on `listener` until `shutdown` resolves, then waits for
/// open connections to finish.
///
/// Open streams end when their requests do, so stop or drain the driver when
/// `shutdown` fires; otherwise a long generation keeps its connection open.
pub async fn serve(
    listener: TcpListener,
    runtime: RuntimeHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    axum::serve(listener, router(runtime))
        .with_graceful_shutdown(shutdown)
        .await
}

/// Body of `POST /v1/generate`.
///
/// Exactly one of `prompt` and `tokens` is required. Generation is greedy;
/// sampling controls are not accepted until the runtime supports them.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerateBody {
    model: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    tokens: Option<Vec<u32>>,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    /// Relative priority; larger is more urgent for priority schedulers.
    #[serde(default)]
    priority: i32,
    /// Deadline relative to arrival, for deadline-aware schedulers.
    #[serde(default)]
    deadline_ms: Option<u64>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

const fn default_max_new_tokens() -> usize {
    128
}

#[derive(Debug, Serialize)]
struct GenerateResponse {
    id: u64,
    text: String,
    token_ids: Vec<u32>,
    finish_reason: &'static str,
    usage: Usage,
    timing: TimingJson,
    /// When each token was produced, in microseconds since the server
    /// started.
    token_us: Vec<u64>,
    /// Gaps between consecutive tokens; absent with fewer than two tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    inter_token: Option<InterTokenJson>,
}

#[derive(Debug, Serialize)]
struct InterTokenJson {
    gaps: usize,
    mean_us: u64,
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    max_us: u64,
}

impl From<TokenGaps> for InterTokenJson {
    fn from(gaps: TokenGaps) -> Self {
        Self {
            gaps: gaps.gaps,
            mean_us: gaps.mean_micros,
            p50_us: gaps.p50_micros,
            p90_us: gaps.p90_micros,
            p99_us: gaps.p99_micros,
            max_us: gaps.max_micros,
        }
    }
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    generated_tokens: usize,
}

/// Lifecycle timestamps in microseconds since the server started.
#[derive(Debug, Serialize)]
struct TimingJson {
    queued_us: u64,
    started_us: Option<u64>,
    first_token_us: Option<u64>,
    completed_us: Option<u64>,
}

impl From<RequestTiming> for TimingJson {
    fn from(timing: RequestTiming) -> Self {
        Self {
            queued_us: timing.queued_at.as_micros(),
            started_us: timing.started_at.map(|at| at.as_micros()),
            first_token_us: timing.first_token_at.map(|at| at.as_micros()),
            completed_us: timing.completed_at.map(|at| at.as_micros()),
        }
    }
}

impl From<GenerationOutput> for GenerateResponse {
    fn from(output: GenerationOutput) -> Self {
        let inter_token = TokenGaps::of(&output.token_times).map(Into::into);
        Self {
            token_us: output.token_times.iter().map(|at| at.as_micros()).collect(),
            inter_token,
            id: output.request_id.get(),
            text: output.text,
            token_ids: output.token_ids,
            finish_reason: match output.finish_reason {
                FinishReason::EndOfSequence => "end_of_sequence",
                FinishReason::Length => "length",
            },
            usage: Usage {
                prompt_tokens: output.metrics.prompt_tokens,
                generated_tokens: output.metrics.generated_tokens,
            },
            timing: output.timing.into(),
        }
    }
}

#[derive(Debug, Serialize)]
struct RequestJson {
    id: u64,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    generated_tokens: usize,
    timing: TimingJson,
}

impl From<RequestSnapshot> for RequestJson {
    fn from(snapshot: RequestSnapshot) -> Self {
        let (status, error) = match snapshot.status {
            RequestStatus::Queued => ("queued", None),
            RequestStatus::Running => ("running", None),
            RequestStatus::Completed => ("completed", None),
            RequestStatus::Cancelled => ("cancelled", None),
            RequestStatus::Failed(message) => ("failed", Some(message)),
        };
        Self {
            id: snapshot.id.get(),
            status,
            error,
            generated_tokens: snapshot.generated_tokens,
            timing: snapshot.timing.into(),
        }
    }
}

#[derive(Debug, Serialize)]
struct HealthJson {
    state: &'static str,
    in_flight: usize,
    max_in_flight: usize,
    queued: usize,
    active: usize,
    kv_cache_bytes: u64,
    reserved_kv_bytes: u64,
    iterations: u64,
}

impl From<Health> for HealthJson {
    fn from(health: Health) -> Self {
        Self {
            state: match health.state {
                ServiceState::Ready => "ready",
                ServiceState::NoModels => "no_models",
                ServiceState::Draining => "draining",
                ServiceState::Stopped => "stopped",
            },
            in_flight: health.in_flight,
            max_in_flight: health.max_in_flight,
            queued: health.queued,
            active: health.active,
            kv_cache_bytes: health.kv_cache_bytes,
            reserved_kv_bytes: health.reserved_kv_bytes,
            iterations: health.iterations,
        }
    }
}

#[derive(Debug, Serialize)]
struct ModelJson {
    id: String,
    architecture: String,
    executable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_cache: Option<KvCacheJson>,
}

#[derive(Debug, Serialize)]
struct KvCacheJson {
    bytes_per_token: u64,
    tokens_per_block: usize,
    capacity_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorJson,
}

#[derive(Debug, Serialize)]
struct ErrorJson {
    kind: &'static str,
    message: String,
}

/// An error answered by the HTTP layer.
#[derive(Debug)]
enum ApiError {
    /// The body or path could not be understood; the driver never saw it.
    Transport(String),
    /// The runtime refused or failed the request.
    Serve(ServeError),
}

impl From<ServeError> for ApiError {
    fn from(error: ServeError) -> Self {
        Self::Serve(error)
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self::Transport(rejection.body_text())
    }
}

impl ApiError {
    fn parts(&self) -> (StatusCode, &'static str, String) {
        match self {
            Self::Transport(message) => (StatusCode::BAD_REQUEST, "bad_request", message.clone()),
            Self::Serve(error) => {
                let (status, kind) = classify(error);
                (status, kind, error.to_string())
            }
        }
    }

    fn body(&self) -> ErrorBody {
        let (_, kind, message) = self.parts();
        ErrorBody {
            error: ErrorJson { kind, message },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.parts().0;
        let mut response = (status, Json(self.body())).into_response();
        if retryable(status) {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

/// Returns whether a client should retry after a short wait.
fn retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status == StatusCode::SERVICE_UNAVAILABLE
}

/// Maps each serving error to an HTTP status and a stable machine-readable
/// kind.
const fn classify(error: &ServeError) -> (StatusCode, &'static str) {
    match error {
        ServeError::Invalid(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
        ServeError::UnknownModel(_) => (StatusCode::NOT_FOUND, "unknown_model"),
        ServeError::UnknownRequest(_) => (StatusCode::NOT_FOUND, "unknown_request"),
        ServeError::Overloaded { .. } => (StatusCode::TOO_MANY_REQUESTS, "overloaded"),
        ServeError::Rejected(_) => (StatusCode::UNPROCESSABLE_ENTITY, "rejected"),
        ServeError::Execution(_) => (StatusCode::INTERNAL_SERVER_ERROR, "execution_failed"),
        ServeError::Cancelled => (StatusCode::CONFLICT, "cancelled"),
        ServeError::NotReady => (StatusCode::SERVICE_UNAVAILABLE, "not_ready"),
        ServeError::ShuttingDown => (StatusCode::SERVICE_UNAVAILABLE, "shutting_down"),
        ServeError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
    }
}

async fn health(State(runtime): State<RuntimeHandle>) -> Json<HealthJson> {
    Json(runtime.health().into())
}

async fn ready(State(runtime): State<RuntimeHandle>) -> (StatusCode, Json<HealthJson>) {
    let health = runtime.health();
    let status = if health.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(health.into()))
}

async fn models(
    State(runtime): State<RuntimeHandle>,
) -> Json<BTreeMap<&'static str, Vec<ModelJson>>> {
    let models = runtime
        .models()
        .iter()
        .map(|model| ModelJson {
            id: model.descriptor.id.to_string(),
            architecture: model.descriptor.architecture.to_string(),
            executable: model.executable,
            kv_cache: model.kv_cache_profile.map(|profile| KvCacheJson {
                bytes_per_token: profile.bytes_per_token(),
                tokens_per_block: profile.tokens_per_block(),
                capacity_bytes: profile.capacity_bytes(),
            }),
        })
        .collect();
    Json(BTreeMap::from([("models", models)]))
}

async fn status(
    State(runtime): State<RuntimeHandle>,
    Path(id): Path<u64>,
) -> Result<Json<RequestJson>, ApiError> {
    Ok(Json(runtime.status(RequestId::new(id)).await?.into()))
}

async fn cancel(
    State(runtime): State<RuntimeHandle>,
    Path(id): Path<u64>,
) -> Result<StatusCode, ApiError> {
    runtime.cancel(RequestId::new(id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn generate(
    State(runtime): State<RuntimeHandle>,
    body: Result<Json<GenerateBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body?;
    let stream = body.stream;
    let spec = request_spec(body, &runtime)?;
    // Dropping the handle cancels the request, so a client that disconnects
    // while this handler or its stream is pending releases its session.
    let handle = runtime.submit(spec).await?;
    if stream {
        return Ok(Sse::new(events(handle))
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    let output = handle.result().await?;
    Ok(Json(GenerateResponse::from(output)).into_response())
}

fn request_spec(body: GenerateBody, runtime: &RuntimeHandle) -> Result<RequestSpec, ApiError> {
    let input = match (body.prompt, body.tokens) {
        (Some(prompt), None) => Input::Text(prompt),
        (None, Some(tokens)) => Input::Tokens(tokens),
        _ => {
            return Err(ApiError::Transport(
                "exactly one of `prompt` and `tokens` is required".into(),
            ));
        }
    };
    let deadline = body
        .deadline_ms
        .map(|millis| {
            runtime
                .now()
                .checked_add(std::time::Duration::from_millis(millis))
                .ok_or_else(|| ApiError::Transport("`deadline_ms` is too large".into()))
        })
        .transpose()?;
    Ok(RequestSpec {
        model: body.model.into(),
        inputs: vec![input],
        generation: GenerationConfig {
            max_new_tokens: body.max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        },
        priority: Priority::new(body.priority),
        deadline,
        attributes: body.attributes,
    })
}

#[derive(Serialize)]
struct StartJson {
    id: u64,
}

#[derive(Serialize)]
struct TokenJson {
    token_id: u32,
    text: String,
    phase: &'static str,
    generated_tokens: usize,
}

/// Turns a request's events into server-sent events: `start` with the
/// request ID, one `token` per token, then `done` with the full result or
/// `error`. Dropping the stream (a disconnect) drops the handle, which
/// cancels the request.
fn events(handle: RequestHandle) -> impl Stream<Item = Result<Event, Infallible>> {
    let start = sse(
        "start",
        &StartJson {
            id: handle.id().get(),
        },
    );
    let rest = stream::unfold(Some(handle), |handle| async move {
        let mut handle = handle?;
        let event = handle.next().await?;
        let (event, more) = match event {
            RequestEvent::Token {
                token_id,
                text,
                phase,
                generated_tokens,
            } => (
                sse(
                    "token",
                    &TokenJson {
                        token_id,
                        text,
                        phase: match phase {
                            ExecutionPhase::Prefill => "prefill",
                            ExecutionPhase::Decode => "decode",
                        },
                        generated_tokens,
                    },
                ),
                true,
            ),
            RequestEvent::Finished(output) => (sse("done", &GenerateResponse::from(output)), false),
            RequestEvent::Failed(error) => (sse("error", &ApiError::Serve(error).body()), false),
            RequestEvent::Cancelled => (
                sse("error", &ApiError::Serve(ServeError::Cancelled).body()),
                false,
            ),
        };
        Some((Ok(event), more.then_some(handle)))
    });
    stream::once(async move { Ok(start) }).chain(rest)
}

fn sse(name: &str, data: &impl Serialize) -> Event {
    Event::default()
        .event(name)
        .json_data(data)
        .expect("response types always serialize")
}
