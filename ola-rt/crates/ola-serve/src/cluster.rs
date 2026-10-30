//! HTTP gateway backed by networked [`ola_cluster`] workers.

use std::{
    collections::{BTreeMap, VecDeque},
    convert::Infallible,
    future::Future,
    sync::Arc,
};

use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use ola_cluster::{
    AdaptiveClusterPolicy, ClientEvent, TcpCoordinator, TcpCoordinatorError, WorkerHost,
    WorkerSession, WorkerSessionError,
};
use ola_core::{
    Clock, ClusterRequestId, GenerationConfig, Input, MonotonicClock, Priority, RequestSpec,
    WorkerId,
};
use serde::{Deserialize, Serialize};
use tokio::{net::TcpListener, sync::Mutex};

/// Shared state for a networked gateway.
#[derive(Clone)]
pub struct ClusterGateway {
    state: Arc<Mutex<GatewayState>>,
}

struct GatewayState {
    coordinator: TcpCoordinator,
    workers: Vec<WorkerId>,
    clock: MonotonicClock,
    buffered: BTreeMap<ClusterRequestId, VecDeque<ClientEvent>>,
}

/// Errors raised while connecting or serving the cluster gateway.
#[derive(Debug)]
pub enum ClusterGatewayError {
    /// The worker/coordinator transport failed.
    Cluster(TcpCoordinatorError),
    /// The HTTP body was malformed.
    BadRequest(String),
    /// A request could not be admitted or completed.
    Request(String),
    /// The worker session failed while serving a connection.
    Worker(WorkerSessionError),
}

impl std::fmt::Display for ClusterGatewayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cluster(error) => error.fmt(f),
            Self::BadRequest(message) | Self::Request(message) => f.write_str(message),
            Self::Worker(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ClusterGatewayError {}

impl From<TcpCoordinatorError> for ClusterGatewayError {
    fn from(error: TcpCoordinatorError) -> Self {
        Self::Cluster(error)
    }
}

impl From<WorkerSessionError> for ClusterGatewayError {
    fn from(error: WorkerSessionError) -> Self {
        Self::Worker(error)
    }
}

/// Serves one worker process endpoint using an existing configured executor.
///
/// The worker accepts one coordinator connection and runs until that
/// connection closes or `shutdown` resolves. A supervisor may call this
/// function again after a restart with a new epoch.
pub async fn serve_worker(
    listener: TcpListener,
    worker_id: impl Into<ola_core::WorkerId>,
    executor: ola_runtime::Executor,
    epoch: u64,
    heartbeat: std::time::Duration,
    shutdown: impl Future<Output = ()> + Send,
) -> Result<(), ClusterGatewayError> {
    let (peer, _) = ola_net::TcpPeer::accept(&listener)
        .await
        .map_err(ola_cluster::WorkerSessionError::from)?;
    let session = WorkerSession::new(peer, WorkerHost::new(worker_id, executor), epoch, heartbeat);
    tokio::select! {
        result = session.run() => { result.map(|_| ()).map_err(Into::into) }
        _ = shutdown => Ok(()),
    }
}

impl ClusterGateway {
    /// Connects to all configured workers and completes their handshakes.
    pub async fn connect(
        workers: impl IntoIterator<Item = (String, Option<WorkerId>)>,
    ) -> Result<Self, ClusterGatewayError> {
        let mut coordinator = TcpCoordinator::new(AdaptiveClusterPolicy::default());
        let mut ids = Vec::new();
        for (address, expected) in workers {
            ids.push(coordinator.connect_worker(address, expected).await?);
        }
        Ok(Self {
            state: Arc::new(Mutex::new(GatewayState {
                coordinator,
                workers: ids,
                clock: MonotonicClock::start(),
                buffered: BTreeMap::new(),
            })),
        })
    }

    /// Builds the gateway's native HTTP router.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/v1/generate", post(generate))
            .with_state(self.clone())
    }

    /// Serves the gateway until `shutdown` resolves.
    pub async fn serve(
        &self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown)
            .await
    }
}

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
    #[serde(default)]
    priority: i32,
    #[serde(default)]
    stream: bool,
}

const fn default_max_new_tokens() -> usize {
    128
}

#[derive(Debug, Serialize)]
struct GenerateResponse {
    id: u64,
    text: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

async fn health(State(gateway): State<ClusterGateway>) -> Json<serde_json::Value> {
    let state = gateway.state.lock().await;
    Json(serde_json::json!({"state":"ready", "workers":state.workers.len()}))
}

async fn ready(State(gateway): State<ClusterGateway>) -> (StatusCode, Json<serde_json::Value>) {
    let state = gateway.state.lock().await;
    let ready = !state.workers.is_empty();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(
            serde_json::json!({"state": if ready {"ready"} else {"not_ready"}, "workers":state.workers.len()}),
        ),
    )
}

async fn generate(
    State(gateway): State<ClusterGateway>,
    body: Result<Json<GenerateBody>, JsonRejection>,
) -> Result<Response, (StatusCode, Json<ErrorBody>)> {
    let Json(body) =
        body.map_err(|error| error_response(StatusCode::BAD_REQUEST, error.body_text()))?;
    let stream_response = body.stream;
    let spec =
        request_spec(body).map_err(|error| error_response(StatusCode::BAD_REQUEST, error))?;
    let (request, first) = {
        let mut state = gateway.state.lock().await;
        let now = state.clock.now();
        let (request, output) = state
            .coordinator
            .submit_and_plan(spec, now)
            .await
            .map_err(cluster_error)?;
        for (id, event) in output.client {
            state.buffered.entry(id).or_default().push_back(event);
        }
        (
            request,
            state
                .coordinator
                .status(request)
                .and_then(|status| status.owner().cloned())
                .ok_or_else(|| {
                    error_response(StatusCode::SERVICE_UNAVAILABLE, "request was not assigned")
                })?,
        )
    };
    let _ = first;
    if stream_response {
        let events = stream::unfold(Some((gateway, request)), |state| async move {
            let (gateway, request) = state?;
            let event = match next_event(&gateway, request).await {
                Ok(Some(event)) => event,
                Ok(None) | Err(_) => return None,
            };
            let terminal = matches!(
                event,
                ClientEvent::Finished { .. } | ClientEvent::Failed { .. } | ClientEvent::Cancelled
            );
            let output = match event {
                ClientEvent::Token { token_id, generated_tokens, .. } => Event::default().event("token").json_data(serde_json::json!({"token_id":token_id,"generated_tokens":generated_tokens})).unwrap(),
                ClientEvent::Finished { token_ids, text } => Event::default().event("done").json_data(GenerateResponse { id: request.get(), text, token_ids }).unwrap(),
                ClientEvent::Failed { message } => Event::default().event("error").json_data(ErrorBody { error: message }).unwrap(),
                ClientEvent::Cancelled => Event::default().event("cancelled"),
            };
            Some((
                Ok::<_, Infallible>(output),
                (!terminal).then_some((gateway, request)),
            ))
        });
        return Ok(Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    loop {
        match next_event(&gateway, request)
            .await
            .map_err(|error| error_response(StatusCode::SERVICE_UNAVAILABLE, error.to_string()))?
        {
            Some(ClientEvent::Token { .. }) => {}
            Some(ClientEvent::Finished { token_ids, text }) => {
                return Ok(Json(GenerateResponse {
                    id: request.get(),
                    text,
                    token_ids,
                })
                .into_response())
            }
            Some(ClientEvent::Failed { message }) => {
                return Err(error_response(StatusCode::INTERNAL_SERVER_ERROR, message))
            }
            Some(ClientEvent::Cancelled) => {
                return Err(error_response(StatusCode::CONFLICT, "cancelled"))
            }
            None => {
                return Err(error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "worker connection closed",
                ))
            }
        }
    }
}

async fn next_event(
    gateway: &ClusterGateway,
    request: ClusterRequestId,
) -> Result<Option<ClientEvent>, ClusterGatewayError> {
    loop {
        let mut state = gateway.state.lock().await;
        if let Some(event) = state
            .buffered
            .get_mut(&request)
            .and_then(VecDeque::pop_front)
        {
            return Ok(Some(event));
        }
        let owner = state
            .coordinator
            .status(request)
            .and_then(|status| status.owner().cloned());
        let Some(owner) = owner else {
            return Ok(None);
        };
        let now = state.clock.now();
        let output = state.coordinator.receive(&owner, now).await?;
        for (id, event) in output.client {
            state.buffered.entry(id).or_default().push_back(event);
        }
    }
}

fn request_spec(body: GenerateBody) -> Result<RequestSpec, String> {
    let input = match (body.prompt, body.tokens) {
        (Some(prompt), None) => Input::Text(prompt),
        (None, Some(tokens)) => Input::Tokens(tokens),
        _ => return Err("exactly one of `prompt` and `tokens` is required".into()),
    };
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
        deadline: None,
        attributes: BTreeMap::new(),
    })
}

fn error_response(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<ErrorBody>) {
    (
        status,
        Json(ErrorBody {
            error: message.into(),
        }),
    )
}

fn cluster_error(error: TcpCoordinatorError) -> (StatusCode, Json<ErrorBody>) {
    error_response(StatusCode::SERVICE_UNAVAILABLE, error.to_string())
}
