//! HTTP gateway backed by networked [`ola_cluster`] workers.
//!
//! One driver task owns the [`TcpCoordinator`]. It reads every worker link
//! continuously, so heartbeats renew leases and summaries reach routing even
//! when no request is waiting on that worker. On a fixed tick it expires
//! silent workers, recovers their requests, and replans requests deferred
//! while no worker was ready. HTTP handlers talk to the driver over
//! channels: a submission gets its own event stream, and nothing holds a lock
//! across network I/O.

use std::{collections::BTreeMap, convert::Infallible, future::Future, time::Duration};

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};
use futures_util::stream;
use ola_cluster::{
    AdaptiveClusterPolicy, ClientEvent, ClusterPolicy, CoordinatorError, CoordinatorOutput,
    TcpCoordinator, TcpCoordinatorError, WorkerHealth, WorkerHost, WorkerSession,
    WorkerSessionError,
};
use ola_core::{
    Clock, ClusterRequestId, GenerationConfig, Input, MonotonicClock, Priority, RequestSpec,
    WorkerId,
};
use serde::{Deserialize, Serialize};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
};

/// How the gateway's driver supervises workers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusterGatewayConfig {
    /// A worker silent for this long is lost: its requests are recovered on
    /// other workers by recompute. Keep it several worker heartbeats long.
    pub lease_timeout: Duration,
    /// How often the driver checks leases and replans deferred requests.
    pub tick: Duration,
}

impl Default for ClusterGatewayConfig {
    /// A 2 s lease checked every 50 ms, for workers heartbeating every 100 ms
    /// (the `ola serve --role worker` default).
    fn default() -> Self {
        Self {
            lease_timeout: Duration::from_secs(2),
            tick: Duration::from_millis(50),
        }
    }
}

/// Handle to a networked gateway. Clones, including routers, share one
/// driver; when the last is dropped the driver says goodbye to every worker
/// and stops.
#[derive(Clone)]
pub struct ClusterGateway {
    commands: mpsc::UnboundedSender<Command>,
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
/// connection closes or `shutdown` resolves. `epoch` must grow across
/// restarts of the same worker ID, so a coordinator can tell a restarted
/// process from a delayed heartbeat of the old one; a start timestamp works.
///
/// # Errors
///
/// Fails if accepting the connection or serving it fails.
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
    /// Connects to all configured workers with the default configuration
    /// and the adaptive routing policy.
    ///
    /// # Errors
    ///
    /// Fails if any worker cannot be reached or fails its handshake.
    pub async fn connect(
        workers: impl IntoIterator<Item = (String, Option<WorkerId>)>,
    ) -> Result<Self, ClusterGatewayError> {
        Self::connect_with(
            workers,
            AdaptiveClusterPolicy::default(),
            ClusterGatewayConfig::default(),
        )
        .await
    }

    /// Connects to all configured workers, routing with `policy`.
    ///
    /// # Errors
    ///
    /// Fails if any worker cannot be reached or fails its handshake.
    pub async fn connect_with<P: ClusterPolicy + 'static>(
        workers: impl IntoIterator<Item = (String, Option<WorkerId>)>,
        policy: P,
        config: ClusterGatewayConfig,
    ) -> Result<Self, ClusterGatewayError> {
        let mut coordinator = TcpCoordinator::new(policy);
        for (address, expected) in workers {
            coordinator.connect_worker(address, expected).await?;
        }
        let (commands, receiver) = mpsc::unbounded_channel();
        tokio::spawn(drive(coordinator, receiver, config));
        Ok(Self { commands })
    }

    /// Builds the gateway's native HTTP router.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/ready", get(ready))
            .route("/v1/generate", post(generate))
            .route("/v1/completions", post(completions))
            .route("/v1/requests/{id}", delete(cancel))
            .with_state(self.clone())
    }

    /// Serves the gateway until `shutdown` resolves.
    ///
    /// # Errors
    ///
    /// Fails if the HTTP server fails.
    pub async fn serve(
        &self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown)
            .await
    }

    async fn ask<T>(&self, command: impl FnOnce(oneshot::Sender<T>) -> Command) -> Option<T> {
        let (reply, response) = oneshot::channel();
        self.commands.send(command(reply)).ok()?;
        response.await.ok()
    }
}

impl crate::InferenceBackend for ClusterGateway {
    fn router(&self) -> Router {
        self.router()
    }
}

/// A request from an HTTP handler to the driver.
enum Command {
    Submit {
        spec: RequestSpec,
        reply: oneshot::Sender<Result<Subscription, SubmitError>>,
    },
    Cancel {
        request: ClusterRequestId,
        reply: oneshot::Sender<Result<(), CancelError>>,
    },
    Health {
        reply: oneshot::Sender<GatewayHealth>,
    },
}

/// A submitted request and the events the driver forwards for it.
struct Subscription {
    request: ClusterRequestId,
    events: mpsc::UnboundedReceiver<ClientEvent>,
}

enum SubmitError {
    Invalid(String),
    Unavailable(String),
}

enum CancelError {
    Unknown,
    Failed(String),
}

#[derive(Clone, Copy)]
struct GatewayHealth {
    connected: usize,
    ready: usize,
}

/// Owns the coordinator: applies worker frames, handler commands, and the
/// supervision tick, one at a time, and forwards client events.
async fn drive(
    mut coordinator: TcpCoordinator,
    mut commands: mpsc::UnboundedReceiver<Command>,
    config: ClusterGatewayConfig,
) {
    let clock = MonotonicClock::start();
    let mut subscribers = BTreeMap::<ClusterRequestId, mpsc::UnboundedSender<ClientEvent>>::new();
    let mut tick = tokio::time::interval(config.tick);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        // Every branch's future is cancel-safe; the work happens after one
        // is chosen.
        let output = tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break };
                let now = clock.now();
                match command {
                    Command::Submit { spec, reply } => {
                        match coordinator.submit_and_plan(spec, now).await {
                            Ok((request, output)) => {
                                let (sender, events) = mpsc::unbounded_channel();
                                subscribers.insert(request, sender);
                                // A handler that has gone away is cancelled
                                // when its first event cannot be delivered.
                                let _ = reply.send(Ok(Subscription { request, events }));
                                Ok(output)
                            }
                            Err(error) => {
                                let message = error.to_string();
                                let _ = reply.send(Err(match error {
                                    TcpCoordinatorError::Coordinator(
                                        CoordinatorError::InvalidRequest(_),
                                    ) => SubmitError::Invalid(message),
                                    _ => SubmitError::Unavailable(message),
                                }));
                                Ok(CoordinatorOutput::default())
                            }
                        }
                    }
                    Command::Cancel { request, reply } => {
                        let result = coordinator.cancel(request, now).await;
                        let answer = match &result {
                            Ok(_) => Ok(()),
                            Err(TcpCoordinatorError::Coordinator(
                                CoordinatorError::UnknownRequest(_),
                            )) => Err(CancelError::Unknown),
                            Err(error) => Err(CancelError::Failed(error.to_string())),
                        };
                        let _ = reply.send(answer);
                        result
                    }
                    Command::Health { reply } => {
                        let _ = reply.send(GatewayHealth {
                            connected: coordinator.connected().count(),
                            ready: coordinator
                                .coordinator()
                                .workers()
                                .filter(|summary| summary.health == WorkerHealth::Ready)
                                .count(),
                        });
                        Ok(CoordinatorOutput::default())
                    }
                }
            }
            frame = coordinator.next_frame() => coordinator.apply_frame(frame, clock.now()).await,
            _ = tick.tick() => coordinator.recover_lost(clock.now(), config.lease_timeout).await,
        };
        // A rejected worker message or failed dispatch affects that one
        // event; the driver keeps serving everything else.
        let Ok(output) = output else { continue };
        let mut abandoned = Vec::new();
        for (request, event) in output.client {
            let terminal = matches!(
                event,
                ClientEvent::Finished { .. } | ClientEvent::Failed { .. } | ClientEvent::Cancelled
            );
            let delivered = subscribers
                .get(&request)
                .is_some_and(|sender| sender.send(event).is_ok());
            if terminal {
                subscribers.remove(&request);
            } else if !delivered && subscribers.remove(&request).is_some() {
                abandoned.push(request);
            }
        }
        // Nobody is listening any more, so stop spending a worker on it.
        for request in abandoned {
            let _ = coordinator.cancel(request, clock.now()).await;
        }
    }
    let workers = coordinator.connected().cloned().collect::<Vec<_>>();
    for worker in workers {
        let _ = coordinator.close_worker(&worker).await;
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionBody {
    model: String,
    prompt: CompletionPrompt,
    #[serde(default = "default_completion_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
}

/// One prompt, as text or token IDs.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
}

const fn default_completion_tokens() -> usize {
    16
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: CompletionUsage,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    text: String,
    index: u8,
    finish_reason: &'static str,
}

/// `length` when generation stopped at the token limit, else `stop`.
const fn finish_reason(completion_tokens: usize, max_tokens: usize) -> &'static str {
    if completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

const fn usage(prompt_tokens: usize, completion_tokens: usize) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

#[derive(Debug, Serialize)]
struct CompletionUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

type HttpError = (StatusCode, Json<ErrorBody>);

async fn gateway_health(gateway: &ClusterGateway) -> Result<GatewayHealth, HttpError> {
    gateway
        .ask(|reply| Command::Health { reply })
        .await
        .ok_or_else(stopped)
}

async fn health(
    State(gateway): State<ClusterGateway>,
) -> Result<Json<serde_json::Value>, HttpError> {
    let health = gateway_health(&gateway).await?;
    Ok(Json(serde_json::json!({
        "state": "ready",
        "workers": health.connected,
        "ready_workers": health.ready,
    })))
}

async fn ready(
    State(gateway): State<ClusterGateway>,
) -> Result<(StatusCode, Json<serde_json::Value>), HttpError> {
    let health = gateway_health(&gateway).await?;
    let ready = health.ready > 0;
    Ok((
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(serde_json::json!({
            "state": if ready { "ready" } else { "not_ready" },
            "workers": health.connected,
            "ready_workers": health.ready,
        })),
    ))
}

async fn generate(
    State(gateway): State<ClusterGateway>,
    body: Result<Json<GenerateBody>, JsonRejection>,
) -> Result<Response, HttpError> {
    let Json(body) =
        body.map_err(|error| error_response(StatusCode::BAD_REQUEST, error.body_text()))?;
    let stream_response = body.stream;
    let spec =
        request_spec(body).map_err(|error| error_response(StatusCode::BAD_REQUEST, error))?;
    let Subscription {
        request,
        mut events,
    } = submit(&gateway, spec).await?;
    if stream_response {
        let events = stream::unfold(Some(events), move |events| async move {
            let mut events = events?;
            let event = events.recv().await?;
            let terminal = matches!(
                event,
                ClientEvent::Finished { .. } | ClientEvent::Failed { .. } | ClientEvent::Cancelled
            );
            let output = match event {
                ClientEvent::Token {
                    token_id,
                    text,
                    generated_tokens,
                    ..
                } => Event::default()
                    .event("token")
                    .json_data(serde_json::json!({
                        "token_id": token_id,
                        "text": text,
                        "generated_tokens": generated_tokens,
                    }))
                    .expect("token events serialize"),
                ClientEvent::Finished {
                    token_ids, text, ..
                } => Event::default()
                    .event("done")
                    .json_data(GenerateResponse {
                        id: request.get(),
                        text,
                        token_ids,
                    })
                    .expect("results serialize"),
                ClientEvent::Failed { message } => Event::default()
                    .event("error")
                    .json_data(ErrorBody { error: message })
                    .expect("errors serialize"),
                ClientEvent::Cancelled => Event::default().event("cancelled"),
            };
            Some((Ok::<_, Infallible>(output), (!terminal).then_some(events)))
        });
        return Ok(Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    loop {
        match events.recv().await {
            Some(ClientEvent::Token { .. }) => {}
            Some(ClientEvent::Finished {
                token_ids, text, ..
            }) => {
                return Ok(Json(GenerateResponse {
                    id: request.get(),
                    text,
                    token_ids,
                })
                .into_response());
            }
            Some(ClientEvent::Failed { message }) => {
                return Err(error_response(StatusCode::INTERNAL_SERVER_ERROR, message));
            }
            Some(ClientEvent::Cancelled) => {
                return Err(error_response(StatusCode::CONFLICT, "cancelled"));
            }
            None => return Err(stopped()),
        }
    }
}

async fn cancel(
    State(gateway): State<ClusterGateway>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Result<StatusCode, HttpError> {
    let request = ClusterRequestId::new(id);
    match gateway
        .ask(|reply| Command::Cancel { request, reply })
        .await
        .ok_or_else(stopped)?
    {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(CancelError::Unknown) => Err(error_response(
            StatusCode::NOT_FOUND,
            format!("unknown request `{id}`"),
        )),
        Err(CancelError::Failed(message)) => {
            Err(error_response(StatusCode::SERVICE_UNAVAILABLE, message))
        }
    }
}

/// OpenAI-style text completion. Streaming sends one chunk per non-empty
/// text delta (workers decode their tokens), then a final chunk with
/// `finish_reason` and `usage`, then `[DONE]`.
async fn completions(
    State(gateway): State<ClusterGateway>,
    body: Result<Json<CompletionBody>, JsonRejection>,
) -> Result<Response, HttpError> {
    let Json(body) =
        body.map_err(|error| error_response(StatusCode::BAD_REQUEST, error.body_text()))?;
    let CompletionBody {
        model,
        prompt,
        max_tokens,
        stream: stream_response,
    } = body;
    let spec = RequestSpec {
        model: model.clone().into(),
        inputs: vec![match prompt {
            CompletionPrompt::Text(text) => Input::Text(text),
            CompletionPrompt::Tokens(tokens) => Input::Tokens(tokens),
        }],
        generation: GenerationConfig {
            max_new_tokens: max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        },
        priority: Priority::NORMAL,
        deadline: None,
        attributes: BTreeMap::new(),
    };
    let Subscription {
        request,
        mut events,
    } = submit(&gateway, spec).await?;
    let id = format!("cmpl-{}", request.get());
    if stream_response {
        let events = stream::unfold(Some((events, id, model, false)), move |state| async move {
            let (mut events, id, model, done) = state?;
            if done {
                return Some((Ok::<_, Infallible>(Event::default().data("[DONE]")), None));
            }
            let chunk =
                |text: &str, finish_reason: Option<&str>, usage: Option<CompletionUsage>| {
                    let mut chunk = serde_json::json!({
                        "id": id,
                        "object": "text_completion",
                        "created": 0,
                        "model": model,
                        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
                    });
                    if let Some(usage) = usage {
                        chunk["usage"] = serde_json::to_value(usage).expect("usage serializes");
                    }
                    Event::default().data(chunk.to_string())
                };
            loop {
                match events.recv().await? {
                    // One chunk per non-empty text delta.
                    ClientEvent::Token { text, .. } if !text.is_empty() => {
                        let event = chunk(&text, None, None);
                        return Some((Ok(event), Some((events, id, model, false))));
                    }
                    ClientEvent::Token { .. } => {}
                    ClientEvent::Finished {
                        token_ids,
                        prompt_tokens,
                        ..
                    } => {
                        let event = chunk(
                            "",
                            Some(finish_reason(token_ids.len(), max_tokens)),
                            Some(usage(prompt_tokens, token_ids.len())),
                        );
                        return Some((Ok(event), Some((events, id, model, true))));
                    }
                    ClientEvent::Failed { message } => {
                        let error = serde_json::json!({"error": message});
                        return Some((Ok(Event::default().data(error.to_string())), None));
                    }
                    ClientEvent::Cancelled => return None,
                }
            }
        });
        return Ok(Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    loop {
        match events.recv().await {
            Some(ClientEvent::Token { .. }) => {}
            Some(ClientEvent::Finished {
                token_ids,
                text,
                prompt_tokens,
            }) => {
                return Ok(Json(CompletionResponse {
                    id,
                    object: "text_completion",
                    created: 0,
                    model,
                    choices: vec![CompletionChoice {
                        text,
                        index: 0,
                        finish_reason: finish_reason(token_ids.len(), max_tokens),
                    }],
                    usage: usage(prompt_tokens, token_ids.len()),
                })
                .into_response());
            }
            Some(ClientEvent::Failed { message }) => {
                return Err(error_response(StatusCode::INTERNAL_SERVER_ERROR, message));
            }
            Some(ClientEvent::Cancelled) => {
                return Err(error_response(StatusCode::CONFLICT, "cancelled"));
            }
            None => return Err(stopped()),
        }
    }
}

/// Submits a request. It may wait in the coordinator until a worker is
/// ready; a model no worker serves fails through the event stream.
async fn submit(gateway: &ClusterGateway, spec: RequestSpec) -> Result<Subscription, HttpError> {
    gateway
        .ask(|reply| Command::Submit { spec, reply })
        .await
        .ok_or_else(stopped)?
        .map_err(|error| match error {
            SubmitError::Invalid(message) => error_response(StatusCode::BAD_REQUEST, message),
            SubmitError::Unavailable(message) => {
                error_response(StatusCode::SERVICE_UNAVAILABLE, message)
            }
        })
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

fn error_response(status: StatusCode, message: impl Into<String>) -> HttpError {
    (
        status,
        Json(ErrorBody {
            error: message.into(),
        }),
    )
}

fn stopped() -> HttpError {
    error_response(StatusCode::SERVICE_UNAVAILABLE, "gateway driver stopped")
}
