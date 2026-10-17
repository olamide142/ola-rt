//! A dedicated thread that owns an [`Executor`] and serves it over channels.

use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    io::{self, Write},
    ops::ControlFlow,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, RecvTimeoutError, TryRecvError},
    },
    task::{Context, Poll},
    thread::{self, JoinHandle},
    time::Duration,
};

use ola_core::{Clock, ModelDescriptor, ModelId, RequestId, RequestSpec, Timestamp};
use ola_model::{IncrementalDecoder, KvCacheProfile, SessionMetrics};
use ola_observe::{TraceRecord, Tracer, trace::write_jsonl};
use ola_runtime::{
    ExecutionPhase, Executor, ExecutorError, ExecutorEvent, RequestStatus, RequestTiming,
};
use tokio::sync::{mpsc as stream, oneshot};

use crate::ServeError;

/// Limits and pacing for a [`Driver`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DriverConfig {
    /// Most requests that may be submitted but unfinished at once. Further
    /// submissions fail immediately with [`ServeError::Overloaded`].
    pub max_in_flight: usize,
    /// How long the driver waits for a command after an iteration that made
    /// no progress, such as when every queued request is deferred.
    pub stall_backoff: Duration,
    /// Most finished requests whose final status stays queryable. Older
    /// ones are forgotten, oldest first, and look unknown afterwards.
    pub retain_finished: usize,
    /// Keep each request's trace for [`RuntimeHandle::trace`], for live
    /// requests and as long as a finished one's status is retained.
    pub trace: bool,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 256,
            stall_backoff: Duration::from_millis(1),
            retain_finished: 1024,
            trace: false,
        }
    }
}

/// Final output of a request that completed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationOutput {
    /// Request that produced this output.
    pub request_id: RequestId,
    /// Generated token IDs in emission order.
    pub token_ids: Vec<u32>,
    /// Decoded text of `token_ids`.
    pub text: String,
    /// Model-session measurements captured before cache release.
    pub metrics: SessionMetrics,
    /// Lifecycle timestamps in the driver clock's epoch.
    pub timing: RequestTiming,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// When each token in `token_ids` was produced, in the driver clock's
    /// epoch.
    pub token_times: Vec<Timestamp>,
}

/// Why a completed request stopped generating.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// The model emitted its end-of-sequence token.
    EndOfSequence,
    /// The request reached `max_new_tokens`.
    Length,
}

/// Something that happened to one request.
///
/// A stream carries any number of [`Token`](Self::Token) events followed by
/// exactly one terminal event.
#[derive(Debug, PartialEq)]
pub enum RequestEvent {
    /// One token was generated.
    Token {
        /// Selected token ID.
        token_id: u32,
        /// Text completed by this token. It may be empty when a character
        /// spans several tokens; concatenated deltas equal the final text.
        text: String,
        /// Model phase that produced the token.
        phase: ExecutionPhase,
        /// Output tokens generated so far, including this one.
        generated_tokens: usize,
    },
    /// The request completed.
    Finished(GenerationOutput),
    /// The request failed or was ended by shutdown.
    Failed(ServeError),
    /// The request was cancelled.
    Cancelled,
}

impl RequestEvent {
    /// Returns whether no further events follow this one.
    pub const fn is_terminal(&self) -> bool {
        !matches!(self, Self::Token { .. })
    }
}

/// Point-in-time view of one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSnapshot {
    /// Request identifier.
    pub id: RequestId,
    /// Current lifecycle state.
    pub status: RequestStatus,
    /// Lifecycle timestamps in the driver clock's epoch.
    pub timing: RequestTiming,
    /// Output tokens generated so far.
    pub generated_tokens: usize,
}

/// Whether the driver is taking work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceState {
    /// Accepting and executing requests.
    Ready,
    /// Running, but no model has an executable attached, so every request
    /// would fail.
    NoModels,
    /// Finishing accepted requests before stopping; new work is refused.
    Draining,
    /// The driver thread has stopped or is about to.
    Stopped,
}

impl ServiceState {
    const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Ready,
            1 => Self::NoModels,
            2 => Self::Draining,
            _ => Self::Stopped,
        }
    }

    const fn as_u8(self) -> u8 {
        match self {
            Self::Ready => 0,
            Self::NoModels => 1,
            Self::Draining => 2,
            Self::Stopped => 3,
        }
    }
}

/// Liveness and load, read without waiting for the driver thread.
///
/// Gauges are published after every command batch and iteration, so they
/// stay readable, if slightly stale, while a long iteration runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Health {
    /// Whether the driver is taking work.
    pub state: ServiceState,
    /// Submitted requests that have not finished.
    pub in_flight: usize,
    /// Configured limit on `in_flight`.
    pub max_in_flight: usize,
    /// Requests waiting for admission.
    pub queued: usize,
    /// Requests with a live generation session.
    pub active: usize,
    /// Physical KV bytes held by live sessions.
    pub kv_cache_bytes: u64,
    /// Worst-case KV bytes committed to live requests.
    pub reserved_kv_bytes: u64,
    /// Executor iterations completed since the driver started.
    pub iterations: u64,
    /// Whether writing to the trace sink failed; the driver then stops
    /// writing to it, and requests are unaffected.
    pub trace_write_failed: bool,
}

impl Health {
    /// Returns whether new submissions would be accepted, capacity
    /// permitting.
    pub fn is_ready(&self) -> bool {
        self.state == ServiceState::Ready
    }
}

/// A registered model and whether it can execute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelStatus {
    /// Registered metadata.
    pub descriptor: ModelDescriptor,
    /// Whether an executable model is attached; requests for a model
    /// without one fail.
    pub executable: bool,
    /// Physical KV-pool geometry, when the architecture publishes one.
    pub kv_cache_profile: Option<KvCacheProfile>,
}

type Responder<T> = oneshot::Sender<Result<T, ServeError>>;

enum Command {
    Submit {
        spec: RequestSpec,
        arrival_at: Timestamp,
        events: stream::UnboundedSender<RequestEvent>,
        reply: Responder<RequestId>,
    },
    Cancel {
        id: RequestId,
        reply: Option<Responder<()>>,
    },
    Detach {
        id: RequestId,
    },
    Status {
        id: RequestId,
        reply: Responder<RequestSnapshot>,
    },
    Trace {
        id: RequestId,
        reply: Responder<Vec<TraceRecord>>,
    },
    Stop {
        /// Finish accepted work until this time, or stop at once when `None`.
        drain_until: Option<Timestamp>,
    },
}

struct Shared {
    clock: Arc<dyn Clock>,
    max_in_flight: usize,
    in_flight: AtomicUsize,
    state: AtomicU8,
    queued: AtomicUsize,
    active: AtomicUsize,
    kv_cache_bytes: AtomicU64,
    reserved_kv_bytes: AtomicU64,
    iterations: AtomicU64,
    trace_write_failed: AtomicBool,
    models: Vec<ModelStatus>,
}

impl Shared {
    fn release(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }

    fn state(&self) -> ServiceState {
        ServiceState::from_u8(self.state.load(Ordering::SeqCst))
    }

    fn set_state(&self, state: ServiceState) {
        self.state.store(state.as_u8(), Ordering::SeqCst);
    }
}

/// Owns the thread that drives an [`Executor`].
///
/// Dropping the driver shuts it down as [`shutdown`](Self::shutdown) does.
pub struct Driver {
    handle: RuntimeHandle,
    thread: Option<JoinHandle<Executor>>,
}

impl Driver {
    /// Starts a thread that owns `executor`, reading time from `clock`.
    ///
    /// The clock is also installed on the executor, so tokens and lifecycle
    /// events are stamped when the model invocation producing them finishes.
    /// Models must be registered on the executor before it is handed over.
    pub fn spawn(
        executor: Executor,
        clock: impl Clock + 'static,
        config: DriverConfig,
    ) -> io::Result<Self> {
        Self::start(executor, Arc::new(clock), config, None)
    }

    /// Like [`spawn`](Self::spawn), and also writes every request's trace to
    /// `sink` as JSON Lines as it happens, flushing after each iteration.
    pub fn spawn_with_trace_sink(
        executor: Executor,
        clock: impl Clock + 'static,
        config: DriverConfig,
        sink: impl Write + Send + 'static,
    ) -> io::Result<Self> {
        Self::start(executor, Arc::new(clock), config, Some(Box::new(sink)))
    }

    fn start(
        executor: Executor,
        clock: Arc<dyn Clock>,
        config: DriverConfig,
        sink: Option<Box<dyn Write + Send>>,
    ) -> io::Result<Self> {
        let executor = executor.with_clock(Arc::clone(&clock));
        let (commands, receiver) = mpsc::channel();
        let models = executor
            .state()
            .models()
            .iter()
            .map(|descriptor| {
                let executable = executor.model(&descriptor.id);
                ModelStatus {
                    descriptor: descriptor.clone(),
                    executable: executable.is_some(),
                    kv_cache_profile: executable.and_then(|model| model.kv_cache_profile()),
                }
            })
            .collect::<Vec<_>>();
        let state = if models.iter().any(|model| model.executable) {
            ServiceState::Ready
        } else {
            ServiceState::NoModels
        };
        let shared = Arc::new(Shared {
            clock,
            max_in_flight: config.max_in_flight,
            in_flight: AtomicUsize::new(0),
            state: AtomicU8::new(state.as_u8()),
            queued: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            kv_cache_bytes: AtomicU64::new(0),
            reserved_kv_bytes: AtomicU64::new(0),
            iterations: AtomicU64::new(0),
            trace_write_failed: AtomicBool::new(false),
            models,
        });
        let worker = Worker {
            executor,
            commands: receiver,
            shared: Arc::clone(&shared),
            live: BTreeMap::new(),
            finished: BTreeMap::new(),
            finished_order: VecDeque::new(),
            retain_finished: config.retain_finished,
            stall_backoff: config.stall_backoff,
            drain_until: None,
            tracer: (config.trace || sink.is_some()).then(Tracer::default),
            traces: config.trace.then(BTreeMap::new),
            sink,
        };
        worker.publish();
        let thread = thread::Builder::new()
            .name("ola-driver".into())
            .spawn(move || worker.run())?;
        Ok(Self {
            handle: RuntimeHandle { commands, shared },
            thread: Some(thread),
        })
    }

    /// Returns a cloneable handle for submitting and controlling requests.
    pub fn handle(&self) -> RuntimeHandle {
        self.handle.clone()
    }

    /// Stops accepting work, ends every unfinished request with
    /// [`ServeError::ShuttingDown`], and returns the executor.
    ///
    /// An iteration already running completes first.
    pub fn shutdown(mut self) -> Result<Executor, ServeError> {
        self.stop(None)
    }

    /// Stops accepting work and lets accepted requests finish for up to
    /// `grace` on the driver's clock, then ends the rest with
    /// [`ServeError::ShuttingDown`] and returns the executor.
    pub fn drain(mut self, grace: Duration) -> Result<Executor, ServeError> {
        let now = self.handle.now();
        let until = now
            .checked_add(grace)
            .unwrap_or(Timestamp::from_micros(u64::MAX));
        self.stop(Some(until))
    }

    fn stop(&mut self, drain_until: Option<Timestamp>) -> Result<Executor, ServeError> {
        let thread = self.thread.take().ok_or(ServeError::ShuttingDown)?;
        // The command goes first so that once the state reads as not ready
        // the driver is guaranteed to see the stop at its next command check.
        let _ = self.handle.commands.send(Command::Stop { drain_until });
        if self.handle.shared.state() != ServiceState::Stopped {
            self.handle.shared.set_state(if drain_until.is_some() {
                ServiceState::Draining
            } else {
                ServiceState::Stopped
            });
        }
        thread
            .join()
            .map_err(|_| ServeError::Internal("the driver thread panicked".into()))
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        if self.thread.is_some() {
            let _ = self.stop(None);
        }
    }
}

/// Cloneable client of a [`Driver`].
#[derive(Clone)]
pub struct RuntimeHandle {
    commands: mpsc::Sender<Command>,
    shared: Arc<Shared>,
}

impl RuntimeHandle {
    /// Submits a request, stamping its arrival with the driver's clock.
    ///
    /// Capacity is checked here, so [`ServeError::Overloaded`] and
    /// [`ServeError::ShuttingDown`] resolve without waiting for the driver.
    /// Dropping the returned [`Submission`] unresolved cancels the request.
    pub fn submit(&self, spec: RequestSpec) -> Submission {
        let arrival_at = self.shared.clock.now();
        let rejected = |error| Submission {
            reply: Reply::ready(Err(error)),
            events: None,
            runtime: self.clone(),
        };
        match self.shared.state() {
            ServiceState::Ready => {}
            ServiceState::NoModels => return rejected(ServeError::NotReady),
            ServiceState::Draining | ServiceState::Stopped => {
                return rejected(ServeError::ShuttingDown);
            }
        }
        let limit = self.shared.max_in_flight;
        if self
            .shared
            .in_flight
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                (count < limit).then_some(count + 1)
            })
            .is_err()
        {
            return rejected(ServeError::Overloaded { limit });
        }
        let (events, receiver) = stream::unbounded_channel();
        let (reply, response) = oneshot::channel();
        let command = Command::Submit {
            spec,
            arrival_at,
            events,
            reply,
        };
        if self.commands.send(command).is_err() {
            self.shared.release();
            return rejected(ServeError::ShuttingDown);
        }
        Submission {
            reply: Reply::waiting(response),
            events: Some(receiver),
            runtime: self.clone(),
        }
    }

    /// Cancels a request. Cancelling a request that already ended succeeds.
    pub fn cancel(&self, id: RequestId) -> Reply<()> {
        self.request(|reply| Command::Cancel {
            id,
            reply: Some(reply),
        })
    }

    /// Returns a request's current state.
    pub fn status(&self, id: RequestId) -> Reply<RequestSnapshot> {
        self.request(|reply| Command::Status { id, reply })
    }

    /// Returns whether the driver accepts new submissions.
    pub fn is_accepting(&self) -> bool {
        self.shared.state() == ServiceState::Ready
    }

    /// Returns liveness and load without waiting for the driver thread.
    pub fn health(&self) -> Health {
        let shared = &self.shared;
        Health {
            state: shared.state(),
            in_flight: shared.in_flight.load(Ordering::SeqCst),
            max_in_flight: shared.max_in_flight,
            queued: shared.queued.load(Ordering::SeqCst),
            active: shared.active.load(Ordering::SeqCst),
            kv_cache_bytes: shared.kv_cache_bytes.load(Ordering::SeqCst),
            reserved_kv_bytes: shared.reserved_kv_bytes.load(Ordering::SeqCst),
            iterations: shared.iterations.load(Ordering::SeqCst),
            trace_write_failed: shared.trace_write_failed.load(Ordering::SeqCst),
        }
    }

    /// Returns a live or retained request's trace so far.
    ///
    /// Fails with [`ServeError::TracingDisabled`] unless the driver was
    /// configured with [`DriverConfig::trace`].
    pub fn trace(&self, id: RequestId) -> Reply<Vec<TraceRecord>> {
        self.request(|reply| Command::Trace { id, reply })
    }

    /// Returns every registered model in stable identifier order.
    pub fn models(&self) -> &[ModelStatus] {
        &self.shared.models
    }

    /// Returns the number of submitted requests that have not finished.
    pub fn in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::SeqCst)
    }

    /// Returns the current time on the driver's clock.
    pub fn now(&self) -> Timestamp {
        self.shared.clock.now()
    }

    fn request<T>(&self, command: impl FnOnce(Responder<T>) -> Command) -> Reply<T> {
        let (reply, response) = oneshot::channel();
        if self.commands.send(command(reply)).is_err() {
            return Reply::ready(Err(ServeError::ShuttingDown));
        }
        Reply::waiting(response)
    }
}

/// The driver's answer to one command.
///
/// Await it from async code or call [`wait`](Self::wait) from a plain
/// thread; `wait` panics inside an async runtime, as blocking there would.
pub struct Reply<T>(ReplyState<T>);

enum ReplyState<T> {
    Ready(Option<Result<T, ServeError>>),
    Waiting(oneshot::Receiver<Result<T, ServeError>>),
}

impl<T> Reply<T> {
    const fn ready(result: Result<T, ServeError>) -> Self {
        Self(ReplyState::Ready(Some(result)))
    }

    const fn waiting(receiver: oneshot::Receiver<Result<T, ServeError>>) -> Self {
        Self(ReplyState::Waiting(receiver))
    }

    /// Blocks the current thread until the driver answers.
    pub fn wait(self) -> Result<T, ServeError> {
        match self.0 {
            ReplyState::Ready(result) => result.expect("a reply resolves once"),
            ReplyState::Waiting(receiver) => receiver
                .blocking_recv()
                .unwrap_or(Err(ServeError::ShuttingDown)),
        }
    }
}

// The reply never pins `T` structurally; it only moves results out.
impl<T> Unpin for Reply<T> {}

impl<T> Future for Reply<T> {
    type Output = Result<T, ServeError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.get_mut().0 {
            ReplyState::Ready(result) => Poll::Ready(result.take().expect("a reply resolves once")),
            ReplyState::Waiting(receiver) => Pin::new(receiver)
                .poll(context)
                .map(|result| result.unwrap_or(Err(ServeError::ShuttingDown))),
        }
    }
}

/// A pending submission that resolves to a [`RequestHandle`].
pub struct Submission {
    reply: Reply<RequestId>,
    events: Option<stream::UnboundedReceiver<RequestEvent>>,
    runtime: RuntimeHandle,
}

impl Submission {
    /// Blocks the current thread until the driver accepts or refuses the
    /// request.
    pub fn wait(mut self) -> Result<RequestHandle, ServeError> {
        let events = self.events.take();
        let id = self.reply.wait()?;
        Ok(RequestHandle::new(id, events, self.runtime))
    }
}

impl Future for Submission {
    type Output = Result<RequestHandle, ServeError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        Pin::new(&mut this.reply).poll(context).map(|result| {
            let id = result?;
            Ok(RequestHandle::new(
                id,
                this.events.take(),
                this.runtime.clone(),
            ))
        })
    }
}

/// A submitted request: its event stream and controls.
///
/// Dropping the handle before the request ends cancels it, so a client
/// disconnect releases the request's session and KV cache. Call
/// [`detach`](Self::detach) to let it run without a listener.
pub struct RequestHandle {
    id: RequestId,
    events: stream::UnboundedReceiver<RequestEvent>,
    runtime: RuntimeHandle,
    ended: bool,
}

impl RequestHandle {
    fn new(
        id: RequestId,
        events: Option<stream::UnboundedReceiver<RequestEvent>>,
        runtime: RuntimeHandle,
    ) -> Self {
        Self {
            id,
            events: events.expect("an accepted submission carries its event stream"),
            runtime,
            ended: false,
        }
    }

    /// Returns the request's identifier.
    pub const fn id(&self) -> RequestId {
        self.id
    }

    /// Waits for the next event; `None` after the terminal event.
    pub async fn next(&mut self) -> Option<RequestEvent> {
        if self.ended {
            return None;
        }
        let event = self.events.recv().await;
        Some(self.observe(event))
    }

    /// Blocks for the next event; `None` after the terminal event.
    pub fn blocking_next(&mut self) -> Option<RequestEvent> {
        if self.ended {
            return None;
        }
        let event = self.events.blocking_recv();
        Some(self.observe(event))
    }

    /// Waits for the request to end, discarding token events.
    pub async fn result(mut self) -> Result<GenerationOutput, ServeError> {
        loop {
            if let Some(result) = Self::outcome(self.next().await) {
                return result;
            }
        }
    }

    /// Blocks until the request ends, discarding token events.
    pub fn blocking_result(mut self) -> Result<GenerationOutput, ServeError> {
        loop {
            if let Some(result) = Self::outcome(self.blocking_next()) {
                return result;
            }
        }
    }

    /// Asks the driver to cancel the request.
    pub fn cancel(&self) -> Reply<()> {
        self.runtime.cancel(self.id)
    }

    /// Returns the request's current state.
    pub fn status(&self) -> Reply<RequestSnapshot> {
        self.runtime.status(self.id)
    }

    /// Lets the request run to completion without a listener and returns
    /// its identifier for later status lookups.
    pub fn detach(mut self) -> RequestId {
        if !self.ended {
            let _ = self.runtime.commands.send(Command::Detach { id: self.id });
            self.ended = true;
        }
        self.id
    }

    fn observe(&mut self, event: Option<RequestEvent>) -> RequestEvent {
        // A closed stream without a terminal event means the driver is gone.
        let event = event.unwrap_or(RequestEvent::Failed(ServeError::ShuttingDown));
        self.ended = event.is_terminal();
        event
    }

    fn outcome(event: Option<RequestEvent>) -> Option<Result<GenerationOutput, ServeError>> {
        match event? {
            RequestEvent::Token { .. } => None,
            RequestEvent::Finished(output) => Some(Ok(output)),
            RequestEvent::Failed(error) => Some(Err(error)),
            RequestEvent::Cancelled => Some(Err(ServeError::Cancelled)),
        }
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if !self.ended {
            let _ = self.runtime.commands.send(Command::Cancel {
                id: self.id,
                reply: None,
            });
        }
    }
}

/// Per-request state the driver keeps while a request is unfinished.
struct Live {
    model: ModelId,
    events: stream::UnboundedSender<RequestEvent>,
    decoder: IncrementalDecoder,
    detached: bool,
}

struct Worker {
    executor: Executor,
    commands: mpsc::Receiver<Command>,
    shared: Arc<Shared>,
    live: BTreeMap<RequestId, Live>,
    /// Final status of recently finished requests, bounded by
    /// `retain_finished` and evicted in `finished_order`.
    finished: BTreeMap<RequestId, RequestSnapshot>,
    finished_order: VecDeque<RequestId>,
    retain_finished: usize,
    stall_backoff: Duration,
    /// Set while draining: accepted work may run until this time.
    drain_until: Option<Timestamp>,
    /// Present when traces are kept or written.
    tracer: Option<Tracer>,
    /// Kept traces of live and retained requests, when configured.
    traces: Option<BTreeMap<RequestId, Vec<TraceRecord>>>,
    sink: Option<Box<dyn Write + Send>>,
}

/// Marks the service stopped however the driver thread exits, including by
/// panic, so health never reports a dead driver as ready.
struct StoppedOnExit(Arc<Shared>);

impl Drop for StoppedOnExit {
    fn drop(&mut self) {
        self.0.set_state(ServiceState::Stopped);
    }
}

impl Worker {
    fn run(mut self) -> Executor {
        let _stopped = StoppedOnExit(Arc::clone(&self.shared));
        while self.serve().is_continue() {}
        self.abort_all(|| RequestEvent::Failed(ServeError::ShuttingDown));
        self.publish();
        self.executor
    }

    /// Runs one round: wait for work if there is none, apply commands, reap
    /// abandoned requests, and advance one iteration.
    fn serve(&mut self) -> ControlFlow<()> {
        if let Some(until) = self.drain_until
            && (!self.has_work() || self.shared.clock.now() >= until)
        {
            return ControlFlow::Break(());
        }
        if !self.has_work() {
            match self.commands.recv() {
                Ok(command) => self.apply(command)?,
                Err(_) => return ControlFlow::Break(()),
            }
        }
        // Requests whose listeners are gone are found before draining
        // commands, so a `Detach` sent just before its handle was dropped
        // is applied before the request would be cancelled.
        let abandoned = self
            .live
            .iter()
            .filter(|(_, live)| !live.detached && live.events.is_closed())
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        let applied = self.apply_pending();
        self.publish();
        applied?;
        for id in abandoned {
            if self.live.get(&id).is_some_and(|live| !live.detached) {
                let _ = self.cancel(id);
            }
        }
        if self.has_work() && !self.step() {
            match self.commands.recv_timeout(self.stall_backoff) {
                Ok(command) => self.apply(command)?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return ControlFlow::Break(()),
            }
        }
        self.publish();
        ControlFlow::Continue(())
    }

    /// Publishes load gauges for [`RuntimeHandle::health`].
    fn publish(&self) {
        let shared = &self.shared;
        let state = self.executor.state();
        shared.queued.store(state.queued_len(), Ordering::SeqCst);
        shared
            .active
            .store(self.executor.active_request_count(), Ordering::SeqCst);
        shared
            .kv_cache_bytes
            .store(state.memory.kv_cache_bytes, Ordering::SeqCst);
        shared
            .reserved_kv_bytes
            .store(self.executor.reserved_kv_bytes(), Ordering::SeqCst);
    }

    fn apply_pending(&mut self) -> ControlFlow<()> {
        loop {
            match self.commands.try_recv() {
                Ok(command) => self.apply(command)?,
                Err(TryRecvError::Empty) => return ControlFlow::Continue(()),
                Err(TryRecvError::Disconnected) => return ControlFlow::Break(()),
            }
        }
    }

    fn has_work(&self) -> bool {
        self.executor.state().queued_len() > 0 || self.executor.has_active_request()
    }

    fn apply(&mut self, command: Command) -> ControlFlow<()> {
        match command {
            Command::Submit {
                spec,
                arrival_at,
                events,
                reply,
            } => {
                if self.drain_until.is_some() {
                    self.shared.release();
                    let _ = reply.send(Err(ServeError::ShuttingDown));
                    return ControlFlow::Continue(());
                }
                let model = spec.model.clone();
                match self.executor.submit(spec, arrival_at) {
                    Ok(id) => {
                        self.live.insert(
                            id,
                            Live {
                                model,
                                events,
                                decoder: IncrementalDecoder::new(),
                                detached: false,
                            },
                        );
                        let _ = reply.send(Ok(id));
                    }
                    Err(error) => {
                        self.shared.release();
                        let error = match error {
                            ExecutorError::Runtime(error) => ServeError::from_submission(error),
                            error => ServeError::Internal(error.to_string()),
                        };
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Cancel { id, reply } => {
                let result = self.cancel(id);
                if let Some(reply) = reply {
                    let _ = reply.send(result);
                }
            }
            Command::Detach { id } => {
                if let Some(live) = self.live.get_mut(&id) {
                    live.detached = true;
                }
            }
            Command::Status { id, reply } => {
                let _ = reply.send(self.snapshot(id));
            }
            Command::Trace { id, reply } => {
                let trace = match &self.traces {
                    None => Err(ServeError::TracingDisabled),
                    Some(traces) => traces
                        .get(&id)
                        .cloned()
                        .ok_or(ServeError::UnknownRequest(id)),
                };
                let _ = reply.send(trace);
            }
            Command::Stop { drain_until: None } => return ControlFlow::Break(()),
            Command::Stop {
                drain_until: Some(until),
            } => self.drain_until = Some(until),
        }
        ControlFlow::Continue(())
    }

    fn cancel(&mut self, id: RequestId) -> Result<(), ServeError> {
        if !self.live.contains_key(&id) {
            return if self.finished.contains_key(&id) {
                Ok(())
            } else {
                Err(ServeError::UnknownRequest(id))
            };
        }
        match self.executor.cancel(id, self.shared.clock.now()) {
            Ok(()) => {
                self.trace_cancellation(id);
                self.end(id, RequestEvent::Cancelled);
                Ok(())
            }
            Err(error) => Err(ServeError::Internal(error.to_string())),
        }
    }

    /// Advances one iteration and returns whether any request progressed.
    fn step(&mut self) -> bool {
        let iteration = match self.executor.step_iteration(self.shared.clock.now()) {
            Ok(iteration) => {
                self.shared.iterations.fetch_add(1, Ordering::SeqCst);
                // Traced before events are routed, while every terminal
                // request's runtime record still exists.
                if let Some(tracer) = &mut self.tracer {
                    let records = tracer.observe_iteration(&iteration, self.executor.state());
                    self.record(records);
                }
                iteration
            }
            Err(error) => {
                // Planning failed before any work was committed. Retrying
                // would fail the same way, so end every request rather than
                // spin.
                let message = error.to_string();
                self.abort_all(|| RequestEvent::Failed(ServeError::Internal(message.clone())));
                return false;
            }
        };
        let mut progressed = false;
        for event in iteration.events {
            match event {
                ExecutorEvent::Token {
                    request_id,
                    token,
                    generated_tokens,
                    phase,
                    finished,
                    ..
                } => {
                    progressed = true;
                    let finished = finished.then_some(if token.is_end_of_sequence {
                        FinishReason::EndOfSequence
                    } else {
                        FinishReason::Length
                    });
                    self.deliver_token(request_id, token.id, generated_tokens, phase, finished);
                }
                ExecutorEvent::PrefillProgress { .. } => progressed = true,
                ExecutorEvent::Failed {
                    request_id,
                    message,
                } => {
                    progressed = true;
                    self.end(
                        request_id,
                        RequestEvent::Failed(ServeError::Execution(message)),
                    );
                }
                ExecutorEvent::Rejected { request_id, reason } => {
                    progressed = true;
                    self.end(
                        request_id,
                        RequestEvent::Failed(ServeError::Rejected(reason)),
                    );
                }
                ExecutorEvent::Idle
                | ExecutorEvent::Deferred { .. }
                | ExecutorEvent::BatchDeferred { .. } => {}
            }
        }
        progressed
    }

    fn deliver_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        generated_tokens: usize,
        phase: ExecutionPhase,
        finished: Option<FinishReason>,
    ) {
        let Some(live) = self.live.get_mut(&id) else {
            return;
        };
        // Text deltas are best effort: a decoding error leaves this delta
        // empty, and the final output still carries the executor's text.
        let mut text = String::new();
        if let Some(tokenizer) = self
            .executor
            .model(&live.model)
            .map(|model| model.tokenizer())
        {
            text = live.decoder.push(tokenizer, token_id).unwrap_or_default();
            if finished.is_some() {
                text.push_str(&live.decoder.finish(tokenizer).unwrap_or_default());
            }
        }
        let _ = live.events.send(RequestEvent::Token {
            token_id,
            text,
            phase,
            generated_tokens,
        });
        let Some(finish_reason) = finished else {
            return;
        };
        let timing = self
            .executor
            .state()
            .request(id)
            .map(|record| record.timing)
            .unwrap_or_default();
        let terminal = match self.executor.take_result(id) {
            Some(result) => RequestEvent::Finished(GenerationOutput {
                request_id: id,
                token_ids: result.token_ids,
                text: result.text,
                metrics: result.metrics,
                timing,
                finish_reason,
                token_times: result.token_times,
            }),
            None => RequestEvent::Failed(ServeError::Internal(
                "a finished request had no result".into(),
            )),
        };
        self.end(id, terminal);
    }

    /// Sends a request's terminal event and releases its capacity.
    ///
    /// Capacity is released first, so a client that sees the terminal event
    /// can immediately submit again.
    ///
    /// The request's runtime record moves into the bounded table of
    /// finished requests, so the executor retains nothing for it.
    fn end(&mut self, id: RequestId, terminal: RequestEvent) {
        let Some(live) = self.live.remove(&id) else {
            return;
        };
        self.shared.release();
        match self.executor.forget(id) {
            Ok((record, _)) if self.retain_finished > 0 => self.retain(RequestSnapshot {
                id,
                status: record.status,
                timing: record.timing,
                generated_tokens: live.decoder.tokens().len(),
            }),
            _ => self.drop_trace(id),
        }
        // Gauges are republished first, so a client that has seen the
        // terminal event never reads health that still counts the request.
        self.publish();
        let _ = live.events.send(terminal);
    }

    fn retain(&mut self, snapshot: RequestSnapshot) {
        if self.retain_finished == 0 {
            return;
        }
        while self.finished_order.len() >= self.retain_finished {
            if let Some(oldest) = self.finished_order.pop_front() {
                self.finished.remove(&oldest);
                self.drop_trace(oldest);
            }
        }
        self.finished_order.push_back(snapshot.id);
        self.finished.insert(snapshot.id, snapshot);
    }

    fn snapshot(&self, id: RequestId) -> Result<RequestSnapshot, ServeError> {
        if let Some(live) = self.live.get(&id)
            && let Some(record) = self.executor.state().request(id)
        {
            return Ok(RequestSnapshot {
                id,
                status: record.status.clone(),
                timing: record.timing,
                generated_tokens: live.decoder.tokens().len(),
            });
        }
        self.finished
            .get(&id)
            .cloned()
            .ok_or(ServeError::UnknownRequest(id))
    }

    /// Cancels every unfinished request in the executor, releasing its
    /// session and KV cache, and ends its stream with `terminal`.
    fn abort_all(&mut self, terminal: impl Fn() -> RequestEvent) {
        let now = self.shared.clock.now();
        for id in self.live.keys().copied().collect::<Vec<_>>() {
            if self.executor.cancel(id, now).is_ok() {
                self.trace_cancellation(id);
            }
            self.end(id, terminal());
        }
    }

    fn trace_cancellation(&mut self, id: RequestId) {
        if let Some(tracer) = &mut self.tracer {
            let records = tracer.observe_cancellation(id, self.executor.state());
            self.record(records);
        }
    }

    /// Writes records to the sink and keeps them, as configured. A sink
    /// that fails is dropped and reported in health; serving continues.
    fn record(&mut self, records: Vec<TraceRecord>) {
        if records.is_empty() {
            return;
        }
        if let Some(sink) = &mut self.sink
            && write_jsonl(&records, &mut *sink)
                .and_then(|()| sink.flush())
                .is_err()
        {
            self.sink = None;
            self.shared.trace_write_failed.store(true, Ordering::SeqCst);
        }
        if let Some(traces) = &mut self.traces {
            for record in records {
                traces
                    .entry(RequestId::new(record.request))
                    .or_default()
                    .push(record);
            }
        }
    }

    fn drop_trace(&mut self, id: RequestId) {
        if let Some(traces) = &mut self.traces {
            traces.remove(&id);
        }
    }
}
