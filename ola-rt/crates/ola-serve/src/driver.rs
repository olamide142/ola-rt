//! A dedicated thread that owns an [`Executor`] and serves it over channels.

use std::{
    collections::BTreeMap,
    future::Future,
    io,
    ops::ControlFlow,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, RecvTimeoutError, TryRecvError},
    },
    task::{Context, Poll},
    thread::{self, JoinHandle},
    time::Duration,
};

use ola_core::{Clock, ModelId, RequestId, RequestSpec, Timestamp};
use ola_model::{IncrementalDecoder, SessionMetrics};
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
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 256,
            stall_backoff: Duration::from_millis(1),
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
    Shutdown,
}

struct Shared {
    clock: Arc<dyn Clock>,
    max_in_flight: usize,
    in_flight: AtomicUsize,
    accepting: AtomicBool,
}

impl Shared {
    fn release(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
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
    /// Models must be registered on the executor before it is handed over.
    pub fn spawn(
        executor: Executor,
        clock: impl Clock + 'static,
        config: DriverConfig,
    ) -> io::Result<Self> {
        let (commands, receiver) = mpsc::channel();
        let shared = Arc::new(Shared {
            clock: Arc::new(clock),
            max_in_flight: config.max_in_flight,
            in_flight: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
        });
        let worker = Worker {
            executor,
            commands: receiver,
            shared: Arc::clone(&shared),
            live: BTreeMap::new(),
            stall_backoff: config.stall_backoff,
        };
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
        self.stop()
    }

    fn stop(&mut self) -> Result<Executor, ServeError> {
        let thread = self.thread.take().ok_or(ServeError::ShuttingDown)?;
        // The command goes first so that once `is_accepting` reads false the
        // driver is guaranteed to stop at its next command check.
        let _ = self.handle.commands.send(Command::Shutdown);
        self.handle.shared.accepting.store(false, Ordering::SeqCst);
        thread
            .join()
            .map_err(|_| ServeError::Internal("the driver thread panicked".into()))
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        if self.thread.is_some() {
            let _ = self.stop();
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
        if !self.is_accepting() {
            return rejected(ServeError::ShuttingDown);
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
        self.shared.accepting.load(Ordering::SeqCst)
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
    stall_backoff: Duration,
}

impl Worker {
    fn run(mut self) -> Executor {
        while self.serve().is_continue() {}
        self.abort_all(|| RequestEvent::Failed(ServeError::ShuttingDown));
        self.executor
    }

    /// Runs one round: wait for work if there is none, apply commands, reap
    /// abandoned requests, and advance one iteration.
    fn serve(&mut self) -> ControlFlow<()> {
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
        self.drain()?;
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
        ControlFlow::Continue(())
    }

    fn drain(&mut self) -> ControlFlow<()> {
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
                let snapshot = self
                    .executor
                    .state()
                    .request(id)
                    .map(|record| RequestSnapshot {
                        id,
                        status: record.status.clone(),
                        timing: record.timing,
                    })
                    .ok_or(ServeError::UnknownRequest(id));
                let _ = reply.send(snapshot);
            }
            Command::Shutdown => return ControlFlow::Break(()),
        }
        ControlFlow::Continue(())
    }

    fn cancel(&mut self, id: RequestId) -> Result<(), ServeError> {
        let Some(record) = self.executor.state().request(id) else {
            return Err(ServeError::UnknownRequest(id));
        };
        if record.status.is_terminal() {
            return Ok(());
        }
        match self.executor.cancel(id, self.shared.clock.now()) {
            Ok(()) => {
                self.end(id, RequestEvent::Cancelled);
                Ok(())
            }
            Err(error) => Err(ServeError::Internal(error.to_string())),
        }
    }

    /// Advances one iteration and returns whether any request progressed.
    fn step(&mut self) -> bool {
        let iteration = match self.executor.step_iteration(self.shared.clock.now()) {
            Ok(iteration) => iteration,
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
        finished: bool,
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
            if finished {
                text.push_str(&live.decoder.finish(tokenizer).unwrap_or_default());
            }
        }
        let _ = live.events.send(RequestEvent::Token {
            token_id,
            text,
            phase,
            generated_tokens,
        });
        if !finished {
            return;
        }
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
    fn end(&mut self, id: RequestId, terminal: RequestEvent) {
        if let Some(live) = self.live.remove(&id) {
            self.shared.release();
            let _ = live.events.send(terminal);
        }
    }

    /// Cancels every unfinished request in the executor, releasing its
    /// session and KV cache, and ends its stream with `terminal`.
    fn abort_all(&mut self, terminal: impl Fn() -> RequestEvent) {
        let now = self.shared.clock.now();
        for id in self.live.keys().copied().collect::<Vec<_>>() {
            let _ = self.executor.cancel(id, now);
            self.end(id, terminal());
        }
    }
}
