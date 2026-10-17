use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    num::NonZeroUsize,
    sync::Arc,
};

use ola_core::{
    Clock, Input, ModelDescriptor, ModelId, Request, RequestId, RequestSpec, Timestamp,
};
use ola_model::{
    CausalLanguageModel, GeneratedToken, GenerationSession, ModelExecutionError, PrefillOutcome,
    SessionMetrics,
};

use crate::{
    AdmissionReason, BatchExecutionMode, BatchPhase, ComposedPolicy, DecisionRecord, DecodeView,
    ExecutionBatch, ExecutionPlan, ExecutionPolicy, ExecutorLimits, IterationId, LifecycleAction,
    ModelSnapshot, PlanDivergence, PlanError, PlannedBatch, PlannedDeferral, PlannedPhase,
    PlanningSnapshot, PolicyDecision, PolicyDescriptor, PolicyError, PolicyMeasurements,
    PreemptionReason, PrefillView, QueuedRequest, ReleaseReason, RequestRecord, Runtime,
    RuntimeError, RuntimeState, composed::is_complete_prompt,
};

/// Final output retained after successful execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionResult {
    /// Request that produced this output.
    pub request_id: RequestId,
    /// Generated token IDs in emission order.
    pub token_ids: Vec<u32>,
    /// Artifact-tokenizer decoding of `token_ids`.
    pub text: String,
    /// Final model-session measurements captured before cache release.
    pub metrics: SessionMetrics,
    /// When each token in `token_ids` was produced: the finish time of the
    /// invocation that selected it.
    pub token_times: Vec<Timestamp>,
}

/// Model work represented by a token event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPhase {
    /// Prompt processing and first-token selection.
    Prefill,
    /// One cached autoregressive token step.
    Decode,
}

/// One observable outcome from a synchronous executor step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorEvent {
    /// Neither queued nor active work was available.
    Idle,
    /// One resumable prompt segment completed without generating a token.
    PrefillProgress {
        /// Request whose prompt cursor advanced.
        request_id: RequestId,
        /// Prompt tokens accepted by the session so far.
        processed_tokens: usize,
        /// Total tokens in the tokenized prompt.
        total_tokens: usize,
        /// Physical KV bytes currently attributed to the request.
        kv_cache_bytes: u64,
    },
    /// One token was emitted by prefill or decode.
    Token {
        /// Request that produced the token.
        request_id: RequestId,
        /// Selected token and EOS indication.
        token: GeneratedToken,
        /// Number of output tokens emitted for this request so far.
        generated_tokens: usize,
        /// Physical KV bytes currently attributed to the request.
        kv_cache_bytes: u64,
        /// Model phase that produced this token.
        phase: ExecutionPhase,
        /// Whether the request completed with this token.
        finished: bool,
        /// When the token was produced: its invocation's finish time.
        at: Timestamp,
    },
    /// Model preparation or execution failed and the request became terminal.
    Failed {
        /// Request that failed.
        request_id: RequestId,
        /// Stable human-readable diagnostic retained in runtime state.
        message: String,
    },
    /// Admission policy left a scheduler-selected request queued.
    Deferred {
        /// Request that remains at the scheduler boundary.
        request_id: RequestId,
        /// Typed resource constraint that prevented admission.
        reason: AdmissionReason,
    },
    /// Batch policy left otherwise eligible work for a later iteration.
    BatchDeferred {
        /// Request that remains queued or active.
        request_id: RequestId,
        /// Model phase whose token cost did not fit the current plan.
        phase: ExecutionPhase,
        /// Input token positions that the operation would process.
        token_cost: usize,
    },
    /// A live request returned to the queue; it resumes later where it
    /// stopped, without repeating tokens.
    Preempted {
        /// Request moved back to the queue.
        request_id: RequestId,
        /// Why it was chosen.
        reason: PreemptionReason,
        /// Output tokens it had generated, all kept.
        generated_tokens: usize,
    },
    /// Admission policy terminated a request that can never fit.
    Rejected {
        /// Request moved directly from queued to failed.
        request_id: RequestId,
        /// Typed permanent resource constraint.
        reason: AdmissionReason,
    },
}

/// Observable result of one iteration-level scheduling boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorIteration {
    /// Correlation ID shared with `execution_plan` and the iteration's
    /// invocations, events, and traces.
    pub id: IterationId,
    /// Caller-supplied monotonic iteration timestamp.
    pub at: Timestamp,
    /// Per-request outcomes in deterministic execution order.
    pub events: Vec<ExecutorEvent>,
    /// Validated plan committed by this iteration.
    pub execution_plan: ExecutionPlan,
    /// Why the policy planned it: ranking, admission inputs and outcomes,
    /// batch selections, grouping alternatives, and capability fallbacks.
    pub decision: DecisionRecord,
    /// Ways execution differed from `execution_plan`: lifecycle failures;
    /// then, batch by batch, a tensor fallback followed by KV-capacity
    /// failures in row order; then unpredicted releases in event order.
    /// Empty when execution matched the plan exactly.
    pub divergences: Vec<PlanDivergence>,
    /// Model invocations used to execute the selected plan.
    pub execution_batches: Vec<ExecutionBatch>,
    /// Live generation sessions after the iteration.
    pub active_requests: usize,
    /// Requests still waiting for admission after the iteration.
    pub queued_requests: usize,
    /// Physical KV bytes attributed to all live sessions.
    pub kv_cache_bytes: u64,
    /// Worst-case KV bytes committed to all live requests.
    pub reserved_kv_bytes: u64,
}

impl ExecutorIteration {
    /// Returns true when the iteration observed no queued or active work.
    pub fn is_idle(&self) -> bool {
        self.events == [ExecutorEvent::Idle]
    }
}

/// Failure to configure or control the executor itself.
#[derive(Debug, Eq, PartialEq)]
pub enum ExecutorError {
    /// Runtime admission or lifecycle operation failed.
    Runtime(RuntimeError),
    /// Executable model attachment referenced absent metadata.
    ModelNotRegistered(ModelId),
    /// An executable was attached more than once under one model ID.
    DuplicateExecutableModel(ModelId),
    /// A multi-request executor requires at least one active slot.
    InvalidActiveRequestLimit(usize),
    /// A resumable prefill segment must contain at least one token.
    InvalidPrefillChunkSize(usize),
    /// The execution policy returned an impossible decision; nothing was
    /// mutated.
    Policy(PolicyError),
    /// The executor built a plan that failed validation; nothing was mutated.
    InvalidPlan(PlanError),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::ModelNotRegistered(model) => {
                write!(
                    formatter,
                    "model `{model}` has no registered runtime metadata"
                )
            }
            Self::DuplicateExecutableModel(model) => {
                write!(
                    formatter,
                    "model `{model}` already has an executable attached"
                )
            }
            Self::InvalidActiveRequestLimit(limit) => {
                write!(
                    formatter,
                    "active request limit must be positive, got {limit}"
                )
            }
            Self::InvalidPrefillChunkSize(size) => {
                write!(formatter, "prefill chunk size must be positive, got {size}")
            }
            Self::Policy(error) => error.fmt(formatter),
            Self::InvalidPlan(error) => {
                write!(formatter, "executor built an invalid plan: {error}")
            }
        }
    }
}

impl Error for ExecutorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            Self::InvalidPlan(error) => Some(error),
            Self::Policy(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RuntimeError> for ExecutorError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

struct ActiveGeneration {
    request_id: RequestId,
    model_id: ModelId,
    model: CausalLanguageModel,
    session: Box<dyn GenerationSession>,
    prompt: Vec<u32>,
    prompt_tokens: usize,
    generated: Vec<u32>,
    token_times: Vec<Timestamp>,
    last_token: GeneratedToken,
    max_new_tokens: usize,
    last_step_at: Timestamp,
    reserved_kv_bytes: u64,
}

impl ActiveGeneration {
    /// Absolute position occupied by the next decode input, the last token.
    fn next_position(&self) -> usize {
        self.prompt_tokens
            .saturating_add(self.generated.len())
            .saturating_sub(1)
    }
}

struct PendingPrefill {
    request_id: RequestId,
    model_id: ModelId,
    model: CausalLanguageModel,
    session: Box<dyn GenerationSession>,
    prompt_tokens: Vec<u32>,
    cursor: usize,
    max_new_tokens: usize,
    last_step_at: Timestamp,
    reserved_kv_bytes: u64,
}

/// What a preempted request keeps while queued, to resume without
/// repeating output.
struct Resumed {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    token_times: Vec<Timestamp>,
}

#[derive(Default)]
struct BatchOutcome {
    indexed_events: Vec<(usize, ExecutorEvent)>,
    execution_batches: Vec<ExecutionBatch>,
    divergences: Vec<PlanDivergence>,
}

impl BatchOutcome {
    fn extend(&mut self, other: Self) {
        self.indexed_events.extend(other.indexed_events);
        self.execution_batches.extend(other.execution_batches);
        self.divergences.extend(other.divergences);
    }
}

/// Deterministic synchronous model executor.
///
/// [`step_iteration`](Self::step_iteration) is the only way to advance it: the
/// configured [`ExecutionPolicy`] plans the iteration from an immutable
/// snapshot, the plan is validated, and only then is it committed. Every
/// session active at the iteration boundary advances once, then free slots
/// are filled.
/// Compatible prefill and decode operations may share tensor invocations;
/// models without a matching batch hook retain the individual correctness
/// path.
pub struct Executor {
    runtime: Runtime,
    models: BTreeMap<ModelId, CausalLanguageModel>,
    active: BTreeMap<RequestId, ActiveGeneration>,
    ready: VecDeque<RequestId>,
    prefilling: BTreeMap<RequestId, PendingPrefill>,
    prefill_ready: VecDeque<RequestId>,
    max_active_requests: usize,
    prefill_chunk_size: Option<NonZeroUsize>,
    policy: Box<dyn ExecutionPolicy>,
    results: BTreeMap<RequestId, ExecutionResult>,
    prepared: BTreeMap<RequestId, Result<Vec<u32>, String>>,
    measurements: PolicyMeasurements,
    generation: u64,
    clock: Option<Arc<dyn Clock>>,
    iterations: u64,
    resumed: BTreeMap<RequestId, Resumed>,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new(Runtime::default())
    }
}

impl Executor {
    /// Wraps a runtime with no attached executable models.
    pub fn new(runtime: Runtime) -> Self {
        Self {
            runtime,
            models: BTreeMap::new(),
            active: BTreeMap::new(),
            ready: VecDeque::new(),
            prefilling: BTreeMap::new(),
            prefill_ready: VecDeque::new(),
            max_active_requests: 1,
            prefill_chunk_size: None,
            policy: Box::new(ComposedPolicy::default()),
            results: BTreeMap::new(),
            prepared: BTreeMap::new(),
            measurements: PolicyMeasurements::default(),
            generation: 0,
            clock: None,
            iterations: 0,
            resumed: BTreeMap::new(),
        }
    }

    /// Reads `clock` around every model invocation.
    ///
    /// Each [`ExecutionBatch`] then records when its invocation started and
    /// finished, and tokens, first-token, completion, and failure times are
    /// stamped when the invocation that produced them finished rather than
    /// at the iteration's planning time. The clock must share the epoch of
    /// the timestamps passed to [`step_iteration`](Self::step_iteration);
    /// readings earlier than the iteration time are clamped to it.
    ///
    /// Without a clock every time in an iteration is its planning time,
    /// which keeps replays and tests deterministic.
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Returns the clock reading, or `at` without a clock, never earlier
    /// than `at`.
    fn now(&self, at: Timestamp) -> Timestamp {
        self.clock.as_ref().map_or(at, |clock| clock.now().max(at))
    }

    /// Wraps a runtime with a bounded number of concurrently live sessions.
    pub fn with_max_active_requests(
        runtime: Runtime,
        max_active_requests: usize,
    ) -> Result<Self, ExecutorError> {
        if max_active_requests == 0 {
            return Err(ExecutorError::InvalidActiveRequestLimit(0));
        }
        let mut executor = Self::new(runtime);
        executor.max_active_requests = max_active_requests;
        Ok(executor)
    }

    /// Replaces the execution policy that plans every iteration.
    ///
    /// The default is [`ComposedPolicy::default`]: FIFO scheduling,
    /// unconstrained admission, unbounded batching, and one prefill group
    /// per model.
    pub fn with_policy<P>(mut self, policy: P) -> Self
    where
        P: ExecutionPolicy + 'static,
    {
        self.policy = Box::new(policy);
        self
    }

    /// Returns the configured policy's stable name and configuration.
    pub fn policy_descriptor(&self) -> PolicyDescriptor {
        self.policy.descriptor()
    }

    /// Supplies measured inputs, such as calibrated invocation overhead,
    /// that every planning snapshot carries to the policy.
    #[must_use]
    pub const fn with_policy_measurements(mut self, measurements: PolicyMeasurements) -> Self {
        self.measurements = measurements;
        self
    }

    /// Bounds resumable prompt work performed for one request per iteration.
    ///
    /// Architectures that do not advertise chunk support remain on the
    /// complete-prompt compatibility path. Admission still reserves the
    /// request's worst-case KV demand before its first chunk executes.
    pub fn with_prefill_chunk_size(mut self, tokens: usize) -> Result<Self, ExecutorError> {
        self.prefill_chunk_size =
            Some(NonZeroUsize::new(tokens).ok_or(ExecutorError::InvalidPrefillChunkSize(tokens))?);
        Ok(self)
    }

    /// Registers runtime metadata and its corresponding executable atomically.
    pub fn register_model(
        &mut self,
        descriptor: ModelDescriptor,
        model: CausalLanguageModel,
    ) -> Result<(), ExecutorError> {
        let id = descriptor.id.clone();
        if self.models.contains_key(&id) {
            return Err(ExecutorError::DuplicateExecutableModel(id));
        }
        self.runtime.register_model(descriptor)?;
        self.models.insert(id, model);
        self.bump_generation();
        Ok(())
    }

    /// Attaches an executable to metadata already registered on the runtime.
    pub fn attach_model(
        &mut self,
        id: ModelId,
        model: CausalLanguageModel,
    ) -> Result<(), ExecutorError> {
        if !self.runtime.state().models().contains(&id) {
            return Err(ExecutorError::ModelNotRegistered(id));
        }
        if self.models.contains_key(&id) {
            return Err(ExecutorError::DuplicateExecutableModel(id));
        }
        self.models.insert(id, model);
        self.bump_generation();
        Ok(())
    }

    /// Returns the executable attached for `id`, if any.
    pub fn model(&self, id: &ModelId) -> Option<&CausalLanguageModel> {
        self.models.get(id)
    }

    /// Validates and queues a request on the owned runtime.
    pub fn submit(
        &mut self,
        spec: RequestSpec,
        arrival_at: Timestamp,
    ) -> Result<RequestId, ExecutorError> {
        let request_id = self.runtime.submit(spec, arrival_at)?;
        self.bump_generation();
        Ok(request_id)
    }

    /// Cancels queued or active work and releases request-local model state.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
        cancelled_at: Timestamp,
    ) -> Result<(), ExecutorError> {
        self.runtime.cancel(request_id, cancelled_at)?;
        self.prepared.remove(&request_id);
        self.resumed.remove(&request_id);
        self.bump_generation();
        if self.active.remove(&request_id).is_some() {
            self.ready.retain(|ready| *ready != request_id);
            self.refresh_kv_memory();
        } else if self.prefilling.remove(&request_id).is_some() {
            self.prefill_ready.retain(|ready| *ready != request_id);
            self.refresh_kv_memory();
        }
        Ok(())
    }

    /// Advances one deterministic generation iteration.
    ///
    /// The iteration is planned, validated, and committed as one
    /// [`ExecutionPlan`]. Every request active at entry receives one decode
    /// step. Free slots, including those of requests that must finish by
    /// reaching `max_new_tokens`, are filled from the configured scheduler;
    /// slots freed by EOS or failure are refilled next iteration. Newly
    /// admitted requests receive prefill but are not decoded again in the
    /// same iteration.
    pub fn step_iteration(&mut self, at: Timestamp) -> Result<ExecutorIteration, ExecutorError> {
        self.validate_iteration_time(at)?;
        self.prepare_queued();
        let snapshot = self.snapshot(at);
        let PolicyDecision {
            plan: mut execution_plan,
            record: decision,
        } = self.policy.plan(&snapshot).map_err(ExecutorError::Policy)?;
        // Numbered only once a plan exists, so failed planning consumes no ID.
        let id = IterationId::new(self.iterations + 1);
        execution_plan.iteration = id;
        execution_plan
            .validate(&snapshot, self.policy.enforces_model_kv_capacity())
            .map_err(ExecutorError::InvalidPlan)?;
        let committed = self.commit_plan(&execution_plan, at)?;
        self.iterations = id.get();
        let mut events = committed
            .indexed_events
            .into_iter()
            .map(|(_, event)| event)
            .collect::<Vec<_>>();
        if events.is_empty() {
            events.push(ExecutorEvent::Idle);
        }
        self.bump_generation();

        let iteration = ExecutorIteration {
            id,
            at,
            events,
            execution_plan,
            decision,
            divergences: committed.divergences,
            execution_batches: committed.execution_batches,
            active_requests: self.live_request_count(),
            queued_requests: self.runtime.state().queued_len(),
            kv_cache_bytes: self.runtime.state().memory.kv_cache_bytes,
            reserved_kv_bytes: self.reserved_kv_bytes(),
        };
        self.policy.observe(&iteration);
        Ok(iteration)
    }

    /// Returns runtime lifecycle and memory state.
    pub fn state(&self) -> &RuntimeState {
        self.runtime.state()
    }

    /// Returns a completed result retained by request ID.
    pub fn result(&self, request_id: RequestId) -> Option<&ExecutionResult> {
        self.results.get(&request_id)
    }

    /// Moves a completed result out of the executor.
    ///
    /// Drivers that deliver results to their own handles call this so the
    /// executor does not retain every output for its lifetime.
    pub fn take_result(&mut self, request_id: RequestId) -> Option<ExecutionResult> {
        self.results.remove(&request_id)
    }

    /// Removes everything retained for a terminal request: its runtime
    /// record and any untaken result.
    pub fn forget(
        &mut self,
        request_id: RequestId,
    ) -> Result<(RequestRecord, Option<ExecutionResult>), ExecutorError> {
        let record = self.runtime.forget_terminal(request_id)?;
        Ok((record, self.results.remove(&request_id)))
    }

    /// Returns whether one request currently owns a generation session.
    pub fn has_active_request(&self) -> bool {
        self.live_request_count() > 0
    }

    /// Returns the number of request-local generation sessions currently live.
    pub fn active_request_count(&self) -> usize {
        self.live_request_count()
    }

    /// Returns the configured upper bound on concurrently live sessions.
    pub const fn max_active_requests(&self) -> usize {
        self.max_active_requests
    }

    /// Returns a counter that changes whenever executor state may change.
    ///
    /// Plans record the generation they were built from and are rejected as
    /// stale when it no longer matches.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns worst-case page-rounded KV bytes committed to live requests.
    pub fn reserved_kv_bytes(&self) -> u64 {
        self.active
            .values()
            .map(|active| active.reserved_kv_bytes)
            .chain(
                self.prefilling
                    .values()
                    .map(|pending| pending.reserved_kv_bytes),
            )
            .sum()
    }

    /// Borrows the underlying runtime for read-only inspection.
    pub const fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Tokenizes queued requests for attached models once, caching the
    /// result until the request starts, fails, or is cancelled.
    fn prepare_queued(&mut self) {
        let state = self.runtime.state();
        for request_id in state.queued_ids() {
            if self.prepared.contains_key(&request_id) {
                continue;
            }
            let request = &state
                .request(request_id)
                .expect("queued requests retain their runtime record")
                .request;
            if let Some(model) = self.models.get(&request.spec.model) {
                let prompt = prepare_prompt(model, request).map_err(|error| error.to_string());
                self.prepared.insert(request_id, prompt);
            }
        }
    }

    /// Builds the immutable input for one planning pass.
    fn snapshot(&self, at: Timestamp) -> PlanningSnapshot {
        let state = self.runtime.state();
        let arrival = |request_id: &RequestId| {
            state
                .request(*request_id)
                .map_or((ola_core::Priority::NORMAL, Timestamp::ZERO), |record| {
                    (record.request.spec.priority, record.timing.queued_at)
                })
        };
        PlanningSnapshot {
            at,
            generation: self.generation,
            limits: ExecutorLimits {
                max_active_requests: self.max_active_requests,
                prefill_chunk_size: self.prefill_chunk_size.map(NonZeroUsize::get),
            },
            queued: state
                .queued_ids()
                .map(|request_id| {
                    let record = state
                        .request(request_id)
                        .expect("queued requests retain their runtime record");
                    let spec = &record.request.spec;
                    // A cache hit must leave a token to compute, and resuming
                    // after it needs resumable prefill.
                    let cached =
                        match (self.prepared.get(&request_id), self.models.get(&spec.model)) {
                            (Some(Ok(tokens)), Some(model))
                                if model.supports_chunked_prefill() && tokens.len() > 1 =>
                            {
                                model.cached_prefix(&tokens[..tokens.len() - 1])
                            }
                            _ => ola_model::CachedPrefix::default(),
                        };
                    QueuedRequest {
                        request_id,
                        cached_prefix_tokens: cached.tokens,
                        cached_prefix_evictable_blocks: cached.evictable_blocks,
                        model_id: spec.model.clone(),
                        prompt_tokens: match self.prepared.get(&request_id) {
                            Some(Ok(tokens)) => Ok(tokens.len()),
                            Some(Err(message)) => Err(message.clone()),
                            None => {
                                Err(format!("model `{}` has no executable attached", spec.model))
                            }
                        },
                        max_new_tokens: spec.generation.max_new_tokens.saturating_sub(
                            self.resumed
                                .get(&request_id)
                                .map_or(0, |resumed| resumed.generated.len()),
                        ),
                        priority: spec.priority,
                        deadline: spec.deadline,
                        queued_at: record.timing.queued_at,
                    }
                })
                .collect(),
            prefilling: self
                .prefilling
                .iter()
                .map(|(request_id, pending)| {
                    (
                        *request_id,
                        PrefillView {
                            model_id: pending.model_id.clone(),
                            cursor: pending.cursor,
                            prompt_tokens: pending.prompt_tokens.len(),
                            max_new_tokens: pending.max_new_tokens.saturating_sub(
                                self.resumed
                                    .get(request_id)
                                    .map_or(0, |resumed| resumed.generated.len()),
                            ),
                            reserved_kv_bytes: pending.reserved_kv_bytes,
                            priority: arrival(request_id).0,
                            arrived_at: arrival(request_id).1,
                        },
                    )
                })
                .collect(),
            decoding: self
                .active
                .iter()
                .map(|(request_id, active)| {
                    (
                        *request_id,
                        DecodeView {
                            model_id: active.model_id.clone(),
                            position: active.next_position(),
                            generated_tokens: active.generated.len(),
                            max_new_tokens: active.max_new_tokens,
                            reserved_kv_bytes: active.reserved_kv_bytes,
                            priority: arrival(request_id).0,
                            arrived_at: arrival(request_id).1,
                        },
                    )
                })
                .collect(),
            decode_order: self.ready.iter().copied().collect(),
            prefill_order: self.prefill_ready.iter().copied().collect(),
            models: state
                .models()
                .iter()
                .map(|descriptor| {
                    let snapshot =
                        self.models
                            .get(&descriptor.id)
                            .map_or(ModelSnapshot::DETACHED, |model| ModelSnapshot {
                                attached: true,
                                batch: model.batch_capabilities(),
                                chunked_prefill: model.supports_chunked_prefill(),
                                kv_cache: model.kv_cache_profile(),
                                kv_pool: model.kv_pool_status(),
                            });
                    (descriptor.id.clone(), snapshot)
                })
                .collect(),
            memory: state.memory,
            device: None,
            measurements: self.measurements,
        }
    }

    /// Commits a validated plan: lifecycle actions first, then every
    /// invocation in order. Returns events sorted by planned index and every
    /// divergence from the plan.
    fn commit_plan(
        &mut self,
        plan: &ExecutionPlan,
        at: Timestamp,
    ) -> Result<BatchOutcome, ExecutorError> {
        let mut outcome = BatchOutcome::default();
        let mut admitted = BTreeMap::new();
        let admit_events = plan
            .batches
            .iter()
            .filter(|batch| batch.phase == PlannedPhase::Prefill)
            .flat_map(|batch| &batch.rows)
            .map(|row| (row.request_id, (row.event_index, row.positions.start)))
            .collect::<BTreeMap<_, _>>();
        for action in &plan.lifecycle {
            match action {
                LifecycleAction::Admit {
                    request_id,
                    model_id,
                    reserved_kv_bytes,
                } => {
                    let model = self.models[model_id].clone();
                    let prompt_tokens = self
                        .prepared
                        .remove(request_id)
                        .and_then(Result::ok)
                        .expect("planned admissions have prepared prompts");
                    let generation = self
                        .runtime
                        .state()
                        .request(*request_id)
                        .expect("admitted requests retain their runtime record")
                        .request
                        .spec
                        .generation
                        .clone();
                    let max_new_tokens = generation.max_new_tokens;
                    let lease = self.runtime.start_selected(*request_id, at)?;
                    let (event_index, cached) = admit_events[request_id];
                    // The plan starts the first prefill after the cached
                    // prefix it saw; attach exactly that much or fail.
                    let session = model.create_session(generation).and_then(|mut session| {
                        if cached == 0 {
                            return Ok(session);
                        }
                        let attached = session.attach_prefix(&prompt_tokens[..cached])?;
                        if attached == cached {
                            Ok(session)
                        } else {
                            Err(ModelExecutionError::new(
                                ola_model::ExecutionStage::SessionCreation,
                                format!(
                                    "planned {cached} cached prefix tokens but attached {attached}"
                                ),
                            ))
                        }
                    });
                    match session {
                        Ok(session) => {
                            admitted.insert(
                                *request_id,
                                PendingPrefill {
                                    request_id: lease.request_id,
                                    model_id: model_id.clone(),
                                    model,
                                    session,
                                    prompt_tokens,
                                    cursor: cached,
                                    max_new_tokens,
                                    last_step_at: at,
                                    reserved_kv_bytes: *reserved_kv_bytes,
                                },
                            );
                        }
                        Err(error) => {
                            let event = self.fail_started(*request_id, error.to_string(), at)?;
                            outcome.indexed_events.push((event_index, event));
                            outcome
                                .divergences
                                .push(PlanDivergence::SessionCreationFailed {
                                    request_id: *request_id,
                                });
                        }
                    }
                }
                LifecycleAction::Reject {
                    request_id,
                    reason,
                    event_index,
                } => {
                    self.prepared.remove(request_id);
                    self.resumed.remove(request_id);
                    self.runtime.reject(*request_id, reason.to_string(), at)?;
                    outcome.indexed_events.push((
                        *event_index,
                        ExecutorEvent::Rejected {
                            request_id: *request_id,
                            reason: reason.clone(),
                        },
                    ));
                }
                LifecycleAction::Fail {
                    request_id,
                    message,
                    event_index,
                } => {
                    self.prepared.remove(request_id);
                    self.runtime.start_selected(*request_id, at)?;
                    let event = self.fail_started(*request_id, message.clone(), at)?;
                    outcome.indexed_events.push((*event_index, event));
                }
                LifecycleAction::Preempt {
                    request_id,
                    reason,
                    event_index,
                } => {
                    let event = self.preempt(*request_id, *reason, at)?;
                    outcome.indexed_events.push((*event_index, event));
                }
            }
        }
        for deferral in &plan.deferrals {
            match deferral {
                PlannedDeferral::Budget {
                    request_id,
                    phase,
                    token_cost,
                    event_index,
                } => outcome.indexed_events.push((
                    *event_index,
                    ExecutorEvent::BatchDeferred {
                        request_id: *request_id,
                        phase: execution_phase(*phase),
                        token_cost: *token_cost,
                    },
                )),
                PlannedDeferral::Admission {
                    request_id,
                    reason,
                    event_index,
                } => outcome.indexed_events.push((
                    *event_index,
                    ExecutorEvent::Deferred {
                        request_id: *request_id,
                        reason: reason.clone(),
                    },
                )),
                PlannedDeferral::Slots { .. } => {}
            }
        }
        let planned = plan
            .batches
            .iter()
            .flat_map(|batch| &batch.rows)
            .map(|row| row.request_id)
            .collect::<BTreeSet<_>>();
        self.ready
            .retain(|request_id| !planned.contains(request_id));
        self.prefill_ready
            .retain(|request_id| !planned.contains(request_id));
        for (batch_index, batch) in plan.batches.iter().enumerate() {
            let mut planned_tensor = batch.mode == BatchExecutionMode::Tensor;
            let executed = match batch.phase {
                PlannedPhase::Decode => {
                    let members = batch
                        .rows
                        .iter()
                        .map(|row| {
                            self.active
                                .remove(&row.request_id)
                                .expect("planned decodes retain active sessions")
                        })
                        .collect();
                    self.execute_decode_batch(batch, members, at)?
                }
                PlannedPhase::ContinuedPrefill => {
                    let members = batch
                        .rows
                        .iter()
                        .map(|row| {
                            self.prefilling
                                .remove(&row.request_id)
                                .expect("planned prefills retain their sessions")
                        })
                        .collect();
                    self.execute_prefill_batch(batch, members, at)?
                }
                PlannedPhase::Prefill => {
                    // Rows whose session creation failed already emitted an
                    // event; the rest still execute together.
                    let mut batch = batch.clone();
                    batch
                        .rows
                        .retain(|row| admitted.contains_key(&row.request_id));
                    if batch.rows.is_empty() {
                        continue;
                    }
                    if batch.rows.len() == 1 {
                        // Already recorded as a session-creation failure.
                        batch.mode = BatchExecutionMode::Individual;
                        planned_tensor = false;
                    }
                    let members = batch
                        .rows
                        .iter()
                        .map(|row| {
                            admitted
                                .remove(&row.request_id)
                                .expect("admitted rows retain their sessions")
                        })
                        .collect();
                    self.execute_prefill_batch(&batch, members, at)?
                }
            };
            if planned_tensor
                && executed
                    .execution_batches
                    .iter()
                    .all(|executed| executed.mode == BatchExecutionMode::Individual)
            {
                outcome
                    .divergences
                    .push(PlanDivergence::ModeFallback { batch_index });
            }
            outcome.extend(executed);
        }
        outcome.indexed_events.sort_by_key(|(index, _)| *index);
        let releases = unplanned_releases(plan, &outcome);
        outcome.divergences.extend(releases);
        Ok(outcome)
    }

    /// Executes one planned prompt invocation. A declared tensor hook that
    /// returns `None` falls back to individual execution.
    fn execute_prefill_batch(
        &mut self,
        batch: &PlannedBatch,
        mut members: Vec<PendingPrefill>,
        at: Timestamp,
    ) -> Result<BatchOutcome, ExecutorError> {
        let mut outcome = BatchOutcome::default();
        let model = members[0].model.clone();
        let prompts = batch
            .rows
            .iter()
            .zip(&members)
            .map(|(row, pending)| pending.prompt_tokens[row.positions.clone()].to_vec())
            .collect::<Vec<_>>();
        // Only a tensor invocation is timed as a whole; individual
        // invocations below read the clock around each call.
        let started_at = if batch.mode == BatchExecutionMode::Tensor {
            self.now(at)
        } else {
            at
        };
        let batched = if batch.mode == BatchExecutionMode::Tensor {
            let prompt_slices = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let final_chunks = batch
                .rows
                .iter()
                .map(|row| row.final_prompt_chunk)
                .collect::<Vec<_>>();
            let mut sessions: Vec<&mut dyn GenerationSession> = Vec::with_capacity(members.len());
            for pending in &mut members {
                sessions.push(pending.session.as_mut());
            }
            if batch.rows.iter().all(is_complete_prompt) {
                model
                    .prefill_batch(&mut sessions, &prompt_slices)
                    .map(|results| {
                        results
                            .into_iter()
                            .map(|result| result.map(PrefillOutcome::Token))
                            .collect()
                    })
            } else {
                model.prefill_chunk_batch(&mut sessions, &prompt_slices, &final_chunks)
            }
        } else {
            None
        };

        if let Some(mut results) = batched {
            if results.len() != members.len() {
                let message = format!(
                    "model returned {} batched prefill results for {} sessions",
                    results.len(),
                    members.len()
                );
                results = (0..members.len())
                    .map(|_| {
                        Err(ModelExecutionError::new(
                            ola_model::ExecutionStage::Prefill,
                            message.clone(),
                        ))
                    })
                    .collect();
            }
            let finished_at = self.now(at);
            outcome.execution_batches.push(ExecutionBatch {
                phase: BatchPhase::Prefill,
                request_ids: members.iter().map(|pending| pending.request_id).collect(),
                mode: BatchExecutionMode::Tensor,
                input_tokens: batch.input_tokens(),
                padding_tokens: batch.padding_tokens(),
                started_at,
                finished_at,
            });
            for (((row, mut pending), prompt), result) in
                batch.rows.iter().zip(members).zip(prompts).zip(results)
            {
                if result.is_ok() {
                    pending.cursor = pending.cursor.saturating_add(prompt.len());
                }
                if result
                    .as_ref()
                    .is_err_and(ModelExecutionError::is_kv_capacity_exhausted)
                {
                    outcome
                        .divergences
                        .push(PlanDivergence::KvCapacityExhausted {
                            request_id: pending.request_id,
                        });
                    if self.recovers_kv_exhaustion(
                        &pending.model,
                        pending.cursor.saturating_add(row.input_tokens()),
                    ) {
                        let event = self.requeue_pending(
                            pending,
                            PreemptionReason::AllocationFailed,
                            finished_at,
                        )?;
                        outcome.indexed_events.push((row.event_index, event));
                        continue;
                    }
                }
                let event = self.finish_prefill(pending, result, finished_at)?;
                outcome.indexed_events.push((row.event_index, event));
            }
        } else {
            for (row, mut pending) in batch.rows.iter().zip(members) {
                let started_at = self.now(at);
                let result = execute_prefill_segment(&mut pending, row.input_tokens());
                let finished_at = self.now(at);
                outcome.execution_batches.push(ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![pending.request_id],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: row.input_tokens(),
                    padding_tokens: 0,
                    started_at,
                    finished_at,
                });
                if result
                    .as_ref()
                    .is_err_and(ModelExecutionError::is_kv_capacity_exhausted)
                {
                    outcome
                        .divergences
                        .push(PlanDivergence::KvCapacityExhausted {
                            request_id: pending.request_id,
                        });
                    if self.recovers_kv_exhaustion(
                        &pending.model,
                        pending.cursor.saturating_add(row.input_tokens()),
                    ) {
                        let event = self.requeue_pending(
                            pending,
                            PreemptionReason::AllocationFailed,
                            finished_at,
                        )?;
                        outcome.indexed_events.push((row.event_index, event));
                        continue;
                    }
                }
                let event = self.finish_prefill(pending, result, finished_at)?;
                outcome.indexed_events.push((row.event_index, event));
            }
        }
        Ok(outcome)
    }

    fn finish_prefill(
        &mut self,
        mut pending: PendingPrefill,
        result: Result<PrefillOutcome, ModelExecutionError>,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                return self.fail_started(pending.request_id, error.to_string(), at);
            }
        };
        if let PrefillOutcome::Pending = outcome {
            if pending.cursor >= pending.prompt_tokens.len() {
                return self.fail_started(
                    pending.request_id,
                    "model left a completed prompt pending without selecting a token".into(),
                    at,
                );
            }
            pending.last_step_at = at;
            let request_id = pending.request_id;
            let metrics = pending.session.metrics();
            pending.reserved_kv_bytes = pending.reserved_kv_bytes.max(metrics.kv_cache_bytes);
            let event = ExecutorEvent::PrefillProgress {
                request_id,
                processed_tokens: pending.cursor,
                total_tokens: pending.prompt_tokens.len(),
                kv_cache_bytes: metrics.kv_cache_bytes,
            };
            self.prefilling.insert(request_id, pending);
            self.prefill_ready.push_back(request_id);
            self.refresh_kv_memory();
            return Ok(event);
        }
        let PrefillOutcome::Token(token) = outcome else {
            unreachable!()
        };
        if pending.cursor != pending.prompt_tokens.len() {
            return self.fail_started(
                pending.request_id,
                "model selected a token before consuming the complete prompt".into(),
                at,
            );
        }
        // A resumed request prefilled prompt and earlier output together, so
        // this token continues its output; its first token was long ago.
        let (prompt, generated, token_times) = match self.resumed.remove(&pending.request_id) {
            Some(resumed) => {
                let mut generated = resumed.generated;
                generated.push(token.id);
                (resumed.prompt, generated, resumed.token_times)
            }
            None => {
                self.runtime.mark_first_token(pending.request_id, at)?;
                (pending.prompt_tokens, vec![token.id], Vec::new())
            }
        };
        let active = ActiveGeneration {
            request_id: pending.request_id,
            model_id: pending.model_id,
            model: pending.model,
            session: pending.session,
            prompt_tokens: prompt.len(),
            prompt,
            generated,
            token_times,
            last_token: token,
            max_new_tokens: pending.max_new_tokens,
            last_step_at: at,
            reserved_kv_bytes: pending.reserved_kv_bytes,
        };
        self.observe_token(active, ExecutionPhase::Prefill, at)
    }

    /// Executes one planned decode invocation. A declared tensor hook that
    /// returns `None` falls back to individual execution.
    fn execute_decode_batch(
        &mut self,
        batch: &PlannedBatch,
        mut members: Vec<ActiveGeneration>,
        at: Timestamp,
    ) -> Result<BatchOutcome, ExecutorError> {
        let mut outcome = BatchOutcome::default();
        let input_tokens = members
            .iter()
            .map(|active| active.last_token.id)
            .collect::<Vec<_>>();
        let model = members[0].model.clone();
        // Only a tensor invocation is timed as a whole; individual
        // invocations below read the clock around each call.
        let started_at = if batch.mode == BatchExecutionMode::Tensor {
            self.now(at)
        } else {
            at
        };
        let batched = if batch.mode == BatchExecutionMode::Tensor {
            let mut sessions: Vec<&mut dyn GenerationSession> = Vec::with_capacity(members.len());
            for active in &mut members {
                sessions.push(active.session.as_mut());
            }
            model.decode_batch(&mut sessions, &input_tokens)
        } else {
            None
        };

        if let Some(mut results) = batched {
            if results.len() != members.len() {
                let message = format!(
                    "model returned {} batched decode results for {} sessions",
                    results.len(),
                    members.len()
                );
                results = (0..members.len())
                    .map(|_| {
                        Err(ModelExecutionError::new(
                            ola_model::ExecutionStage::Decode,
                            message.clone(),
                        ))
                    })
                    .collect();
            }
            let finished_at = self.now(at);
            outcome.execution_batches.push(ExecutionBatch {
                phase: BatchPhase::Decode,
                request_ids: members.iter().map(|active| active.request_id).collect(),
                mode: BatchExecutionMode::Tensor,
                input_tokens: members.len(),
                padding_tokens: 0,
                started_at,
                finished_at,
            });
            for ((row, active), result) in batch.rows.iter().zip(members).zip(results) {
                if result
                    .as_ref()
                    .is_err_and(ModelExecutionError::is_kv_capacity_exhausted)
                {
                    outcome
                        .divergences
                        .push(PlanDivergence::KvCapacityExhausted {
                            request_id: active.request_id,
                        });
                    if self.recovers_kv_exhaustion(
                        &active.model,
                        active.next_position().saturating_add(1),
                    ) {
                        let event = self.requeue_active(
                            active,
                            PreemptionReason::AllocationFailed,
                            finished_at,
                        )?;
                        outcome.indexed_events.push((row.event_index, event));
                        continue;
                    }
                }
                let event = self.finish_decode(active, result, finished_at)?;
                outcome.indexed_events.push((row.event_index, event));
            }
        } else {
            for (row, mut active) in batch.rows.iter().zip(members) {
                let started_at = self.now(at);
                let result = active.session.decode(active.last_token.id);
                let finished_at = self.now(at);
                outcome.execution_batches.push(ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![active.request_id],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                    started_at,
                    finished_at,
                });
                if result
                    .as_ref()
                    .is_err_and(ModelExecutionError::is_kv_capacity_exhausted)
                {
                    outcome
                        .divergences
                        .push(PlanDivergence::KvCapacityExhausted {
                            request_id: active.request_id,
                        });
                    if self.recovers_kv_exhaustion(
                        &active.model,
                        active.next_position().saturating_add(1),
                    ) {
                        let event = self.requeue_active(
                            active,
                            PreemptionReason::AllocationFailed,
                            finished_at,
                        )?;
                        outcome.indexed_events.push((row.event_index, event));
                        continue;
                    }
                }
                let event = self.finish_decode(active, result, finished_at)?;
                outcome.indexed_events.push((row.event_index, event));
            }
        }
        Ok(outcome)
    }

    fn finish_decode(
        &mut self,
        mut active: ActiveGeneration,
        result: Result<GeneratedToken, ModelExecutionError>,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let token = match result {
            Ok(token) => token,
            Err(error) => {
                return self.fail_started(active.request_id, error.to_string(), at);
            }
        };
        active.generated.push(token.id);
        active.last_token = token;
        active.last_step_at = at;
        self.observe_token(active, ExecutionPhase::Decode, at)
    }

    fn observe_token(
        &mut self,
        mut active: ActiveGeneration,
        phase: ExecutionPhase,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let metrics = active.session.metrics();
        active.reserved_kv_bytes = active.reserved_kv_bytes.max(metrics.kv_cache_bytes);
        active.token_times.push(at);
        let finished =
            active.last_token.is_end_of_sequence || active.generated.len() >= active.max_new_tokens;
        let event = ExecutorEvent::Token {
            request_id: active.request_id,
            token: active.last_token,
            generated_tokens: active.generated.len(),
            kv_cache_bytes: metrics.kv_cache_bytes,
            phase,
            finished,
            at,
        };

        if finished {
            let text = match active.model.tokenizer().decode(&active.generated, true) {
                Ok(text) => text,
                Err(error) => return self.fail_started(active.request_id, error.to_string(), at),
            };
            self.runtime.complete(active.request_id, at)?;
            self.results.insert(
                active.request_id,
                ExecutionResult {
                    request_id: active.request_id,
                    token_ids: active.generated,
                    text,
                    metrics,
                    token_times: active.token_times,
                },
            );
            drop(active.session);
            self.refresh_kv_memory();
        } else {
            let request_id = active.request_id;
            self.active.insert(request_id, active);
            self.ready.push_back(request_id);
            self.refresh_kv_memory();
        }
        Ok(event)
    }

    /// Releases a live request's session and returns it to the queue,
    /// keeping its output so it resumes by prefilling prompt and output.
    fn preempt(
        &mut self,
        request_id: RequestId,
        reason: PreemptionReason,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        if let Some(active) = self.active.remove(&request_id) {
            self.ready.retain(|ready| *ready != request_id);
            return self.requeue_active(active, reason, at);
        }
        if let Some(pending) = self.prefilling.remove(&request_id) {
            self.prefill_ready.retain(|ready| *ready != request_id);
            return self.requeue_pending(pending, reason, at);
        }
        // Validation admits only live requests, so this reports the
        // runtime's own transition error.
        self.runtime.preempt(request_id, at)?;
        unreachable!("a preempted request is live")
    }

    /// Returns a decoding request to the queue, dropping its session and KV
    /// blocks but keeping its output so it resumes by prefilling prompt and
    /// output together.
    fn requeue_active(
        &mut self,
        active: ActiveGeneration,
        reason: PreemptionReason,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let ActiveGeneration {
            request_id,
            session,
            prompt,
            generated,
            token_times,
            ..
        } = active;
        drop(session);
        self.runtime.preempt(request_id, at)?;
        self.bump_generation();
        let mut resume_prompt = prompt.clone();
        resume_prompt.extend_from_slice(&generated);
        self.prepared.insert(request_id, Ok(resume_prompt));
        let generated_tokens = generated.len();
        self.resumed.insert(
            request_id,
            Resumed {
                prompt,
                generated,
                token_times,
            },
        );
        self.refresh_kv_memory();
        Ok(ExecutorEvent::Preempted {
            request_id,
            reason,
            generated_tokens,
        })
    }

    /// Returns a request with an unfinished prompt to the queue; its prompt
    /// (with any earlier output) is prefilled again from the start, and an
    /// existing resume record stays as it is.
    fn requeue_pending(
        &mut self,
        pending: PendingPrefill,
        reason: PreemptionReason,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let PendingPrefill {
            request_id,
            session,
            prompt_tokens,
            ..
        } = pending;
        drop(session);
        self.runtime.preempt(request_id, at)?;
        self.bump_generation();
        self.prepared.insert(request_id, Ok(prompt_tokens));
        self.refresh_kv_memory();
        Ok(ExecutorEvent::Preempted {
            request_id,
            reason,
            generated_tokens: self
                .resumed
                .get(&request_id)
                .map_or(0, |resumed| resumed.generated.len()),
        })
    }

    /// Whether a KV-capacity failure on `model` should requeue the request
    /// instead of failing it: the policy must opt in, and the request's
    /// cache after the failed operation (`required_tokens` positions) must
    /// fit the whole pool, so it can run once other work releases blocks. A
    /// request that could never fit fails. The test depends only on the
    /// request, not on which batch peers were requeued first.
    fn recovers_kv_exhaustion(&self, model: &CausalLanguageModel, required_tokens: usize) -> bool {
        self.policy.recovers_kv_exhaustion()
            && model.kv_pool_status().is_some_and(|pool| {
                required_tokens.div_ceil(pool.tokens_per_block.max(1)) <= pool.total_blocks
            })
    }

    fn fail_started(
        &mut self,
        request_id: RequestId,
        message: String,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        self.runtime.fail(request_id, message.clone(), at)?;
        self.resumed.remove(&request_id);
        self.refresh_kv_memory();
        Ok(ExecutorEvent::Failed {
            request_id,
            message,
        })
    }

    fn refresh_kv_memory(&mut self) {
        self.runtime.state_mut().memory.kv_cache_bytes = self
            .active
            .values()
            .map(|active| active.session.metrics().kv_cache_bytes)
            .chain(
                self.prefilling
                    .values()
                    .map(|pending| pending.session.metrics().kv_cache_bytes),
            )
            .sum();
    }

    fn validate_iteration_time(&self, at: Timestamp) -> Result<(), ExecutorError> {
        if let Some(active) = self.active.values().find(|active| at < active.last_step_at) {
            return Err(RuntimeError::TimestampRegression {
                request: active.request_id,
                event: "executor step",
                previous: active.last_step_at,
                observed: at,
            }
            .into());
        }
        if let Some(pending) = self
            .prefilling
            .values()
            .find(|pending| at < pending.last_step_at)
        {
            return Err(RuntimeError::TimestampRegression {
                request: pending.request_id,
                event: "executor prefill step",
                previous: pending.last_step_at,
                observed: at,
            }
            .into());
        }
        Ok(())
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn live_request_count(&self) -> usize {
        self.active.len().saturating_add(self.prefilling.len())
    }
}

/// Planned rows whose request left the live set without being predicted.
fn unplanned_releases(plan: &ExecutionPlan, outcome: &BatchOutcome) -> Vec<PlanDivergence> {
    let planned = plan
        .batches
        .iter()
        .flat_map(|batch| &batch.rows)
        .map(|row| row.request_id)
        .collect::<BTreeSet<_>>();
    let excluded = plan
        .predicted_releases
        .iter()
        .copied()
        .chain(
            outcome
                .divergences
                .iter()
                .filter_map(|divergence| match divergence {
                    PlanDivergence::SessionCreationFailed { request_id } => Some(*request_id),
                    _ => None,
                }),
        )
        .collect::<BTreeSet<_>>();
    outcome
        .indexed_events
        .iter()
        .filter_map(|(_, event)| {
            let (request_id, reason) = match event {
                ExecutorEvent::Token {
                    request_id,
                    token,
                    finished: true,
                    ..
                } => (
                    *request_id,
                    if token.is_end_of_sequence {
                        ReleaseReason::EndOfSequence
                    } else {
                        ReleaseReason::OutputLimit
                    },
                ),
                ExecutorEvent::Failed { request_id, .. } => (*request_id, ReleaseReason::Failure),
                _ => return None,
            };
            (planned.contains(&request_id) && !excluded.contains(&request_id))
                .then_some(PlanDivergence::UnplannedRelease { request_id, reason })
        })
        .collect()
}

const fn execution_phase(phase: BatchPhase) -> ExecutionPhase {
    match phase {
        BatchPhase::Prefill => ExecutionPhase::Prefill,
        BatchPhase::Decode => ExecutionPhase::Decode,
    }
}

fn execute_prefill_segment(
    pending: &mut PendingPrefill,
    chunk_len: usize,
) -> Result<PrefillOutcome, ModelExecutionError> {
    debug_assert!(chunk_len > 0);
    let end = pending.cursor.saturating_add(chunk_len);
    debug_assert!(end <= pending.prompt_tokens.len());
    let is_final = end == pending.prompt_tokens.len();
    let prompt_chunk = pending.prompt_tokens[pending.cursor..end].to_vec();
    let result = if pending.cursor == 0 && is_final {
        pending
            .session
            .prefill(&prompt_chunk)
            .map(PrefillOutcome::Token)
    } else {
        pending.session.prefill_chunk(&prompt_chunk, is_final)
    };
    if result.is_ok() {
        pending.cursor = end;
    }
    result
}

fn prepare_prompt(
    model: &CausalLanguageModel,
    request: &Request,
) -> Result<Vec<u32>, ModelExecutionError> {
    match request.spec.inputs.as_slice() {
        [Input::Text(text)] => model.tokenizer().encode(text, true),
        [Input::Tokens(tokens)] if !tokens.is_empty() => Ok(tokens.clone()),
        [Input::Tokens(_)] => Err(ModelExecutionError::new(
            ola_model::ExecutionStage::Tokenize,
            "token input cannot be empty",
        )),
        [_] => Err(ModelExecutionError::new(
            ola_model::ExecutionStage::Tokenize,
            "the initial executor supports text or token input only",
        )),
        _ => Err(ModelExecutionError::new(
            ola_model::ExecutionStage::Tokenize,
            "the initial executor requires exactly one input",
        )),
    }
}

#[cfg(test)]
mod tests {
    use ola_core::GenerationConfig;
    use ola_model::{
        BatchCapabilities, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
        KvCacheProfile, SessionMetrics, Tokenizer,
    };

    use super::*;
    use crate::{PlannedRow, RequestStatus};

    #[derive(Clone, Copy)]
    struct NumericTokenizer;

    impl Tokenizer for NumericTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> Result<Vec<u32>, ModelExecutionError> {
            text.split_whitespace()
                .map(|piece| {
                    piece.parse().map_err(|_| {
                        ModelExecutionError::new(ExecutionStage::Tokenize, "expected integers")
                    })
                })
                .collect()
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> Result<String, ModelExecutionError> {
            Ok(token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    struct IncrementModel;

    impl CausalModel for IncrementModel {
        fn supports_chunked_prefill(&self) -> bool {
            true
        }

        fn batch_capabilities(&self) -> BatchCapabilities {
            BatchCapabilities::ALL
        }

        fn create_session(
            &self,
            _generation: GenerationConfig,
        ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
            Ok(Box::new(IncrementSession::default()))
        }

        fn prefill_batch(
            &self,
            sessions: &mut [&mut dyn GenerationSession],
            prompt_tokens: &[&[u32]],
        ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
            Some(
                sessions
                    .iter_mut()
                    .zip(prompt_tokens.iter().copied())
                    .map(|(session, prompt)| session.prefill(prompt))
                    .collect(),
            )
        }

        fn prefill_chunk_batch(
            &self,
            sessions: &mut [&mut dyn GenerationSession],
            prompt_chunks: &[&[u32]],
            is_final: &[bool],
        ) -> Option<Vec<Result<PrefillOutcome, ModelExecutionError>>> {
            Some(
                sessions
                    .iter_mut()
                    .zip(prompt_chunks.iter().copied())
                    .zip(is_final.iter().copied())
                    .map(|((session, chunk), is_final)| session.prefill_chunk(chunk, is_final))
                    .collect(),
            )
        }

        fn decode_batch(
            &self,
            sessions: &mut [&mut dyn GenerationSession],
            input_tokens: &[u32],
        ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
            Some(
                sessions
                    .iter_mut()
                    .zip(input_tokens.iter().copied())
                    .map(|(session, input_token)| session.decode(input_token))
                    .collect(),
            )
        }
    }

    struct IndividualIncrementModel;

    impl CausalModel for IndividualIncrementModel {
        fn create_session(
            &self,
            _generation: GenerationConfig,
        ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
            Ok(Box::new(IncrementSession::default()))
        }
    }

    #[derive(Default)]
    struct IncrementSession {
        metrics: SessionMetrics,
        ready: bool,
    }

    impl GenerationSession for IncrementSession {
        fn prefill(
            &mut self,
            prompt_tokens: &[u32],
        ) -> Result<GeneratedToken, ModelExecutionError> {
            let last = prompt_tokens.last().copied().ok_or_else(|| {
                ModelExecutionError::new(ExecutionStage::Prefill, "prompt cannot be empty")
            })?;
            self.ready = true;
            self.metrics.prompt_tokens = prompt_tokens.len();
            self.metrics.generated_tokens = 1;
            self.metrics.kv_cache_bytes = prompt_tokens.len() as u64 * 8;
            Ok(GeneratedToken {
                id: last + 1,
                is_end_of_sequence: false,
            })
        }

        fn prefill_chunk(
            &mut self,
            prompt_tokens: &[u32],
            is_final: bool,
        ) -> Result<PrefillOutcome, ModelExecutionError> {
            let last = prompt_tokens.last().copied().ok_or_else(|| {
                ModelExecutionError::new(ExecutionStage::Prefill, "chunk cannot be empty")
            })?;
            if self.ready {
                return Err(ModelExecutionError::new(
                    ExecutionStage::Prefill,
                    "prefill already completed",
                ));
            }
            self.metrics.prompt_tokens += prompt_tokens.len();
            self.metrics.kv_cache_bytes += prompt_tokens.len() as u64 * 8;
            if is_final {
                self.ready = true;
                self.metrics.generated_tokens = 1;
                Ok(PrefillOutcome::Token(GeneratedToken {
                    id: last + 1,
                    is_end_of_sequence: false,
                }))
            } else {
                Ok(PrefillOutcome::Pending)
            }
        }

        fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
            if !self.ready {
                return Err(ModelExecutionError::new(
                    ExecutionStage::Decode,
                    "decode requires prefill",
                ));
            }
            self.metrics.generated_tokens += 1;
            self.metrics.kv_cache_bytes += 8;
            Ok(GeneratedToken {
                id: input_token + 1,
                is_end_of_sequence: input_token + 1 == TEST_EOS_TOKEN,
            })
        }

        fn metrics(&self) -> SessionMetrics {
            self.metrics
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    /// Decoding this token ends a test sequence early.
    const TEST_EOS_TOKEN: u32 = 999;

    fn artifact() -> CausalLanguageModel {
        CausalLanguageModel::new(NumericTokenizer, IncrementModel)
    }

    fn individual_artifact() -> CausalLanguageModel {
        CausalLanguageModel::new(NumericTokenizer, IndividualIncrementModel)
    }

    /// Declares every batch hook but implements none, and rejects sessions
    /// with a non-zero temperature, so execution diverges from plans.
    struct DivergentModel;

    impl CausalModel for DivergentModel {
        fn batch_capabilities(&self) -> BatchCapabilities {
            BatchCapabilities::ALL
        }

        fn create_session(
            &self,
            generation: GenerationConfig,
        ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
            if generation.temperature != 0.0 {
                return Err(ModelExecutionError::new(
                    ExecutionStage::SessionCreation,
                    "sampling is unsupported",
                ));
            }
            Ok(Box::new(IncrementSession::default()))
        }
    }

    fn profiled_artifact(capacity_bytes: u64) -> CausalLanguageModel {
        let profile = KvCacheProfile::new(8, 4, capacity_bytes).unwrap();
        artifact().with_kv_cache_profile(profile)
    }

    /// Runs one iteration of a single-request test and returns its only event.
    fn step_one(executor: &mut Executor, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        let mut events = executor.step_iteration(at)?.events;
        assert_eq!(events.len(), 1, "single-request tests expect one event");
        Ok(events.remove(0))
    }

    /// Scheduled rows as `(request, phase, input tokens)` in event order.
    fn scheduled(plan: &ExecutionPlan) -> Vec<(RequestId, PlannedPhase, usize)> {
        let mut rows = plan
            .batches
            .iter()
            .flat_map(|batch| {
                batch.rows.iter().map(move |row| {
                    (
                        row.event_index,
                        row.request_id,
                        batch.phase,
                        row.input_tokens(),
                    )
                })
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row.0);
        rows.into_iter()
            .map(|(_, request_id, phase, tokens)| (request_id, phase, tokens))
            .collect()
    }

    fn request(prompt: &str, max_new_tokens: usize) -> RequestSpec {
        let mut request = RequestSpec::text("tiny", prompt);
        request.generation = GenerationConfig {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        };
        request
    }

    #[test]
    fn drives_prefill_and_decode_to_a_retained_result() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("1 2", 3), Timestamp::from_micros(10))
            .unwrap();

        let first = step_one(&mut executor, Timestamp::from_micros(20)).unwrap();
        assert!(matches!(
            first,
            ExecutorEvent::Token {
                token: GeneratedToken { id: 3, .. },
                generated_tokens: 1,
                kv_cache_bytes: 16,
                finished: false,
                ..
            }
        ));
        assert_eq!(executor.state().memory.kv_cache_bytes, 16);
        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Running
        );

        step_one(&mut executor, Timestamp::from_micros(30)).unwrap();
        let last = step_one(&mut executor, Timestamp::from_micros(40)).unwrap();
        assert!(matches!(
            last,
            ExecutorEvent::Token {
                token: GeneratedToken { id: 5, .. },
                generated_tokens: 3,
                kv_cache_bytes: 32,
                finished: true,
                ..
            }
        ));
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Completed
        );
        assert_eq!(
            executor.result(id),
            Some(&ExecutionResult {
                request_id: id,
                token_ids: vec![3, 4, 5],
                text: "3 4 5".into(),
                metrics: SessionMetrics {
                    prompt_tokens: 2,
                    generated_tokens: 3,
                    kv_cache_bytes: 32,
                },
                token_times: [20, 30, 40].map(Timestamp::from_micros).to_vec(),
            })
        );
    }

    #[test]
    fn chunked_prefill_advances_a_bounded_cursor_before_emitting_a_token() {
        let mut executor = Executor::default().with_prefill_chunk_size(2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("1 2 3 4 5", 1), Timestamp::ZERO)
            .unwrap();

        let first = executor.step_iteration(Timestamp::from_micros(1)).unwrap();
        assert_eq!(scheduled(&first.execution_plan)[0].2, 2);
        assert!(matches!(
            first.events.as_slice(),
            [ExecutorEvent::PrefillProgress {
                request_id,
                processed_tokens: 2,
                total_tokens: 5,
                kv_cache_bytes: 16,
            }] if *request_id == id
        ));
        assert_eq!(first.execution_batches[0].input_tokens, 2);
        assert_eq!(first.active_requests, 1);
        assert_eq!(executor.state().memory.kv_cache_bytes, 16);

        let second = executor.step_iteration(Timestamp::from_micros(2)).unwrap();
        assert!(matches!(
            second.events.as_slice(),
            [ExecutorEvent::PrefillProgress {
                processed_tokens: 4,
                total_tokens: 5,
                ..
            }]
        ));

        let third = executor.step_iteration(Timestamp::from_micros(3)).unwrap();
        assert_eq!(scheduled(&third.execution_plan)[0].2, 1);
        assert!(matches!(
            third.events.as_slice(),
            [ExecutorEvent::Token {
                token: GeneratedToken { id: 6, .. },
                generated_tokens: 1,
                finished: true,
                phase: ExecutionPhase::Prefill,
                ..
            }]
        ));
        assert_eq!(executor.result(id).unwrap().metrics.prompt_tokens, 5);
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
    }

    #[test]
    fn results_can_be_taken_and_terminal_requests_forgotten() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let done = executor.submit(request("1 2", 1), Timestamp::ZERO).unwrap();
        let waiting = executor.submit(request("1 2", 1), Timestamp::ZERO).unwrap();
        executor.step_iteration(Timestamp::from_micros(1)).unwrap();

        assert!(matches!(
            executor.forget(waiting),
            Err(ExecutorError::Runtime(
                RuntimeError::InvalidTransition { .. }
            ))
        ));
        let taken = executor.take_result(done).unwrap();
        assert_eq!(taken.token_ids, vec![3]);
        assert_eq!(executor.result(done), None);
        let (record, result) = executor.forget(done).unwrap();
        assert_eq!(record.status, RequestStatus::Completed);
        assert_eq!(result, None);
        assert!(executor.state().request(done).is_none());

        executor.cancel(waiting, Timestamp::from_micros(2)).unwrap();
        let (record, _) = executor.forget(waiting).unwrap();
        assert_eq!(record.status, RequestStatus::Cancelled);
    }

    /// Advances by 10 µs on every reading, so each read is distinguishable.
    #[derive(Default)]
    struct TickingClock(std::sync::atomic::AtomicU64);

    impl TickingClock {
        fn starting_at(micros: u64) -> Arc<Self> {
            Arc::new(Self(std::sync::atomic::AtomicU64::new(micros)))
        }

        fn peek(&self) -> Timestamp {
            Timestamp::from_micros(self.0.load(std::sync::atomic::Ordering::SeqCst))
        }
    }

    impl Clock for TickingClock {
        fn now(&self) -> Timestamp {
            let previous = self.0.fetch_add(10, std::sync::atomic::Ordering::SeqCst);
            Timestamp::from_micros(previous + 10)
        }
    }

    #[test]
    fn a_clock_stamps_work_when_its_invocation_finishes() {
        let clock = TickingClock::starting_at(1_000);
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_clock(clock.clone());
        executor
            .register_model(
                ModelDescriptor::language("tiny", "increment"),
                individual_artifact(),
            )
            .unwrap();
        let first = executor.submit(request("1 2", 2), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("5", 2), Timestamp::ZERO).unwrap();

        let admitted = executor.step_iteration(clock.peek()).unwrap();
        // Individual invocations each get their own interval, in order.
        let [a, b] = admitted.execution_batches.as_slice() else {
            panic!("expected two individual prefills");
        };
        assert_eq!(
            (a.started_at, a.finished_at, b.started_at, b.finished_at),
            (
                Timestamp::from_micros(1_010),
                Timestamp::from_micros(1_020),
                Timestamp::from_micros(1_030),
                Timestamp::from_micros(1_040),
            )
        );
        assert_eq!(a.latency_micros(), 10);
        let token_times = admitted
            .events
            .iter()
            .map(|event| match event {
                ExecutorEvent::Token { request_id, at, .. } => (*request_id, *at),
                other => panic!("unexpected {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            token_times,
            vec![(first, a.finished_at), (second, b.finished_at)]
        );
        let timing = executor.state().request(second).unwrap().timing;
        assert_eq!(timing.started_at, Some(Timestamp::from_micros(1_000)));
        assert_eq!(timing.first_token_at, Some(b.finished_at));

        let decoded = executor.step_iteration(clock.peek()).unwrap();
        let finished = decoded.execution_batches[0].finished_at;
        let result = executor.result(first).unwrap();
        assert_eq!(result.token_times, vec![a.finished_at, finished]);
        assert_eq!(
            executor.state().request(first).unwrap().timing.completed_at,
            Some(finished)
        );
    }

    #[test]
    fn iterations_and_their_plans_share_consecutive_ids() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        executor.submit(request("1 2", 5), Timestamp::ZERO).unwrap();

        let ids = (1..=3)
            .map(|step| {
                let iteration = executor
                    .step_iteration(Timestamp::from_micros(step))
                    .unwrap();
                assert_eq!(iteration.execution_plan.iteration, iteration.id);
                iteration.id.get()
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, [1, 2, 3]);

        // A rejected step (time went backwards with a live session)
        // consumes no ID.
        assert!(executor.step_iteration(Timestamp::ZERO).is_err());
        let next = executor.step_iteration(Timestamp::from_micros(4)).unwrap();
        assert_eq!(next.id.get(), 4);
    }

    #[test]
    fn without_a_clock_every_time_is_the_planning_time() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor.submit(request("1 2", 2), Timestamp::ZERO).unwrap();
        let at = Timestamp::from_micros(50);
        let iteration = executor.step_iteration(at).unwrap();

        let batch = &iteration.execution_batches[0];
        assert_eq!((batch.started_at, batch.finished_at), (at, at));
        assert!(matches!(
            iteration.events[0],
            ExecutorEvent::Token { at: token_at, .. } if token_at == at
        ));
        assert_eq!(
            executor.state().request(id).unwrap().timing.first_token_at,
            Some(at)
        );
    }

    #[test]
    fn chunked_prefill_and_decode_share_an_iteration_budget() {
        let policy = crate::FifoTokenBudgetPolicy::new(3).unwrap();
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_batch(policy))
            .with_prefill_chunk_size(2)
            .unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let short = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let long = executor
            .submit(request("1 2 3 4", 1), Timestamp::ZERO)
            .unwrap();

        let first = executor.step_iteration(Timestamp::from_micros(1)).unwrap();
        assert!(
            matches!(first.events[0], ExecutorEvent::Token { request_id, .. } if request_id == short)
        );
        assert!(
            matches!(first.events[1], ExecutorEvent::PrefillProgress { request_id, .. } if request_id == long)
        );

        let second = executor.step_iteration(Timestamp::from_micros(2)).unwrap();
        assert_eq!(
            scheduled(&second.execution_plan)
                .into_iter()
                .map(|(request_id, _, tokens)| (request_id, tokens))
                .collect::<Vec<_>>(),
            vec![(short, 1), (long, 2)]
        );
        assert!(
            matches!(second.events[0], ExecutorEvent::Token { request_id, .. } if request_id == short)
        );
        assert!(
            matches!(second.events[1], ExecutorEvent::Token { request_id, finished: true, .. } if request_id == long)
        );
    }

    #[test]
    fn resumable_chunks_tensor_batch_at_unequal_absolute_positions() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_prefill_chunk_size(2)
            .unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor
            .submit(request("1 2 3 4 5", 1), Timestamp::ZERO)
            .unwrap();
        executor.step_iteration(Timestamp::from_micros(1)).unwrap();
        let second = executor
            .submit(request("5 6 7 8 9", 1), Timestamp::from_micros(1))
            .unwrap();
        executor.step_iteration(Timestamp::from_micros(2)).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(3)).unwrap();

        assert_eq!(iteration.execution_batches.len(), 1);
        assert_eq!(
            iteration.execution_batches[0],
            ExecutionBatch {
                phase: BatchPhase::Prefill,
                request_ids: vec![first, second],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 3,
                padding_tokens: 1,
                started_at: Timestamp::from_micros(3),
                finished_at: Timestamp::from_micros(3),
            }
        );
        assert!(matches!(
            iteration.events[0],
            ExecutorEvent::Token {
                request_id,
                finished: true,
                ..
            } if request_id == first
        ));
        assert!(matches!(
            iteration.events[1],
            ExecutorEvent::PrefillProgress {
                request_id,
                processed_tokens: 4,
                ..
            } if request_id == second
        ));
    }

    #[test]
    fn cancelling_chunked_prefill_releases_its_live_session_memory() {
        let mut executor = Executor::default().with_prefill_chunk_size(2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("1 2 3 4", 2), Timestamp::ZERO)
            .unwrap();
        executor.step_iteration(Timestamp::from_micros(1)).unwrap();
        assert_eq!(executor.state().memory.kv_cache_bytes, 16);

        executor.cancel(id, Timestamp::from_micros(2)).unwrap();

        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
        assert_eq!(executor.active_request_count(), 0);
        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Cancelled
        );
    }

    #[test]
    fn models_without_resumable_prefill_keep_the_complete_prompt_path() {
        let mut executor = Executor::default().with_prefill_chunk_size(2).unwrap();
        executor
            .register_model(
                ModelDescriptor::language("tiny", "increment"),
                individual_artifact(),
            )
            .unwrap();
        executor
            .submit(request("1 2 3 4", 1), Timestamp::ZERO)
            .unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(1)).unwrap();

        assert_eq!(scheduled(&iteration.execution_plan)[0].2, 4);
        assert_eq!(iteration.execution_batches[0].input_tokens, 4);
        assert!(matches!(
            iteration.events.as_slice(),
            [ExecutorEvent::Token { finished: true, .. }]
        ));
    }

    #[test]
    fn rejects_a_zero_prefill_chunk_size() {
        assert!(matches!(
            Executor::default().with_prefill_chunk_size(0),
            Err(ExecutorError::InvalidPrefillChunkSize(0))
        ));
    }

    #[test]
    fn execution_failure_becomes_terminal_without_leaking_cache_accounting() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("not-a-number", 2), Timestamp::ZERO)
            .unwrap();

        let event = step_one(&mut executor, Timestamp::from_micros(1)).unwrap();

        assert!(matches!(event, ExecutorEvent::Failed { request_id, .. } if request_id == id));
        assert!(matches!(
            &executor.state().request(id).unwrap().status,
            RequestStatus::Failed(message) if message.contains("expected integers")
        ));
        assert!(!executor.has_active_request());
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
    }

    #[test]
    fn cancelling_active_work_drops_session_memory() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor.submit(request("1 2", 3), Timestamp::ZERO).unwrap();
        step_one(&mut executor, Timestamp::from_micros(1)).unwrap();

        executor.cancel(id, Timestamp::from_micros(2)).unwrap();

        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Cancelled
        );
        assert!(!executor.has_active_request());
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
    }

    #[test]
    fn rejects_a_regressive_step_without_dropping_active_state() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("1 2", 2), Timestamp::from_micros(10))
            .unwrap();
        step_one(&mut executor, Timestamp::from_micros(20)).unwrap();

        let error = step_one(&mut executor, Timestamp::from_micros(19)).unwrap_err();

        assert_eq!(
            error,
            ExecutorError::Runtime(RuntimeError::TimestampRegression {
                request: id,
                event: "executor step",
                previous: Timestamp::from_micros(20),
                observed: Timestamp::from_micros(19),
            })
        );
        assert!(executor.has_active_request());
        assert_eq!(executor.state().memory.kv_cache_bytes, 16);
        assert!(matches!(
            step_one(&mut executor, Timestamp::from_micros(30)).unwrap(),
            ExecutorEvent::Token { finished: true, .. }
        ));
    }

    #[test]
    fn iteration_advances_live_sessions_and_refills_freed_slots() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 2), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        let third = executor.submit(request("20", 1), Timestamp::ZERO).unwrap();

        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();
        assert!(matches!(
            admitted.events.as_slice(),
            [
                ExecutorEvent::Token {
                    request_id: first_id,
                    phase: ExecutionPhase::Prefill,
                    ..
                },
                ExecutorEvent::Token {
                    request_id: second_id,
                    phase: ExecutionPhase::Prefill,
                    ..
                }
            ] if *first_id == first && *second_id == second
        ));
        assert_eq!(admitted.active_requests, 2);
        assert_eq!(admitted.queued_requests, 1);
        assert_eq!(admitted.kv_cache_bytes, 16);
        assert_eq!(
            admitted.execution_batches,
            vec![ExecutionBatch {
                phase: BatchPhase::Prefill,
                request_ids: vec![first, second],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 2,
                padding_tokens: 0,
                started_at: Timestamp::from_micros(10),
                finished_at: Timestamp::from_micros(10),
            }]
        );

        let refilled = executor.step_iteration(Timestamp::from_micros(20)).unwrap();
        assert!(matches!(
            refilled.events.as_slice(),
            [
                ExecutorEvent::Token {
                    request_id: first_id,
                    phase: ExecutionPhase::Decode,
                    finished: true,
                    ..
                },
                ExecutorEvent::Token {
                    request_id: second_id,
                    phase: ExecutionPhase::Decode,
                    finished: false,
                    ..
                },
                ExecutorEvent::Token {
                    request_id: third_id,
                    phase: ExecutionPhase::Prefill,
                    finished: true,
                    ..
                }
            ] if *first_id == first && *second_id == second && *third_id == third
        ));
        assert_eq!(refilled.active_requests, 1);
        assert_eq!(refilled.queued_requests, 0);
        assert_eq!(refilled.kv_cache_bytes, 16);
        assert_eq!(
            refilled.execution_batches,
            vec![
                ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![first, second],
                    mode: BatchExecutionMode::Tensor,
                    input_tokens: 2,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(20),
                    finished_at: Timestamp::from_micros(20),
                },
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![third],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(20),
                    finished_at: Timestamp::from_micros(20),
                },
            ]
        );
        assert_eq!(
            executor.state().active_ids().collect::<Vec<_>>(),
            vec![second]
        );

        let completed = executor.step_iteration(Timestamp::from_micros(30)).unwrap();
        assert!(matches!(
            completed.events.as_slice(),
            [ExecutorEvent::Token {
                request_id,
                phase: ExecutionPhase::Decode,
                finished: true,
                ..
            }] if *request_id == second
        ));
        assert_eq!(completed.active_requests, 0);
        assert_eq!(completed.kv_cache_bytes, 0);
        assert_eq!(executor.result(first).unwrap().token_ids, vec![2, 3]);
        assert_eq!(executor.result(second).unwrap().token_ids, vec![11, 12, 13]);
        assert_eq!(executor.result(third).unwrap().token_ids, vec![21]);
    }

    #[test]
    fn iteration_falls_back_to_individual_execution_when_model_has_no_batch_path() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(
                ModelDescriptor::language("tiny", "individual-increment"),
                individual_artifact(),
            )
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(
            admitted.execution_batches,
            vec![
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![first],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(10),
                    finished_at: Timestamp::from_micros(10),
                },
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![second],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(10),
                    finished_at: Timestamp::from_micros(10),
                },
            ]
        );

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();

        // Without declared batch capabilities the plan itself is individual.
        assert_eq!(decoded.execution_plan.batches.len(), 2);
        assert!(
            decoded
                .execution_plan
                .batches
                .iter()
                .all(|batch| batch.mode == BatchExecutionMode::Individual)
        );
        assert_eq!(
            decoded.execution_batches,
            vec![
                ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![first],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(20),
                    finished_at: Timestamp::from_micros(20),
                },
                ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![second],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(20),
                    finished_at: Timestamp::from_micros(20),
                },
            ]
        );
    }

    #[test]
    fn cancellation_releases_only_the_selected_live_session() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        executor.cancel(first, Timestamp::from_micros(11)).unwrap();

        assert_eq!(executor.active_request_count(), 1);
        assert_eq!(executor.state().memory.kv_cache_bytes, 8);
        assert_eq!(
            executor.state().request(first).unwrap().status,
            RequestStatus::Cancelled
        );
        assert_eq!(
            executor.state().active_ids().collect::<Vec<_>>(),
            vec![second]
        );
        assert!(matches!(
            executor
                .step_iteration(Timestamp::from_micros(20))
                .unwrap()
                .events
                .as_slice(),
            [ExecutorEvent::Token { request_id, .. }] if *request_id == second
        ));
    }

    #[test]
    fn rejects_a_zero_active_request_limit() {
        let result = Executor::with_max_active_requests(Runtime::default(), 0);

        assert!(matches!(
            result,
            Err(ExecutorError::InvalidActiveRequestLimit(0))
        ));
    }

    #[test]
    fn token_budget_records_selected_and_deferred_iteration_work() {
        let policy = crate::FifoTokenBudgetPolicy::new(2).unwrap();
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_batch(policy));
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        let third = executor.submit(request("20", 1), Timestamp::ZERO).unwrap();

        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(admitted.execution_plan.estimates().input_tokens, 2);
        assert_eq!(
            scheduled(&admitted.execution_plan),
            vec![
                (first, PlannedPhase::Prefill, 1),
                (second, PlannedPhase::Prefill, 1),
            ]
        );
        assert_eq!(
            admitted.execution_plan.deferrals,
            vec![PlannedDeferral::Budget {
                request_id: third,
                phase: BatchPhase::Prefill,
                token_cost: 1,
                event_index: 2,
            }]
        );
        assert!(matches!(
            admitted.events.last(),
            Some(ExecutorEvent::BatchDeferred {
                request_id,
                phase: ExecutionPhase::Prefill,
                token_cost: 1,
            }) if *request_id == third
        ));
        assert_eq!(admitted.active_requests, 2);
        assert_eq!(admitted.queued_requests, 1);

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();
        assert_eq!(decoded.execution_plan.estimates().input_tokens, 2);
        assert!(
            scheduled(&decoded.execution_plan)
                .iter()
                .all(|(_, phase, _)| *phase == PlannedPhase::Decode)
        );
        assert_eq!(decoded.execution_plan.deferrals[0].request_id(), third);
        assert_eq!(
            decoded.execution_batches,
            vec![ExecutionBatch {
                phase: BatchPhase::Decode,
                request_ids: vec![first, second],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 2,
                padding_tokens: 0,
                started_at: Timestamp::from_micros(20),
                finished_at: Timestamp::from_micros(20),
            }]
        );
        assert_eq!(
            executor.state().request(third).unwrap().status,
            RequestStatus::Queued
        );
    }

    #[test]
    fn execution_plan_describes_live_decodes_before_they_run() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 2), Timestamp::ZERO).unwrap();
        executor.step_iteration(Timestamp::from_micros(10)).unwrap();
        let generation = executor.generation();

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();

        let plan = &decoded.execution_plan;
        assert_eq!(plan.generation, generation);
        assert!(executor.generation() > generation);
        assert_eq!(
            plan.batches,
            vec![PlannedBatch {
                phase: PlannedPhase::Decode,
                model_id: "tiny".into(),
                mode: BatchExecutionMode::Tensor,
                rows: vec![
                    PlannedRow {
                        request_id: first,
                        event_index: 0,
                        positions: 1..2,
                        final_prompt_chunk: false,
                    },
                    PlannedRow {
                        request_id: second,
                        event_index: 1,
                        positions: 1..2,
                        final_prompt_chunk: false,
                    },
                ],
            }]
        );
        // The second request's output limit is two, so this decode ends it.
        assert_eq!(plan.predicted_releases, vec![second]);
        assert!(matches!(
            decoded.events.as_slice(),
            [
                ExecutorEvent::Token { finished: false, .. },
                ExecutorEvent::Token { request_id, finished: true, .. },
            ] if *request_id == second
        ));
        assert_eq!(plan.estimates().input_tokens, 2);
        assert_eq!(
            decoded.execution_batches[0].request_ids,
            plan.batches[0]
                .rows
                .iter()
                .map(|row| row.request_id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn execution_plan_advances_continued_prefill_cursors() {
        let mut executor = Executor::default().with_prefill_chunk_size(1).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request_id = executor
            .submit(request("1 2 3", 2), Timestamp::ZERO)
            .unwrap();
        executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        for (at, positions, final_prompt_chunk) in [(20, 1..2, false), (30, 2..3, true)] {
            let iteration = executor.step_iteration(Timestamp::from_micros(at)).unwrap();
            assert_eq!(
                iteration.execution_plan.batches,
                vec![PlannedBatch {
                    phase: PlannedPhase::ContinuedPrefill,
                    model_id: "tiny".into(),
                    mode: BatchExecutionMode::Individual,
                    rows: vec![PlannedRow {
                        request_id,
                        event_index: 0,
                        positions,
                        final_prompt_chunk,
                    }],
                }]
            );
        }
    }

    #[test]
    fn execution_plan_records_decode_budget_deferrals() {
        struct FirstOnly;

        impl crate::BatchPolicy for FirstOnly {
            fn descriptor(&self) -> crate::PolicyDescriptor {
                crate::PolicyDescriptor::new("first-only")
            }

            fn select(&self, context: &crate::BatchContext) -> Vec<RequestId> {
                context
                    .candidates
                    .first()
                    .map(|candidate| candidate.request_id)
                    .into_iter()
                    .collect()
            }
        }

        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_batch(FirstOnly));
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();

        assert_eq!(decoded.execution_plan.batches.len(), 1);
        assert_eq!(decoded.execution_plan.batches[0].rows[0].request_id, first);
        assert_eq!(
            decoded.execution_plan.deferrals,
            vec![PlannedDeferral::Budget {
                request_id: second,
                phase: BatchPhase::Decode,
                token_cost: 1,
                event_index: 1,
            }]
        );
        assert!(matches!(
            decoded.events.as_slice(),
            [
                ExecutorEvent::Token { .. },
                ExecutorEvent::BatchDeferred { request_id, phase: ExecutionPhase::Decode, .. },
            ] if *request_id == second
        ));
    }

    #[test]
    fn eos_freed_slots_refill_on_the_next_iteration() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 1).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let early = executor.submit(request("997", 5), Timestamp::ZERO).unwrap();
        let waiting = executor.submit(request("1", 2), Timestamp::ZERO).unwrap();
        executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        // EOS cannot be known when the plan is built, so the freed slot is
        // not reused in the same iteration and the wait is recorded.
        let finished = executor.step_iteration(Timestamp::from_micros(20)).unwrap();
        assert!(matches!(
            finished.events.as_slice(),
            [ExecutorEvent::Token { request_id, finished: true, .. }] if *request_id == early
        ));
        assert!(finished.execution_plan.predicted_releases.is_empty());
        assert_eq!(
            finished.execution_plan.deferrals,
            vec![PlannedDeferral::Slots {
                request_id: waiting
            }]
        );
        assert_eq!(finished.queued_requests, 1);
        assert_eq!(
            finished.divergences,
            vec![PlanDivergence::UnplannedRelease {
                request_id: early,
                reason: ReleaseReason::EndOfSequence,
            }]
        );

        let refilled = executor.step_iteration(Timestamp::from_micros(30)).unwrap();
        assert!(matches!(
            refilled.events.as_slice(),
            [ExecutorEvent::Token { request_id, phase: ExecutionPhase::Prefill, .. }]
                if *request_id == waiting
        ));
    }

    #[test]
    fn divergences_record_fallbacks_and_session_failures() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 3).unwrap();
        executor
            .register_model(
                ModelDescriptor::language("tiny", "divergent"),
                CausalLanguageModel::new(NumericTokenizer, DivergentModel),
            )
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let mut sampled = request("5", 3);
        sampled.generation.temperature = 1.0;
        let sampled = executor.submit(sampled, Timestamp::ZERO).unwrap();
        let third = executor.submit(request("9", 3), Timestamp::ZERO).unwrap();

        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(admitted.execution_plan.batches.len(), 1);
        assert_eq!(
            admitted.execution_plan.batches[0].mode,
            BatchExecutionMode::Tensor
        );
        assert_eq!(
            admitted.divergences,
            vec![
                PlanDivergence::SessionCreationFailed {
                    request_id: sampled
                },
                PlanDivergence::ModeFallback { batch_index: 0 },
            ]
        );
        assert_eq!(executor.active_request_count(), 2);
        assert!(
            admitted
                .execution_batches
                .iter()
                .all(|batch| batch.mode == BatchExecutionMode::Individual)
        );
        assert_eq!(
            admitted
                .execution_batches
                .iter()
                .flat_map(|batch| batch.request_ids.clone())
                .collect::<Vec<_>>(),
            vec![first, third]
        );
    }

    #[test]
    fn one_token_admissions_are_unplanned_output_limit_releases() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request_id = executor.submit(request("1", 1), Timestamp::ZERO).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(
            iteration.divergences,
            vec![PlanDivergence::UnplannedRelease {
                request_id,
                reason: ReleaseReason::OutputLimit,
            }]
        );
        let decoded = executor.step_iteration(Timestamp::from_micros(20));
        assert!(decoded.unwrap().divergences.is_empty());
    }

    #[test]
    fn waiting_requests_are_tokenized_once() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        struct CountingTokenizer(Arc<AtomicUsize>);

        impl Tokenizer for CountingTokenizer {
            fn encode(
                &self,
                text: &str,
                add_special_tokens: bool,
            ) -> Result<Vec<u32>, ModelExecutionError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                NumericTokenizer.encode(text, add_special_tokens)
            }

            fn decode(
                &self,
                token_ids: &[u32],
                skip_special_tokens: bool,
            ) -> Result<String, ModelExecutionError> {
                NumericTokenizer.decode(token_ids, skip_special_tokens)
            }
        }

        // The first request fills the 32-byte pool, so admission defers the
        // second on every iteration until the first finishes.
        let encodes = Arc::new(AtomicUsize::new(0));
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_admission(crate::MemoryAwareAdmission));
        executor
            .register_model(
                ModelDescriptor::language("tiny", "increment"),
                CausalLanguageModel::new(CountingTokenizer(encodes.clone()), IncrementModel)
                    .with_kv_cache_profile(KvCacheProfile::new(8, 4, 32).unwrap()),
            )
            .unwrap();
        executor.submit(request("1", 4), Timestamp::ZERO).unwrap();
        let waiting = executor.submit(request("10", 4), Timestamp::ZERO).unwrap();

        for step in 1..=3 {
            let iteration = executor
                .step_iteration(Timestamp::from_micros(step * 10))
                .unwrap();
            assert!(
                iteration
                    .execution_plan
                    .deferrals
                    .iter()
                    .any(|deferral| matches!(
                        deferral,
                        PlannedDeferral::Admission { request_id, .. } if *request_id == waiting
                    ))
            );
        }

        assert_eq!(encodes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn default_policy_has_a_stable_composed_descriptor() {
        assert_eq!(
            Executor::default().policy_descriptor().to_string(),
            "composed{head_of_line=block}[fifo,unconstrained-admission,unbounded-batch,unbucketed-prefill]"
        );
    }

    #[test]
    fn custom_policies_observe_commits_and_cannot_mutate_through_bad_plans() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };

        /// Delegates to the default policy, counts observed iterations, and
        /// can corrupt its plan on demand.
        struct Probe {
            inner: ComposedPolicy,
            observed: Arc<AtomicUsize>,
            corrupt: Arc<AtomicBool>,
        }

        impl ExecutionPolicy for Probe {
            fn descriptor(&self) -> PolicyDescriptor {
                PolicyDescriptor::new("probe").with_component(self.inner.descriptor())
            }

            fn plan(
                &self,
                snapshot: &PlanningSnapshot,
            ) -> Result<crate::PolicyDecision, PolicyError> {
                let mut decision = self.inner.plan(snapshot)?;
                if self.corrupt.load(Ordering::SeqCst) {
                    decision.plan.generation += 1;
                }
                Ok(decision)
            }

            fn enforces_model_kv_capacity(&self) -> bool {
                self.inner.enforces_model_kv_capacity()
            }

            fn observe(&mut self, _iteration: &ExecutorIteration) {
                self.observed.fetch_add(1, Ordering::SeqCst);
            }
        }

        let observed = Arc::new(AtomicUsize::new(0));
        let corrupt = Arc::new(AtomicBool::new(false));
        let mut executor = Executor::default().with_policy(Probe {
            inner: ComposedPolicy::default(),
            observed: observed.clone(),
            corrupt: corrupt.clone(),
        });
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request_id = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        executor.step_iteration(Timestamp::from_micros(10)).unwrap();
        assert_eq!(observed.load(Ordering::SeqCst), 1);

        corrupt.store(true, Ordering::SeqCst);
        let generation = executor.generation();
        let kv_cache_bytes = executor.state().memory.kv_cache_bytes;
        let error = executor
            .step_iteration(Timestamp::from_micros(20))
            .unwrap_err();

        assert!(matches!(
            error,
            ExecutorError::InvalidPlan(PlanError::Stale { .. })
        ));
        assert_eq!(executor.generation(), generation);
        assert_eq!(executor.state().memory.kv_cache_bytes, kv_cache_bytes);
        assert_eq!(executor.active_request_count(), 1);
        assert_eq!(observed.load(Ordering::SeqCst), 1);

        corrupt.store(false, Ordering::SeqCst);
        let resumed = executor.step_iteration(Timestamp::from_micros(30)).unwrap();
        assert!(matches!(
            resumed.events.as_slice(),
            [ExecutorEvent::Token { request_id: id, generated_tokens: 2, .. }] if *id == request_id
        ));
        assert_eq!(observed.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn decision_records_ranking_admission_and_budget_selection() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
            .unwrap()
            .with_policy(
                ComposedPolicy::default().with_batch(crate::FifoTokenBudgetPolicy::new(2).unwrap()),
            );
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        let third = executor.submit(request("20", 1), Timestamp::ZERO).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();
        let decision = &iteration.decision;

        assert_eq!(
            decision.policy.to_string(),
            "composed{head_of_line=block}[fifo,unconstrained-admission,fifo-token-budget{max_tokens=2},unbucketed-prefill]"
        );
        assert_eq!(
            decision.ranking,
            crate::RankingRecord {
                slots_available: 3,
                limit: 4,
                ranked: vec![first, second, third],
            }
        );
        let budget = decision
            .admissions
            .iter()
            .map(|record| match &record.outcome {
                crate::AdmissionOutcome::Decided {
                    budget_selected, ..
                } => (record.request_id, *budget_selected),
                other => panic!("unexpected outcome {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            budget,
            vec![
                (first, Some(true)),
                (second, Some(true)),
                (third, Some(false))
            ]
        );
        let admission_selections = decision
            .selections
            .iter()
            .filter(|selection| selection.stage == crate::SelectionStage::Admission)
            .collect::<Vec<_>>();
        assert_eq!(admission_selections.len(), 3);
        assert_eq!(admission_selections[2].context.scheduled_tokens(), 2);
        assert!(admission_selections[2].selected.is_empty());
    }

    #[test]
    fn decision_records_grouping_alternatives_and_capability_fallbacks() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(
                ComposedPolicy::default().with_grouping(crate::CostAwarePrefillBatcher::new(32)),
            );
        executor
            .register_model(
                ModelDescriptor::language("tiny", "individual-increment"),
                individual_artifact(),
            )
            .unwrap();
        executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        executor
            .submit(request("1 2 3 4 5", 3), Timestamp::ZERO)
            .unwrap();

        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        let grouping = &admitted.decision.grouping;
        assert_eq!(grouping.len(), 1);
        assert_eq!(grouping[0].phase, PlannedPhase::Prefill);
        assert_eq!(grouping[0].prompt_tokens, vec![1, 5]);
        let estimates = grouping[0].decision.cost_estimates.unwrap();
        assert_eq!(
            estimates.selected,
            crate::PrefillGroupingStrategy::Unbucketed
        );
        assert_eq!(estimates.unbucketed.dense_input_tokens, 10);
        assert_eq!(estimates.power_of_two.invocations, 2);
        assert_eq!(
            admitted.decision.capability_fallbacks,
            vec![crate::CapabilityFallback {
                phase: PlannedPhase::Prefill,
                model_id: "tiny".into(),
                rows: 2,
                missing: crate::BatchHook::Prefill,
            }]
        );

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();
        assert_eq!(
            decoded.decision.capability_fallbacks[0].missing,
            crate::BatchHook::Decode
        );
    }

    #[test]
    fn decision_records_the_exact_admission_context() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_admission(crate::MemoryAwareAdmission));
        executor
            .register_model(
                ModelDescriptor::language("tiny", "increment"),
                profiled_artifact(32),
            )
            .unwrap();
        executor.submit(request("1", 4), Timestamp::ZERO).unwrap();
        let deferred = executor.submit(request("10", 4), Timestamp::ZERO).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        let record = &iteration.decision.admissions[1];
        assert_eq!(record.request_id, deferred);
        let crate::AdmissionOutcome::Decided {
            context,
            decision,
            budget_selected,
        } = &record.outcome
        else {
            panic!("admission should have run");
        };
        assert_eq!(context.model_reserved_kv_bytes, 32);
        assert_eq!(context.prompt_tokens, 1);
        assert!(matches!(decision, crate::AdmissionDecision::Defer(_)));
        assert_eq!(*budget_selected, None);
    }

    #[test]
    fn priority_scheduling_admits_urgent_work_first() {
        let mut executor = Executor::default()
            .with_policy(ComposedPolicy::default().with_scheduler(crate::PriorityScheduler::new()));
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let normal = executor.submit(request("1", 1), Timestamp::ZERO).unwrap();
        let mut urgent = request("10", 1);
        urgent.priority = ola_core::Priority::new(5);
        let urgent = executor.submit(urgent, Timestamp::from_micros(1)).unwrap();

        let first = executor.step_iteration(Timestamp::from_micros(10)).unwrap();
        let second = executor.step_iteration(Timestamp::from_micros(20)).unwrap();

        assert_eq!(first.decision.ranking.ranked, vec![urgent, normal]);
        assert!(matches!(
            first.events.as_slice(),
            [ExecutorEvent::Token { request_id, finished: true, .. }] if *request_id == urgent
        ));
        assert!(matches!(
            second.events.as_slice(),
            [ExecutorEvent::Token { request_id, finished: true, .. }] if *request_id == normal
        ));
    }

    #[test]
    fn partial_prompts_on_non_resumable_models_are_rejected_before_execution() {
        /// Shortens every first prompt row to one token, which a model
        /// without resumable prefill cannot execute.
        struct Truncating;

        impl ExecutionPolicy for Truncating {
            fn descriptor(&self) -> PolicyDescriptor {
                PolicyDescriptor::new("truncating")
            }

            fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
                let mut decision = ComposedPolicy::default().plan(snapshot)?;
                for row in decision
                    .plan
                    .batches
                    .iter_mut()
                    .flat_map(|batch| &mut batch.rows)
                {
                    row.positions = row.positions.start..row.positions.start + 1;
                    row.final_prompt_chunk = false;
                }
                Ok(decision)
            }

            fn enforces_model_kv_capacity(&self) -> bool {
                false
            }
        }

        let mut executor = Executor::default().with_policy(Truncating);
        executor
            .register_model(
                ModelDescriptor::language("tiny", "individual-increment"),
                individual_artifact(),
            )
            .unwrap();
        let request_id = executor
            .submit(request("1 2 3", 2), Timestamp::ZERO)
            .unwrap();

        let error = executor
            .step_iteration(Timestamp::from_micros(10))
            .unwrap_err();

        assert_eq!(
            error,
            ExecutorError::InvalidPlan(PlanError::UnsupportedPartialPrefill(request_id))
        );
        assert_eq!(
            executor.state().request(request_id).unwrap().status,
            RequestStatus::Queued
        );
    }

    #[test]
    fn policy_measurements_reach_the_planning_snapshot() {
        use std::sync::{Arc, Mutex};

        struct Recording(Arc<Mutex<Option<crate::PolicyMeasurements>>>);

        impl ExecutionPolicy for Recording {
            fn descriptor(&self) -> PolicyDescriptor {
                PolicyDescriptor::new("recording")
            }

            fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
                *self.0.lock().unwrap() = Some(snapshot.measurements);
                ComposedPolicy::default().plan(snapshot)
            }

            fn enforces_model_kv_capacity(&self) -> bool {
                false
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let measured = crate::PolicyMeasurements {
            prefill_invocation_overhead_tokens: Some(42),
        };
        let mut executor = Executor::default()
            .with_policy(Recording(seen.clone()))
            .with_policy_measurements(measured);
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        executor.submit(request("1", 1), Timestamp::ZERO).unwrap();

        executor.step_iteration(Timestamp::from_micros(1)).unwrap();

        assert_eq!(*seen.lock().unwrap(), Some(measured));
    }

    #[test]
    fn admissions_and_preparation_failures_are_planned_before_execution() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let malformed = executor.submit(request("abc", 2), Timestamp::ZERO).unwrap();
        let admitted = executor.submit(request("1 2", 2), Timestamp::ZERO).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        let plan = &iteration.execution_plan;
        assert!(matches!(
            plan.lifecycle.as_slice(),
            [
                LifecycleAction::Fail { request_id: failed, event_index: 0, .. },
                LifecycleAction::Admit { request_id: started, .. },
            ] if *failed == malformed && *started == admitted
        ));
        assert_eq!(
            plan.batches,
            vec![PlannedBatch {
                phase: PlannedPhase::Prefill,
                model_id: "tiny".into(),
                mode: BatchExecutionMode::Individual,
                rows: vec![PlannedRow {
                    request_id: admitted,
                    event_index: 1,
                    positions: 0..2,
                    final_prompt_chunk: true,
                }],
            }]
        );
        assert!(matches!(
            iteration.events.as_slice(),
            [
                ExecutorEvent::Failed { request_id: failed, .. },
                ExecutorEvent::Token { request_id: started, .. },
            ] if *failed == malformed && *started == admitted
        ));
    }

    #[test]
    fn power_of_two_prefill_buckets_reduce_padding_without_reordering_events() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_grouping(crate::PowerOfTwoPrefillBatcher));
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor
            .submit(request("1 2 3 4 5", 1), Timestamp::ZERO)
            .unwrap();
        let second = executor
            .submit(request("1 2 3 4 5 6 7 8 9", 1), Timestamp::ZERO)
            .unwrap();
        let third = executor
            .submit(request("1 2 3 4 5 6 7", 1), Timestamp::ZERO)
            .unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert!(matches!(
            iteration.events.as_slice(),
            [
                ExecutorEvent::Token { request_id: first_id, .. },
                ExecutorEvent::Token { request_id: second_id, .. },
                ExecutorEvent::Token { request_id: third_id, .. },
            ] if *first_id == first && *second_id == second && *third_id == third
        ));
        assert_eq!(
            iteration.execution_batches,
            vec![
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![first, third],
                    mode: BatchExecutionMode::Tensor,
                    input_tokens: 12,
                    padding_tokens: 2,
                    started_at: Timestamp::from_micros(10),
                    finished_at: Timestamp::from_micros(10),
                },
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![second],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 9,
                    padding_tokens: 0,
                    started_at: Timestamp::from_micros(10),
                    finished_at: Timestamp::from_micros(10),
                },
            ]
        );
        assert_eq!(
            iteration
                .execution_batches
                .iter()
                .map(|batch| batch.padding_tokens)
                .sum::<usize>(),
            2
        );
    }

    #[test]
    fn cost_aware_prefill_applies_one_workload_level_decision() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 4)
            .unwrap()
            .with_policy(
                ComposedPolicy::default().with_grouping(crate::CostAwarePrefillBatcher::new(32)),
            );
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let mut ids = Vec::new();
        for length in [5, 7, 9, 63] {
            let mut spec = request("1", 1);
            spec.inputs = vec![Input::Tokens(vec![1; length])];
            ids.push(executor.submit(spec, Timestamp::ZERO).unwrap());
        }

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(
            iteration
                .events
                .iter()
                .filter(|event| matches!(event, ExecutorEvent::Token { .. }))
                .count(),
            4
        );
        assert_eq!(iteration.execution_batches.len(), 3);
        assert_eq!(
            iteration.execution_batches[0].request_ids,
            vec![ids[0], ids[1]]
        );
        assert_eq!(iteration.execution_batches[0].padding_tokens, 2);
        assert_eq!(iteration.execution_batches[1].request_ids, vec![ids[2]]);
        assert_eq!(iteration.execution_batches[2].request_ids, vec![ids[3]]);
        assert_eq!(
            iteration
                .execution_batches
                .iter()
                .map(|batch| batch.padding_tokens)
                .sum::<usize>(),
            2
        );
    }

    #[test]
    fn oversized_prefill_runs_alone_and_reports_its_actual_cost() {
        let policy = crate::FifoTokenBudgetPolicy::new(2).unwrap();
        let mut executor =
            Executor::default().with_policy(ComposedPolicy::default().with_batch(policy));
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request = executor
            .submit(request("1 2 3", 1), Timestamp::ZERO)
            .unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(iteration.execution_plan.estimates().input_tokens, 3);
        assert_eq!(
            scheduled(&iteration.execution_plan),
            vec![(request, PlannedPhase::Prefill, 3)]
        );
        assert!(iteration.execution_plan.deferrals.is_empty());
        assert_eq!(executor.state().queued_len(), 0);
        assert!(executor.result(request).is_some());
    }

    #[test]
    fn memory_aware_admission_defers_until_a_model_pool_reservation_is_released() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_admission(crate::MemoryAwareAdmission));
        executor
            .register_model(
                ModelDescriptor::language("tiny", "increment"),
                profiled_artifact(32),
            )
            .unwrap();
        let first = executor.submit(request("1", 4), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 4), Timestamp::ZERO).unwrap();

        let initial = executor.step_iteration(Timestamp::from_micros(10)).unwrap();
        assert!(matches!(
            initial.events.as_slice(),
            [
                ExecutorEvent::Token {
                    request_id,
                    phase: ExecutionPhase::Prefill,
                    ..
                },
                ExecutorEvent::Deferred {
                    request_id: deferred,
                    reason: AdmissionReason::ModelKvCapacity {
                        required_bytes: 32,
                        available_bytes: 0,
                    },
                }
            ] if *request_id == first && *deferred == second
        ));
        assert_eq!(initial.active_requests, 1);
        assert_eq!(initial.queued_requests, 1);
        assert_eq!(initial.kv_cache_bytes, 8);
        assert_eq!(initial.reserved_kv_bytes, 32);
        assert_eq!(
            executor.state().request(second).unwrap().status,
            RequestStatus::Queued
        );

        executor.step_iteration(Timestamp::from_micros(20)).unwrap();
        executor.step_iteration(Timestamp::from_micros(30)).unwrap();
        let refilled = executor.step_iteration(Timestamp::from_micros(40)).unwrap();

        assert!(matches!(
            refilled.events.as_slice(),
            [
                ExecutorEvent::Token {
                    request_id,
                    phase: ExecutionPhase::Decode,
                    finished: true,
                    ..
                },
                ExecutorEvent::Token {
                    request_id: admitted,
                    phase: ExecutionPhase::Prefill,
                    ..
                }
            ] if *request_id == first && *admitted == second
        ));
        assert_eq!(refilled.active_requests, 1);
        assert_eq!(refilled.queued_requests, 0);
        assert_eq!(refilled.reserved_kv_bytes, 32);
    }

    #[test]
    fn memory_aware_admission_can_enforce_a_device_wide_budget() {
        let mut runtime = Runtime::default();
        runtime.state_mut().memory.capacity_bytes = 80;
        runtime.state_mut().memory.weights_bytes = 32;
        let mut executor = Executor::with_max_active_requests(runtime, 2)
            .unwrap()
            .with_policy(ComposedPolicy::default().with_admission(crate::MemoryAwareAdmission));
        executor
            .register_model(
                ModelDescriptor::language("tiny", "increment"),
                profiled_artifact(128),
            )
            .unwrap();
        executor.submit(request("1", 4), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 4), Timestamp::ZERO).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert!(matches!(
            iteration.events.last(),
            Some(ExecutorEvent::Deferred {
                request_id,
                reason: AdmissionReason::DeviceMemoryCapacity {
                    required_bytes: 32,
                    available_bytes: 16,
                },
            }) if *request_id == second
        ));
    }

    #[test]
    fn strict_memory_policy_rejects_models_without_a_resource_profile() {
        let mut executor = Executor::default()
            .with_policy(ComposedPolicy::default().with_admission(crate::MemoryAwareAdmission));
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request = executor.submit(request("1", 2), Timestamp::ZERO).unwrap();

        let event = step_one(&mut executor, Timestamp::from_micros(1)).unwrap();

        assert_eq!(
            event,
            ExecutorEvent::Rejected {
                request_id: request,
                reason: AdmissionReason::MissingKvCacheProfile,
            }
        );
        assert!(matches!(
            &executor.state().request(request).unwrap().status,
            RequestStatus::Failed(message) if message.contains("no KV-cache resource profile")
        ));
    }
}
