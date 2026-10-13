use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
};

use ola_core::{Input, ModelDescriptor, ModelId, Request, RequestId, RequestSpec, Timestamp};
use ola_model::{
    CausalLanguageModel, GeneratedToken, GenerationSession, ModelExecutionError, SessionMetrics,
};

use crate::{
    AdmissionContext, AdmissionDecision, AdmissionPolicy, AdmissionReason, BatchCandidate,
    BatchContext, BatchExecutionMode, BatchPhase, BatchPolicy, ExecutionBatch, FifoScheduler,
    IterationPlan, PrefillBatcher, Runtime, RuntimeError, RuntimeState, Scheduler,
    UnboundedBatchPolicy, UnbucketedPrefillBatcher, UnconstrainedAdmission,
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
    /// Caller-supplied monotonic iteration timestamp.
    pub at: Timestamp,
    /// Per-request outcomes in deterministic execution order.
    pub events: Vec<ExecutorEvent>,
    /// Auditable batch-policy decision for this iteration.
    pub plan: IterationPlan,
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
    /// A batch policy selected an unknown request or selected one twice.
    InvalidBatchSelection(RequestId),
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
            Self::InvalidBatchSelection(request) => {
                write!(
                    formatter,
                    "batch policy returned invalid or duplicate request `{request}`"
                )
            }
        }
    }
}

impl Error for ExecutorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
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
    generated: Vec<u32>,
    last_token: GeneratedToken,
    max_new_tokens: usize,
    last_step_at: Timestamp,
    reserved_kv_bytes: u64,
}

struct PendingPrefill {
    request_id: RequestId,
    model_id: ModelId,
    model: CausalLanguageModel,
    session: Box<dyn GenerationSession>,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    reserved_kv_bytes: u64,
}

enum PrefillPreparation {
    Ready(PendingPrefill),
    Event(ExecutorEvent),
}

struct PrefillBatchOutcome {
    indexed_events: Vec<(usize, ExecutorEvent)>,
    execution_batches: Vec<ExecutionBatch>,
}

struct PrefillGroup {
    model_id: ModelId,
    bucket: usize,
    members: Vec<(usize, PendingPrefill)>,
}

/// Deterministic synchronous model executor.
///
/// Each [`step`](Self::step) performs at most one model operation: a queued
/// request's prefill or an active request's single-token decode.
/// [`step_iteration`](Self::step_iteration) advances every session that was
/// active at the iteration boundary once, then fills newly available slots.
/// Compatible prefill and decode operations may share tensor invocations;
/// models without a matching batch hook retain the individual correctness
/// path.
pub struct Executor<S = FifoScheduler> {
    runtime: Runtime<S>,
    models: BTreeMap<ModelId, CausalLanguageModel>,
    active: BTreeMap<RequestId, ActiveGeneration>,
    ready: VecDeque<RequestId>,
    max_active_requests: usize,
    admission: Box<dyn AdmissionPolicy>,
    batch_policy: Box<dyn BatchPolicy>,
    prefill_batcher: Box<dyn PrefillBatcher>,
    results: BTreeMap<RequestId, ExecutionResult>,
}

impl Default for Executor<FifoScheduler> {
    fn default() -> Self {
        Self::new(Runtime::default())
    }
}

impl<S: Scheduler> Executor<S> {
    /// Wraps a runtime with no attached executable models.
    pub fn new(runtime: Runtime<S>) -> Self {
        Self {
            runtime,
            models: BTreeMap::new(),
            active: BTreeMap::new(),
            ready: VecDeque::new(),
            max_active_requests: 1,
            admission: Box::new(UnconstrainedAdmission),
            batch_policy: Box::new(UnboundedBatchPolicy),
            prefill_batcher: Box::new(UnbucketedPrefillBatcher),
            results: BTreeMap::new(),
        }
    }

    /// Wraps a runtime with a bounded number of concurrently live sessions.
    pub fn with_max_active_requests(
        runtime: Runtime<S>,
        max_active_requests: usize,
    ) -> Result<Self, ExecutorError> {
        if max_active_requests == 0 {
            return Err(ExecutorError::InvalidActiveRequestLimit(0));
        }
        let mut executor = Self::new(runtime);
        executor.max_active_requests = max_active_requests;
        Ok(executor)
    }

    /// Replaces the compatibility admission rule with an explicit policy.
    pub fn with_admission_policy<P>(mut self, policy: P) -> Self
    where
        P: AdmissionPolicy + 'static,
    {
        self.admission = Box::new(policy);
        self
    }

    /// Replaces the unbounded iteration composer with an explicit policy.
    ///
    /// This affects [`step_iteration`](Self::step_iteration). The single-work
    /// compatibility method [`step`](Self::step) continues to execute one
    /// available operation directly.
    pub fn with_batch_policy<P>(mut self, policy: P) -> Self
    where
        P: BatchPolicy + 'static,
    {
        self.batch_policy = Box::new(policy);
        self
    }

    /// Replaces the compatibility prefill grouper with a length-bucketing rule.
    ///
    /// This affects tensor shape only after admission and batch-policy
    /// selection. It does not reorder lifecycle events or allow different
    /// loaded models to share an invocation.
    pub fn with_prefill_batcher<P>(mut self, batcher: P) -> Self
    where
        P: PrefillBatcher + 'static,
    {
        self.prefill_batcher = Box::new(batcher);
        self
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
        Ok(())
    }

    /// Validates and queues a request on the owned runtime.
    pub fn submit(
        &mut self,
        spec: RequestSpec,
        arrival_at: Timestamp,
    ) -> Result<RequestId, ExecutorError> {
        Ok(self.runtime.submit(spec, arrival_at)?)
    }

    /// Cancels queued or active work and releases request-local model state.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
        cancelled_at: Timestamp,
    ) -> Result<(), ExecutorError> {
        self.runtime.cancel(request_id, cancelled_at)?;
        if self.active.remove(&request_id).is_some() {
            self.ready.retain(|ready| *ready != request_id);
            self.refresh_kv_memory();
        }
        Ok(())
    }

    /// Performs one prefill or decode operation at the supplied timestamp.
    pub fn step(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        let can_admit =
            self.active.len() < self.max_active_requests && self.runtime.state().queued_len() > 0;
        if can_admit || self.ready.is_empty() {
            self.start_next(at)
        } else {
            self.decode_next(at)
        }
    }

    /// Advances one deterministic generation iteration.
    ///
    /// Every request active at entry receives one decode step. Slots freed by
    /// completion or failure are then filled from the configured scheduler;
    /// newly admitted requests receive prefill but are not decoded again in the
    /// same iteration.
    pub fn step_iteration(&mut self, at: Timestamp) -> Result<ExecutorIteration, ExecutorError> {
        self.validate_iteration_time(at)?;
        let mut plan = IterationPlan::default();
        let mut events = Vec::with_capacity(self.max_active_requests);
        let mut execution_batches = Vec::new();
        let decode_candidates = self
            .ready
            .iter()
            .copied()
            .map(|request_id| BatchCandidate {
                request_id,
                phase: BatchPhase::Decode,
                token_cost: 1,
            })
            .collect::<Vec<_>>();
        let selected = self.select_batch_candidates(&plan, &decode_candidates)?;
        let mut selected_decodes = Vec::new();
        let mut indexed_events = Vec::new();
        for (index, candidate) in decode_candidates.into_iter().enumerate() {
            if selected.contains(&candidate.request_id) {
                plan.scheduled.push(candidate);
                selected_decodes.push((index, candidate.request_id));
            } else {
                plan.deferred.push(candidate);
                indexed_events.push((
                    index,
                    ExecutorEvent::BatchDeferred {
                        request_id: candidate.request_id,
                        phase: ExecutionPhase::Decode,
                        token_cost: candidate.token_cost,
                    },
                ));
            }
        }
        let selected_ids = selected_decodes
            .iter()
            .map(|(_, request_id)| *request_id)
            .collect::<Vec<_>>();
        let (decode_events, decode_batches) = self.decode_selected_batch(&selected_ids, at)?;
        indexed_events.extend(
            selected_decodes
                .into_iter()
                .zip(decode_events)
                .map(|((index, _), event)| (index, event)),
        );
        indexed_events.sort_by_key(|(index, _)| *index);
        events.extend(indexed_events.into_iter().map(|(_, event)| event));
        execution_batches.extend(decode_batches);

        let available_slots = self.max_active_requests.saturating_sub(self.active.len());
        let admission_count = available_slots.min(self.runtime.state().queued_len());
        let mut pending = Vec::with_capacity(admission_count);
        let mut admission_events = Vec::new();
        let mut pending_reserved_kv_bytes = 0_u64;
        let mut pending_model_reservations = BTreeMap::<ModelId, u64>::new();
        for index in 0..admission_count {
            let preparation = self.prepare_next_prefill(
                at,
                &mut plan,
                pending_reserved_kv_bytes,
                &pending_model_reservations,
            )?;
            match preparation {
                PrefillPreparation::Ready(prepared) => {
                    pending_reserved_kv_bytes =
                        pending_reserved_kv_bytes.saturating_add(prepared.reserved_kv_bytes);
                    *pending_model_reservations
                        .entry(prepared.model_id.clone())
                        .or_default() += prepared.reserved_kv_bytes;
                    pending.push((index, prepared));
                }
                PrefillPreparation::Event(event) => {
                    let deferred = matches!(
                        event,
                        ExecutorEvent::Deferred { .. } | ExecutorEvent::BatchDeferred { .. }
                    );
                    admission_events.push((index, event));
                    if deferred {
                        break;
                    }
                }
            }
        }
        let prefill = self.prefill_selected_batch(pending, at)?;
        admission_events.extend(prefill.indexed_events);
        admission_events.sort_by_key(|(index, _)| *index);
        events.extend(admission_events.into_iter().map(|(_, event)| event));
        execution_batches.extend(prefill.execution_batches);
        if events.is_empty() {
            events.push(ExecutorEvent::Idle);
        }

        Ok(ExecutorIteration {
            at,
            events,
            plan,
            execution_batches,
            active_requests: self.active.len(),
            queued_requests: self.runtime.state().queued_len(),
            kv_cache_bytes: self.runtime.state().memory.kv_cache_bytes,
            reserved_kv_bytes: self.reserved_kv_bytes(),
        })
    }

    /// Returns runtime lifecycle and memory state.
    pub fn state(&self) -> &RuntimeState {
        self.runtime.state()
    }

    /// Returns a completed result retained by request ID.
    pub fn result(&self, request_id: RequestId) -> Option<&ExecutionResult> {
        self.results.get(&request_id)
    }

    /// Returns whether one request currently owns a generation session.
    pub fn has_active_request(&self) -> bool {
        !self.active.is_empty()
    }

    /// Returns the number of request-local generation sessions currently live.
    pub fn active_request_count(&self) -> usize {
        self.active.len()
    }

    /// Returns the configured upper bound on concurrently live sessions.
    pub const fn max_active_requests(&self) -> usize {
        self.max_active_requests
    }

    /// Returns worst-case page-rounded KV bytes committed to live requests.
    pub fn reserved_kv_bytes(&self) -> u64 {
        self.active
            .values()
            .map(|active| active.reserved_kv_bytes)
            .sum()
    }

    /// Borrows the underlying runtime for read-only inspection.
    pub const fn runtime(&self) -> &Runtime<S> {
        &self.runtime
    }

    /// Validates, admits, and creates one request-local session without running
    /// its prompt. Delaying execution lets the iteration group newly admitted
    /// requests into model-compatible prefill batches.
    fn prepare_next_prefill(
        &mut self,
        at: Timestamp,
        plan: &mut IterationPlan,
        pending_reserved_kv_bytes: u64,
        pending_model_reservations: &BTreeMap<ModelId, u64>,
    ) -> Result<PrefillPreparation, ExecutorError> {
        let Some(request_id) = self.runtime.select_next()? else {
            return Ok(PrefillPreparation::Event(ExecutorEvent::Idle));
        };
        let request = self
            .runtime
            .state()
            .request(request_id)
            .expect("a selected request retains its runtime record")
            .request
            .clone();
        let model_id = request.spec.model.clone();
        let Some(model) = self.models.get(&model_id).cloned() else {
            self.runtime.start_selected(request_id, at)?;
            let event = self.fail_started(
                request_id,
                format!("model `{model_id}` has no executable attached"),
                at,
            )?;
            return Ok(PrefillPreparation::Event(event));
        };
        let prompt_tokens = match prepare_prompt(&model, &request) {
            Ok(tokens) => tokens,
            Err(error) => {
                self.runtime.start_selected(request_id, at)?;
                let event = self.fail_started(request_id, error.to_string(), at)?;
                return Ok(PrefillPreparation::Event(event));
            }
        };
        let context = AdmissionContext {
            request_id,
            model_id: model_id.clone(),
            prompt_tokens: prompt_tokens.len(),
            max_new_tokens: request.spec.generation.max_new_tokens,
            kv_cache: model.kv_cache_profile(),
            memory: self.runtime.state().memory,
            reserved_kv_bytes: self
                .reserved_kv_bytes()
                .saturating_add(pending_reserved_kv_bytes),
            model_reserved_kv_bytes: self.reserved_kv_bytes_for_model(&model_id).saturating_add(
                pending_model_reservations
                    .get(&model_id)
                    .copied()
                    .unwrap_or(0),
            ),
        };
        let reserved_kv_bytes = match self.admission.decide(&context) {
            AdmissionDecision::Admit { reserved_kv_bytes } => reserved_kv_bytes,
            AdmissionDecision::Defer(reason) => {
                return Ok(PrefillPreparation::Event(ExecutorEvent::Deferred {
                    request_id,
                    reason,
                }));
            }
            AdmissionDecision::Reject(reason) => {
                self.runtime.reject(request_id, reason.to_string(), at)?;
                return Ok(PrefillPreparation::Event(ExecutorEvent::Rejected {
                    request_id,
                    reason,
                }));
            }
        };
        let candidate = BatchCandidate {
            request_id,
            phase: BatchPhase::Prefill,
            token_cost: prompt_tokens.len(),
        };
        let selected = self.select_batch_candidates(plan, &[candidate])?;
        if selected.is_empty() {
            plan.deferred.push(candidate);
            return Ok(PrefillPreparation::Event(ExecutorEvent::BatchDeferred {
                request_id,
                phase: ExecutionPhase::Prefill,
                token_cost: candidate.token_cost,
            }));
        }
        plan.scheduled.push(candidate);

        let lease = self.runtime.start_selected(request_id, at)?;
        let session = match model.create_session(request.spec.generation.clone()) {
            Ok(session) => session,
            Err(error) => {
                let event = self.fail_started(lease.request_id, error.to_string(), at)?;
                return Ok(PrefillPreparation::Event(event));
            }
        };
        Ok(PrefillPreparation::Ready(PendingPrefill {
            request_id: lease.request_id,
            model_id,
            model,
            session,
            prompt_tokens,
            max_new_tokens: request.spec.generation.max_new_tokens,
            reserved_kv_bytes,
        }))
    }

    /// Executes prepared prompts in stable model groups and returns events
    /// tagged with their original admission positions.
    fn prefill_selected_batch(
        &mut self,
        prepared: Vec<(usize, PendingPrefill)>,
        at: Timestamp,
    ) -> Result<PrefillBatchOutcome, ExecutorError> {
        let mut groups: Vec<PrefillGroup> = Vec::new();
        for (index, pending) in prepared {
            let bucket = self.prefill_batcher.bucket(pending.prompt_tokens.len());
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.model_id == pending.model_id && group.bucket == bucket)
            {
                group.members.push((index, pending));
            } else {
                groups.push(PrefillGroup {
                    model_id: pending.model_id.clone(),
                    bucket,
                    members: vec![(index, pending)],
                });
            }
        }

        let mut indexed_events = Vec::new();
        let mut execution_batches = Vec::new();
        for group in groups {
            let mut members = group.members;
            let model = members[0].1.model.clone();
            let prompts = members
                .iter()
                .map(|(_, pending)| pending.prompt_tokens.clone())
                .collect::<Vec<_>>();
            let prompt_slices = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let batched = if members.len() > 1 {
                let mut sessions: Vec<&mut dyn GenerationSession> =
                    Vec::with_capacity(members.len());
                for (_, pending) in &mut members {
                    sessions.push(pending.session.as_mut());
                }
                model.prefill_batch(&mut sessions, &prompt_slices)
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
                execution_batches.push(ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: members
                        .iter()
                        .map(|(_, pending)| pending.request_id)
                        .collect(),
                    mode: BatchExecutionMode::Tensor,
                    input_tokens: prompts.iter().map(Vec::len).sum(),
                    padding_tokens: prompts
                        .iter()
                        .map(Vec::len)
                        .max()
                        .unwrap_or(0)
                        .saturating_mul(prompts.len())
                        .saturating_sub(prompts.iter().map(Vec::len).sum()),
                });
                for ((index, pending), result) in members.into_iter().zip(results) {
                    let event = self.finish_prefill(pending, result, at)?;
                    indexed_events.push((index, event));
                }
            } else {
                for (index, mut pending) in members {
                    execution_batches.push(ExecutionBatch {
                        phase: BatchPhase::Prefill,
                        request_ids: vec![pending.request_id],
                        mode: BatchExecutionMode::Individual,
                        input_tokens: pending.prompt_tokens.len(),
                        padding_tokens: 0,
                    });
                    let result = pending.session.prefill(&pending.prompt_tokens);
                    let event = self.finish_prefill(pending, result, at)?;
                    indexed_events.push((index, event));
                }
            }
        }
        Ok(PrefillBatchOutcome {
            indexed_events,
            execution_batches,
        })
    }

    fn finish_prefill(
        &mut self,
        pending: PendingPrefill,
        result: Result<GeneratedToken, ModelExecutionError>,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let token = match result {
            Ok(token) => token,
            Err(error) => {
                return self.fail_started(pending.request_id, error.to_string(), at);
            }
        };
        self.runtime.mark_first_token(pending.request_id, at)?;
        let active = ActiveGeneration {
            request_id: pending.request_id,
            model_id: pending.model_id,
            model: pending.model,
            session: pending.session,
            generated: vec![token.id],
            last_token: token,
            max_new_tokens: pending.max_new_tokens,
            last_step_at: at,
            reserved_kv_bytes: pending.reserved_kv_bytes,
        };
        self.observe_token(active, ExecutionPhase::Prefill, at)
    }

    fn start_next(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        self.start_next_with_plan(at, None)
    }

    fn start_next_with_plan(
        &mut self,
        at: Timestamp,
        plan: Option<&mut IterationPlan>,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let Some(request_id) = self.runtime.select_next()? else {
            return Ok(ExecutorEvent::Idle);
        };
        let request = self
            .runtime
            .state()
            .request(request_id)
            .expect("a selected request retains its runtime record")
            .request
            .clone();
        let model_id = request.spec.model.clone();
        let Some(model) = self.models.get(&model_id).cloned() else {
            self.runtime.start_selected(request_id, at)?;
            return self.fail_started(
                request_id,
                format!("model `{model_id}` has no executable attached"),
                at,
            );
        };

        let prompt_tokens = match prepare_prompt(&model, &request) {
            Ok(tokens) => tokens,
            Err(error) => {
                self.runtime.start_selected(request_id, at)?;
                return self.fail_started(request_id, error.to_string(), at);
            }
        };
        let context = AdmissionContext {
            request_id,
            model_id: model_id.clone(),
            prompt_tokens: prompt_tokens.len(),
            max_new_tokens: request.spec.generation.max_new_tokens,
            kv_cache: model.kv_cache_profile(),
            memory: self.runtime.state().memory,
            reserved_kv_bytes: self.reserved_kv_bytes(),
            model_reserved_kv_bytes: self.reserved_kv_bytes_for_model(&model_id),
        };
        let reserved_kv_bytes = match self.admission.decide(&context) {
            AdmissionDecision::Admit { reserved_kv_bytes } => reserved_kv_bytes,
            AdmissionDecision::Defer(reason) => {
                return Ok(ExecutorEvent::Deferred { request_id, reason });
            }
            AdmissionDecision::Reject(reason) => {
                self.runtime.reject(request_id, reason.to_string(), at)?;
                return Ok(ExecutorEvent::Rejected { request_id, reason });
            }
        };
        if let Some(plan) = plan {
            let candidate = BatchCandidate {
                request_id,
                phase: BatchPhase::Prefill,
                token_cost: prompt_tokens.len(),
            };
            let selected = self.select_batch_candidates(plan, &[candidate])?;
            if selected.is_empty() {
                plan.deferred.push(candidate);
                return Ok(ExecutorEvent::BatchDeferred {
                    request_id,
                    phase: ExecutionPhase::Prefill,
                    token_cost: candidate.token_cost,
                });
            }
            plan.scheduled.push(candidate);
        }
        let lease = self.runtime.start_selected(request_id, at)?;
        let mut session = match model.create_session(request.spec.generation.clone()) {
            Ok(session) => session,
            Err(error) => return self.fail_started(lease.request_id, error.to_string(), at),
        };
        let token = match session.prefill(&prompt_tokens) {
            Ok(token) => token,
            Err(error) => return self.fail_started(lease.request_id, error.to_string(), at),
        };
        self.runtime.mark_first_token(lease.request_id, at)?;
        let active = ActiveGeneration {
            request_id: lease.request_id,
            model_id,
            model,
            session,
            generated: vec![token.id],
            last_token: token,
            max_new_tokens: request.spec.generation.max_new_tokens,
            last_step_at: at,
            reserved_kv_bytes,
        };
        self.observe_token(active, ExecutionPhase::Prefill, at)
    }

    fn decode_next(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        let Some(request_id) = self.ready.pop_front() else {
            return Ok(ExecutorEvent::Idle);
        };
        self.decode_removed(request_id, 0, at)
    }

    fn decode_removed(
        &mut self,
        request_id: RequestId,
        ready_position: usize,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let mut active = self
            .active
            .remove(&request_id)
            .expect("ready requests always retain an active generation session");
        if at < active.last_step_at {
            let previous = active.last_step_at;
            self.active.insert(request_id, active);
            self.ready.insert(ready_position, request_id);
            return Err(RuntimeError::TimestampRegression {
                request: request_id,
                event: "executor step",
                previous,
                observed: at,
            }
            .into());
        }
        let token = match active.session.decode(active.last_token.id) {
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

    fn decode_selected_batch(
        &mut self,
        request_ids: &[RequestId],
        at: Timestamp,
    ) -> Result<(Vec<ExecutorEvent>, Vec<ExecutionBatch>), ExecutorError> {
        if request_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let selected = request_ids.iter().copied().collect::<BTreeSet<_>>();
        self.ready
            .retain(|request_id| !selected.contains(request_id));

        let mut groups: Vec<(ModelId, Vec<(usize, ActiveGeneration)>)> = Vec::new();
        for (index, request_id) in request_ids.iter().copied().enumerate() {
            let active = self
                .active
                .remove(&request_id)
                .expect("selected decode requests retain active sessions");
            if let Some((_, members)) = groups
                .iter_mut()
                .find(|(model_id, _)| model_id == &active.model_id)
            {
                members.push((index, active));
            } else {
                groups.push((active.model_id.clone(), vec![(index, active)]));
            }
        }

        let mut indexed_events = Vec::with_capacity(request_ids.len());
        let mut execution_batches = Vec::new();
        for (_, mut members) in groups {
            let input_tokens = members
                .iter()
                .map(|(_, active)| active.last_token.id)
                .collect::<Vec<_>>();
            let model = members[0].1.model.clone();
            let batched = if members.len() > 1 {
                {
                    let mut sessions: Vec<&mut dyn GenerationSession> =
                        Vec::with_capacity(members.len());
                    for (_, active) in &mut members {
                        sessions.push(active.session.as_mut());
                    }
                    model.decode_batch(&mut sessions, &input_tokens)
                }
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
                execution_batches.push(ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: members
                        .iter()
                        .map(|(_, active)| active.request_id)
                        .collect(),
                    mode: BatchExecutionMode::Tensor,
                    input_tokens: members.len(),
                    padding_tokens: 0,
                });
                for ((index, active), result) in members.into_iter().zip(results) {
                    let event = self.finish_decode(active, result, at)?;
                    indexed_events.push((index, event));
                }
            } else {
                for (index, mut active) in members {
                    execution_batches.push(ExecutionBatch {
                        phase: BatchPhase::Decode,
                        request_ids: vec![active.request_id],
                        mode: BatchExecutionMode::Individual,
                        input_tokens: 1,
                        padding_tokens: 0,
                    });
                    let result = active.session.decode(active.last_token.id);
                    let event = self.finish_decode(active, result, at)?;
                    indexed_events.push((index, event));
                }
            }
        }
        indexed_events.sort_by_key(|(index, _)| *index);
        Ok((
            indexed_events.into_iter().map(|(_, event)| event).collect(),
            execution_batches,
        ))
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

    fn select_batch_candidates(
        &mut self,
        plan: &IterationPlan,
        candidates: &[BatchCandidate],
    ) -> Result<BTreeSet<RequestId>, ExecutorError> {
        let context = BatchContext {
            scheduled: plan.scheduled.clone(),
            candidates: candidates.to_vec(),
        };
        let mut selected = BTreeSet::new();
        for request_id in self.batch_policy.select(&context) {
            let is_candidate = candidates
                .iter()
                .any(|candidate| candidate.request_id == request_id);
            if !is_candidate || !selected.insert(request_id) {
                return Err(ExecutorError::InvalidBatchSelection(request_id));
            }
        }
        Ok(selected)
    }

    fn observe_token(
        &mut self,
        mut active: ActiveGeneration,
        phase: ExecutionPhase,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let metrics = active.session.metrics();
        active.reserved_kv_bytes = active.reserved_kv_bytes.max(metrics.kv_cache_bytes);
        let finished =
            active.last_token.is_end_of_sequence || active.generated.len() >= active.max_new_tokens;
        let event = ExecutorEvent::Token {
            request_id: active.request_id,
            token: active.last_token,
            generated_tokens: active.generated.len(),
            kv_cache_bytes: metrics.kv_cache_bytes,
            phase,
            finished,
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

    fn fail_started(
        &mut self,
        request_id: RequestId,
        message: String,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        self.runtime.fail(request_id, message.clone(), at)?;
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
        Ok(())
    }

    fn reserved_kv_bytes_for_model(&self, model_id: &ModelId) -> u64 {
        self.active
            .values()
            .filter(|active| &active.model_id == model_id)
            .map(|active| active.reserved_kv_bytes)
            .sum()
    }
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
        CausalModel, ExecutionStage, GeneratedToken, GenerationSession, KvCacheProfile,
        SessionMetrics, Tokenizer,
    };

    use super::*;
    use crate::RequestStatus;

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
                is_end_of_sequence: false,
            })
        }

        fn metrics(&self) -> SessionMetrics {
            self.metrics
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    fn artifact() -> CausalLanguageModel {
        CausalLanguageModel::new(NumericTokenizer, IncrementModel)
    }

    fn individual_artifact() -> CausalLanguageModel {
        CausalLanguageModel::new(NumericTokenizer, IndividualIncrementModel)
    }

    fn profiled_artifact(capacity_bytes: u64) -> CausalLanguageModel {
        let profile = KvCacheProfile::new(8, 4, capacity_bytes).unwrap();
        artifact().with_kv_cache_profile(profile)
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

        let first = executor.step(Timestamp::from_micros(20)).unwrap();
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

        executor.step(Timestamp::from_micros(30)).unwrap();
        let last = executor.step(Timestamp::from_micros(40)).unwrap();
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
            })
        );
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

        let event = executor.step(Timestamp::from_micros(1)).unwrap();

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
        executor.step(Timestamp::from_micros(1)).unwrap();

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
        executor.step(Timestamp::from_micros(20)).unwrap();

        let error = executor.step(Timestamp::from_micros(19)).unwrap_err();

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
            executor.step(Timestamp::from_micros(30)).unwrap(),
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
                },
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![third],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
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
                },
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![second],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                },
            ]
        );

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();

        assert_eq!(
            decoded.execution_batches,
            vec![
                ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![first],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                },
                ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![second],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
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
            .with_batch_policy(policy);
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let first = executor.submit(request("1", 3), Timestamp::ZERO).unwrap();
        let second = executor.submit(request("10", 3), Timestamp::ZERO).unwrap();
        let third = executor.submit(request("20", 1), Timestamp::ZERO).unwrap();

        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(admitted.plan.scheduled_tokens(), 2);
        assert_eq!(
            admitted.plan.scheduled,
            vec![
                BatchCandidate {
                    request_id: first,
                    phase: BatchPhase::Prefill,
                    token_cost: 1,
                },
                BatchCandidate {
                    request_id: second,
                    phase: BatchPhase::Prefill,
                    token_cost: 1,
                },
            ]
        );
        assert_eq!(
            admitted.plan.deferred,
            vec![BatchCandidate {
                request_id: third,
                phase: BatchPhase::Prefill,
                token_cost: 1,
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
        assert_eq!(decoded.plan.scheduled_tokens(), 2);
        assert!(
            decoded
                .plan
                .scheduled
                .iter()
                .all(|candidate| candidate.phase == BatchPhase::Decode)
        );
        assert_eq!(decoded.plan.deferred[0].request_id, third);
        assert_eq!(
            decoded.execution_batches,
            vec![ExecutionBatch {
                phase: BatchPhase::Decode,
                request_ids: vec![first, second],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 2,
                padding_tokens: 0,
            }]
        );
        assert_eq!(
            executor.state().request(third).unwrap().status,
            RequestStatus::Queued
        );
    }

    #[test]
    fn power_of_two_prefill_buckets_reduce_padding_without_reordering_events() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
            .unwrap()
            .with_prefill_batcher(crate::PowerOfTwoPrefillBatcher);
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
                },
                ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![second],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 9,
                    padding_tokens: 0,
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
    fn oversized_prefill_runs_alone_and_reports_its_actual_cost() {
        let policy = crate::FifoTokenBudgetPolicy::new(2).unwrap();
        let mut executor = Executor::default().with_batch_policy(policy);
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request = executor
            .submit(request("1 2 3", 1), Timestamp::ZERO)
            .unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(iteration.plan.scheduled_tokens(), 3);
        assert_eq!(
            iteration.plan.scheduled,
            vec![BatchCandidate {
                request_id: request,
                phase: BatchPhase::Prefill,
                token_cost: 3,
            }]
        );
        assert!(iteration.plan.deferred.is_empty());
        assert_eq!(executor.state().queued_len(), 0);
        assert!(executor.result(request).is_some());
    }

    #[test]
    fn memory_aware_admission_defers_until_a_model_pool_reservation_is_released() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_admission_policy(crate::MemoryAwareAdmission);
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
            .with_admission_policy(crate::MemoryAwareAdmission);
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
        let mut executor = Executor::default().with_admission_policy(crate::MemoryAwareAdmission);
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let request = executor.submit(request("1", 2), Timestamp::ZERO).unwrap();

        let event = executor.step(Timestamp::from_micros(1)).unwrap();

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
