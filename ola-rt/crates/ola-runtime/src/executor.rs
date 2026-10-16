use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    num::NonZeroUsize,
};

use ola_core::{
    GenerationConfig, Input, ModelDescriptor, ModelId, Request, RequestId, RequestSpec, Timestamp,
};
use ola_model::{
    BatchCapabilities, CausalLanguageModel, GeneratedToken, GenerationSession, ModelExecutionError,
    PrefillOutcome, SessionMetrics,
};

use crate::{
    AdmissionContext, AdmissionDecision, AdmissionPolicy, AdmissionReason, BatchCandidate,
    BatchContext, BatchExecutionMode, BatchPhase, BatchPolicy, DecodeView, ExecutionBatch,
    ExecutionPlan, FifoScheduler, LifecycleAction, PlanDivergence, PlanError, PlannedBatch,
    PlannedDeferral, PlannedPhase, PlannedRow, PlanningView, PrefillBatcher, PrefillView,
    QueuedView, ReleaseReason, Runtime, RuntimeError, RuntimeState, Scheduler,
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
    /// Validated plan committed by this iteration.
    pub execution_plan: ExecutionPlan,
    /// Ways execution differed from `execution_plan`: lifecycle failures,
    /// then tensor fallbacks in batch order, then unpredicted releases in
    /// event order. Empty when execution matched the plan exactly.
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
    /// A batch policy selected an unknown request or selected one twice.
    InvalidBatchSelection(RequestId),
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
            Self::InvalidBatchSelection(request) => {
                write!(
                    formatter,
                    "batch policy returned invalid or duplicate request `{request}`"
                )
            }
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
    prompt_tokens: usize,
    generated: Vec<u32>,
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

    /// Whether the next decode must finish by reaching `max_new_tokens`.
    fn reaches_limit(&self) -> bool {
        self.generated.len().saturating_add(1) >= self.max_new_tokens
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

/// Inputs prepared while planning an admission and consumed at commit.
struct PlannedAdmission {
    model: CausalLanguageModel,
    prompt_tokens: Vec<u32>,
    generation: GenerationConfig,
    max_new_tokens: usize,
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
    prefilling: BTreeMap<RequestId, PendingPrefill>,
    prefill_ready: VecDeque<RequestId>,
    max_active_requests: usize,
    prefill_chunk_size: Option<NonZeroUsize>,
    admission: Box<dyn AdmissionPolicy>,
    batch_policy: Box<dyn BatchPolicy>,
    prefill_batcher: Box<dyn PrefillBatcher>,
    results: BTreeMap<RequestId, ExecutionResult>,
    generation: u64,
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
            prefilling: BTreeMap::new(),
            prefill_ready: VecDeque::new(),
            max_active_requests: 1,
            prefill_chunk_size: None,
            admission: Box::new(UnconstrainedAdmission),
            batch_policy: Box::new(UnboundedBatchPolicy),
            prefill_batcher: Box::new(UnbucketedPrefillBatcher),
            results: BTreeMap::new(),
            generation: 0,
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

    /// Performs one prefill or decode operation at the supplied timestamp.
    pub fn step(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        self.bump_generation();
        if !self.prefill_ready.is_empty() {
            return self.prefill_next(at);
        }
        let can_admit = self.live_request_count() < self.max_active_requests
            && self.runtime.state().queued_len() > 0;
        if can_admit || self.ready.is_empty() {
            self.start_next(at)
        } else {
            self.decode_next(at)
        }
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
        let mut scheduled = Vec::new();
        let (execution_plan, mut admissions) = self.plan_iteration(at, &mut scheduled)?;
        execution_plan
            .validate(&self.planning_view(&admissions))
            .map_err(ExecutorError::InvalidPlan)?;
        let committed = self.commit_plan(&execution_plan, &mut admissions, at)?;
        let mut events = committed
            .indexed_events
            .into_iter()
            .map(|(_, event)| event)
            .collect::<Vec<_>>();
        if events.is_empty() {
            events.push(ExecutorEvent::Idle);
        }
        self.bump_generation();

        Ok(ExecutorIteration {
            at,
            events,
            execution_plan,
            divergences: committed.divergences,
            execution_batches: committed.execution_batches,
            active_requests: self.live_request_count(),
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
    pub const fn runtime(&self) -> &Runtime<S> {
        &self.runtime
    }

    /// Plans one complete iteration before anything executes.
    ///
    /// Event indices follow the historical order: decodes, continued
    /// prefills, then admissions. Slots and reservations of requests that
    /// must finish by reaching their output limit are reused for admission;
    /// slots freed by EOS or failure are refilled next iteration.
    fn plan_iteration(
        &mut self,
        at: Timestamp,
        scheduled: &mut Vec<BatchCandidate>,
    ) -> Result<(ExecutionPlan, BTreeMap<RequestId, PlannedAdmission>), ExecutorError> {
        let mut execution_plan = ExecutionPlan {
            generation: self.generation,
            at,
            ..ExecutionPlan::default()
        };
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
        let decode_count = decode_candidates.len();
        let selected = self.select_batch_candidates(scheduled, &decode_candidates)?;
        let mut decode_rows = Vec::new();
        for (event_index, candidate) in decode_candidates.into_iter().enumerate() {
            if !selected.contains(&candidate.request_id) {
                execution_plan
                    .deferrals
                    .push(budget_deferral(candidate, event_index));
                continue;
            }
            scheduled.push(candidate);
            let active = self
                .active
                .get(&candidate.request_id)
                .expect("ready requests always retain an active generation session");
            if active.reaches_limit() {
                execution_plan.predicted_releases.push(candidate.request_id);
            }
            let position = active.next_position();
            decode_rows.push((
                active.model_id.clone(),
                PlannedRow {
                    request_id: candidate.request_id,
                    event_index,
                    positions: position..position.saturating_add(1),
                    final_prompt_chunk: false,
                },
            ));
        }
        execution_plan
            .batches
            .extend(self.plan_decode_batches(decode_rows));

        let prefill_candidates = self
            .prefill_ready
            .iter()
            .map(|request_id| {
                let pending = self
                    .prefilling
                    .get(request_id)
                    .expect("ready prefill requests retain their session");
                BatchCandidate {
                    request_id: *request_id,
                    phase: BatchPhase::Prefill,
                    token_cost: self.next_prefill_chunk_len(pending),
                }
            })
            .collect::<Vec<_>>();
        let prefill_candidates_len = prefill_candidates.len();
        let selected = self.select_batch_candidates(scheduled, &prefill_candidates)?;
        let mut prefill_rows = Vec::new();
        for (index, candidate) in prefill_candidates.into_iter().enumerate() {
            let event_index = decode_count.saturating_add(index);
            if !selected.contains(&candidate.request_id) {
                execution_plan
                    .deferrals
                    .push(budget_deferral(candidate, event_index));
                continue;
            }
            scheduled.push(candidate);
            let pending = self
                .prefilling
                .get(&candidate.request_id)
                .expect("ready prefill requests retain their session");
            let end = pending.cursor.saturating_add(candidate.token_cost);
            let final_prompt_chunk = end == pending.prompt_tokens.len();
            if final_prompt_chunk && pending.max_new_tokens <= 1 {
                execution_plan.predicted_releases.push(candidate.request_id);
            }
            prefill_rows.push((
                pending.model_id.clone(),
                PlannedRow {
                    request_id: candidate.request_id,
                    event_index,
                    positions: pending.cursor..end,
                    final_prompt_chunk,
                },
            ));
        }
        let prefill_count = prefill_candidates_len;
        execution_plan
            .batches
            .extend(self.plan_prefill_batches(PlannedPhase::ContinuedPrefill, prefill_rows));

        let admissions = self.plan_admissions(
            scheduled,
            &mut execution_plan,
            decode_count.saturating_add(prefill_count),
        )?;
        Ok((execution_plan, admissions))
    }

    /// Ranks queued work and plans admissions into the remaining slots.
    ///
    /// Each ranked request consumes one admission attempt whether it is
    /// admitted, rejected, or fails preparation. Planning stops at the first
    /// admission or budget deferral, preserving FIFO head-of-line semantics.
    fn plan_admissions(
        &mut self,
        scheduled: &mut Vec<BatchCandidate>,
        execution_plan: &mut ExecutionPlan,
        first_event_index: usize,
    ) -> Result<BTreeMap<RequestId, PlannedAdmission>, ExecutorError> {
        let mut released_total = 0_u64;
        let mut released_by_model = BTreeMap::<ModelId, u64>::new();
        for request_id in &execution_plan.predicted_releases {
            let (model_id, bytes) = if let Some(active) = self.active.get(request_id) {
                (&active.model_id, active.reserved_kv_bytes)
            } else {
                let pending = &self.prefilling[request_id];
                (&pending.model_id, pending.reserved_kv_bytes)
            };
            released_total = released_total.saturating_add(bytes);
            *released_by_model.entry(model_id.clone()).or_default() += bytes;
        }
        let live_after = self
            .live_request_count()
            .saturating_sub(execution_plan.predicted_releases.len());
        let admission_count = self
            .max_active_requests
            .saturating_sub(live_after)
            .min(self.runtime.state().queued_len());
        let ranked = self.runtime.rank_next(admission_count.saturating_add(1))?;

        let mut admissions = BTreeMap::new();
        let mut rows = Vec::new();
        let mut pending_total = 0_u64;
        let mut pending_by_model = BTreeMap::<ModelId, u64>::new();
        for (offset, request_id) in ranked.into_iter().enumerate() {
            if offset == admission_count {
                execution_plan
                    .deferrals
                    .push(PlannedDeferral::Slots { request_id });
                break;
            }
            let event_index = first_event_index.saturating_add(offset);
            let request = self
                .runtime
                .state()
                .request(request_id)
                .expect("a ranked request retains its runtime record")
                .request
                .clone();
            let model_id = request.spec.model.clone();
            let Some(model) = self.models.get(&model_id).cloned() else {
                execution_plan.lifecycle.push(LifecycleAction::Fail {
                    request_id,
                    message: format!("model `{model_id}` has no executable attached"),
                    event_index,
                });
                continue;
            };
            let prompt_tokens = match prepare_prompt(&model, &request) {
                Ok(tokens) => tokens,
                Err(error) => {
                    execution_plan.lifecycle.push(LifecycleAction::Fail {
                        request_id,
                        message: error.to_string(),
                        event_index,
                    });
                    continue;
                }
            };
            let model_released = released_by_model.get(&model_id).copied().unwrap_or(0);
            let model_pending = pending_by_model.get(&model_id).copied().unwrap_or(0);
            let context = AdmissionContext {
                request_id,
                model_id: model_id.clone(),
                prompt_tokens: prompt_tokens.len(),
                max_new_tokens: request.spec.generation.max_new_tokens,
                kv_cache: model.kv_cache_profile(),
                memory: self.runtime.state().memory,
                reserved_kv_bytes: self
                    .reserved_kv_bytes()
                    .saturating_sub(released_total)
                    .saturating_add(pending_total),
                model_reserved_kv_bytes: self
                    .reserved_kv_bytes_for_model(&model_id)
                    .saturating_sub(model_released)
                    .saturating_add(model_pending),
            };
            let reserved_kv_bytes = match self.admission.decide(&context) {
                AdmissionDecision::Admit { reserved_kv_bytes } => reserved_kv_bytes,
                AdmissionDecision::Defer(reason) => {
                    execution_plan.deferrals.push(PlannedDeferral::Admission {
                        request_id,
                        reason,
                        event_index,
                    });
                    break;
                }
                AdmissionDecision::Reject(reason) => {
                    execution_plan.lifecycle.push(LifecycleAction::Reject {
                        request_id,
                        reason,
                        event_index,
                    });
                    continue;
                }
            };
            let candidate = BatchCandidate {
                request_id,
                phase: BatchPhase::Prefill,
                token_cost: self.initial_prefill_chunk_len(&model, prompt_tokens.len()),
            };
            if self
                .select_batch_candidates(scheduled, &[candidate])?
                .is_empty()
            {
                execution_plan
                    .deferrals
                    .push(budget_deferral(candidate, event_index));
                break;
            }
            scheduled.push(candidate);
            pending_total = pending_total.saturating_add(reserved_kv_bytes);
            *pending_by_model.entry(model_id.clone()).or_default() += reserved_kv_bytes;
            execution_plan.lifecycle.push(LifecycleAction::Admit {
                request_id,
                model_id: model_id.clone(),
                reserved_kv_bytes,
            });
            rows.push((
                model_id,
                PlannedRow {
                    request_id,
                    event_index,
                    positions: 0..candidate.token_cost,
                    final_prompt_chunk: candidate.token_cost == prompt_tokens.len(),
                },
            ));
            admissions.insert(
                request_id,
                PlannedAdmission {
                    model,
                    prompt_tokens,
                    max_new_tokens: request.spec.generation.max_new_tokens,
                    generation: request.spec.generation,
                },
            );
        }
        execution_plan
            .batches
            .extend(self.plan_prefill_batches(PlannedPhase::Prefill, rows));
        Ok(admissions)
    }

    /// Snapshot of live work that plans are validated against.
    fn planning_view(&self, admissions: &BTreeMap<RequestId, PlannedAdmission>) -> PlanningView {
        let state = self.runtime.state();
        PlanningView {
            generation: self.generation,
            max_active_requests: self.max_active_requests,
            queued: state
                .queued_ids()
                .filter_map(|request_id| {
                    let record = state.request(request_id)?;
                    Some((
                        request_id,
                        QueuedView {
                            model_id: record.request.spec.model.clone(),
                            prompt_tokens: admissions
                                .get(&request_id)
                                .map(|admission| admission.prompt_tokens.len()),
                        },
                    ))
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
                            max_new_tokens: pending.max_new_tokens,
                            reserved_kv_bytes: pending.reserved_kv_bytes,
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
                        },
                    )
                })
                .collect(),
            model_kv_capacity: self
                .models
                .iter()
                .filter(|_| self.admission.enforces_model_kv_capacity())
                .filter_map(|(model_id, model)| {
                    model
                        .kv_cache_profile()
                        .map(|profile| (model_id.clone(), profile.capacity_bytes()))
                })
                .collect(),
        }
    }

    /// Commits a validated plan: lifecycle actions first, then every
    /// invocation in order. Returns events sorted by planned index and every
    /// divergence from the plan.
    fn commit_plan(
        &mut self,
        plan: &ExecutionPlan,
        admissions: &mut BTreeMap<RequestId, PlannedAdmission>,
        at: Timestamp,
    ) -> Result<BatchOutcome, ExecutorError> {
        let mut outcome = BatchOutcome::default();
        let mut admitted = BTreeMap::new();
        let admit_events = plan
            .batches
            .iter()
            .filter(|batch| batch.phase == PlannedPhase::Prefill)
            .flat_map(|batch| &batch.rows)
            .map(|row| (row.request_id, row.event_index))
            .collect::<BTreeMap<_, _>>();
        for action in &plan.lifecycle {
            match action {
                LifecycleAction::Admit {
                    request_id,
                    model_id,
                    reserved_kv_bytes,
                } => {
                    let admission = admissions
                        .remove(request_id)
                        .expect("planned admissions retain their prepared inputs");
                    let lease = self.runtime.start_selected(*request_id, at)?;
                    match admission.model.create_session(admission.generation) {
                        Ok(session) => {
                            admitted.insert(
                                *request_id,
                                PendingPrefill {
                                    request_id: lease.request_id,
                                    model_id: model_id.clone(),
                                    model: admission.model,
                                    session,
                                    prompt_tokens: admission.prompt_tokens,
                                    cursor: 0,
                                    max_new_tokens: admission.max_new_tokens,
                                    last_step_at: at,
                                    reserved_kv_bytes: *reserved_kv_bytes,
                                },
                            );
                        }
                        Err(error) => {
                            let event = self.fail_started(*request_id, error.to_string(), at)?;
                            outcome
                                .indexed_events
                                .push((admit_events[request_id], event));
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
                    self.runtime.start_selected(*request_id, at)?;
                    let event = self.fail_started(*request_id, message.clone(), at)?;
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

    /// Groups decode rows by loaded model in first-appearance order.
    fn plan_decode_batches(&self, rows: Vec<(ModelId, PlannedRow)>) -> Vec<PlannedBatch> {
        let mut groups: Vec<PlannedBatch> = Vec::new();
        for (model_id, row) in rows {
            if let Some(group) = groups.iter_mut().find(|group| group.model_id == model_id) {
                group.rows.push(row);
            } else {
                groups.push(PlannedBatch {
                    phase: PlannedPhase::Decode,
                    model_id,
                    mode: BatchExecutionMode::Individual,
                    rows: vec![row],
                });
            }
        }
        self.assign_batch_modes(groups)
    }

    /// Groups prompt rows by loaded model and prefill-batcher bucket.
    ///
    /// The batcher sees each model's complete row set, so workload-aware
    /// policies make one decision per model. Groups keep first-appearance
    /// order, so bucketing never reorders events.
    fn plan_prefill_batches(
        &mut self,
        phase: PlannedPhase,
        rows: Vec<(ModelId, PlannedRow)>,
    ) -> Vec<PlannedBatch> {
        let mut model_positions: Vec<(ModelId, Vec<usize>)> = Vec::new();
        for (position, (model_id, _)) in rows.iter().enumerate() {
            if let Some((_, positions)) = model_positions
                .iter_mut()
                .find(|(existing, _)| existing == model_id)
            {
                positions.push(position);
            } else {
                model_positions.push((model_id.clone(), vec![position]));
            }
        }
        let mut assigned_buckets = vec![0; rows.len()];
        for (_, positions) in model_positions {
            let lengths = positions
                .iter()
                .map(|position| rows[*position].1.input_tokens())
                .collect::<Vec<_>>();
            let mut buckets = vec![0; positions.len()];
            self.prefill_batcher.buckets(&lengths, &mut buckets);
            for (position, bucket) in positions.into_iter().zip(buckets) {
                assigned_buckets[position] = bucket;
            }
        }

        let mut groups: Vec<(usize, PlannedBatch)> = Vec::new();
        for ((model_id, row), bucket) in rows.into_iter().zip(assigned_buckets) {
            if let Some((_, group)) = groups
                .iter_mut()
                .find(|(key, group)| group.model_id == model_id && *key == bucket)
            {
                group.rows.push(row);
            } else {
                groups.push((
                    bucket,
                    PlannedBatch {
                        phase,
                        model_id,
                        mode: BatchExecutionMode::Individual,
                        rows: vec![row],
                    },
                ));
            }
        }
        self.assign_batch_modes(groups.into_iter().map(|(_, group)| group).collect())
    }

    /// Marks compatible groups as tensor invocations when the model declares
    /// the matching hook, and splits the rest into one-row invocations.
    fn assign_batch_modes(&self, groups: Vec<PlannedBatch>) -> Vec<PlannedBatch> {
        let mut batches = Vec::with_capacity(groups.len());
        for group in groups {
            let capabilities = self.models.get(&group.model_id).map_or(
                BatchCapabilities::NONE,
                CausalLanguageModel::batch_capabilities,
            );
            let supported = match group.phase {
                PlannedPhase::Decode => capabilities.decode,
                PlannedPhase::Prefill | PlannedPhase::ContinuedPrefill => {
                    if group.rows.iter().all(is_complete_prompt) {
                        capabilities.prefill
                    } else {
                        capabilities.chunked_prefill
                    }
                }
            };
            if supported && group.rows.len() > 1 {
                batches.push(PlannedBatch {
                    mode: BatchExecutionMode::Tensor,
                    ..group
                });
                continue;
            }
            let PlannedBatch {
                phase,
                model_id,
                rows,
                ..
            } = group;
            batches.extend(rows.into_iter().map(|row| PlannedBatch {
                phase,
                model_id: model_id.clone(),
                mode: BatchExecutionMode::Individual,
                rows: vec![row],
            }));
        }
        batches
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
            outcome.execution_batches.push(ExecutionBatch {
                phase: BatchPhase::Prefill,
                request_ids: members.iter().map(|pending| pending.request_id).collect(),
                mode: BatchExecutionMode::Tensor,
                input_tokens: batch.input_tokens(),
                padding_tokens: batch.padding_tokens(),
            });
            for (((row, mut pending), prompt), result) in
                batch.rows.iter().zip(members).zip(prompts).zip(results)
            {
                if result.is_ok() {
                    pending.cursor = pending.cursor.saturating_add(prompt.len());
                }
                let event = self.finish_prefill(pending, result, at)?;
                outcome.indexed_events.push((row.event_index, event));
            }
        } else {
            for (row, mut pending) in batch.rows.iter().zip(members) {
                outcome.execution_batches.push(ExecutionBatch {
                    phase: BatchPhase::Prefill,
                    request_ids: vec![pending.request_id],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: row.input_tokens(),
                    padding_tokens: 0,
                });
                let result = execute_prefill_segment(&mut pending, row.input_tokens());
                let event = self.finish_prefill(pending, result, at)?;
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
        self.runtime.mark_first_token(pending.request_id, at)?;
        let active = ActiveGeneration {
            request_id: pending.request_id,
            model_id: pending.model_id,
            model: pending.model,
            session: pending.session,
            prompt_tokens: pending.prompt_tokens.len(),
            generated: vec![token.id],
            last_token: token,
            max_new_tokens: pending.max_new_tokens,
            last_step_at: at,
            reserved_kv_bytes: pending.reserved_kv_bytes,
        };
        self.observe_token(active, ExecutionPhase::Prefill, at)
    }

    fn start_next(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
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
        let lease = self.runtime.start_selected(request_id, at)?;
        let session = match model.create_session(request.spec.generation.clone()) {
            Ok(session) => session,
            Err(error) => return self.fail_started(lease.request_id, error.to_string(), at),
        };
        let mut pending = PendingPrefill {
            request_id: lease.request_id,
            model_id,
            model,
            session,
            prompt_tokens,
            cursor: 0,
            max_new_tokens: request.spec.generation.max_new_tokens,
            last_step_at: at,
            reserved_kv_bytes,
        };
        let chunk_len = self.next_prefill_chunk_len(&pending);
        let result = execute_prefill_segment(&mut pending, chunk_len);
        self.finish_prefill(pending, result, at)
    }

    fn prefill_next(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        let Some(request_id) = self.prefill_ready.pop_front() else {
            return Ok(ExecutorEvent::Idle);
        };
        let mut pending = self
            .prefilling
            .remove(&request_id)
            .expect("ready prefill requests retain their session");
        if at < pending.last_step_at {
            let previous = pending.last_step_at;
            self.prefilling.insert(request_id, pending);
            self.prefill_ready.push_front(request_id);
            return Err(RuntimeError::TimestampRegression {
                request: request_id,
                event: "executor prefill step",
                previous,
                observed: at,
            }
            .into());
        }
        let chunk_len = self.next_prefill_chunk_len(&pending);
        let result = execute_prefill_segment(&mut pending, chunk_len);
        self.finish_prefill(pending, result, at)
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
            outcome.execution_batches.push(ExecutionBatch {
                phase: BatchPhase::Decode,
                request_ids: members.iter().map(|active| active.request_id).collect(),
                mode: BatchExecutionMode::Tensor,
                input_tokens: members.len(),
                padding_tokens: 0,
            });
            for ((row, active), result) in batch.rows.iter().zip(members).zip(results) {
                let event = self.finish_decode(active, result, at)?;
                outcome.indexed_events.push((row.event_index, event));
            }
        } else {
            for (row, mut active) in batch.rows.iter().zip(members) {
                outcome.execution_batches.push(ExecutionBatch {
                    phase: BatchPhase::Decode,
                    request_ids: vec![active.request_id],
                    mode: BatchExecutionMode::Individual,
                    input_tokens: 1,
                    padding_tokens: 0,
                });
                let result = active.session.decode(active.last_token.id);
                let event = self.finish_decode(active, result, at)?;
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

    fn select_batch_candidates(
        &mut self,
        scheduled: &[BatchCandidate],
        candidates: &[BatchCandidate],
    ) -> Result<BTreeSet<RequestId>, ExecutorError> {
        let context = BatchContext {
            scheduled: scheduled.to_vec(),
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

    fn reserved_kv_bytes_for_model(&self, model_id: &ModelId) -> u64 {
        self.active
            .values()
            .filter(|active| &active.model_id == model_id)
            .map(|active| active.reserved_kv_bytes)
            .chain(
                self.prefilling
                    .values()
                    .filter(|pending| &pending.model_id == model_id)
                    .map(|pending| pending.reserved_kv_bytes),
            )
            .sum()
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn live_request_count(&self) -> usize {
        self.active.len().saturating_add(self.prefilling.len())
    }

    fn initial_prefill_chunk_len(
        &self,
        model: &CausalLanguageModel,
        prompt_tokens: usize,
    ) -> usize {
        if model.supports_chunked_prefill() {
            self.prefill_chunk_size
                .map_or(prompt_tokens, |chunk| prompt_tokens.min(chunk.get()))
        } else {
            prompt_tokens
        }
    }

    fn next_prefill_chunk_len(&self, pending: &PendingPrefill) -> usize {
        let remaining = pending.prompt_tokens.len().saturating_sub(pending.cursor);
        self.initial_prefill_chunk_len(&pending.model, remaining)
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

const fn budget_deferral(candidate: BatchCandidate, event_index: usize) -> PlannedDeferral {
    PlannedDeferral::Budget {
        request_id: candidate.request_id,
        phase: candidate.phase,
        token_cost: candidate.token_cost,
        event_index,
    }
}

const fn execution_phase(phase: BatchPhase) -> ExecutionPhase {
    match phase {
        BatchPhase::Prefill => ExecutionPhase::Prefill,
        BatchPhase::Decode => ExecutionPhase::Decode,
    }
}

fn is_complete_prompt(row: &PlannedRow) -> bool {
    row.positions.start == 0 && row.final_prompt_chunk
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
    fn chunked_prefill_and_decode_share_an_iteration_budget() {
        let policy = crate::FifoTokenBudgetPolicy::new(3).unwrap();
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_batch_policy(policy)
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
            .with_batch_policy(FirstOnly);
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
    fn cost_aware_prefill_applies_one_workload_level_decision() {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 4)
            .unwrap()
            .with_prefill_batcher(crate::CostAwarePrefillBatcher::new(32));
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
        let mut executor = Executor::default().with_batch_policy(policy);
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
