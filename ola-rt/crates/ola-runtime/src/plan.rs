//! Validated description of one executor iteration.
//!
//! See `EXECUTION_PLAN.md` in this crate for the design. A plan is pure data:
//! it is built from read-only state, checked by [`ExecutionPlan::validate`],
//! and only then committed by the executor.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::ops::Range;

use ola_core::{ModelId, Priority, RequestId, Timestamp};

use crate::{AdmissionReason, BatchExecutionMode, BatchPhase, PlanningSnapshot};

/// Model work performed by one planned invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlannedPhase {
    /// First prompt segment of a request admitted by this plan.
    Prefill,
    /// Later prompt segment of a request whose session already exists.
    ContinuedPrefill,
    /// One cached autoregressive position.
    Decode,
}

/// Lifecycle change applied before any planned batch executes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleAction {
    /// Start a queued request and hold its KV reservation while live.
    ///
    /// The request's event is produced by its [`PlannedPhase::Prefill`] row.
    Admit {
        /// Queued request that becomes live.
        request_id: RequestId,
        /// Model whose KV pool owns the reservation.
        model_id: ModelId,
        /// Worst-case KV bytes chosen by the admission policy.
        reserved_kv_bytes: u64,
    },
    /// Terminate a queued request that can never fit this configuration.
    Reject {
        /// Queued request moved to failed.
        request_id: RequestId,
        /// Typed permanent resource constraint.
        reason: AdmissionReason,
        /// Position of the resulting event in the iteration.
        event_index: usize,
    },
    /// Terminate a queued request that could not be prepared.
    Fail {
        /// Queued request moved to failed.
        request_id: RequestId,
        /// Diagnostic such as a missing model or tokenization error.
        message: String,
        /// Position of the resulting event in the iteration.
        event_index: usize,
    },
    /// Return a live request to the queue, releasing its session, KV
    /// blocks, slot, and reservation before any invocation runs. Its
    /// generated tokens are kept; it resumes by prefilling prompt and
    /// generated tokens together.
    Preempt {
        /// Live request moved back to the queue.
        request_id: RequestId,
        /// Why it was chosen.
        reason: PreemptionReason,
        /// Position of the resulting event in the iteration.
        event_index: usize,
    },
}

/// Why a live request was preempted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreemptionReason {
    /// The model's KV pool could not hold this iteration's growth.
    KvShortfall {
        /// Blocks missing before any preemption.
        shortfall_blocks: usize,
    },
    /// An allocation failed during execution, which the planner did not
    /// predict; the request is requeued instead of failed. Never planned.
    AllocationFailed,
}

impl LifecycleAction {
    /// Returns the request affected by this action.
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Admit { request_id, .. }
            | Self::Reject { request_id, .. }
            | Self::Fail { request_id, .. }
            | Self::Preempt { request_id, .. } => *request_id,
        }
    }
}

/// One request's share of a planned invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedRow {
    /// Request whose session executes this row.
    pub request_id: RequestId,
    /// Position of the resulting event in the iteration.
    pub event_index: usize,
    /// Absolute sequence positions processed, which also enter the KV cache.
    pub positions: Range<usize>,
    /// Whether this row consumes the end of the request's prompt.
    pub final_prompt_chunk: bool,
}

impl PlannedRow {
    /// Returns the real input positions processed by this row.
    pub fn input_tokens(&self) -> usize {
        self.positions.len()
    }
}

/// One planned model invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedBatch {
    /// Work performed by every row.
    pub phase: PlannedPhase,
    /// Loaded model shared by every row.
    pub model_id: ModelId,
    /// Intended execution mechanism.
    pub mode: BatchExecutionMode,
    /// Rows in tensor order.
    pub rows: Vec<PlannedRow>,
}

impl PlannedBatch {
    /// Returns the real input positions processed by the invocation.
    pub fn input_tokens(&self) -> usize {
        self.rows.iter().map(PlannedRow::input_tokens).sum()
    }

    /// Returns right-padding positions needed to make tensor rows rectangular.
    pub fn padding_tokens(&self) -> usize {
        let widest = self
            .rows
            .iter()
            .map(PlannedRow::input_tokens)
            .max()
            .unwrap_or(0);
        widest
            .saturating_mul(self.rows.len())
            .saturating_sub(self.input_tokens())
    }
}

/// Eligible work left for a later iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlannedDeferral {
    /// Admission policy left a queued request waiting for resources.
    Admission {
        /// Request that remains queued.
        request_id: RequestId,
        /// Typed resource constraint.
        reason: AdmissionReason,
        /// Position of the resulting event in the iteration.
        event_index: usize,
    },
    /// Batch policy left work outside this iteration's token budget.
    Budget {
        /// Request that remains queued or live.
        request_id: RequestId,
        /// Phase whose cost did not fit.
        phase: BatchPhase,
        /// Input positions the operation would process.
        token_cost: usize,
        /// Position of the resulting event in the iteration.
        event_index: usize,
    },
    /// No slot was free at the iteration boundary.
    ///
    /// Slots freed by an unpredicted EOS during this iteration are refilled by
    /// the next one; this records that wait. It produces no event.
    Slots {
        /// Request that remains queued.
        request_id: RequestId,
    },
}

impl PlannedDeferral {
    /// Returns the deferred request.
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Admission { request_id, .. }
            | Self::Budget { request_id, .. }
            | Self::Slots { request_id } => *request_id,
        }
    }

    const fn event_index(&self) -> Option<usize> {
        match self {
            Self::Admission { event_index, .. } | Self::Budget { event_index, .. } => {
                Some(*event_index)
            }
            Self::Slots { .. } => None,
        }
    }
}

/// Why a request left the live set without being a predicted release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseReason {
    /// The model selected an end-of-sequence token.
    EndOfSequence,
    /// The request reached `max_new_tokens` in a row the planner does not
    /// reuse slots for, such as the first prefill of a one-token request.
    OutputLimit,
    /// Model execution or output decoding failed.
    Failure,
}

/// One way committed execution differed from its validated plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanDivergence {
    /// A planned tensor invocation ran as individual invocations because the
    /// model's declared hook returned `None`.
    ModeFallback {
        /// Index of the batch in [`ExecutionPlan::batches`].
        batch_index: usize,
    },
    /// An admitted request could not create its session; its prefill row was
    /// dropped from the batch.
    SessionCreationFailed {
        /// Request that failed instead of starting.
        request_id: RequestId,
    },
    /// A planned row ended its request, freeing a slot and reservation that
    /// the plan could not reuse this iteration.
    UnplannedRelease {
        /// Request that left the live set.
        request_id: RequestId,
        /// Observed cause.
        reason: ReleaseReason,
    },
    /// A planned row failed because the model's KV pool ran out of blocks.
    KvCapacityExhausted {
        /// Request whose allocation failed.
        request_id: RequestId,
    },
}

/// Deterministic work estimates derived from a plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlanEstimates {
    /// Real input positions across every invocation.
    pub input_tokens: usize,
    /// Right-padding positions across every invocation.
    pub padding_tokens: usize,
    /// Planned model invocations.
    pub invocations: usize,
    /// Rows performing first or continued prompt work.
    pub prefill_rows: usize,
    /// Rows performing one decode step.
    pub decode_rows: usize,
    /// Operations left out by the batch policy's token budget.
    pub budget_deferrals: usize,
    /// KV bytes newly reserved by admitted requests.
    pub admitted_kv_bytes: u64,
    /// Requests whose final token is known to be produced by this plan.
    pub predicted_releases: usize,
}

/// Correlates an executor iteration, its plan, its invocations, and the
/// events and traces it produced.
///
/// The executor numbers committed iterations from 1; 0 means "not yet
/// assigned", as in a plan a policy has just built.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IterationId(u64);

impl IterationId {
    /// Wraps a raw iteration number.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw iteration number.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for IterationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Complete, validated description of one executor iteration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionPlan {
    /// Iteration that committed the plan. Assigned by the executor before
    /// validation, overwriting whatever the policy set.
    pub iteration: IterationId,
    /// Executor state generation the plan was built from.
    pub generation: u64,
    /// Caller-supplied iteration timestamp.
    pub at: Timestamp,
    /// Lifecycle changes applied before any batch executes.
    pub lifecycle: Vec<LifecycleAction>,
    /// Model invocations in execution order.
    pub batches: Vec<PlannedBatch>,
    /// Eligible work left for a later iteration.
    pub deferrals: Vec<PlannedDeferral>,
    /// Live requests whose row in this plan must finish them by reaching
    /// `max_new_tokens`: limit-reaching decodes, and final continued-prefill
    /// chunks with a one-token limit. Their slots and reservations may be
    /// reused by admissions in the same plan.
    pub predicted_releases: Vec<RequestId>,
}

/// Live request with an unfinished prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillView {
    /// Model that owns the session.
    pub model_id: ModelId,
    /// Prompt positions already accepted by the session.
    pub cursor: usize,
    /// Total tokenized prompt length.
    pub prompt_tokens: usize,
    /// Output tokens still allowed: the caller's limit minus any generated
    /// before a preemption.
    pub max_new_tokens: usize,
    /// KV bytes reserved for the request.
    pub reserved_kv_bytes: u64,
    /// Caller-supplied priority.
    pub priority: Priority,
    /// Time the request first entered the queue.
    pub arrived_at: Timestamp,
}

/// Live request ready for one decode step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeView {
    /// Model that owns the session.
    pub model_id: ModelId,
    /// Absolute position the next decode input occupies.
    pub position: usize,
    /// Output tokens emitted so far.
    pub generated_tokens: usize,
    /// Caller output limit.
    pub max_new_tokens: usize,
    /// KV bytes reserved for the request.
    pub reserved_kv_bytes: u64,
    /// Caller-supplied priority.
    pub priority: Priority,
    /// Time the request first entered the queue.
    pub arrived_at: Timestamp,
}

impl DecodeView {
    /// Returns whether the next decode must finish the request.
    pub const fn reaches_limit(&self) -> bool {
        self.generated_tokens.saturating_add(1) >= self.max_new_tokens
    }
}

/// Lookup tables built from a [`PlanningSnapshot`] for validation.
struct ValidationView<'a> {
    generation: u64,
    max_active_requests: usize,
    queued: BTreeMap<RequestId, QueuedView<'a>>,
    prefilling: &'a BTreeMap<RequestId, PrefillView>,
    decoding: &'a BTreeMap<RequestId, DecodeView>,
    model_kv_capacity: BTreeMap<&'a ModelId, u64>,
    chunked_models: BTreeSet<&'a ModelId>,
    prefill_chunk_size: Option<usize>,
}

struct QueuedView<'a> {
    model_id: &'a ModelId,
    prompt_tokens: Option<usize>,
}

impl<'a> ValidationView<'a> {
    fn new(snapshot: &'a PlanningSnapshot, enforce_model_kv_capacity: bool) -> Self {
        Self {
            generation: snapshot.generation,
            max_active_requests: snapshot.limits.max_active_requests,
            queued: snapshot
                .queued
                .iter()
                .map(|request| {
                    (
                        request.request_id,
                        QueuedView {
                            model_id: &request.model_id,
                            prompt_tokens: request.prompt_tokens.as_ref().ok().copied(),
                        },
                    )
                })
                .collect(),
            prefilling: &snapshot.prefilling,
            decoding: &snapshot.decoding,
            model_kv_capacity: snapshot
                .models
                .iter()
                .filter(|_| enforce_model_kv_capacity)
                .filter_map(|(model_id, model)| {
                    model
                        .kv_cache
                        .map(|profile| (model_id, profile.capacity_bytes()))
                })
                .collect(),
            chunked_models: snapshot
                .models
                .iter()
                .filter(|(_, model)| model.chunked_prefill)
                .map(|(model_id, _)| model_id)
                .collect(),
            prefill_chunk_size: snapshot.limits.prefill_chunk_size,
        }
    }

    fn live_requests(&self) -> usize {
        self.prefilling.len().saturating_add(self.decoding.len())
    }
}

/// Reason a plan cannot be committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// The plan was built from an older executor state.
    Stale {
        /// Generation recorded in the plan.
        planned: u64,
        /// Current executor generation.
        current: u64,
    },
    /// A request appears in more than one plan role.
    DuplicateRequest(RequestId),
    /// A request is not in the state its plan role requires.
    InvalidStatus {
        /// Offending request.
        request_id: RequestId,
        /// State the plan role requires.
        expected: &'static str,
    },
    /// An admitted request has no prefill row, or a prefill row no admission.
    UnpairedAdmission(RequestId),
    /// A batch contains no rows.
    EmptyBatch(usize),
    /// An individual invocation contains more than one row.
    IndividualBatchWidth(usize),
    /// A row's or admission's model differs from the request's model.
    ModelMismatch {
        /// Offending request.
        request_id: RequestId,
        /// Model that owns the request.
        expected: ModelId,
        /// Model named by the plan.
        planned: ModelId,
    },
    /// A row's positions or final-chunk flag contradict the request state.
    InvalidPositions(RequestId),
    /// A partial prompt row targets a model without resumable prefill, which
    /// would fail the request at execution.
    UnsupportedPartialPrefill(RequestId),
    /// A prompt row on a resumable model exceeds the executor's chunk size.
    ChunkTooLong {
        /// Offending request.
        request_id: RequestId,
        /// Positions the row processes.
        tokens: usize,
        /// Configured chunk size.
        limit: usize,
    },
    /// Event indices are not exactly `0..n`.
    InvalidEventOrder,
    /// A predicted release is not a row in this plan that must finish its
    /// request.
    InvalidRelease(RequestId),
    /// Admissions exceed the slots available after predicted releases.
    SlotCapacity {
        /// Live sessions after the plan commits.
        live_after: usize,
        /// Configured limit.
        max_active_requests: usize,
    },
    /// Reservations exceed a model's physical KV pool.
    KvCapacity {
        /// Model whose pool is exceeded.
        model_id: ModelId,
        /// Bytes reserved after the plan commits.
        reserved_bytes: u64,
        /// Pool capacity.
        capacity_bytes: u64,
    },
}

impl fmt::Display for ExecutionPlan {
    /// Writes a one-line summary such as
    /// `gen 7: 1 admitted, 3 invocations (1 tensor), 8+0 tokens, 1 deferred, 1 predicted release`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let estimates = self.estimates();
        let admitted = self
            .lifecycle
            .iter()
            .filter(|action| matches!(action, LifecycleAction::Admit { .. }))
            .count();
        let tensor = self
            .batches
            .iter()
            .filter(|batch| batch.mode == BatchExecutionMode::Tensor)
            .count();
        write!(
            formatter,
            "gen {}: {admitted} admitted, {} terminated, {} invocations ({tensor} tensor), \
             {}+{} tokens, {} deferred, {} predicted release{}",
            self.generation,
            self.lifecycle.len().saturating_sub(admitted),
            estimates.invocations,
            estimates.input_tokens,
            estimates.padding_tokens,
            self.deferrals.len(),
            estimates.predicted_releases,
            if estimates.predicted_releases == 1 {
                ""
            } else {
                "s"
            },
        )
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stale { planned, current } => write!(
                formatter,
                "plan was built at generation {planned} but the executor is at {current}"
            ),
            Self::DuplicateRequest(id) => {
                write!(
                    formatter,
                    "request {id:?} appears more than once in the plan"
                )
            }
            Self::InvalidStatus {
                request_id,
                expected,
            } => write!(formatter, "request {request_id:?} is not {expected}"),
            Self::UnpairedAdmission(id) => write!(
                formatter,
                "request {id:?} must have both an admission and a first prefill row"
            ),
            Self::EmptyBatch(index) => write!(formatter, "planned batch {index} has no rows"),
            Self::IndividualBatchWidth(index) => write!(
                formatter,
                "individual planned batch {index} has more than one row"
            ),
            Self::ModelMismatch {
                request_id,
                expected,
                planned,
            } => write!(
                formatter,
                "request {request_id:?} belongs to model `{expected}` but was planned on `{planned}`"
            ),
            Self::InvalidPositions(id) => write!(
                formatter,
                "planned positions for request {id:?} contradict its sequence state"
            ),
            Self::UnsupportedPartialPrefill(id) => write!(
                formatter,
                "request {id:?} has a partial prompt row but its model cannot resume prefill"
            ),
            Self::ChunkTooLong {
                request_id,
                tokens,
                limit,
            } => write!(
                formatter,
                "request {request_id:?} processes {tokens} prompt tokens but the chunk size is {limit}"
            ),
            Self::InvalidEventOrder => {
                formatter.write_str("planned event indices are not a contiguous permutation")
            }
            Self::InvalidRelease(id) => write!(
                formatter,
                "request {id:?} is not guaranteed to finish in this plan"
            ),
            Self::SlotCapacity {
                live_after,
                max_active_requests,
            } => write!(
                formatter,
                "plan leaves {live_after} live sessions but the limit is {max_active_requests}"
            ),
            Self::KvCapacity {
                model_id,
                reserved_bytes,
                capacity_bytes,
            } => write!(
                formatter,
                "model `{model_id}` would reserve {reserved_bytes} KV bytes of {capacity_bytes}"
            ),
        }
    }
}

impl Error for PlanError {}

impl ExecutionPlan {
    /// Returns deterministic work estimates for the plan.
    pub fn estimates(&self) -> PlanEstimates {
        PlanEstimates {
            input_tokens: self.batches.iter().map(PlannedBatch::input_tokens).sum(),
            padding_tokens: self.batches.iter().map(PlannedBatch::padding_tokens).sum(),
            invocations: self.batches.len(),
            prefill_rows: self.rows_in(|phase| phase != PlannedPhase::Decode),
            decode_rows: self.rows_in(|phase| phase == PlannedPhase::Decode),
            budget_deferrals: self
                .deferrals
                .iter()
                .filter(|deferral| matches!(deferral, PlannedDeferral::Budget { .. }))
                .count(),
            admitted_kv_bytes: self
                .lifecycle
                .iter()
                .map(|action| match action {
                    LifecycleAction::Admit {
                        reserved_kv_bytes, ..
                    } => *reserved_kv_bytes,
                    _ => 0,
                })
                .sum(),
            predicted_releases: self.predicted_releases.len(),
        }
    }

    fn rows_in(&self, phase: impl Fn(PlannedPhase) -> bool) -> usize {
        self.batches
            .iter()
            .filter(|batch| phase(batch.phase))
            .map(|batch| batch.rows.len())
            .sum()
    }

    /// Checks that the plan can be committed against `snapshot` without
    /// partial mutation.
    ///
    /// `enforce_model_kv_capacity` rejects admissions that push a model past
    /// its KV pool; pass the admission policy's own guarantee, since
    /// permissive policies leave allocation failures to execution.
    ///
    /// # Errors
    ///
    /// Returns the first violated rule; see [`PlanError`].
    pub fn validate(
        &self,
        snapshot: &PlanningSnapshot,
        enforce_model_kv_capacity: bool,
    ) -> Result<(), PlanError> {
        let view = &ValidationView::new(snapshot, enforce_model_kv_capacity);
        if self.generation != view.generation {
            return Err(PlanError::Stale {
                planned: self.generation,
                current: view.generation,
            });
        }

        // Every request has exactly one role, except that an admission is
        // paired with its first prefill row.
        let mut claimed = BTreeSet::new();
        let mut claim = |request_id| {
            if claimed.insert(request_id) {
                Ok(())
            } else {
                Err(PlanError::DuplicateRequest(request_id))
            }
        };
        let mut admitted = BTreeMap::new();
        let mut event_indices = Vec::new();
        let mut preempted = BTreeSet::new();
        for action in &self.lifecycle {
            let request_id = action.request_id();
            if let LifecycleAction::Preempt { event_index, .. } = action {
                if !view.decoding.contains_key(&request_id)
                    && !view.prefilling.contains_key(&request_id)
                {
                    return Err(PlanError::InvalidStatus {
                        request_id,
                        expected: "decoding or prefilling",
                    });
                }
                claim(request_id)?;
                event_indices.push(*event_index);
                preempted.insert(request_id);
                continue;
            }
            let Some(queued) = view.queued.get(&request_id) else {
                return Err(PlanError::InvalidStatus {
                    request_id,
                    expected: "queued",
                });
            };
            match action {
                LifecycleAction::Admit {
                    model_id,
                    reserved_kv_bytes,
                    ..
                } => {
                    if admitted
                        .insert(request_id, (model_id, *reserved_kv_bytes))
                        .is_some()
                    {
                        return Err(PlanError::DuplicateRequest(request_id));
                    }
                    if model_id != queued.model_id {
                        return Err(PlanError::ModelMismatch {
                            request_id,
                            expected: queued.model_id.clone(),
                            planned: model_id.clone(),
                        });
                    }
                }
                LifecycleAction::Reject { event_index, .. }
                | LifecycleAction::Fail { event_index, .. } => {
                    claim(request_id)?;
                    event_indices.push(*event_index);
                }
                LifecycleAction::Preempt { .. } => unreachable!("handled above"),
            }
        }

        let mut prefilled = BTreeSet::new();
        let mut finishing = BTreeSet::new();
        for (index, batch) in self.batches.iter().enumerate() {
            if batch.rows.is_empty() {
                return Err(PlanError::EmptyBatch(index));
            }
            if batch.mode == BatchExecutionMode::Individual && batch.rows.len() > 1 {
                return Err(PlanError::IndividualBatchWidth(index));
            }
            for row in &batch.rows {
                claim(row.request_id)?;
                event_indices.push(row.event_index);
                validate_row(view, batch, row)?;
                match batch.phase {
                    PlannedPhase::Prefill => {
                        prefilled.insert(row.request_id);
                    }
                    PlannedPhase::Decode => {
                        if view
                            .decoding
                            .get(&row.request_id)
                            .is_some_and(DecodeView::reaches_limit)
                        {
                            finishing.insert(row.request_id);
                        }
                    }
                    PlannedPhase::ContinuedPrefill => {
                        if row.final_prompt_chunk
                            && view
                                .prefilling
                                .get(&row.request_id)
                                .is_some_and(|pending| pending.max_new_tokens <= 1)
                        {
                            finishing.insert(row.request_id);
                        }
                    }
                }
            }
        }
        for request_id in admitted.keys().chain(&prefilled) {
            if admitted.contains_key(request_id) != prefilled.contains(request_id) {
                return Err(PlanError::UnpairedAdmission(*request_id));
            }
        }

        for deferral in &self.deferrals {
            let request_id = deferral.request_id();
            claim(request_id)?;
            event_indices.extend(deferral.event_index());
            let (eligible, expected) = match deferral {
                PlannedDeferral::Budget {
                    phase: BatchPhase::Decode,
                    ..
                } => (view.decoding.contains_key(&request_id), "decoding"),
                PlannedDeferral::Budget {
                    phase: BatchPhase::Prefill,
                    ..
                } => (
                    view.queued.contains_key(&request_id)
                        || view.prefilling.contains_key(&request_id),
                    "queued or prefilling",
                ),
                PlannedDeferral::Admission { .. } | PlannedDeferral::Slots { .. } => {
                    (view.queued.contains_key(&request_id), "queued")
                }
            };
            if !eligible {
                return Err(PlanError::InvalidStatus {
                    request_id,
                    expected,
                });
            }
        }

        event_indices.sort_unstable();
        if event_indices
            .iter()
            .enumerate()
            .any(|(expected, index)| expected != *index)
        {
            return Err(PlanError::InvalidEventOrder);
        }

        let mut released = BTreeSet::new();
        for request_id in &self.predicted_releases {
            if !finishing.contains(request_id) || !released.insert(*request_id) {
                return Err(PlanError::InvalidRelease(*request_id));
            }
        }
        // Preemption frees a slot and a reservation before admissions run.
        released.extend(preempted);

        let live_after = view
            .live_requests()
            .saturating_sub(released.len())
            .saturating_add(admitted.len());
        if live_after > view.max_active_requests {
            return Err(PlanError::SlotCapacity {
                live_after,
                max_active_requests: view.max_active_requests,
            });
        }

        let mut reserved = BTreeMap::<&ModelId, u64>::new();
        let live = view
            .prefilling
            .iter()
            .map(|(id, view)| (id, &view.model_id, view.reserved_kv_bytes))
            .chain(
                view.decoding
                    .iter()
                    .map(|(id, view)| (id, &view.model_id, view.reserved_kv_bytes)),
            );
        for (request_id, model_id, bytes) in live {
            if !released.contains(request_id) {
                *reserved.entry(model_id).or_default() += bytes;
            }
        }
        for (model_id, bytes) in admitted.values() {
            *reserved.entry(model_id).or_default() += bytes;
        }
        // Only admissions in this plan are checked: a permissive admission
        // policy may already have committed live reservations past capacity.
        let admitting = admitted
            .values()
            .map(|(model_id, _)| *model_id)
            .collect::<BTreeSet<_>>();
        for (model_id, reserved_bytes) in reserved {
            if admitting.contains(model_id)
                && let Some(&capacity_bytes) = view.model_kv_capacity.get(&model_id)
                && reserved_bytes > capacity_bytes
            {
                return Err(PlanError::KvCapacity {
                    model_id: model_id.clone(),
                    reserved_bytes,
                    capacity_bytes,
                });
            }
        }
        Ok(())
    }
}

fn validate_row(
    view: &ValidationView<'_>,
    batch: &PlannedBatch,
    row: &PlannedRow,
) -> Result<(), PlanError> {
    let request_id = row.request_id;
    let (expected, prompt_range) = match batch.phase {
        PlannedPhase::Prefill => {
            let queued = view
                .queued
                .get(&request_id)
                .ok_or(PlanError::InvalidStatus {
                    request_id,
                    expected: "queued",
                })?;
            let prompt = queued
                .prompt_tokens
                .ok_or(PlanError::InvalidPositions(request_id))?;
            (queued.model_id, Some((0, prompt)))
        }
        PlannedPhase::ContinuedPrefill => {
            let pending = view
                .prefilling
                .get(&request_id)
                .ok_or(PlanError::InvalidStatus {
                    request_id,
                    expected: "prefilling",
                })?;
            (
                &pending.model_id,
                Some((pending.cursor, pending.prompt_tokens)),
            )
        }
        PlannedPhase::Decode => {
            let decode = view
                .decoding
                .get(&request_id)
                .ok_or(PlanError::InvalidStatus {
                    request_id,
                    expected: "decoding",
                })?;
            if row.positions != (decode.position..decode.position.saturating_add(1))
                || row.final_prompt_chunk
            {
                return Err(PlanError::InvalidPositions(request_id));
            }
            (&decode.model_id, None)
        }
    };
    if expected != &batch.model_id {
        return Err(PlanError::ModelMismatch {
            request_id,
            expected: expected.clone(),
            planned: batch.model_id.clone(),
        });
    }
    if let Some((cursor, prompt)) = prompt_range {
        let valid = row.positions.start == cursor
            && !row.positions.is_empty()
            && row.positions.end <= prompt
            && row.final_prompt_chunk == (row.positions.end == prompt);
        if !valid {
            return Err(PlanError::InvalidPositions(request_id));
        }
        let resumable = view.chunked_models.contains(expected);
        if !row.final_prompt_chunk && !resumable {
            return Err(PlanError::UnsupportedPartialPrefill(request_id));
        }
        // Models without resumable prefill always process whole prompts, so
        // the chunk size only bounds resumable models.
        if let Some(limit) = view.prefill_chunk_size
            && resumable
            && row.input_tokens() > limit
        {
            return Err(PlanError::ChunkTooLong {
                request_id,
                tokens: row.input_tokens(),
                limit,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ola_core::Priority;
    use ola_model::{BatchCapabilities, KvCacheProfile};

    use super::*;
    use crate::{ExecutorLimits, MemoryState, ModelSnapshot, PolicyMeasurements, QueuedRequest};

    fn id(value: u64) -> RequestId {
        RequestId::new(value)
    }

    fn model() -> ModelId {
        "tiny".into()
    }

    /// One queued request (1), one prefilling request (2), and two decoding
    /// requests (3 reaches its limit, 4 does not), with two free slots and a
    /// 400-byte KV pool.
    fn view() -> PlanningSnapshot {
        PlanningSnapshot {
            at: Timestamp::from_micros(10),
            generation: 7,
            limits: ExecutorLimits {
                max_active_requests: 4,
                prefill_chunk_size: None,
            },
            queued: vec![QueuedRequest {
                request_id: id(1),
                model_id: model(),
                prompt_tokens: Ok(5),
                max_new_tokens: 4,
                priority: Priority::default(),
                deadline: None,
                queued_at: Timestamp::ZERO,
            }],
            prefilling: BTreeMap::from([(
                id(2),
                PrefillView {
                    model_id: model(),
                    cursor: 3,
                    prompt_tokens: 7,
                    max_new_tokens: 4,
                    reserved_kv_bytes: 100,
                    priority: ola_core::Priority::NORMAL,
                    arrived_at: Timestamp::ZERO,
                },
            )]),
            decoding: BTreeMap::from([
                (
                    id(3),
                    DecodeView {
                        model_id: model(),
                        position: 9,
                        generated_tokens: 3,
                        max_new_tokens: 4,
                        reserved_kv_bytes: 100,
                        priority: ola_core::Priority::NORMAL,
                        arrived_at: Timestamp::ZERO,
                    },
                ),
                (
                    id(4),
                    DecodeView {
                        model_id: model(),
                        position: 6,
                        generated_tokens: 1,
                        max_new_tokens: 8,
                        reserved_kv_bytes: 100,
                        priority: ola_core::Priority::NORMAL,
                        arrived_at: Timestamp::ZERO,
                    },
                ),
            ]),
            decode_order: vec![id(3), id(4)],
            prefill_order: vec![id(2)],
            models: BTreeMap::from([(
                model(),
                ModelSnapshot {
                    attached: true,
                    batch: BatchCapabilities::ALL,
                    chunked_prefill: true,
                    kv_cache: KvCacheProfile::new(1, 1, 400),
                    kv_pool: None,
                },
            )]),
            memory: MemoryState::default(),
            device: None,
            measurements: PolicyMeasurements::default(),
        }
    }

    fn row(request: u64, event_index: usize, positions: Range<usize>, last: bool) -> PlannedRow {
        PlannedRow {
            request_id: id(request),
            event_index,
            positions,
            final_prompt_chunk: last,
        }
    }

    fn plan() -> ExecutionPlan {
        ExecutionPlan {
            iteration: IterationId::default(),
            generation: 7,
            at: Timestamp::from_micros(10),
            lifecycle: vec![LifecycleAction::Admit {
                request_id: id(1),
                model_id: model(),
                reserved_kv_bytes: 100,
            }],
            batches: vec![
                PlannedBatch {
                    phase: PlannedPhase::Decode,
                    model_id: model(),
                    mode: BatchExecutionMode::Tensor,
                    rows: vec![row(3, 0, 9..10, false), row(4, 1, 6..7, false)],
                },
                PlannedBatch {
                    phase: PlannedPhase::ContinuedPrefill,
                    model_id: model(),
                    mode: BatchExecutionMode::Individual,
                    rows: vec![row(2, 2, 3..7, true)],
                },
                PlannedBatch {
                    phase: PlannedPhase::Prefill,
                    model_id: model(),
                    mode: BatchExecutionMode::Individual,
                    rows: vec![row(1, 3, 0..2, false)],
                },
            ],
            deferrals: Vec::new(),
            predicted_releases: vec![id(3)],
        }
    }

    #[test]
    fn accepts_a_consistent_plan_and_derives_estimates() {
        let plan = plan();
        assert_eq!(plan.validate(&view(), true), Ok(()));
        assert_eq!(
            plan.estimates(),
            PlanEstimates {
                input_tokens: 8,
                padding_tokens: 0,
                invocations: 3,
                prefill_rows: 2,
                decode_rows: 2,
                budget_deferrals: 0,
                admitted_kv_bytes: 100,
                predicted_releases: 1,
            }
        );
        assert_eq!(
            plan.to_string(),
            "gen 7: 1 admitted, 0 terminated, 3 invocations (1 tensor), 8+0 tokens, \
             0 deferred, 1 predicted release"
        );
    }

    #[test]
    fn padding_is_derived_from_rectangular_tensor_rows() {
        let batch = PlannedBatch {
            phase: PlannedPhase::Prefill,
            model_id: model(),
            mode: BatchExecutionMode::Tensor,
            rows: vec![row(1, 0, 0..5, true), row(2, 1, 0..2, true)],
        };
        assert_eq!(batch.input_tokens(), 7);
        assert_eq!(batch.padding_tokens(), 3);
    }

    #[test]
    fn rejects_stale_plans() {
        let mut view = view();
        view.generation = 8;
        assert_eq!(
            plan().validate(&view, true),
            Err(PlanError::Stale {
                planned: 7,
                current: 8
            })
        );
    }

    #[test]
    fn rejects_duplicate_requests_across_roles() {
        let mut duplicate_row = plan();
        duplicate_row.batches[0].rows[1].request_id = id(3);
        assert_eq!(
            duplicate_row.validate(&view(), true),
            Err(PlanError::DuplicateRequest(id(3)))
        );

        let mut executed_and_deferred = plan();
        executed_and_deferred
            .deferrals
            .push(PlannedDeferral::Budget {
                request_id: id(4),
                phase: BatchPhase::Decode,
                token_cost: 1,
                event_index: 4,
            });
        assert_eq!(
            executed_and_deferred.validate(&view(), true),
            Err(PlanError::DuplicateRequest(id(4)))
        );
    }

    #[test]
    fn rejects_partial_prompts_on_models_that_cannot_resume_them() {
        // The fixture admits request 1 with a partial row (0..2 of 5).
        let mut snapshot = view();
        snapshot.models.get_mut(&model()).unwrap().chunked_prefill = false;

        assert_eq!(
            plan().validate(&snapshot, true),
            Err(PlanError::UnsupportedPartialPrefill(id(1)))
        );
    }

    #[test]
    fn enforces_the_chunk_size_on_resumable_models_only() {
        // The continued row for request 2 processes four positions.
        let mut limited = view();
        limited.limits.prefill_chunk_size = Some(3);
        assert_eq!(
            plan().validate(&limited, true),
            Err(PlanError::ChunkTooLong {
                request_id: id(2),
                tokens: 4,
                limit: 3,
            })
        );

        limited.limits.prefill_chunk_size = Some(4);
        assert_eq!(plan().validate(&limited, true), Ok(()));

        // A model without resumable prefill processes whole prompts, so a
        // configured chunk size does not bound its complete-prompt rows.
        let mut whole_prompts = plan();
        whole_prompts.batches[2].rows[0].positions = 0..5;
        whole_prompts.batches[2].rows[0].final_prompt_chunk = true;
        let mut unchunked = view();
        unchunked.limits.prefill_chunk_size = Some(2);
        unchunked.models.get_mut(&model()).unwrap().chunked_prefill = false;
        unchunked.prefilling.clear();
        whole_prompts.batches.remove(1);
        whole_prompts.batches[1].rows[0].event_index = 2;
        unchunked
            .decoding
            .get_mut(&id(4))
            .unwrap()
            .reserved_kv_bytes = 0;
        assert_eq!(whole_prompts.validate(&unchunked, true), Ok(()));
    }

    #[test]
    fn rejects_rows_that_contradict_request_state() {
        let mut wrong_cursor = plan();
        wrong_cursor.batches[1].rows[0].positions = 2..7;
        assert_eq!(
            wrong_cursor.validate(&view(), true),
            Err(PlanError::InvalidPositions(id(2)))
        );

        let mut wrong_final_flag = plan();
        wrong_final_flag.batches[1].rows[0].final_prompt_chunk = false;
        assert_eq!(
            wrong_final_flag.validate(&view(), true),
            Err(PlanError::InvalidPositions(id(2)))
        );

        let mut decode_of_queued = plan();
        decode_of_queued.lifecycle.clear();
        decode_of_queued.batches.truncate(2);
        decode_of_queued.batches[0].rows[1].request_id = id(1);
        assert_eq!(
            decode_of_queued.validate(&view(), true),
            Err(PlanError::InvalidStatus {
                request_id: id(1),
                expected: "decoding"
            })
        );

        let mut wrong_model = plan();
        wrong_model.batches[1].model_id = "other".into();
        assert!(matches!(
            wrong_model.validate(&view(), true),
            Err(PlanError::ModelMismatch { .. })
        ));
    }

    #[test]
    fn rejects_malformed_batches_and_admissions() {
        let mut wide_individual = plan();
        wide_individual.batches[0].mode = BatchExecutionMode::Individual;
        assert_eq!(
            wide_individual.validate(&view(), true),
            Err(PlanError::IndividualBatchWidth(0))
        );

        let mut empty = plan();
        empty.batches.push(PlannedBatch {
            phase: PlannedPhase::Decode,
            model_id: model(),
            mode: BatchExecutionMode::Tensor,
            rows: Vec::new(),
        });
        assert_eq!(empty.validate(&view(), true), Err(PlanError::EmptyBatch(3)));

        let mut unpaired = plan();
        unpaired.batches.pop();
        unpaired.batches[1].rows[0].event_index = 2;
        assert_eq!(
            unpaired.validate(&view(), true),
            Err(PlanError::UnpairedAdmission(id(1)))
        );
    }

    #[test]
    fn rejects_gapped_or_repeated_event_indices() {
        let mut gapped = plan();
        gapped.batches[2].rows[0].event_index = 5;
        assert_eq!(
            gapped.validate(&view(), true),
            Err(PlanError::InvalidEventOrder)
        );
    }

    #[test]
    fn rejects_slot_and_kv_over_capacity() {
        let mut no_release = plan();
        no_release.predicted_releases.clear();
        let mut tight = view();
        tight.limits.max_active_requests = 3;
        assert_eq!(
            no_release.validate(&tight, true),
            Err(PlanError::SlotCapacity {
                live_after: 4,
                max_active_requests: 3
            })
        );
        // Predicting request 3's limit-reached completion frees its slot.
        assert_eq!(plan().validate(&tight, true), Ok(()));

        let mut large = plan();
        large.lifecycle[0] = LifecycleAction::Admit {
            request_id: id(1),
            model_id: model(),
            reserved_kv_bytes: 201,
        };
        assert_eq!(
            large.validate(&view(), true),
            Err(PlanError::KvCapacity {
                model_id: model(),
                reserved_bytes: 401,
                capacity_bytes: 400
            })
        );
    }

    #[test]
    fn rejects_releases_that_are_not_limit_reaching_decodes() {
        let mut early = plan();
        early.predicted_releases = vec![id(4)];
        assert_eq!(
            early.validate(&view(), true),
            Err(PlanError::InvalidRelease(id(4)))
        );

        let mut not_executed = plan();
        not_executed.batches[0].rows.remove(0);
        not_executed.batches[0].rows[0].event_index = 0;
        not_executed.batches[1].rows[0].event_index = 1;
        not_executed.batches[2].rows[0].event_index = 2;
        assert_eq!(
            not_executed.validate(&view(), true),
            Err(PlanError::InvalidRelease(id(3)))
        );
    }

    #[test]
    fn final_prefill_chunks_release_only_one_token_requests() {
        let mut plan = plan();
        plan.predicted_releases.push(id(2));
        assert_eq!(
            plan.validate(&view(), true),
            Err(PlanError::InvalidRelease(id(2)))
        );

        let mut one_token = view();
        one_token.prefilling.get_mut(&id(2)).unwrap().max_new_tokens = 1;
        assert_eq!(plan.validate(&one_token, true), Ok(()));
    }

    #[test]
    fn validates_deferral_eligibility() {
        let mut deferred_live = plan();
        deferred_live.lifecycle.clear();
        deferred_live.batches.pop();
        deferred_live
            .deferrals
            .push(PlannedDeferral::Slots { request_id: id(2) });
        assert_eq!(
            deferred_live.validate(&view(), true),
            Err(PlanError::DuplicateRequest(id(2)))
        );

        let mut queued_waits = plan();
        queued_waits.lifecycle.clear();
        queued_waits.batches.pop();
        queued_waits
            .deferrals
            .push(PlannedDeferral::Slots { request_id: id(1) });
        assert_eq!(queued_waits.validate(&view(), true), Ok(()));
    }
}
