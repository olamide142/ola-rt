//! Versioned JSON representation of one executor iteration's plan.
//!
//! `ola-runtime` keeps its plan types free of serialization concerns. This
//! module converts an [`ExecutionPlan`] and its divergences into stable,
//! JSON-friendly records so iterations can be retained beside benchmark
//! reports and compared across revisions.

use std::{error::Error, fmt};

use ola_runtime::{
    AdmissionDecision, AdmissionOutcome, AdmissionRecord, BatchCandidate, BatchExecutionMode,
    BatchHook, BatchPhase, CostAwarePrefillDecision, DecisionRecord, ExecutionPlan,
    ExecutorIteration, LifecycleAction, PlanDivergence, PlanEstimates, PlannedBatch,
    PlannedDeferral, PlannedPhase, PrefillGroupingEstimate, PrefillGroupingStrategy, ReleaseReason,
    SelectionStage,
};
use serde::{Deserialize, Serialize};

/// Current schema version for serialized execution plans.
///
/// Version 2 added [`PlanReport::decision`].
pub const PLAN_REPORT_SCHEMA_VERSION: u32 = 2;

/// One iteration's validated plan and observed divergences.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReport {
    /// Version of this record layout.
    pub schema_version: u32,
    /// Executor state generation the plan was built from.
    pub generation: u64,
    /// Iteration timestamp in microseconds.
    pub at_micros: u64,
    /// Lifecycle changes applied before any invocation.
    pub lifecycle: Vec<PlanReportLifecycle>,
    /// Model invocations in execution order.
    pub batches: Vec<PlanReportBatch>,
    /// Eligible work left for a later iteration.
    pub deferrals: Vec<PlanReportDeferral>,
    /// Requests whose planned row must finish them at their output limit.
    pub predicted_releases: Vec<u64>,
    /// Ways execution differed from the plan.
    pub divergences: Vec<PlanReportDivergence>,
    /// Deterministic estimates derived from the plan.
    pub estimates: PlanReportEstimates,
    /// Why the policy planned it.
    pub decision: PlanReportDecision,
}

/// Serialized [`DecisionRecord`].
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportDecision {
    /// Policy descriptor, rendered as `name{config}[components]`.
    pub policy: String,
    /// Scheduler ranking.
    pub ranking: PlanReportRanking,
    /// Outcome for every ranked request.
    pub admissions: Vec<PlanReportAdmission>,
    /// Admission-deferred requests that backfill skipped past.
    pub backfill_skips: Vec<u64>,
    /// Every batch-policy call.
    pub selections: Vec<PlanReportSelection>,
    /// Every prefill-grouping call.
    pub grouping: Vec<PlanReportGrouping>,
    /// Multi-row groups planned individually for lack of a batch hook.
    pub capability_fallbacks: Vec<PlanReportCapabilityFallback>,
}

/// Serialized ranking record.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportRanking {
    /// Slots free for admission after predicted releases.
    pub slots_available: usize,
    /// Requests asked of the scheduler.
    pub limit: usize,
    /// Ranked request identifiers.
    pub ranked: Vec<u64>,
}

/// Serialized admission outcome.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PlanReportAdmission {
    /// No slot remained.
    NoSlot {
        /// Request identifier.
        request_id: u64,
    },
    /// Preparation failed before admission.
    PreparationFailed {
        /// Request identifier.
        request_id: u64,
        /// Failure diagnostic.
        message: String,
    },
    /// The admission policy decided.
    Decided {
        /// Request identifier.
        request_id: u64,
        /// Prompt length admission saw.
        prompt_tokens: usize,
        /// Output limit admission saw.
        max_new_tokens: usize,
        /// Device-wide reserved KV bytes admission saw.
        reserved_kv_bytes: u64,
        /// Model-pool reserved KV bytes admission saw.
        model_reserved_kv_bytes: u64,
        /// `admit`, `defer`, or `reject`.
        decision: String,
        /// Bytes reserved on admission.
        granted_kv_bytes: Option<u64>,
        /// Reason for a deferral or rejection.
        reason: Option<String>,
        /// Whether an admitted request's first prefill fit the budget.
        budget_selected: Option<bool>,
    },
}

/// Serialized [`SelectionStage`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReportSelectionStage {
    /// Live decodes.
    Decode,
    /// Live unfinished prompts.
    ContinuedPrefill,
    /// One admitted request's first prefill.
    Admission,
}

/// Serialized batch candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportCandidate {
    /// Request identifier.
    pub request_id: u64,
    /// Prefill or decode.
    pub phase: PlanReportPhase,
    /// Input positions the operation processes.
    pub token_cost: usize,
}

/// Serialized batch-policy call.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportSelection {
    /// Stage that asked.
    pub stage: PlanReportSelectionStage,
    /// Work already scheduled before the call.
    pub scheduled: Vec<PlanReportCandidate>,
    /// Queued requests at the iteration boundary.
    pub queued_requests: usize,
    /// Live sessions at the iteration boundary.
    pub live_requests: usize,
    /// Candidates offered.
    pub candidates: Vec<PlanReportCandidate>,
    /// Requests selected.
    pub selected: Vec<u64>,
}

/// Serialized grouping strategy cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportGroupingCost {
    /// Invocations required.
    pub invocations: usize,
    /// Real plus padded positions.
    pub dense_input_tokens: usize,
    /// Dense positions plus invocation overhead.
    pub estimated_cost: usize,
}

/// Serialized cost-aware grouping alternatives.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportGroupingEstimates {
    /// `unbucketed` or `power_of_two`.
    pub selected: PlanReportGroupingStrategy,
    /// One-invocation estimate.
    pub unbucketed: PlanReportGroupingCost,
    /// Power-of-two estimate.
    pub power_of_two: PlanReportGroupingCost,
}

/// Serialized grouping strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReportGroupingStrategy {
    /// One invocation.
    Unbucketed,
    /// Power-of-two buckets.
    PowerOfTwo,
}

/// Serialized prefill-grouping call.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportGrouping {
    /// Prompt phase grouped.
    pub phase: PlanReportPhase,
    /// Model identifier.
    pub model: String,
    /// Row lengths seen.
    pub prompt_tokens: Vec<usize>,
    /// Bucket key per row.
    pub buckets: Vec<usize>,
    /// Alternatives weighed, when the rule reports them.
    pub cost_estimates: Option<PlanReportGroupingEstimates>,
}

/// Serialized [`BatchHook`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReportBatchHook {
    /// Complete-prompt batching.
    Prefill,
    /// Resumable-prompt batching.
    ChunkedPrefill,
    /// Decode batching.
    Decode,
}

/// Serialized capability fallback.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportCapabilityFallback {
    /// Phase of the group.
    pub phase: PlanReportPhase,
    /// Model identifier.
    pub model: String,
    /// Rows planned individually.
    pub rows: usize,
    /// Hook the model did not declare.
    pub missing: PlanReportBatchHook,
}

/// Serialized [`LifecycleAction`].
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PlanReportLifecycle {
    /// A queued request starts with a KV reservation.
    Admit {
        /// Request identifier.
        request_id: u64,
        /// Model identifier.
        model: String,
        /// Reserved KV bytes.
        reserved_kv_bytes: u64,
    },
    /// A queued request is permanently rejected.
    Reject {
        /// Request identifier.
        request_id: u64,
        /// Human-readable admission reason.
        reason: String,
        /// Position of the resulting event.
        event_index: usize,
    },
    /// A queued request failed preparation.
    Fail {
        /// Request identifier.
        request_id: u64,
        /// Diagnostic message.
        message: String,
        /// Position of the resulting event.
        event_index: usize,
    },
}

/// Serialized [`PlannedPhase`] or budget-deferral phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReportPhase {
    /// First prompt segment.
    Prefill,
    /// Later prompt segment.
    ContinuedPrefill,
    /// One decode step.
    Decode,
}

/// Serialized [`BatchExecutionMode`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReportMode {
    /// One shared tensor invocation.
    Tensor,
    /// One invocation per request.
    Individual,
}

/// Serialized [`PlannedBatch`].
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportBatch {
    /// Work performed by every row.
    pub phase: PlanReportPhase,
    /// Model identifier.
    pub model: String,
    /// Intended execution mechanism.
    pub mode: PlanReportMode,
    /// Rows in tensor order.
    pub rows: Vec<PlanReportRow>,
    /// Real input positions.
    pub input_tokens: usize,
    /// Right-padding positions.
    pub padding_tokens: usize,
}

/// Serialized planned row.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportRow {
    /// Request identifier.
    pub request_id: u64,
    /// Position of the resulting event.
    pub event_index: usize,
    /// First absolute position processed.
    pub start_position: usize,
    /// One past the last absolute position processed.
    pub end_position: usize,
    /// Whether the row ends its prompt.
    pub final_prompt_chunk: bool,
}

/// Serialized [`PlannedDeferral`].
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanReportDeferral {
    /// The admission policy deferred a queued request.
    Admission {
        /// Request identifier.
        request_id: u64,
        /// Human-readable admission reason.
        reason: String,
        /// Position of the resulting event.
        event_index: usize,
    },
    /// The batch policy's token budget left work out.
    Budget {
        /// Request identifier.
        request_id: u64,
        /// Prefill or decode.
        phase: PlanReportPhase,
        /// Input positions the operation would process.
        token_cost: usize,
        /// Position of the resulting event.
        event_index: usize,
    },
    /// No slot was free at the iteration boundary.
    Slots {
        /// Request identifier.
        request_id: u64,
    },
}

/// Serialized [`ReleaseReason`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReportReleaseReason {
    /// End-of-sequence token.
    EndOfSequence,
    /// Output limit in a row whose slot is not reused.
    OutputLimit,
    /// Execution or decoding failure.
    Failure,
}

/// Serialized [`PlanDivergence`].
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanReportDivergence {
    /// A planned tensor invocation ran individually.
    ModeFallback {
        /// Index into `batches`.
        batch_index: usize,
    },
    /// An admitted request could not create its session.
    SessionCreationFailed {
        /// Request identifier.
        request_id: u64,
    },
    /// A planned row ended its request without a predicted release.
    UnplannedRelease {
        /// Request identifier.
        request_id: u64,
        /// Observed cause.
        reason: PlanReportReleaseReason,
    },
}

/// Serialized [`PlanEstimates`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PlanReportEstimates {
    /// Real input positions.
    pub input_tokens: usize,
    /// Right-padding positions.
    pub padding_tokens: usize,
    /// Model invocations.
    pub invocations: usize,
    /// Prompt rows.
    pub prefill_rows: usize,
    /// Decode rows.
    pub decode_rows: usize,
    /// Budget deferrals.
    pub budget_deferrals: usize,
    /// KV bytes reserved by admissions.
    pub admitted_kv_bytes: u64,
    /// Predicted releases.
    pub predicted_releases: usize,
}

impl PlanReport {
    /// Converts a plan, the decision behind it, and the divergences observed
    /// while committing it.
    pub fn new(
        plan: &ExecutionPlan,
        decision: &DecisionRecord,
        divergences: &[PlanDivergence],
    ) -> Self {
        Self {
            schema_version: PLAN_REPORT_SCHEMA_VERSION,
            generation: plan.generation,
            at_micros: plan.at.as_micros(),
            lifecycle: plan.lifecycle.iter().map(lifecycle).collect(),
            batches: plan.batches.iter().map(batch).collect(),
            deferrals: plan.deferrals.iter().map(deferral).collect(),
            predicted_releases: plan
                .predicted_releases
                .iter()
                .map(|request_id| request_id.get())
                .collect(),
            divergences: divergences.iter().map(divergence).collect(),
            estimates: estimates(plan.estimates()),
            decision: decision_record(decision),
        }
    }

    /// Converts one committed executor iteration.
    pub fn from_iteration(iteration: &ExecutorIteration) -> Self {
        Self::new(
            &iteration.execution_plan,
            &iteration.decision,
            &iteration.divergences,
        )
    }

    /// Encodes the report as human-readable JSON.
    pub fn to_json_pretty(&self) -> Result<String, PlanReportError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Decodes a report and rejects schema versions this build cannot read.
    pub fn from_json(json: &str) -> Result<Self, PlanReportError> {
        let report: Self = serde_json::from_str(json)?;
        if report.schema_version != PLAN_REPORT_SCHEMA_VERSION {
            return Err(PlanReportError::UnsupportedSchema(report.schema_version));
        }
        Ok(report)
    }
}

fn lifecycle(action: &LifecycleAction) -> PlanReportLifecycle {
    match action {
        LifecycleAction::Admit {
            request_id,
            model_id,
            reserved_kv_bytes,
        } => PlanReportLifecycle::Admit {
            request_id: request_id.get(),
            model: model_id.as_str().to_owned(),
            reserved_kv_bytes: *reserved_kv_bytes,
        },
        LifecycleAction::Reject {
            request_id,
            reason,
            event_index,
        } => PlanReportLifecycle::Reject {
            request_id: request_id.get(),
            reason: reason.to_string(),
            event_index: *event_index,
        },
        LifecycleAction::Fail {
            request_id,
            message,
            event_index,
        } => PlanReportLifecycle::Fail {
            request_id: request_id.get(),
            message: message.clone(),
            event_index: *event_index,
        },
    }
}

const fn planned_phase(phase: PlannedPhase) -> PlanReportPhase {
    match phase {
        PlannedPhase::Prefill => PlanReportPhase::Prefill,
        PlannedPhase::ContinuedPrefill => PlanReportPhase::ContinuedPrefill,
        PlannedPhase::Decode => PlanReportPhase::Decode,
    }
}

const fn batch_phase(phase: BatchPhase) -> PlanReportPhase {
    match phase {
        BatchPhase::Prefill => PlanReportPhase::Prefill,
        BatchPhase::Decode => PlanReportPhase::Decode,
    }
}

fn decision_record(record: &DecisionRecord) -> PlanReportDecision {
    let candidates = |candidates: &[BatchCandidate]| {
        candidates
            .iter()
            .map(|candidate| PlanReportCandidate {
                request_id: candidate.request_id.get(),
                phase: batch_phase(candidate.phase),
                token_cost: candidate.token_cost,
            })
            .collect()
    };
    PlanReportDecision {
        policy: record.policy.to_string(),
        ranking: PlanReportRanking {
            slots_available: record.ranking.slots_available,
            limit: record.ranking.limit,
            ranked: record.ranking.ranked.iter().map(|id| id.get()).collect(),
        },
        admissions: record.admissions.iter().map(admission).collect(),
        backfill_skips: record.backfill_skips.iter().map(|id| id.get()).collect(),
        selections: record
            .selections
            .iter()
            .map(|selection| PlanReportSelection {
                stage: match selection.stage {
                    SelectionStage::Decode => PlanReportSelectionStage::Decode,
                    SelectionStage::ContinuedPrefill => PlanReportSelectionStage::ContinuedPrefill,
                    SelectionStage::Admission => PlanReportSelectionStage::Admission,
                },
                scheduled: candidates(&selection.context.scheduled),
                queued_requests: selection.context.pressure.queued_requests,
                live_requests: selection.context.pressure.live_requests,
                candidates: candidates(&selection.context.candidates),
                selected: selection.selected.iter().map(|id| id.get()).collect(),
            })
            .collect(),
        grouping: record
            .grouping
            .iter()
            .map(|grouping| PlanReportGrouping {
                phase: planned_phase(grouping.phase),
                model: grouping.model_id.as_str().to_owned(),
                prompt_tokens: grouping.prompt_tokens.clone(),
                buckets: grouping.decision.buckets.clone(),
                cost_estimates: grouping.decision.cost_estimates.map(grouping_estimates),
            })
            .collect(),
        capability_fallbacks: record
            .capability_fallbacks
            .iter()
            .map(|fallback| PlanReportCapabilityFallback {
                phase: planned_phase(fallback.phase),
                model: fallback.model_id.as_str().to_owned(),
                rows: fallback.rows,
                missing: match fallback.missing {
                    BatchHook::Prefill => PlanReportBatchHook::Prefill,
                    BatchHook::ChunkedPrefill => PlanReportBatchHook::ChunkedPrefill,
                    BatchHook::Decode => PlanReportBatchHook::Decode,
                },
            })
            .collect(),
    }
}

fn admission(record: &AdmissionRecord) -> PlanReportAdmission {
    let request_id = record.request_id.get();
    match &record.outcome {
        AdmissionOutcome::NoSlot => PlanReportAdmission::NoSlot { request_id },
        AdmissionOutcome::PreparationFailed { message } => PlanReportAdmission::PreparationFailed {
            request_id,
            message: message.clone(),
        },
        AdmissionOutcome::Decided {
            context,
            decision,
            budget_selected,
        } => {
            let (decision, granted_kv_bytes, reason) = match decision {
                AdmissionDecision::Admit { reserved_kv_bytes } => {
                    ("admit", Some(*reserved_kv_bytes), None)
                }
                AdmissionDecision::Defer(reason) => ("defer", None, Some(reason.to_string())),
                AdmissionDecision::Reject(reason) => ("reject", None, Some(reason.to_string())),
            };
            PlanReportAdmission::Decided {
                request_id,
                prompt_tokens: context.prompt_tokens,
                max_new_tokens: context.max_new_tokens,
                reserved_kv_bytes: context.reserved_kv_bytes,
                model_reserved_kv_bytes: context.model_reserved_kv_bytes,
                decision: decision.to_owned(),
                granted_kv_bytes,
                reason,
                budget_selected: *budget_selected,
            }
        }
    }
}

fn grouping_estimates(decision: CostAwarePrefillDecision) -> PlanReportGroupingEstimates {
    let cost = |estimate: PrefillGroupingEstimate| PlanReportGroupingCost {
        invocations: estimate.invocations,
        dense_input_tokens: estimate.dense_input_tokens,
        estimated_cost: estimate.estimated_cost,
    };
    PlanReportGroupingEstimates {
        selected: match decision.selected {
            PrefillGroupingStrategy::Unbucketed => PlanReportGroupingStrategy::Unbucketed,
            PrefillGroupingStrategy::PowerOfTwo => PlanReportGroupingStrategy::PowerOfTwo,
        },
        unbucketed: cost(decision.unbucketed),
        power_of_two: cost(decision.power_of_two),
    }
}

fn batch(batch: &PlannedBatch) -> PlanReportBatch {
    PlanReportBatch {
        phase: planned_phase(batch.phase),
        model: batch.model_id.as_str().to_owned(),
        mode: match batch.mode {
            BatchExecutionMode::Tensor => PlanReportMode::Tensor,
            BatchExecutionMode::Individual => PlanReportMode::Individual,
        },
        rows: batch
            .rows
            .iter()
            .map(|row| PlanReportRow {
                request_id: row.request_id.get(),
                event_index: row.event_index,
                start_position: row.positions.start,
                end_position: row.positions.end,
                final_prompt_chunk: row.final_prompt_chunk,
            })
            .collect(),
        input_tokens: batch.input_tokens(),
        padding_tokens: batch.padding_tokens(),
    }
}

fn deferral(deferral: &PlannedDeferral) -> PlanReportDeferral {
    match deferral {
        PlannedDeferral::Admission {
            request_id,
            reason,
            event_index,
        } => PlanReportDeferral::Admission {
            request_id: request_id.get(),
            reason: reason.to_string(),
            event_index: *event_index,
        },
        PlannedDeferral::Budget {
            request_id,
            phase,
            token_cost,
            event_index,
        } => PlanReportDeferral::Budget {
            request_id: request_id.get(),
            phase: batch_phase(*phase),
            token_cost: *token_cost,
            event_index: *event_index,
        },
        PlannedDeferral::Slots { request_id } => PlanReportDeferral::Slots {
            request_id: request_id.get(),
        },
    }
}

fn divergence(divergence: &PlanDivergence) -> PlanReportDivergence {
    match divergence {
        PlanDivergence::ModeFallback { batch_index } => PlanReportDivergence::ModeFallback {
            batch_index: *batch_index,
        },
        PlanDivergence::SessionCreationFailed { request_id } => {
            PlanReportDivergence::SessionCreationFailed {
                request_id: request_id.get(),
            }
        }
        PlanDivergence::UnplannedRelease { request_id, reason } => {
            PlanReportDivergence::UnplannedRelease {
                request_id: request_id.get(),
                reason: match reason {
                    ReleaseReason::EndOfSequence => PlanReportReleaseReason::EndOfSequence,
                    ReleaseReason::OutputLimit => PlanReportReleaseReason::OutputLimit,
                    ReleaseReason::Failure => PlanReportReleaseReason::Failure,
                },
            }
        }
    }
}

const fn estimates(estimates: PlanEstimates) -> PlanReportEstimates {
    PlanReportEstimates {
        input_tokens: estimates.input_tokens,
        padding_tokens: estimates.padding_tokens,
        invocations: estimates.invocations,
        prefill_rows: estimates.prefill_rows,
        decode_rows: estimates.decode_rows,
        budget_deferrals: estimates.budget_deferrals,
        admitted_kv_bytes: estimates.admitted_kv_bytes,
        predicted_releases: estimates.predicted_releases,
    }
}

/// Failure to encode or decode a plan report.
#[derive(Debug)]
pub enum PlanReportError {
    /// The record uses a schema this build cannot read.
    UnsupportedSchema(u32),
    /// JSON encoding or decoding failed.
    Json(serde_json::Error),
}

impl fmt::Display for PlanReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(version) => {
                write!(
                    formatter,
                    "unsupported plan report schema version {version}"
                )
            }
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl Error for PlanReportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::UnsupportedSchema(_) => None,
        }
    }
}

impl From<serde_json::Error> for PlanReportError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[cfg(test)]
mod tests {
    use ola_core::{RequestId, Timestamp};
    use ola_runtime::{AdmissionReason, PlannedRow};

    use super::*;

    fn plan() -> ExecutionPlan {
        ExecutionPlan {
            generation: 3,
            at: Timestamp::from_micros(40),
            lifecycle: vec![
                LifecycleAction::Admit {
                    request_id: RequestId::new(2),
                    model_id: "tiny".into(),
                    reserved_kv_bytes: 64,
                },
                LifecycleAction::Reject {
                    request_id: RequestId::new(3),
                    reason: AdmissionReason::MissingKvCacheProfile,
                    event_index: 2,
                },
            ],
            batches: vec![
                PlannedBatch {
                    phase: PlannedPhase::Decode,
                    model_id: "tiny".into(),
                    mode: BatchExecutionMode::Individual,
                    rows: vec![PlannedRow {
                        request_id: RequestId::new(1),
                        event_index: 0,
                        positions: 4..5,
                        final_prompt_chunk: false,
                    }],
                },
                PlannedBatch {
                    phase: PlannedPhase::Prefill,
                    model_id: "tiny".into(),
                    mode: BatchExecutionMode::Individual,
                    rows: vec![PlannedRow {
                        request_id: RequestId::new(2),
                        event_index: 1,
                        positions: 0..3,
                        final_prompt_chunk: true,
                    }],
                },
            ],
            deferrals: vec![PlannedDeferral::Slots {
                request_id: RequestId::new(4),
            }],
            predicted_releases: vec![RequestId::new(1)],
        }
    }

    #[test]
    fn plan_reports_round_trip_through_stable_json() {
        let divergences = [PlanDivergence::UnplannedRelease {
            request_id: RequestId::new(2),
            reason: ReleaseReason::OutputLimit,
        }];
        let decision = DecisionRecord {
            policy: ola_runtime::PolicyDescriptor::new("composed")
                .with_component(ola_runtime::PolicyDescriptor::new("fifo")),
            grouping: vec![ola_runtime::GroupingRecord {
                phase: PlannedPhase::Prefill,
                model_id: "tiny".into(),
                prompt_tokens: vec![3],
                decision: ola_runtime::GroupingDecision {
                    buckets: vec![0],
                    cost_estimates: Some(ola_runtime::CostAwarePrefillBatcher::new(4).decide(&[3])),
                },
            }],
            admissions: vec![AdmissionRecord {
                request_id: RequestId::new(4),
                outcome: AdmissionOutcome::NoSlot,
            }],
            ..DecisionRecord::default()
        };
        let report = PlanReport::new(&plan(), &decision, &divergences);

        assert_eq!(report.schema_version, PLAN_REPORT_SCHEMA_VERSION);
        assert_eq!(report.estimates.input_tokens, 4);
        assert_eq!(report.estimates.admitted_kv_bytes, 64);
        let json = report.to_json_pretty().unwrap();
        for fragment in [
            r#""action": "admit""#,
            r#""reason": "model has no KV-cache resource profile""#,
            r#""phase": "decode""#,
            r#""kind": "slots""#,
            r#""reason": "output_limit""#,
            r#""policy": "composed[fifo]""#,
            r#""outcome": "no_slot""#,
            r#""selected": "unbucketed""#,
        ] {
            assert!(json.contains(fragment), "missing {fragment} in {json}");
        }
        assert_eq!(PlanReport::from_json(&json).unwrap(), report);
    }

    #[test]
    fn plan_reports_reject_unknown_schema_versions() {
        let mut report = PlanReport::new(&plan(), &DecisionRecord::default(), &[]);
        report.schema_version += 1;
        let json = report.to_json_pretty().unwrap();

        assert!(matches!(
            PlanReport::from_json(&json),
            Err(PlanReportError::UnsupportedSchema(version))
                if version == PLAN_REPORT_SCHEMA_VERSION + 1
        ));
    }
}
