//! Why an execution policy planned what it did.
//!
//! See `EXECUTION_POLICY.md` in this crate. A plan says *what* an iteration
//! runs; a [`DecisionRecord`] keeps every input, alternative, estimate, and
//! outcome the policy's components produced while building it, so policies
//! can be audited and compared without re-running them.

use ola_core::{ModelId, RequestId};

use crate::{
    AdmissionContext, AdmissionDecision, BatchContext, CostAwarePrefillDecision, PlannedPhase,
    PolicyDescriptor,
};

/// Complete audit trail of one planning pass.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DecisionRecord {
    /// Policy that made the decision.
    pub policy: PolicyDescriptor,
    /// How queued work was ranked for admission.
    pub ranking: RankingRecord,
    /// Every ranked request the policy considered, in ranking order.
    pub admissions: Vec<AdmissionRecord>,
    /// Every batch-policy call, in planning order.
    pub selections: Vec<SelectionRecord>,
    /// Every prefill-grouping call, in planning order.
    pub grouping: Vec<GroupingRecord>,
    /// Groups the policy wanted as tensor invocations but planned
    /// individually because the model does not declare the hook.
    pub capability_fallbacks: Vec<CapabilityFallback>,
}

/// Inputs and output of the scheduler's ranking.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RankingRecord {
    /// Slots free for admission after predicted releases.
    pub slots_available: usize,
    /// Number of requests the scheduler was asked for (slots plus one, so
    /// the first waiting request can be recorded).
    pub limit: usize,
    /// Requests in the order the scheduler returned them.
    pub ranked: Vec<RequestId>,
}

/// What happened to one ranked request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionRecord {
    /// Ranked request.
    pub request_id: RequestId,
    /// Outcome and, where admission ran, its complete inputs.
    pub outcome: AdmissionOutcome,
}

/// Outcome of considering one ranked request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionOutcome {
    /// No slot remained; the request waits.
    NoSlot,
    /// Preparation failed before admission could run.
    PreparationFailed {
        /// Diagnostic that becomes the request's failure.
        message: String,
    },
    /// The admission policy decided; for `Admit`, the batch policy then
    /// either selected (`budget_selected`) or deferred the first prefill.
    Decided {
        /// Exactly what the admission policy saw.
        context: AdmissionContext,
        /// What it returned.
        decision: AdmissionDecision,
        /// Whether the first prefill fit the iteration's token budget.
        /// `None` unless the request was admitted.
        budget_selected: Option<bool>,
    },
}

/// Planning stage that consulted the batch policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionStage {
    /// Live decodes.
    Decode,
    /// Live unfinished prompts.
    ContinuedPrefill,
    /// The first prefill of one admitted request.
    Admission,
}

/// Inputs and output of one batch-policy call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionRecord {
    /// Stage that asked.
    pub stage: SelectionStage,
    /// Exactly what the batch policy saw.
    pub context: BatchContext,
    /// Requests it selected, in its returned order.
    pub selected: Vec<RequestId>,
}

/// Output of a prefill-grouping rule for one model's rows.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GroupingDecision {
    /// Opaque bucket key per input row; equal keys may share an invocation.
    pub buckets: Vec<usize>,
    /// Alternatives and cost estimates, for rules that weigh them.
    pub cost_estimates: Option<CostAwarePrefillDecision>,
}

/// Inputs and output of one prefill-grouping call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupingRecord {
    /// Prompt phase being grouped.
    pub phase: PlannedPhase,
    /// Model whose rows were grouped.
    pub model_id: ModelId,
    /// Row lengths the rule saw, in row order.
    pub prompt_tokens: Vec<usize>,
    /// What it returned.
    pub decision: GroupingDecision,
}

/// Batched model hook a capability fallback was missing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchHook {
    /// `CausalModel::prefill_batch`.
    Prefill,
    /// `CausalModel::prefill_chunk_batch`.
    ChunkedPrefill,
    /// `CausalModel::decode_batch`.
    Decode,
}

/// A multi-row group planned as individual invocations because the model
/// does not declare the required batch hook.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityFallback {
    /// Phase of the group.
    pub phase: PlannedPhase,
    /// Model lacking the hook.
    pub model_id: ModelId,
    /// Rows that could otherwise have shared one invocation.
    pub rows: usize,
    /// Hook that was not declared.
    pub missing: BatchHook,
}
