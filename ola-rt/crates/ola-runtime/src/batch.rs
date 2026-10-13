use std::num::NonZeroUsize;

use ola_core::RequestId;

/// Model phase represented by one iteration candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchPhase {
    /// Process a complete prompt and select its first output token.
    Prefill,
    /// Process one cached autoregressive token position.
    Decode,
}

/// Backend execution mechanism used for one compatible operation group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchExecutionMode {
    /// Requests were packed into one model tensor invocation.
    Tensor,
    /// The model was invoked independently for this request.
    Individual,
}

/// One model invocation emitted while executing an iteration plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBatch {
    /// Prefill or decode work performed by the invocation.
    pub phase: BatchPhase,
    /// Requests represented by the invocation, in tensor row order.
    pub request_ids: Vec<RequestId>,
    /// Whether work shared a tensor invocation or used the oracle path.
    pub mode: BatchExecutionMode,
    /// Meaningful input token positions represented by this invocation.
    pub input_tokens: usize,
    /// Artificial prompt positions computed only to make tensor rows rectangular.
    pub padding_tokens: usize,
}

impl ExecutionBatch {
    /// Returns all dense input positions computed by this invocation.
    pub fn dense_input_tokens(&self) -> usize {
        self.input_tokens.saturating_add(self.padding_tokens)
    }
}

/// Replaceable rule assigning prompt lengths to compatible tensor buckets.
///
/// Equal keys may share a prefill tensor invocation. The executor still groups
/// by loaded model first, so a custom implementation cannot combine sessions
/// backed by different artifacts. Prompt lengths are always positive.
pub trait PrefillBatcher: Send {
    /// Returns an opaque deterministic bucket key for one prompt length.
    fn bucket(&mut self, prompt_tokens: usize) -> usize;
}

/// Compatibility rule that places every prompt length in one bucket.
///
/// This maximizes batch width but pads every row to the longest selected prompt.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnbucketedPrefillBatcher;

impl PrefillBatcher for UnbucketedPrefillBatcher {
    fn bucket(&mut self, _prompt_tokens: usize) -> usize {
        0
    }
}

/// Groups prompts by the next power-of-two sequence length.
///
/// For example, lengths `5`, `7`, and `8` share bucket `8`, while length `9`
/// enters bucket `16`. This bounds padding within each bucket while retaining
/// useful batching across nearby lengths.
#[derive(Clone, Copy, Debug, Default)]
pub struct PowerOfTwoPrefillBatcher;

impl PrefillBatcher for PowerOfTwoPrefillBatcher {
    fn bucket(&mut self, prompt_tokens: usize) -> usize {
        prompt_tokens
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
    }
}

/// One unit of model work eligible for the current iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchCandidate {
    /// Request that owns this work.
    pub request_id: RequestId,
    /// Prefill or decode phase.
    pub phase: BatchPhase,
    /// Number of input token positions processed by the operation.
    pub token_cost: usize,
}

/// Read-only inputs supplied to a batch-composition policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchContext {
    /// Work already selected earlier in this iteration.
    pub scheduled: Vec<BatchCandidate>,
    /// Eligible work in deterministic executor order.
    pub candidates: Vec<BatchCandidate>,
}

impl BatchContext {
    /// Returns input token positions already committed this iteration.
    pub fn scheduled_tokens(&self) -> usize {
        self.scheduled
            .iter()
            .map(|candidate| candidate.token_cost)
            .sum()
    }
}

/// Replaceable policy selecting work for one execution iteration.
pub trait BatchPolicy: Send {
    /// Returns request IDs selected from `context.candidates`.
    ///
    /// The executor validates membership and uniqueness before executing the
    /// result. Returning IDs in candidate order makes traces easiest to compare.
    fn select(&mut self, context: &BatchContext) -> Vec<RequestId>;
}

/// Compatibility policy that selects every eligible operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnboundedBatchPolicy;

impl BatchPolicy for UnboundedBatchPolicy {
    fn select(&mut self, context: &BatchContext) -> Vec<RequestId> {
        context
            .candidates
            .iter()
            .map(|candidate| candidate.request_id)
            .collect()
    }
}

/// Deterministic FIFO prefix bounded by input token positions per iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FifoTokenBudgetPolicy {
    max_tokens: NonZeroUsize,
}

impl FifoTokenBudgetPolicy {
    /// Creates a positive per-iteration token budget.
    pub const fn new(max_tokens: usize) -> Option<Self> {
        match NonZeroUsize::new(max_tokens) {
            Some(max_tokens) => Some(Self { max_tokens }),
            None => None,
        }
    }

    /// Returns the configured input-token budget.
    pub const fn max_tokens(self) -> usize {
        self.max_tokens.get()
    }
}

impl BatchPolicy for FifoTokenBudgetPolicy {
    fn select(&mut self, context: &BatchContext) -> Vec<RequestId> {
        let mut used = context.scheduled_tokens();
        let mut selected = Vec::new();
        for candidate in &context.candidates {
            let Some(next) = used.checked_add(candidate.token_cost) else {
                break;
            };
            if next <= self.max_tokens.get() {
                used = next;
                selected.push(candidate.request_id);
                continue;
            }

            // A prompt larger than the budget must still be able to make
            // progress. Run it alone and record its actual cost in the plan.
            if context.scheduled.is_empty()
                && selected.is_empty()
                && candidate.phase == BatchPhase::Prefill
            {
                selected.push(candidate.request_id);
            }
            break;
        }
        selected
    }
}

/// Auditable work selected and deferred during one executor iteration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IterationPlan {
    /// Operations selected for execution, in execution order.
    pub scheduled: Vec<BatchCandidate>,
    /// Eligible operations left for a later iteration by batch policy.
    pub deferred: Vec<BatchCandidate>,
}

impl IterationPlan {
    /// Returns the total input token positions selected for execution.
    pub fn scheduled_tokens(&self) -> usize {
        self.scheduled
            .iter()
            .map(|candidate| candidate.token_cost)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: u64, phase: BatchPhase, token_cost: usize) -> BatchCandidate {
        BatchCandidate {
            request_id: RequestId::new(id),
            phase,
            token_cost,
        }
    }

    #[test]
    fn token_budget_selects_a_fifo_prefix_across_iteration_phases() {
        let mut policy = FifoTokenBudgetPolicy::new(4).unwrap();
        let context = BatchContext {
            scheduled: vec![candidate(1, BatchPhase::Decode, 1)],
            candidates: vec![
                candidate(2, BatchPhase::Decode, 1),
                candidate(3, BatchPhase::Prefill, 2),
                candidate(4, BatchPhase::Prefill, 1),
            ],
        };

        assert_eq!(
            policy.select(&context),
            vec![RequestId::new(2), RequestId::new(3)]
        );
    }

    #[test]
    fn oversized_prefill_runs_alone_to_preserve_progress() {
        let mut policy = FifoTokenBudgetPolicy::new(4).unwrap();
        let context = BatchContext {
            scheduled: Vec::new(),
            candidates: vec![candidate(1, BatchPhase::Prefill, 9)],
        };

        assert_eq!(policy.select(&context), vec![RequestId::new(1)]);
    }

    #[test]
    fn zero_token_budget_is_rejected_at_construction() {
        assert_eq!(FifoTokenBudgetPolicy::new(0), None);
    }

    #[test]
    fn power_of_two_prefill_buckets_group_nearby_lengths() {
        let mut batcher = PowerOfTwoPrefillBatcher;

        assert_eq!(batcher.bucket(1), 1);
        assert_eq!(batcher.bucket(5), 8);
        assert_eq!(batcher.bucket(7), 8);
        assert_eq!(batcher.bucket(8), 8);
        assert_eq!(batcher.bucket(9), 16);
    }

    #[test]
    fn execution_batch_reports_dense_padding_work() {
        let batch = ExecutionBatch {
            phase: BatchPhase::Prefill,
            request_ids: vec![RequestId::new(1), RequestId::new(2)],
            mode: BatchExecutionMode::Tensor,
            input_tokens: 13,
            padding_tokens: 3,
        };

        assert_eq!(batch.dense_input_tokens(), 16);
    }
}
