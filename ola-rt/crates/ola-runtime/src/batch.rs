use std::num::NonZeroUsize;

use ola_core::RequestId;

use crate::PolicyDescriptor;

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
    fn bucket(&self, prompt_tokens: usize) -> usize;

    /// Returns the grouping rule's stable name and configuration.
    fn descriptor(&self) -> PolicyDescriptor;

    /// Assigns bucket keys to one complete same-model prefill set.
    ///
    /// The default preserves the original per-prompt behavior. Workload-aware
    /// policies may override this method. The executor supplies an output
    /// slice with exactly the same length, preventing malformed policies from
    /// changing the cardinality of prepared work.
    fn buckets(&self, prompt_tokens: &[usize], buckets: &mut [usize]) {
        debug_assert_eq!(prompt_tokens.len(), buckets.len());
        for (tokens, bucket) in prompt_tokens.iter().zip(buckets) {
            *bucket = self.bucket(*tokens);
        }
    }
}

/// Compatibility rule that places every prompt length in one bucket.
///
/// This maximizes batch width but pads every row to the longest selected prompt.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnbucketedPrefillBatcher;

impl PrefillBatcher for UnbucketedPrefillBatcher {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("unbucketed-prefill")
    }

    fn bucket(&self, _prompt_tokens: usize) -> usize {
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
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("power-of-two-prefill")
    }

    fn bucket(&self, prompt_tokens: usize) -> usize {
        power_of_two_bucket(prompt_tokens)
    }
}

/// Grouping alternatives considered by [`CostAwarePrefillBatcher`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillGroupingStrategy {
    /// One invocation containing every compatible prompt.
    Unbucketed,
    /// One invocation per populated next-power-of-two length bucket.
    PowerOfTwo,
}

/// Estimated work for one candidate prefill grouping.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefillGroupingEstimate {
    /// Model invocations required by the grouping.
    pub invocations: usize,
    /// Real plus padded token positions processed by dense tensors.
    pub dense_input_tokens: usize,
    /// Dense positions plus configured token-equivalent invocation overhead.
    pub estimated_cost: usize,
}

/// Auditable choice made for one same-model prefill set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostAwarePrefillDecision {
    /// Strategy with the lower estimated cost; ties prefer one invocation.
    pub selected: PrefillGroupingStrategy,
    /// Estimate for placing every prompt in one tensor.
    pub unbucketed: PrefillGroupingEstimate,
    /// Estimate for next-power-of-two length grouping.
    pub power_of_two: PrefillGroupingEstimate,
}

/// Chooses a prefill grouping using dense work plus invocation overhead.
///
/// `invocation_overhead_tokens` expresses one model invocation's fixed cost in
/// token-equivalent units. A value of zero optimizes only dense tensor
/// positions; larger values increasingly favor fewer, wider invocations. The
/// estimate is deliberately simple and deterministic so experiments can
/// calibrate it per model/backend without embedding timing reads in policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostAwarePrefillBatcher {
    invocation_overhead_tokens: usize,
}

impl CostAwarePrefillBatcher {
    /// Creates a workload-aware grouper with an explicit invocation estimate.
    pub const fn new(invocation_overhead_tokens: usize) -> Self {
        Self {
            invocation_overhead_tokens,
        }
    }

    /// Returns the fixed token-equivalent cost charged per model invocation.
    pub const fn invocation_overhead_tokens(self) -> usize {
        self.invocation_overhead_tokens
    }

    /// Estimates both alternatives and returns the selected strategy.
    pub fn decide(self, prompt_tokens: &[usize]) -> CostAwarePrefillDecision {
        let input_tokens = prompt_tokens
            .iter()
            .fold(0_usize, |total, tokens| total.saturating_add(*tokens));
        let unbucketed_dense = prompt_tokens
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_mul(prompt_tokens.len());
        let unbucketed_invocations = usize::from(!prompt_tokens.is_empty());
        let unbucketed = PrefillGroupingEstimate {
            invocations: unbucketed_invocations,
            dense_input_tokens: unbucketed_dense,
            estimated_cost: unbucketed_dense.saturating_add(
                unbucketed_invocations.saturating_mul(self.invocation_overhead_tokens),
            ),
        };

        let mut groups: Vec<(usize, usize, usize)> = Vec::new();
        for &tokens in prompt_tokens {
            let bucket = power_of_two_bucket(tokens);
            if let Some((_, maximum, rows)) = groups.iter_mut().find(|(key, _, _)| *key == bucket) {
                *maximum = (*maximum).max(tokens);
                *rows += 1;
            } else {
                groups.push((bucket, tokens, 1));
            }
        }
        let power_of_two_dense = groups.iter().fold(0_usize, |total, (_, maximum, rows)| {
            total.saturating_add(maximum.saturating_mul(*rows))
        });
        debug_assert!(power_of_two_dense >= input_tokens);
        let power_of_two_invocations = groups.len();
        let power_of_two = PrefillGroupingEstimate {
            invocations: power_of_two_invocations,
            dense_input_tokens: power_of_two_dense,
            estimated_cost: power_of_two_dense.saturating_add(
                power_of_two_invocations.saturating_mul(self.invocation_overhead_tokens),
            ),
        };
        let selected = if power_of_two.estimated_cost < unbucketed.estimated_cost {
            PrefillGroupingStrategy::PowerOfTwo
        } else {
            PrefillGroupingStrategy::Unbucketed
        };
        CostAwarePrefillDecision {
            selected,
            unbucketed,
            power_of_two,
        }
    }
}

impl PrefillBatcher for CostAwarePrefillBatcher {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("cost-aware-prefill").with_config(
            "invocation_overhead_tokens",
            self.invocation_overhead_tokens,
        )
    }

    fn bucket(&self, prompt_tokens: usize) -> usize {
        power_of_two_bucket(prompt_tokens)
    }

    fn buckets(&self, prompt_tokens: &[usize], buckets: &mut [usize]) {
        debug_assert_eq!(prompt_tokens.len(), buckets.len());
        match self.decide(prompt_tokens).selected {
            PrefillGroupingStrategy::Unbucketed => buckets.fill(0),
            PrefillGroupingStrategy::PowerOfTwo => {
                for (tokens, bucket) in prompt_tokens.iter().zip(buckets) {
                    *bucket = power_of_two_bucket(*tokens);
                }
            }
        }
    }
}

fn power_of_two_bucket(prompt_tokens: usize) -> usize {
    prompt_tokens
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
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
    fn select(&self, context: &BatchContext) -> Vec<RequestId>;

    /// Returns the policy's stable name and configuration.
    fn descriptor(&self) -> PolicyDescriptor;
}

/// Compatibility policy that selects every eligible operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnboundedBatchPolicy;

impl BatchPolicy for UnboundedBatchPolicy {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("unbounded-batch")
    }

    fn select(&self, context: &BatchContext) -> Vec<RequestId> {
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
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("fifo-token-budget").with_config("max_tokens", self.max_tokens)
    }

    fn select(&self, context: &BatchContext) -> Vec<RequestId> {
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
        let policy = FifoTokenBudgetPolicy::new(4).unwrap();
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
        let policy = FifoTokenBudgetPolicy::new(4).unwrap();
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
        let batcher = PowerOfTwoPrefillBatcher;

        assert_eq!(batcher.bucket(1), 1);
        assert_eq!(batcher.bucket(5), 8);
        assert_eq!(batcher.bucket(7), 8);
        assert_eq!(batcher.bucket(8), 8);
        assert_eq!(batcher.bucket(9), 16);
    }

    #[test]
    fn cost_aware_prefill_balances_padding_against_extra_invocations() {
        let lengths = [5, 7, 9];
        let low_overhead = CostAwarePrefillBatcher::new(3).decide(&lengths);
        let break_even = CostAwarePrefillBatcher::new(4).decide(&lengths);

        assert_eq!(low_overhead.unbucketed.dense_input_tokens, 27);
        assert_eq!(low_overhead.power_of_two.dense_input_tokens, 23);
        assert_eq!(low_overhead.unbucketed.estimated_cost, 30);
        assert_eq!(low_overhead.power_of_two.estimated_cost, 29);
        assert_eq!(low_overhead.selected, PrefillGroupingStrategy::PowerOfTwo);
        assert_eq!(break_even.unbucketed.estimated_cost, 31);
        assert_eq!(break_even.power_of_two.estimated_cost, 31);
        assert_eq!(break_even.selected, PrefillGroupingStrategy::Unbucketed);
    }

    #[test]
    fn cost_aware_prefill_assigns_the_selected_workload_level_grouping() {
        let lengths = [5, 7, 9, 63];
        let batcher = CostAwarePrefillBatcher::new(32);

        assert_eq!(
            batcher.decide(&lengths).selected,
            PrefillGroupingStrategy::PowerOfTwo
        );
        let mut buckets = [0; 4];
        batcher.buckets(&lengths, &mut buckets);
        assert_eq!(buckets, [8, 8, 16, 64]);

        let clustered = [5, 6, 7, 8];
        assert_eq!(
            batcher.decide(&clustered).selected,
            PrefillGroupingStrategy::Unbucketed
        );
        batcher.buckets(&clustered, &mut buckets);
        assert_eq!(buckets, [0, 0, 0, 0]);
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
