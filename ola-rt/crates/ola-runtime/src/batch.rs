use std::num::NonZeroUsize;

use ola_core::RequestId;

use crate::{GroupingDecision, PolicyDescriptor};

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

    /// Groups one model's rows and reports any alternatives weighed.
    ///
    /// Execution policies call this so decision records keep the rule's
    /// reasoning. The default wraps [`buckets`](Self::buckets) and reports no
    /// alternatives.
    fn group(&self, prompt_tokens: &[usize]) -> GroupingDecision {
        let mut buckets = vec![0; prompt_tokens.len()];
        self.buckets(prompt_tokens, &mut buckets);
        GroupingDecision {
            buckets,
            cost_estimates: None,
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

    fn group(&self, prompt_tokens: &[usize]) -> GroupingDecision {
        let decision = self.decide(prompt_tokens);
        let mut buckets = vec![0; prompt_tokens.len()];
        self.buckets(prompt_tokens, &mut buckets);
        GroupingDecision {
            buckets,
            cost_estimates: Some(decision),
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
    /// Load at the iteration boundary, for budgets that adapt to it.
    pub pressure: BatchPressure,
}

/// Load visible to batch policies, taken from the planning snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchPressure {
    /// Requests waiting for admission.
    pub queued_requests: usize,
    /// Live sessions.
    pub live_requests: usize,
    /// Configured live-session limit.
    pub max_active_requests: usize,
}

impl BatchContext {
    /// Returns input token positions already committed this iteration.
    pub fn scheduled_tokens(&self) -> usize {
        self.scheduled
            .iter()
            .map(|candidate| candidate.token_cost)
            .sum()
    }

    /// Returns prompt token positions already committed this iteration.
    pub fn scheduled_prefill_tokens(&self) -> usize {
        self.scheduled
            .iter()
            .filter(|candidate| candidate.phase == BatchPhase::Prefill)
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
        select_within_budget(context, self.max_tokens.get(), None)
    }
}

/// Token budget that serves decodes first and caps prompt work per
/// iteration, so long prompts cannot crowd out live decodes.
///
/// The first prompt in an iteration may exceed the prompt cap when it fits
/// the total budget; otherwise an unchunked prompt longer than the cap could
/// never run while decodes are live. A prompt larger than the whole budget
/// still runs alone when nothing else is scheduled. With
/// `max_prefill_tokens == max_tokens` this is exactly
/// [`FifoTokenBudgetPolicy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MixedTokenBudgetPolicy {
    max_tokens: NonZeroUsize,
    max_prefill_tokens: NonZeroUsize,
}

impl MixedTokenBudgetPolicy {
    /// Creates a budget; `None` unless `0 < max_prefill_tokens <= max_tokens`.
    pub const fn new(max_tokens: usize, max_prefill_tokens: usize) -> Option<Self> {
        if max_prefill_tokens > max_tokens {
            return None;
        }
        match (
            NonZeroUsize::new(max_tokens),
            NonZeroUsize::new(max_prefill_tokens),
        ) {
            (Some(max_tokens), Some(max_prefill_tokens)) => Some(Self {
                max_tokens,
                max_prefill_tokens,
            }),
            _ => None,
        }
    }
}

impl BatchPolicy for MixedTokenBudgetPolicy {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("mixed-token-budget")
            .with_config("max_tokens", self.max_tokens)
            .with_config("max_prefill_tokens", self.max_prefill_tokens)
    }

    fn select(&self, context: &BatchContext) -> Vec<RequestId> {
        select_within_budget(
            context,
            self.max_tokens.get(),
            Some(self.max_prefill_tokens.get()),
        )
    }
}

/// FIFO token budget that grows with queue depth: small when idle for low
/// latency, larger under load for throughput.
///
/// The budget is `min_tokens + tokens_per_queued × queued_requests`, clamped
/// to `max_tokens`. It is computed from [`BatchPressure`] alone, so equal
/// snapshots give equal budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueAdaptiveTokenBudget {
    min_tokens: NonZeroUsize,
    max_tokens: NonZeroUsize,
    tokens_per_queued: usize,
}

impl QueueAdaptiveTokenBudget {
    /// Creates an adaptive budget; `None` unless `0 < min_tokens <= max_tokens`.
    pub const fn new(
        min_tokens: usize,
        max_tokens: usize,
        tokens_per_queued: usize,
    ) -> Option<Self> {
        if min_tokens > max_tokens {
            return None;
        }
        match (NonZeroUsize::new(min_tokens), NonZeroUsize::new(max_tokens)) {
            (Some(min_tokens), Some(max_tokens)) => Some(Self {
                min_tokens,
                max_tokens,
                tokens_per_queued,
            }),
            _ => None,
        }
    }

    /// Returns the budget for a given load.
    pub fn budget(self, pressure: BatchPressure) -> usize {
        self.tokens_per_queued
            .saturating_mul(pressure.queued_requests)
            .saturating_add(self.min_tokens.get())
            .min(self.max_tokens.get())
    }
}

impl BatchPolicy for QueueAdaptiveTokenBudget {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("queue-adaptive-token-budget")
            .with_config("min_tokens", self.min_tokens)
            .with_config("max_tokens", self.max_tokens)
            .with_config("tokens_per_queued", self.tokens_per_queued)
    }

    fn select(&self, context: &BatchContext) -> Vec<RequestId> {
        select_within_budget(context, self.budget(context.pressure), None)
    }
}

/// Selects the FIFO prefix of candidates that fits `max_tokens`, and
/// `max_prefill_tokens` for prompt work when given.
fn select_within_budget(
    context: &BatchContext,
    max_tokens: usize,
    max_prefill_tokens: Option<usize>,
) -> Vec<RequestId> {
    let mut used = context.scheduled_tokens();
    let mut prefill_used = context.scheduled_prefill_tokens();
    let mut selected = Vec::new();
    for candidate in &context.candidates {
        let Some(next) = used.checked_add(candidate.token_cost) else {
            break;
        };
        let prefill = candidate.phase == BatchPhase::Prefill;
        let within_prefill_cap = !prefill
            || prefill_used == 0
            || max_prefill_tokens
                .is_none_or(|cap| prefill_used.saturating_add(candidate.token_cost) <= cap);
        if next <= max_tokens && within_prefill_cap {
            used = next;
            if prefill {
                prefill_used = prefill_used.saturating_add(candidate.token_cost);
            }
            selected.push(candidate.request_id);
            continue;
        }

        // A prompt larger than the budget must still be able to make
        // progress. Run it alone and record its actual cost in the plan.
        if context.scheduled.is_empty() && selected.is_empty() && prefill {
            selected.push(candidate.request_id);
        }
        break;
    }
    selected
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
            pressure: BatchPressure::default(),
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
            pressure: BatchPressure::default(),
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

    fn context(
        scheduled: Vec<BatchCandidate>,
        candidates: Vec<BatchCandidate>,
        queued_requests: usize,
    ) -> BatchContext {
        BatchContext {
            scheduled,
            candidates,
            pressure: BatchPressure {
                queued_requests,
                live_requests: 0,
                max_active_requests: 8,
            },
        }
    }

    fn ids(selected: Vec<RequestId>) -> Vec<u64> {
        selected.into_iter().map(RequestId::get).collect()
    }

    #[test]
    fn mixed_budget_with_an_uncapped_prompt_share_matches_fifo_budget() {
        let decode = |id| candidate(id, BatchPhase::Decode, 1);
        let prefill = |id, cost| candidate(id, BatchPhase::Prefill, cost);
        let cases = [
            context(
                vec![decode(1)],
                vec![decode(2), prefill(3, 3), prefill(4, 4)],
                0,
            ),
            context(Vec::new(), vec![prefill(1, 20), prefill(2, 1)], 0),
            context(vec![decode(1), decode(2)], vec![prefill(3, 9)], 0),
            context(
                Vec::new(),
                vec![prefill(1, 5), prefill(2, 5), prefill(3, 1)],
                0,
            ),
        ];
        let fifo = FifoTokenBudgetPolicy::new(10).unwrap();
        let mixed = MixedTokenBudgetPolicy::new(10, 10).unwrap();

        for case in &cases {
            assert_eq!(mixed.select(case), fifo.select(case));
        }
    }

    #[test]
    fn mixed_budget_caps_prompt_work_behind_live_decodes() {
        let context = context(
            vec![
                candidate(1, BatchPhase::Decode, 1),
                candidate(2, BatchPhase::Decode, 1),
            ],
            vec![
                candidate(3, BatchPhase::Prefill, 3),
                candidate(4, BatchPhase::Prefill, 3),
            ],
            0,
        );

        assert_eq!(
            ids(FifoTokenBudgetPolicy::new(10).unwrap().select(&context)),
            [3, 4]
        );
        assert_eq!(
            ids(MixedTokenBudgetPolicy::new(10, 4).unwrap().select(&context)),
            [3]
        );
    }

    #[test]
    fn mixed_budget_lets_the_first_prompt_exceed_the_cap_and_oversized_prompts_run_alone() {
        let over_cap = context(
            vec![candidate(1, BatchPhase::Decode, 1)],
            vec![candidate(2, BatchPhase::Prefill, 6)],
            0,
        );
        let oversized = context(Vec::new(), vec![candidate(1, BatchPhase::Prefill, 20)], 0);
        let policy = MixedTokenBudgetPolicy::new(10, 4).unwrap();

        assert_eq!(ids(policy.select(&over_cap)), [2]);
        assert_eq!(ids(policy.select(&oversized)), [1]);
    }

    #[test]
    fn adaptive_budget_grows_with_queue_depth_up_to_its_maximum() {
        let policy = QueueAdaptiveTokenBudget::new(2, 6, 1).unwrap();
        let prompts = || {
            vec![
                candidate(1, BatchPhase::Prefill, 2),
                candidate(2, BatchPhase::Prefill, 2),
                candidate(3, BatchPhase::Prefill, 2),
                candidate(4, BatchPhase::Prefill, 2),
            ]
        };

        assert_eq!(
            policy.budget(context(Vec::new(), Vec::new(), 0).pressure),
            2
        );
        assert_eq!(
            policy.budget(context(Vec::new(), Vec::new(), 3).pressure),
            5
        );
        assert_eq!(
            policy.budget(context(Vec::new(), Vec::new(), 50).pressure),
            6
        );
        assert_eq!(ids(policy.select(&context(Vec::new(), prompts(), 0))), [1]);
        assert_eq!(
            ids(policy.select(&context(Vec::new(), prompts(), 4))),
            [1, 2, 3]
        );
    }

    #[test]
    fn budget_policies_reject_invalid_configuration_and_have_stable_names() {
        assert!(MixedTokenBudgetPolicy::new(4, 5).is_none());
        assert!(MixedTokenBudgetPolicy::new(4, 0).is_none());
        assert!(QueueAdaptiveTokenBudget::new(6, 2, 1).is_none());
        assert!(QueueAdaptiveTokenBudget::new(0, 2, 1).is_none());
        assert_eq!(
            MixedTokenBudgetPolicy::new(16, 8)
                .unwrap()
                .descriptor()
                .to_string(),
            "mixed-token-budget{max_tokens=16,max_prefill_tokens=8}"
        );
        assert_eq!(
            QueueAdaptiveTokenBudget::new(4, 64, 2)
                .unwrap()
                .descriptor()
                .to_string(),
            "queue-adaptive-token-budget{min_tokens=4,max_tokens=64,tokens_per_queued=2}"
        );
    }
}
