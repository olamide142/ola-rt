#![doc = include_str!("../README.md")]

mod calibration;
mod cluster_simulation;
mod experiment;
mod online;
mod plan_report;
mod random;
mod regime_filter;
mod report;
mod scheduling_summary;
mod simulation;
mod synthetic;
mod trace;

use std::{
    error::Error,
    fmt,
    time::{Duration, Instant},
};

use ola_core::{
    GenerationConfig, Input, ModelDescriptor, ModelId, RequestId, RequestSpec, Timestamp,
};
use ola_model::CausalLanguageModel;
use ola_runtime::{
    BatchExecutionMode, BatchPhase, ComposedPolicy, CostAwarePrefillBatcher,
    CostAwarePrefillDecision, Executor, ExecutorError, ExecutorEvent, PowerOfTwoPrefillBatcher,
    PrefillBatcher, Runtime, RuntimeError, UnbucketedPrefillBatcher,
};
use serde::{Deserialize, Serialize};

pub use calibration::{
    CALIBRATION_RESAMPLES, COST_CALIBRATION, CalibrationError, CostFit, CostIntervals,
    CostObservation, DecodeFit, DecodeObservation, PaddedCostFit, PaddedCostIntervals,
    fit_cost_model, fit_decode_cost, fit_padded_cost_model, measure_decode_iteration_costs,
    measure_iteration_costs, measure_mixed_iteration_costs, mixed_length_prompts,
};
pub use cluster_simulation::{
    ClusterRun, ClusterSimulationConfig, ClusterSimulationError, ClusterSummary,
    MESSAGE_HEADER_BYTES, NetworkModel, SUMMARY_BYTES, TOKEN_BYTES, WorkerFailure, WorkerRun,
    run_cluster_trial, summarize_cluster_run,
};
pub use experiment::{
    AdaptivePrefillCost, BOOTSTRAP_RESAMPLES, BuildInfo, Change, ChunkedPrefill, ClusterRouting,
    CompareError, CompareOptions, CostCalibration, Direction, EXPERIMENT_SCHEMA, Estimate,
    Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor, ExperimentError,
    ExperimentIdentity, ExperimentKind, ExperimentOutcome, ExperimentRegistry, ExperimentReport,
    ExperimentRunConfig, ExperimentTrace, FifoWorkload, HOST_STUDY_MIN_REPETITIONS,
    HOST_STUDY_MIN_WARMUPS, IdentityDifference, Interval, KvBlockPressure, KvPageSize, KvPressure,
    LlamaGeneration, MetricComparison, MetricSpec, OnlineAdaptivePolicyStudy,
    PagedAttentionKernelStudy, PagedAttentionOccupancy, ParameterSpec, ParameterValue, Parameters,
    PrefillBucketing, PrefillStrategies, PrefixCache, Protocol, RegimeShiftStudy, ReportComparison,
    ReportError, ReportParts, Requirement, RunMetadata, STUDY_MIN_REPETITIONS, Sample,
    SchedulerComparison, TimeDomain, Unit, Variant, VariantSummary, WorkloadIdentity,
    compare_reports, content_digest, nsight_metrics,
};
pub use online::{
    AdaptationCost, AdaptationMeter, LATENCY_OBJECTIVE, OnlineTuning, PolicyFactory,
    ProposalRecord, RegimeFeatures, RegimeTrigger, ReplayHorizon, ReplayOptimizer, ReplaySelection,
    ReplaySettings, TOKEN_BUDGET_DOMAINS, charged_adaptive_policy, charged_adaptive_policy_over,
    largest_token_budget, serving_adaptive_policy, serving_adaptive_policy_logged,
    smallest_token_budget, token_budget, token_budget_policy,
};
pub use plan_report::{
    PLAN_REPORT_SCHEMA_VERSION, PlanReport, PlanReportAdmission, PlanReportBatch,
    PlanReportBatchHook, PlanReportCandidate, PlanReportCapabilityFallback, PlanReportDecision,
    PlanReportDeferral, PlanReportDivergence, PlanReportError, PlanReportEstimates,
    PlanReportGrouping, PlanReportGroupingCost, PlanReportGroupingEstimates,
    PlanReportGroupingStrategy, PlanReportLifecycle, PlanReportMode, PlanReportPhase,
    PlanReportRanking, PlanReportReleaseReason, PlanReportRow, PlanReportSelection,
    PlanReportSelectionStage,
};
pub use regime_filter::{ArrivalWindows, GaussianHmm, PageHinkley, RegimeFilter, SizeWindows};
pub use report::{
    PREFILL_REPORT_SCHEMA_VERSION, PrefillBenchmarkMetadata, PrefillBenchmarkReport,
    PrefillReportError, PrefillReportProtocol, PrefillReportSample, PrefillReportStrategySummary,
    PrefillReportTrial,
};
pub use scheduling_summary::{
    ClassSummary, DEFAULT_STARVATION_FACTOR, LONG_PROMPT_TOKENS, SchedulingComparison,
    SchedulingSummary, SlowdownSummary, VirtualTimeCandidateEvaluation, VirtualTimeHeldOutScore,
    VirtualTimeHeldOutStudy, VirtualTimeOptimizationError, compare_schedulers,
    compare_schedulers_observed, evaluate_virtual_time_held_out, isolated_end_to_end,
    optimize_virtual_time_candidate, summarize_run,
};
pub use simulation::{
    CostModel, DecodeCost, Perturbation, SchedulingRun, SimulationConfig, SimulationError,
    TrialIteration, TrialOutcome, TrialRequest, run_charged_trial, run_observed_trial,
    run_perturbed_trial, run_scheduling_trial, trace_request_spec,
};
pub use synthetic::{
    KvGeometry, synthetic_artifact, synthetic_artifact_with_prefix_cache, synthetic_group_prompt,
    synthetic_prompt, synthetic_shared_prompt, synthetic_token,
};
pub use trace::{SchedulingTrace, SharedPrefix, TraceParameters, TraceRequest};

#[derive(Clone, Debug, PartialEq)]
/// One request arrival in a workload trace.
pub struct WorkloadEvent {
    /// Arrival time relative to the trace epoch.
    pub at: Timestamp,
    /// Request submitted at that time.
    pub request: RequestSpec,
}

#[derive(Clone, Debug, Default, PartialEq)]
/// Arrival-ordered, replayable request trace.
pub struct Workload {
    events: Vec<WorkloadEvent>,
}

impl Workload {
    /// Creates a trace after validating nondecreasing arrival order.
    pub fn new(events: Vec<WorkloadEvent>) -> Result<Self, WorkloadError> {
        if events.windows(2).any(|window| window[0].at > window[1].at) {
            return Err(WorkloadError::EventsOutOfOrder);
        }
        Ok(Self { events })
    }

    /// Returns trace events in arrival order.
    pub fn events(&self) -> &[WorkloadEvent] {
        &self.events
    }

    /// Submits every event using its recorded arrival timestamp.
    ///
    /// Replay performs admission only; the experiment remains responsible for
    /// advancing scheduling and execution.
    pub fn replay(&self, runtime: &mut Runtime) -> Result<Vec<RequestId>, RuntimeError> {
        self.events
            .iter()
            .map(|event| runtime.submit(event.request.clone(), event.at))
            .collect()
    }
}

#[derive(Clone, Debug)]
/// Deterministic workload with identical request shapes and constant arrivals.
pub struct FixedWorkload {
    /// Registered model used by every request.
    pub model: ModelId,
    /// Number of arrivals in the trace.
    pub requests: usize,
    /// Token count supplied to every request.
    pub prompt_tokens: usize,
    /// Generation limit supplied to every request.
    pub output_tokens: usize,
    /// Constant time between adjacent arrivals.
    pub interarrival: Duration,
}

impl FixedWorkload {
    /// Materializes the fixed trace, rejecting empty or overflowing workloads.
    pub fn build(&self) -> Result<Workload, WorkloadError> {
        if self.requests == 0 {
            return Err(WorkloadError::NoRequests);
        }
        if self.output_tokens == 0 {
            return Err(WorkloadError::NoOutputTokens);
        }

        let interval = u64::try_from(self.interarrival.as_micros())
            .map_err(|_| WorkloadError::TimestampOverflow)?;
        let events = (0..self.requests)
            .map(|index| {
                let index = u64::try_from(index).map_err(|_| WorkloadError::TimestampOverflow)?;
                let at = interval
                    .checked_mul(index)
                    .ok_or(WorkloadError::TimestampOverflow)?;
                let mut request = RequestSpec::text(self.model.clone(), "");
                request.inputs = vec![Input::Tokens(vec![0; self.prompt_tokens])];
                request.generation.max_new_tokens = self.output_tokens;
                Ok(WorkloadEvent {
                    at: Timestamp::from_micros(at),
                    request,
                })
            })
            .collect::<Result<Vec<_>, WorkloadError>>()?;
        Workload::new(events)
    }
}

/// Deterministic compute-shape summary for one prefill grouping strategy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefillBatchSummary {
    /// Prompt rows represented by the trace.
    pub requests: usize,
    /// Model invocations produced after grouping.
    pub invocations: usize,
    /// Invocations containing at least two prompt rows.
    pub tensor_invocations: usize,
    /// Single-row invocations using the individual execution path.
    pub individual_invocations: usize,
    /// Real prompt positions across every invocation.
    pub input_tokens: usize,
    /// Artificial right-padding positions across every invocation.
    pub padding_tokens: usize,
}

impl PrefillBatchSummary {
    /// Returns real plus artificial dense input positions.
    pub fn dense_input_tokens(self) -> usize {
        self.input_tokens.saturating_add(self.padding_tokens)
    }

    /// Returns the fraction of dense positions carrying real prompt tokens.
    pub fn input_token_efficiency(self) -> Option<f64> {
        let dense = self.dense_input_tokens();
        if dense == 0 {
            None
        } else {
            Some(self.input_tokens as f64 / dense as f64)
        }
    }
}

/// Side-by-side deterministic shape comparison for the built-in strategies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillBatchComparison {
    /// All prompts grouped together after model compatibility.
    pub unbucketed: PrefillBatchSummary,
    /// Prompts grouped by next-power-of-two length.
    pub power_of_two: PrefillBatchSummary,
}

impl PrefillBatchComparison {
    /// Returns padding positions avoided by power-of-two grouping.
    pub fn padding_tokens_saved(self) -> usize {
        self.unbucketed
            .padding_tokens
            .saturating_sub(self.power_of_two.padding_tokens)
    }

    /// Returns the fraction of baseline padding avoided by bucketing.
    pub fn padding_reduction(self) -> Option<f64> {
        if self.unbucketed.padding_tokens == 0 {
            None
        } else {
            Some(self.padding_tokens_saved() as f64 / self.unbucketed.padding_tokens as f64)
        }
    }
}

/// Analyzes one same-model prompt-length trace with a caller-supplied grouper.
///
/// This measures deterministic tensor shape, not elapsed performance. Groups
/// retain first-seen bucket order, matching the executor's stable grouping
/// semantics. Every prompt length must be positive.
pub fn summarize_prefill_batches(
    prompt_lengths: &[usize],
    batcher: &mut dyn PrefillBatcher,
) -> Result<PrefillBatchSummary, WorkloadError> {
    if prompt_lengths.is_empty() {
        return Err(WorkloadError::NoPromptLengths);
    }
    if prompt_lengths.contains(&0) {
        return Err(WorkloadError::EmptyPrompt);
    }

    let mut bucket_keys = vec![0; prompt_lengths.len()];
    batcher.buckets(prompt_lengths, &mut bucket_keys);
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (&prompt_tokens, bucket) in prompt_lengths.iter().zip(bucket_keys) {
        if let Some((_, lengths)) = groups.iter_mut().find(|(key, _)| *key == bucket) {
            lengths.push(prompt_tokens);
        } else {
            groups.push((bucket, vec![prompt_tokens]));
        }
    }

    let mut summary = PrefillBatchSummary {
        requests: prompt_lengths.len(),
        invocations: groups.len(),
        ..PrefillBatchSummary::default()
    };
    for (_, lengths) in groups {
        if lengths.len() > 1 {
            summary.tensor_invocations += 1;
        } else {
            summary.individual_invocations += 1;
        }
        let input_tokens = lengths.iter().try_fold(0_usize, |total, length| {
            total
                .checked_add(*length)
                .ok_or(WorkloadError::TokenCountOverflow)
        })?;
        let padded_length = lengths.iter().copied().max().unwrap_or(0);
        let dense_tokens = padded_length
            .checked_mul(lengths.len())
            .ok_or(WorkloadError::TokenCountOverflow)?;
        summary.input_tokens = summary
            .input_tokens
            .checked_add(input_tokens)
            .ok_or(WorkloadError::TokenCountOverflow)?;
        summary.padding_tokens = summary
            .padding_tokens
            .checked_add(dense_tokens - input_tokens)
            .ok_or(WorkloadError::TokenCountOverflow)?;
    }
    Ok(summary)
}

/// Deterministic result of applying the cost-aware grouping policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostAwarePrefillAnalysis {
    /// Token-equivalent fixed cost charged per candidate invocation.
    pub invocation_overhead_tokens: usize,
    /// Candidate estimates and selected grouping strategy.
    pub decision: CostAwarePrefillDecision,
    /// Actual invocation and padding shape produced by the selected grouping.
    pub selected: PrefillBatchSummary,
}

/// Analyzes one trace using workload-aware invocation-versus-padding costs.
pub fn analyze_cost_aware_prefill(
    prompt_lengths: &[usize],
    invocation_overhead_tokens: usize,
) -> Result<CostAwarePrefillAnalysis, WorkloadError> {
    if prompt_lengths.is_empty() {
        return Err(WorkloadError::NoPromptLengths);
    }
    if prompt_lengths.contains(&0) {
        return Err(WorkloadError::EmptyPrompt);
    }
    let mut batcher = CostAwarePrefillBatcher::new(invocation_overhead_tokens);
    let decision = batcher.decide(prompt_lengths);
    let selected = summarize_prefill_batches(prompt_lengths, &mut batcher)?;
    Ok(CostAwarePrefillAnalysis {
        invocation_overhead_tokens,
        decision,
        selected,
    })
}

/// Compares the unbucketed baseline with power-of-two prompt grouping.
pub fn compare_prefill_batching(
    prompt_lengths: &[usize],
) -> Result<PrefillBatchComparison, WorkloadError> {
    let unbucketed = summarize_prefill_batches(prompt_lengths, &mut UnbucketedPrefillBatcher)?;
    let power_of_two = summarize_prefill_batches(prompt_lengths, &mut PowerOfTwoPrefillBatcher)?;
    Ok(PrefillBatchComparison {
        unbucketed,
        power_of_two,
    })
}

/// Configuration for a real model-execution comparison of prefill groupers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillExecutionConfig {
    /// Already-tokenized, non-empty prompts submitted in this order.
    pub prompts: Vec<Vec<u32>>,
    /// Maximum generated tokens per request, including the prefill token.
    pub max_new_tokens: usize,
    /// How runs of different strategies must agree for the comparison to hold.
    pub output_check: OutputCheck,
}

/// How execution-backed comparisons decide that two runs did the same work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutputCheck {
    /// Every request must generate identical tokens.
    #[default]
    Exact,
    /// Every request must generate the same number of tokens.
    ///
    /// For GPU backends whose f32 matmuls run on TF32 tensor cores: kernel
    /// selection depends on the padded shape, which perturbs logits enough
    /// to flip near-tied greedy choices while each strategy still performs
    /// the same work.
    Lengths,
}

impl OutputCheck {
    /// Whether `actual` agrees with `expected` under this check.
    pub fn accepts(self, expected: &[Vec<u32>], actual: &[Vec<u32>]) -> bool {
        match self {
            Self::Exact => expected == actual,
            Self::Lengths => {
                expected.len() == actual.len()
                    && expected
                        .iter()
                        .zip(actual)
                        .all(|(expected, actual)| expected.len() == actual.len())
            }
        }
    }
}

impl PrefillExecutionConfig {
    /// Validates the workload independently of any loaded model.
    pub fn validate(&self) -> Result<(), WorkloadError> {
        if self.prompts.is_empty() {
            return Err(WorkloadError::NoPromptLengths);
        }
        if self.prompts.iter().any(Vec::is_empty) {
            return Err(WorkloadError::EmptyPrompt);
        }
        if self.max_new_tokens == 0 {
            return Err(WorkloadError::NoOutputTokens);
        }
        Ok(())
    }
}

/// Measurements from one execution-backed prefill strategy run.
///
/// Durations are host wall-clock observations around synchronous executor
/// calls. In particular, `prefill_iteration_elapsed` includes executor work in
/// every iteration containing prefill and must not be interpreted as
/// per-request time to first token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillExecutionSummary {
    /// Requests executed by the run.
    pub requests: usize,
    /// Executor iteration boundaries required to finish all requests.
    pub iterations: usize,
    /// Output tokens emitted across all requests.
    pub output_tokens: usize,
    /// Model invocations that performed prefill.
    pub prefill_invocations: usize,
    /// Prefill invocations using a shared tensor call.
    pub tensor_prefill_invocations: usize,
    /// Prefill invocations falling back to one request at a time.
    pub individual_prefill_invocations: usize,
    /// Real prompt positions processed by prefill invocations.
    pub prefill_input_tokens: usize,
    /// Artificial right-padding positions processed by prefill invocations.
    pub prefill_padding_tokens: usize,
    /// Wall time spent in complete executor iterations containing prefill.
    pub prefill_iteration_elapsed: Duration,
    /// Wall time from immediately before the first iteration until completion.
    pub total_elapsed: Duration,
    /// Highest physical KV-cache byte count observed during execution.
    pub peak_kv_cache_bytes: u64,
    /// Highest worst-case KV reservation observed during execution.
    pub peak_reserved_kv_bytes: u64,
    /// Canonical comma-separated executed provider names, when applicable.
    pub kernel: Option<String>,
}

impl PrefillExecutionSummary {
    /// Returns real plus artificial prefill tensor positions.
    pub fn dense_prefill_tokens(&self) -> usize {
        self.prefill_input_tokens
            .saturating_add(self.prefill_padding_tokens)
    }

    /// Returns the fraction of prefill positions carrying real prompt tokens.
    pub fn prefill_token_efficiency(&self) -> Option<f64> {
        let dense = self.dense_prefill_tokens();
        (dense != 0).then(|| self.prefill_input_tokens as f64 / dense as f64)
    }

    /// Returns emitted output tokens per second over the full measured run.
    pub fn output_tokens_per_second(&self) -> Option<f64> {
        let seconds = self.total_elapsed.as_secs_f64();
        (seconds > 0.0).then(|| self.output_tokens as f64 / seconds)
    }
}

/// Execution-backed results for the unbucketed and power-of-two strategies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillExecutionComparison {
    /// Run that places all compatible prompts in one tensor group.
    pub unbucketed: PrefillExecutionSummary,
    /// Run that groups compatible prompts by next-power-of-two length.
    pub power_of_two: PrefillExecutionSummary,
}

/// Execution-backed comparison of complete and resumable prompt processing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedPrefillComparison {
    /// Complete-prompt compatibility execution.
    pub unchunked: PrefillExecutionSummary,
    /// Execution using the configured maximum segment size.
    pub chunked: PrefillExecutionSummary,
    /// Maximum real prompt positions processed per request per iteration.
    pub chunk_size: usize,
}

/// Per-request scheduler progress from one mixed prefill/decode run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillFairnessRequest {
    /// Request index in the configured prompt order.
    pub request_index: usize,
    /// Iteration that emitted this request's first output token.
    pub first_token_iteration: usize,
    /// Iteration that completed this request.
    pub completion_iteration: usize,
    /// Host time from run start to this request's first output token.
    pub ttft: Duration,
    /// Host time from run start until this request completed.
    pub completion_latency: Duration,
}

/// Fairness and completion measurements for one prompt chunk size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillFairnessRun {
    /// `None` means complete-prompt baseline; otherwise the configured segment
    /// token limit.
    pub chunk_size: Option<usize>,
    /// Total scheduler iterations until every request completed.
    pub iterations: usize,
    /// Number of non-final prompt-segment progress events.
    pub pending_segment_events: usize,
    /// Per-request first-token and completion iteration boundaries.
    pub requests: Vec<PrefillFairnessRequest>,
    /// Prefill/decode model work summary for the run.
    pub execution: PrefillExecutionSummary,
}

/// Complete-prompt baseline and chunk-size sweep for a mixed workload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillFairnessComparison {
    /// Complete-prompt scheduler behavior.
    pub unchunked: PrefillFairnessRun,
    /// Runs in the same order as the requested chunk-size list.
    pub chunked: Vec<PrefillFairnessRun>,
}

/// Repeated mixed-workload fairness measurements with per-request latency
/// distributions. Samples retain measured-trial order for every strategy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillFairnessTrials {
    /// Number of complete strategy sweeps discarded before measurement.
    pub warmup_trials: usize,
    /// Strategy order for each measured trial (`None` is complete prefill).
    pub strategy_order_by_trial: Vec<Vec<Option<usize>>>,
    /// Complete-prefill samples in measured-trial order.
    pub unchunked: Vec<PrefillFairnessRun>,
    /// Chunk-size samples in the same order as the requested sizes.
    pub chunked: Vec<Vec<PrefillFairnessRun>>,
}

/// Per-request nearest-rank TTFT and completion-latency summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillFairnessLatency {
    /// Prompt index, stable across all samples.
    pub request_index: usize,
    /// TTFT distribution across retained measured runs.
    pub ttft: LatencySummary,
    /// Completion-latency distribution across retained measured runs.
    pub completion: LatencySummary,
}

impl PrefillFairnessTrials {
    /// Summarizes per-request TTFT/completion latency for complete prefill.
    pub fn unchunked_latency(&self) -> Vec<PrefillFairnessLatency> {
        summarize_fairness_latency(&self.unchunked)
    }

    /// Summarizes per-request TTFT/completion latency for each chunk size.
    pub fn chunked_latency(&self) -> Vec<Vec<PrefillFairnessLatency>> {
        self.chunked
            .iter()
            .map(|runs| summarize_fairness_latency(runs))
            .collect()
    }
}

/// Prefill grouping strategy used by an execution benchmark run.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillStrategy {
    /// Place every compatible prompt in one group.
    Unbucketed,
    /// Group compatible prompts by next-power-of-two length.
    PowerOfTwo,
    /// Select grouping from dense work and calibrated invocation overhead.
    CostAware,
}

/// Warm-up and sampling protocol for a repeated prefill comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillTrialConfig {
    /// Request workload executed by every strategy run.
    pub execution: PrefillExecutionConfig,
    /// Complete paired trials executed and discarded before measurement.
    pub warmup_trials: usize,
    /// Complete paired trials retained in the result.
    pub measured_trials: usize,
}

/// Raw samples and derived latency distributions for one strategy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillStrategyTrials {
    /// Per-run measurements in paired trial order.
    pub samples: Vec<PrefillExecutionSummary>,
    /// Distribution of complete executor iterations containing prefill.
    pub prefill_iteration_latency: LatencySummary,
    /// Distribution of complete run wall times.
    pub total_latency: LatencySummary,
}

impl PrefillStrategyTrials {
    /// Returns aggregate emitted-token throughput across every retained run.
    pub fn output_tokens_per_second(&self) -> Option<f64> {
        let output_tokens = self
            .samples
            .iter()
            .map(|sample| sample.output_tokens)
            .sum::<usize>();
        let elapsed = self
            .samples
            .iter()
            .map(|sample| sample.total_elapsed.as_secs_f64())
            .sum::<f64>();
        (elapsed > 0.0).then(|| output_tokens as f64 / elapsed)
    }

    /// Returns the largest physical KV-cache observation across retained runs.
    pub fn peak_kv_cache_bytes(&self) -> u64 {
        self.samples
            .iter()
            .map(|sample| sample.peak_kv_cache_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Returns the largest conservative KV reservation across retained runs.
    pub fn peak_reserved_kv_bytes(&self) -> u64 {
        self.samples
            .iter()
            .map(|sample| sample.peak_reserved_kv_bytes)
            .max()
            .unwrap_or(0)
    }
}

/// Repeated execution-backed comparison with retained raw paired samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillTrialComparison {
    /// Number of discarded paired trials completed before measurement.
    pub warmup_trials: usize,
    /// Strategy executed first in each retained paired trial.
    pub first_strategy_by_trial: Vec<PrefillStrategy>,
    /// Unbucketed samples and latency summaries.
    pub unbucketed: PrefillStrategyTrials,
    /// Power-of-two samples and latency summaries.
    pub power_of_two: PrefillStrategyTrials,
}

/// Protocol for a fresh, order-balanced three-strategy comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillThreeWayTrialConfig {
    /// Request workload executed by every strategy run.
    pub execution: PrefillExecutionConfig,
    /// Complete three-strategy trials discarded before measurement.
    pub warmup_trials: usize,
    /// Complete three-strategy trials retained in the result.
    pub measured_trials: usize,
    /// Calibrated token-equivalent fixed cost per model invocation.
    pub adaptive_invocation_overhead_tokens: usize,
}

/// Repeated measurements for fixed and adaptive prefill grouping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillThreeWayComparison {
    /// Number of discarded three-strategy trials.
    pub warmup_trials: usize,
    /// Complete strategy execution order for each retained trial.
    pub strategy_order_by_trial: Vec<[PrefillStrategy; 3]>,
    /// Unbucketed measurements.
    pub unbucketed: PrefillStrategyTrials,
    /// Power-of-two measurements.
    pub power_of_two: PrefillStrategyTrials,
    /// Cost-aware measurements using the configured overhead.
    pub cost_aware: PrefillStrategyTrials,
}

/// Invocation-overhead estimate derived from two measured strategies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillCostCalibration {
    /// Rounded token-equivalent fixed cost per model invocation.
    pub invocation_overhead_tokens: usize,
    /// Unbucketed p50 prefill-iteration wall time.
    pub unbucketed_p50_micros: u64,
    /// Power-of-two p50 prefill-iteration wall time.
    pub power_of_two_p50_micros: u64,
    /// Unbucketed dense positions per run.
    pub unbucketed_dense_tokens: usize,
    /// Power-of-two dense positions per run.
    pub power_of_two_dense_tokens: usize,
    /// Unbucketed invocations per run.
    pub unbucketed_invocations: usize,
    /// Power-of-two invocations per run.
    pub power_of_two_invocations: usize,
}

/// Executes identical tokenized requests through both prefill strategies.
///
/// Both runs use cloned handles to the same loaded model, greedy generation,
/// an unconstrained executor, and the same request order. The function rejects
/// the comparison if the two strategies produce different token sequences.
/// Timings are a local operational probe: strategy order, backend warm-up, and
/// host load are not controlled as they would be in a statistical benchmark.
pub fn benchmark_prefill_execution(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
) -> Result<PrefillExecutionComparison, ExecutionBenchmarkError> {
    config.validate()?;
    let (unbucketed, baseline_outputs) =
        execute_prefill_strategy(model, config, UnbucketedPrefillBatcher)?;
    let (power_of_two, bucketed_outputs) =
        execute_prefill_strategy(model, config, PowerOfTwoPrefillBatcher)?;
    if !config
        .output_check
        .accepts(&baseline_outputs, &bucketed_outputs)
    {
        return Err(ExecutionBenchmarkError::OutputMismatch);
    }
    Ok(PrefillExecutionComparison {
        unbucketed,
        power_of_two,
    })
}

/// Executes identical requests with complete and bounded resumable prefill.
///
/// The generated token sequences must match exactly. This comparison exposes
/// the latency/iteration tradeoff of interruption points; it does not claim
/// that smaller chunks improve total throughput.
pub fn benchmark_chunked_prefill_execution(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    chunk_size: usize,
) -> Result<ChunkedPrefillComparison, ExecutionBenchmarkError> {
    config.validate()?;
    if chunk_size == 0 {
        return Err(ExecutionBenchmarkError::InvalidChunkSize(chunk_size));
    }
    if !model.supports_chunked_prefill() {
        return Err(ExecutionBenchmarkError::ChunkedPrefillUnsupported);
    }
    let (unchunked, unchunked_outputs) =
        execute_prefill_strategy(model, config, UnbucketedPrefillBatcher)?;
    let (chunked, chunked_outputs) = execute_prefill_strategy_with_chunk_size(
        model,
        config,
        UnbucketedPrefillBatcher,
        Some(chunk_size),
    )?;
    if !config
        .output_check
        .accepts(&unchunked_outputs, &chunked_outputs)
    {
        return Err(ExecutionBenchmarkError::OutputMismatch);
    }
    Ok(ChunkedPrefillComparison {
        unchunked,
        chunked,
        chunk_size,
    })
}

/// Measures per-request first-token and completion fairness for a prompt-size
/// sweep under the same bounded number of live sessions.
///
/// All requests arrive at the same logical timestamp. Each run is recreated
/// from scratch, and every chunk-size run must produce exactly the same output
/// tokens as the complete-prompt baseline. Iteration counts are deterministic
/// scheduler measurements; wall-clock values remain operational observations.
pub fn benchmark_prefill_fairness(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    max_active_requests: usize,
    chunk_sizes: &[usize],
) -> Result<PrefillFairnessComparison, ExecutionBenchmarkError> {
    config.validate()?;
    if chunk_sizes.is_empty() {
        return Err(ExecutionBenchmarkError::NoChunkSizes);
    }
    if chunk_sizes.contains(&0) {
        return Err(ExecutionBenchmarkError::InvalidChunkSize(0));
    }
    if !model.supports_chunked_prefill() {
        return Err(ExecutionBenchmarkError::ChunkedPrefillUnsupported);
    }
    let (unchunked, baseline_outputs) =
        run_prefill_fairness(model, config, max_active_requests, None)?;
    let mut chunked = Vec::with_capacity(chunk_sizes.len());
    for &chunk_size in chunk_sizes {
        let (run, outputs) =
            run_prefill_fairness(model, config, max_active_requests, Some(chunk_size))?;
        if !config.output_check.accepts(&baseline_outputs, &outputs) {
            return Err(ExecutionBenchmarkError::OutputMismatch);
        }
        chunked.push(run);
    }
    Ok(PrefillFairnessComparison { unchunked, chunked })
}

/// Runs warm-up and measured mixed-workload fairness sweeps with rotating
/// strategy order. Each measured run retains per-request host TTFT and
/// completion latency; outputs must match both within and across trials.
pub fn benchmark_prefill_fairness_trials(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    max_active_requests: usize,
    chunk_sizes: &[usize],
    warmup_trials: usize,
    measured_trials: usize,
) -> Result<PrefillFairnessTrials, ExecutionBenchmarkError> {
    config.validate()?;
    if chunk_sizes.is_empty() {
        return Err(ExecutionBenchmarkError::NoChunkSizes);
    }
    if chunk_sizes.contains(&0) {
        return Err(ExecutionBenchmarkError::InvalidChunkSize(0));
    }
    if measured_trials == 0 {
        return Err(ExecutionBenchmarkError::NoMeasuredTrials);
    }
    if !model.supports_chunked_prefill() {
        return Err(ExecutionBenchmarkError::ChunkedPrefillUnsupported);
    }

    let strategy_count = chunk_sizes.len() + 1;
    let mut expected_outputs: Option<Vec<Vec<u32>>> = None;
    let mut measured_unchunked = Vec::with_capacity(measured_trials);
    let mut measured_chunked = vec![Vec::with_capacity(measured_trials); chunk_sizes.len()];
    let mut strategy_order_by_trial = Vec::with_capacity(measured_trials);

    let total_trials = warmup_trials
        .checked_add(measured_trials)
        .ok_or(ExecutionBenchmarkError::IterationLimit)?;
    for trial in 0..total_trials {
        let rotation = trial % strategy_count;
        let mut order = (0..strategy_count)
            .map(|index| (index + rotation) % strategy_count)
            .collect::<Vec<_>>();
        // Alternate direction as well as first position to distribute order effects.
        if trial % 2 == 1 {
            order.reverse();
        }
        let trial_order = order
            .iter()
            .map(|strategy| (*strategy != 0).then(|| chunk_sizes[*strategy - 1]))
            .collect::<Vec<_>>();
        let mut runs = vec![None; strategy_count];
        let mut trial_outputs: Option<Vec<Vec<u32>>> = None;
        for strategy in order {
            let chunk_size = (strategy != 0).then(|| chunk_sizes[strategy - 1]);
            let (run, outputs) =
                run_prefill_fairness(model, config, max_active_requests, chunk_size)?;
            if trial_outputs
                .as_ref()
                .is_some_and(|expected| !config.output_check.accepts(expected, &outputs))
            {
                return Err(ExecutionBenchmarkError::OutputMismatch);
            }
            trial_outputs.get_or_insert(outputs.clone());
            if expected_outputs
                .as_ref()
                .is_some_and(|expected| !config.output_check.accepts(expected, &outputs))
            {
                return Err(ExecutionBenchmarkError::OutputMismatch);
            }
            expected_outputs.get_or_insert(outputs);
            runs[if strategy == 0 { 0 } else { strategy }] = Some(run);
        }
        if trial >= warmup_trials {
            strategy_order_by_trial.push(trial_order);
            measured_unchunked.push(runs[0].take().expect("baseline strategy ran"));
            for (index, samples) in measured_chunked.iter_mut().enumerate() {
                samples.push(runs[index + 1].take().expect("chunk strategy ran"));
            }
        }
    }
    Ok(PrefillFairnessTrials {
        warmup_trials,
        strategy_order_by_trial,
        unchunked: measured_unchunked,
        chunked: measured_chunked,
    })
}

/// Runs warm-up and measured paired trials with alternating strategy order.
///
/// Every paired trial executes both strategies and verifies identical output
/// token sequences. Warm-up trials are discarded. Measured trial zero runs the
/// unbucketed strategy first, trial one runs power-of-two first, and the order
/// continues alternating. The `samples` vectors retain paired trial order, so
/// entries at the same index were collected in the same trial.
pub fn benchmark_prefill_trials(
    model: &CausalLanguageModel,
    config: &PrefillTrialConfig,
) -> Result<PrefillTrialComparison, ExecutionBenchmarkError> {
    config.execution.validate()?;
    if config.measured_trials == 0 {
        return Err(ExecutionBenchmarkError::NoMeasuredTrials);
    }

    for trial in 0..config.warmup_trials {
        let first = alternating_strategy(trial);
        execute_strategy_pair(model, &config.execution, first)?;
    }

    let mut first_strategy_by_trial = Vec::with_capacity(config.measured_trials);
    let mut unbucketed_samples = Vec::with_capacity(config.measured_trials);
    let mut power_of_two_samples = Vec::with_capacity(config.measured_trials);
    let mut expected_outputs: Option<Vec<Vec<u32>>> = None;
    for trial in 0..config.measured_trials {
        let first = alternating_strategy(trial);
        let (unbucketed, power_of_two, outputs) =
            execute_strategy_pair(model, &config.execution, first)?;
        if expected_outputs
            .as_ref()
            .is_some_and(|expected| !config.execution.output_check.accepts(expected, &outputs))
        {
            return Err(ExecutionBenchmarkError::OutputMismatch);
        }
        expected_outputs.get_or_insert(outputs);
        first_strategy_by_trial.push(first);
        unbucketed_samples.push(unbucketed);
        power_of_two_samples.push(power_of_two);
    }

    Ok(PrefillTrialComparison {
        warmup_trials: config.warmup_trials,
        first_strategy_by_trial,
        unbucketed: summarize_strategy_trials(unbucketed_samples),
        power_of_two: summarize_strategy_trials(power_of_two_samples),
    })
}

/// Calibrates invocation overhead from measured unbucketed and bucketed runs.
///
/// The calibration fits `time = dense_tokens * a + invocations * b` to the two
/// p50 prefill-iteration measurements, then reports `b / a` rounded to the
/// nearest token. It requires power-of-two grouping to use fewer dense tokens
/// and more invocations than the baseline, plus a positive solution for both
/// coefficients. Callers should retain the source measurements because this
/// two-point model is an operational estimate, not a hardware constant.
pub fn calibrate_prefill_invocation_overhead(
    comparison: &PrefillTrialComparison,
) -> Result<PrefillCostCalibration, ExecutionBenchmarkError> {
    let unbucketed = consistent_prefill_shape(&comparison.unbucketed)?;
    let power_of_two = consistent_prefill_shape(&comparison.power_of_two)?;
    if unbucketed.0 <= power_of_two.0 || unbucketed.1 >= power_of_two.1 {
        return Err(ExecutionBenchmarkError::CalibrationUnavailable(
            "strategies do not trade fewer dense tokens for more invocations",
        ));
    }
    let unbucketed_time = comparison.unbucketed.prefill_iteration_latency.p50_micros as u128;
    let power_of_two_time = comparison.power_of_two.prefill_iteration_latency.p50_micros as u128;
    if unbucketed_time == 0 || power_of_two_time == 0 {
        return Err(ExecutionBenchmarkError::CalibrationUnavailable(
            "p50 prefill measurements must be non-zero",
        ));
    }

    let dense_unbucketed = unbucketed.0 as u128;
    let dense_power_of_two = power_of_two.0 as u128;
    let calls_unbucketed = unbucketed.1 as u128;
    let calls_power_of_two = power_of_two.1 as u128;
    let dense_time_numerator = unbucketed_time
        .checked_mul(calls_power_of_two)
        .and_then(|left| {
            power_of_two_time
                .checked_mul(calls_unbucketed)
                .and_then(|right| left.checked_sub(right))
        })
        .ok_or(ExecutionBenchmarkError::CalibrationUnavailable(
            "measurements imply a non-positive dense-token cost",
        ))?;
    let invocation_time_numerator = dense_unbucketed
        .checked_mul(power_of_two_time)
        .and_then(|left| {
            dense_power_of_two
                .checked_mul(unbucketed_time)
                .and_then(|right| left.checked_sub(right))
        })
        .ok_or(ExecutionBenchmarkError::CalibrationUnavailable(
            "measurements imply a non-positive invocation cost",
        ))?;
    if dense_time_numerator == 0 {
        return Err(ExecutionBenchmarkError::CalibrationUnavailable(
            "measurements imply a zero dense-token cost",
        ));
    }
    let rounded = invocation_time_numerator
        .checked_add(dense_time_numerator / 2)
        .map(|value| value / dense_time_numerator)
        .ok_or(ExecutionBenchmarkError::CalibrationUnavailable(
            "calibrated invocation overhead overflowed",
        ))?;
    let invocation_overhead_tokens = usize::try_from(rounded).map_err(|_| {
        ExecutionBenchmarkError::CalibrationUnavailable(
            "calibrated invocation overhead exceeds the platform token range",
        )
    })?;
    Ok(PrefillCostCalibration {
        invocation_overhead_tokens,
        unbucketed_p50_micros: comparison.unbucketed.prefill_iteration_latency.p50_micros,
        power_of_two_p50_micros: comparison.power_of_two.prefill_iteration_latency.p50_micros,
        unbucketed_dense_tokens: unbucketed.0,
        power_of_two_dense_tokens: power_of_two.0,
        unbucketed_invocations: unbucketed.1,
        power_of_two_invocations: power_of_two.1,
    })
}

/// Runs fresh unbucketed, power-of-two, and cost-aware measured trials.
///
/// Strategy order rotates on every trial so each strategy occupies the first,
/// second, and third position equally when the trial count is a multiple of
/// three. Every run must produce the same output token sequences.
pub fn benchmark_three_way_prefill_trials(
    model: &CausalLanguageModel,
    config: &PrefillThreeWayTrialConfig,
) -> Result<PrefillThreeWayComparison, ExecutionBenchmarkError> {
    config.execution.validate()?;
    if config.measured_trials == 0 {
        return Err(ExecutionBenchmarkError::NoMeasuredTrials);
    }
    for trial in 0..config.warmup_trials {
        execute_three_strategy_trial(model, config, three_way_order(trial))?;
    }

    let mut strategy_order_by_trial = Vec::with_capacity(config.measured_trials);
    let mut unbucketed_samples = Vec::with_capacity(config.measured_trials);
    let mut power_of_two_samples = Vec::with_capacity(config.measured_trials);
    let mut cost_aware_samples = Vec::with_capacity(config.measured_trials);
    let mut expected_outputs: Option<Vec<Vec<u32>>> = None;
    for trial in 0..config.measured_trials {
        let order = three_way_order(trial);
        let ThreeWayTrialResult {
            unbucketed,
            power_of_two,
            cost_aware,
            outputs,
        } = execute_three_strategy_trial(model, config, order)?;
        if expected_outputs
            .as_ref()
            .is_some_and(|expected| !config.execution.output_check.accepts(expected, &outputs))
        {
            return Err(ExecutionBenchmarkError::OutputMismatch);
        }
        expected_outputs.get_or_insert(outputs);
        strategy_order_by_trial.push(order);
        unbucketed_samples.push(unbucketed);
        power_of_two_samples.push(power_of_two);
        cost_aware_samples.push(cost_aware);
    }
    Ok(PrefillThreeWayComparison {
        warmup_trials: config.warmup_trials,
        strategy_order_by_trial,
        unbucketed: summarize_strategy_trials(unbucketed_samples),
        power_of_two: summarize_strategy_trials(power_of_two_samples),
        cost_aware: summarize_strategy_trials(cost_aware_samples),
    })
}

fn consistent_prefill_shape(
    trials: &PrefillStrategyTrials,
) -> Result<(usize, usize), ExecutionBenchmarkError> {
    let first = trials
        .samples
        .first()
        .ok_or(ExecutionBenchmarkError::CalibrationUnavailable(
            "calibration requires retained samples",
        ))?;
    let shape = (first.dense_prefill_tokens(), first.prefill_invocations);
    if trials
        .samples
        .iter()
        .any(|sample| (sample.dense_prefill_tokens(), sample.prefill_invocations) != shape)
    {
        return Err(ExecutionBenchmarkError::CalibrationUnavailable(
            "prefill shape changed across measured trials",
        ));
    }
    Ok(shape)
}

fn three_way_order(trial: usize) -> [PrefillStrategy; 3] {
    match trial % 3 {
        0 => [
            PrefillStrategy::Unbucketed,
            PrefillStrategy::PowerOfTwo,
            PrefillStrategy::CostAware,
        ],
        1 => [
            PrefillStrategy::PowerOfTwo,
            PrefillStrategy::CostAware,
            PrefillStrategy::Unbucketed,
        ],
        _ => [
            PrefillStrategy::CostAware,
            PrefillStrategy::Unbucketed,
            PrefillStrategy::PowerOfTwo,
        ],
    }
}

struct ThreeWayTrialResult {
    unbucketed: PrefillExecutionSummary,
    power_of_two: PrefillExecutionSummary,
    cost_aware: PrefillExecutionSummary,
    outputs: Vec<Vec<u32>>,
}

fn execute_three_strategy_trial(
    model: &CausalLanguageModel,
    config: &PrefillThreeWayTrialConfig,
    order: [PrefillStrategy; 3],
) -> Result<ThreeWayTrialResult, ExecutionBenchmarkError> {
    let mut unbucketed = None;
    let mut power_of_two = None;
    let mut cost_aware = None;
    let mut outputs: Option<Vec<Vec<u32>>> = None;
    for strategy in order {
        let (summary, current_outputs) = execute_named_strategy(
            model,
            &config.execution,
            strategy,
            config.adaptive_invocation_overhead_tokens,
        )?;
        if outputs.as_ref().is_some_and(|expected| {
            !config
                .execution
                .output_check
                .accepts(expected, &current_outputs)
        }) {
            return Err(ExecutionBenchmarkError::OutputMismatch);
        }
        outputs.get_or_insert(current_outputs);
        match strategy {
            PrefillStrategy::Unbucketed => unbucketed = Some(summary),
            PrefillStrategy::PowerOfTwo => power_of_two = Some(summary),
            PrefillStrategy::CostAware => cost_aware = Some(summary),
        }
    }
    Ok(ThreeWayTrialResult {
        unbucketed: unbucketed.expect("three-way order includes unbucketed"),
        power_of_two: power_of_two.expect("three-way order includes power-of-two"),
        cost_aware: cost_aware.expect("three-way order includes cost-aware"),
        outputs: outputs.expect("three-way order is non-empty"),
    })
}

fn execute_named_strategy(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    strategy: PrefillStrategy,
    invocation_overhead_tokens: usize,
) -> Result<(PrefillExecutionSummary, Vec<Vec<u32>>), ExecutionBenchmarkError> {
    match strategy {
        PrefillStrategy::Unbucketed => {
            execute_prefill_strategy(model, config, UnbucketedPrefillBatcher)
        }
        PrefillStrategy::PowerOfTwo => {
            execute_prefill_strategy(model, config, PowerOfTwoPrefillBatcher)
        }
        PrefillStrategy::CostAware => execute_prefill_strategy(
            model,
            config,
            CostAwarePrefillBatcher::new(invocation_overhead_tokens),
        ),
    }
}

fn alternating_strategy(trial: usize) -> PrefillStrategy {
    if trial.is_multiple_of(2) {
        PrefillStrategy::Unbucketed
    } else {
        PrefillStrategy::PowerOfTwo
    }
}

fn execute_strategy_pair(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    first: PrefillStrategy,
) -> Result<
    (
        PrefillExecutionSummary,
        PrefillExecutionSummary,
        Vec<Vec<u32>>,
    ),
    ExecutionBenchmarkError,
> {
    let run = |strategy| execute_named_strategy(model, config, strategy, 0);
    let second = match first {
        PrefillStrategy::Unbucketed => PrefillStrategy::PowerOfTwo,
        PrefillStrategy::PowerOfTwo => PrefillStrategy::Unbucketed,
        PrefillStrategy::CostAware => unreachable!("paired order excludes cost-aware grouping"),
    };
    let (first_summary, first_outputs) = run(first)?;
    let (second_summary, second_outputs) = run(second)?;
    if !config.output_check.accepts(&first_outputs, &second_outputs) {
        return Err(ExecutionBenchmarkError::OutputMismatch);
    }
    let (unbucketed, power_of_two) = match first {
        PrefillStrategy::Unbucketed => (first_summary, second_summary),
        PrefillStrategy::PowerOfTwo => (second_summary, first_summary),
        PrefillStrategy::CostAware => unreachable!("paired order excludes cost-aware grouping"),
    };
    Ok((unbucketed, power_of_two, first_outputs))
}

fn summarize_strategy_trials(samples: Vec<PrefillExecutionSummary>) -> PrefillStrategyTrials {
    let prefill = samples
        .iter()
        .map(|sample| sample.prefill_iteration_elapsed)
        .collect::<Vec<_>>();
    let total = samples
        .iter()
        .map(|sample| sample.total_elapsed)
        .collect::<Vec<_>>();
    PrefillStrategyTrials {
        prefill_iteration_latency: summarize_latencies(&prefill),
        total_latency: summarize_latencies(&total),
        samples,
    }
}

fn record_kernel_names(
    batches: &[ola_runtime::ExecutionBatch],
    kernels: &mut std::collections::BTreeSet<String>,
) {
    kernels.extend(batches.iter().filter_map(|batch| {
        batch
            .kernel
            .filter(|kernel| kernel.calls > 0)
            .map(|kernel| kernel.descriptor.name().to_owned())
    }));
}

fn kernel_identity(kernels: std::collections::BTreeSet<String>) -> Option<String> {
    (!kernels.is_empty()).then(|| kernels.into_iter().collect::<Vec<_>>().join(","))
}

fn execute_prefill_strategy<P>(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    batcher: P,
) -> Result<(PrefillExecutionSummary, Vec<Vec<u32>>), ExecutionBenchmarkError>
where
    P: PrefillBatcher + 'static,
{
    execute_prefill_strategy_with_chunk_size(model, config, batcher, None)
}

fn execute_prefill_strategy_with_chunk_size<P>(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    batcher: P,
    chunk_size: Option<usize>,
) -> Result<(PrefillExecutionSummary, Vec<Vec<u32>>), ExecutionBenchmarkError>
where
    P: PrefillBatcher + 'static,
{
    execute_prefill_strategy_with_options(model, config, batcher, chunk_size, config.prompts.len())
}

fn execute_prefill_strategy_with_options<P>(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    batcher: P,
    chunk_size: Option<usize>,
    max_active_requests: usize,
) -> Result<(PrefillExecutionSummary, Vec<Vec<u32>>), ExecutionBenchmarkError>
where
    P: PrefillBatcher + 'static,
{
    const MODEL_ID: &str = "prefill-execution-benchmark";

    let executor = Executor::with_max_active_requests(Runtime::default(), max_active_requests)?
        .with_policy(ComposedPolicy::default().with_grouping(batcher));
    let mut executor = match chunk_size {
        Some(chunk_size) => executor.with_prefill_chunk_size(chunk_size)?,
        None => executor,
    };
    executor.register_model(
        ModelDescriptor::language(MODEL_ID, "execution-benchmark"),
        model.clone(),
    )?;

    let mut request_ids = Vec::with_capacity(config.prompts.len());
    for prompt in &config.prompts {
        let mut request = RequestSpec::text(MODEL_ID, "");
        request.inputs = vec![Input::Tokens(prompt.clone())];
        request.generation = GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        };
        request_ids.push(executor.submit(request, Timestamp::ZERO)?);
    }

    let mut summary = PrefillExecutionSummary {
        requests: config.prompts.len(),
        iterations: 0,
        output_tokens: 0,
        prefill_invocations: 0,
        tensor_prefill_invocations: 0,
        individual_prefill_invocations: 0,
        prefill_input_tokens: 0,
        prefill_padding_tokens: 0,
        prefill_iteration_elapsed: Duration::ZERO,
        total_elapsed: Duration::ZERO,
        peak_kv_cache_bytes: 0,
        peak_reserved_kv_bytes: 0,
        kernel: None,
    };
    let mut kernels = std::collections::BTreeSet::new();
    let prefill_iterations = match chunk_size {
        Some(chunk_size) => config.prompts.iter().try_fold(0_usize, |total, prompt| {
            total.checked_add(prompt.len().div_ceil(chunk_size))
        }),
        None => Some(config.prompts.len()),
    }
    .ok_or(ExecutionBenchmarkError::IterationLimit)?;
    let iteration_limit = config
        .max_new_tokens
        .checked_mul(config.prompts.len())
        .and_then(|limit| limit.checked_add(prefill_iterations))
        .and_then(|limit| limit.checked_add(config.prompts.len()))
        .and_then(|limit| limit.checked_add(2))
        .ok_or(ExecutionBenchmarkError::IterationLimit)?;
    let total_started = Instant::now();

    while executor.has_active_request() || executor.state().queued_len() != 0 {
        if summary.iterations >= iteration_limit {
            return Err(ExecutionBenchmarkError::IterationLimit);
        }
        summary.iterations += 1;
        let at = u64::try_from(summary.iterations)
            .map(Timestamp::from_micros)
            .map_err(|_| ExecutionBenchmarkError::TimestampOverflow)?;
        let iteration_started = Instant::now();
        let iteration = executor.step_iteration(at)?;
        record_kernel_names(&iteration.execution_batches, &mut kernels);
        let iteration_elapsed = iteration_started.elapsed();
        let contains_prefill = iteration
            .execution_batches
            .iter()
            .any(|batch| batch.phase == BatchPhase::Prefill);
        if contains_prefill {
            summary.prefill_iteration_elapsed += iteration_elapsed;
        }

        summary.peak_kv_cache_bytes = summary.peak_kv_cache_bytes.max(iteration.kv_cache_bytes);
        summary.peak_reserved_kv_bytes = summary
            .peak_reserved_kv_bytes
            .max(iteration.reserved_kv_bytes);
        for event in &iteration.events {
            match event {
                ExecutorEvent::Token { kv_cache_bytes, .. } => {
                    summary.output_tokens += 1;
                    summary.peak_kv_cache_bytes = summary.peak_kv_cache_bytes.max(*kv_cache_bytes);
                }
                ExecutorEvent::PrefillProgress { kv_cache_bytes, .. } => {
                    summary.peak_kv_cache_bytes = summary.peak_kv_cache_bytes.max(*kv_cache_bytes);
                }
                ExecutorEvent::Failed { message, .. } => {
                    return Err(ExecutionBenchmarkError::RequestFailed(message.clone()));
                }
                ExecutorEvent::Deferred { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral("admission"));
                }
                ExecutorEvent::BatchDeferred { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral("batch policy"));
                }
                ExecutorEvent::Rejected { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral(
                        "admission rejection",
                    ));
                }
                ExecutorEvent::Preempted { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral("preemption"));
                }
                ExecutorEvent::Idle => {
                    return Err(ExecutionBenchmarkError::UnexpectedIdle);
                }
            }
        }
        for batch in iteration
            .execution_batches
            .iter()
            .filter(|batch| batch.phase == BatchPhase::Prefill)
        {
            summary.prefill_invocations += 1;
            summary.prefill_input_tokens += batch.input_tokens;
            summary.prefill_padding_tokens += batch.padding_tokens;
            match batch.mode {
                BatchExecutionMode::Tensor => summary.tensor_prefill_invocations += 1,
                BatchExecutionMode::Individual => summary.individual_prefill_invocations += 1,
            }
        }
    }
    summary.total_elapsed = total_started.elapsed();
    summary.kernel = kernel_identity(kernels);

    let outputs = request_ids
        .into_iter()
        .map(|request_id| {
            executor
                .result(request_id)
                .map(|result| result.token_ids.clone())
                .ok_or(ExecutionBenchmarkError::MissingResult(request_id))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((summary, outputs))
}

fn run_prefill_fairness(
    model: &CausalLanguageModel,
    config: &PrefillExecutionConfig,
    max_active_requests: usize,
    chunk_size: Option<usize>,
) -> Result<(PrefillFairnessRun, Vec<Vec<u32>>), ExecutionBenchmarkError> {
    const MODEL_ID: &str = "prefill-fairness-benchmark";
    let executor = Executor::with_max_active_requests(Runtime::default(), max_active_requests)?;
    let mut executor = match chunk_size {
        Some(size) => executor.with_prefill_chunk_size(size)?,
        None => executor,
    };
    executor.register_model(
        ModelDescriptor::language(MODEL_ID, "prefill-fairness-benchmark"),
        model.clone(),
    )?;
    let mut request_ids = Vec::with_capacity(config.prompts.len());
    let started = Instant::now();
    for prompt in &config.prompts {
        let mut request = RequestSpec::text(MODEL_ID, "");
        request.inputs = vec![Input::Tokens(prompt.clone())];
        request.generation = GenerationConfig {
            max_new_tokens: config.max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        };
        request_ids.push(executor.submit(request, Timestamp::ZERO)?);
    }
    let index_by_request = request_ids
        .iter()
        .enumerate()
        .map(|(index, request_id)| (*request_id, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut first_tokens: Vec<Option<(usize, Duration)>> = vec![None; request_ids.len()];
    let mut completions: Vec<Option<(usize, Duration)>> = vec![None; request_ids.len()];
    let mut pending_segment_events = 0;
    let mut execution = PrefillExecutionSummary {
        requests: config.prompts.len(),
        iterations: 0,
        output_tokens: 0,
        prefill_invocations: 0,
        tensor_prefill_invocations: 0,
        individual_prefill_invocations: 0,
        prefill_input_tokens: 0,
        prefill_padding_tokens: 0,
        prefill_iteration_elapsed: Duration::ZERO,
        total_elapsed: Duration::ZERO,
        peak_kv_cache_bytes: 0,
        peak_reserved_kv_bytes: 0,
        kernel: None,
    };
    let mut kernels = std::collections::BTreeSet::new();
    let prefill_bound = match chunk_size {
        Some(size) => config.prompts.iter().try_fold(0_usize, |sum, prompt| {
            sum.checked_add(prompt.len().div_ceil(size))
        }),
        None => Some(config.prompts.len()),
    }
    .ok_or(ExecutionBenchmarkError::IterationLimit)?;
    let iteration_limit = config
        .max_new_tokens
        .checked_mul(config.prompts.len())
        .and_then(|value| value.checked_add(prefill_bound))
        .and_then(|value| value.checked_add(config.prompts.len()))
        .and_then(|value| value.checked_add(2))
        .ok_or(ExecutionBenchmarkError::IterationLimit)?;
    while executor.has_active_request() || executor.state().queued_len() != 0 {
        if execution.iterations >= iteration_limit {
            return Err(ExecutionBenchmarkError::IterationLimit);
        }
        execution.iterations += 1;
        let iteration_started = Instant::now();
        let iteration = executor.step_iteration(
            u64::try_from(execution.iterations)
                .map(Timestamp::from_micros)
                .map_err(|_| ExecutionBenchmarkError::TimestampOverflow)?,
        )?;
        record_kernel_names(&iteration.execution_batches, &mut kernels);
        let iteration_elapsed = iteration_started.elapsed();
        let event_elapsed = started.elapsed();
        if iteration
            .execution_batches
            .iter()
            .any(|batch| batch.phase == BatchPhase::Prefill)
        {
            execution.prefill_iteration_elapsed += iteration_elapsed;
        }
        execution.peak_kv_cache_bytes = execution.peak_kv_cache_bytes.max(iteration.kv_cache_bytes);
        execution.peak_reserved_kv_bytes = execution
            .peak_reserved_kv_bytes
            .max(iteration.reserved_kv_bytes);
        for event in iteration.events {
            match event {
                ExecutorEvent::Token {
                    request_id,
                    phase,
                    finished,
                    kv_cache_bytes,
                    ..
                } => {
                    execution.output_tokens += 1;
                    execution.peak_kv_cache_bytes =
                        execution.peak_kv_cache_bytes.max(kv_cache_bytes);
                    let index = index_by_request[&request_id];
                    if phase == ola_runtime::ExecutionPhase::Prefill {
                        first_tokens[index].get_or_insert((execution.iterations, event_elapsed));
                    }
                    if finished {
                        completions[index] = Some((execution.iterations, event_elapsed));
                    }
                }
                ExecutorEvent::PrefillProgress { .. } => pending_segment_events += 1,
                ExecutorEvent::Failed { message, .. } => {
                    return Err(ExecutionBenchmarkError::RequestFailed(message));
                }
                ExecutorEvent::Deferred { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral("admission"));
                }
                ExecutorEvent::BatchDeferred { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral("batch policy"));
                }
                ExecutorEvent::Rejected { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral(
                        "admission rejection",
                    ));
                }
                ExecutorEvent::Preempted { .. } => {
                    return Err(ExecutionBenchmarkError::UnexpectedDeferral("preemption"));
                }
                ExecutorEvent::Idle => return Err(ExecutionBenchmarkError::UnexpectedIdle),
            }
        }
        for batch in iteration
            .execution_batches
            .iter()
            .filter(|batch| batch.phase == BatchPhase::Prefill)
        {
            execution.prefill_invocations += 1;
            execution.prefill_input_tokens += batch.input_tokens;
            execution.prefill_padding_tokens += batch.padding_tokens;
            match batch.mode {
                BatchExecutionMode::Tensor => execution.tensor_prefill_invocations += 1,
                BatchExecutionMode::Individual => execution.individual_prefill_invocations += 1,
            }
        }
    }
    execution.total_elapsed = started.elapsed();
    execution.kernel = kernel_identity(kernels);
    let requests = (0..request_ids.len())
        .map(|request_index| {
            Ok(PrefillFairnessRequest {
                request_index,
                first_token_iteration: first_tokens[request_index]
                    .ok_or(ExecutionBenchmarkError::MissingFirstToken(
                        request_ids[request_index],
                    ))?
                    .0,
                completion_iteration: completions[request_index]
                    .ok_or(ExecutionBenchmarkError::MissingCompletion(
                        request_ids[request_index],
                    ))?
                    .0,
                ttft: first_tokens[request_index]
                    .ok_or(ExecutionBenchmarkError::MissingFirstToken(
                        request_ids[request_index],
                    ))?
                    .1,
                completion_latency: completions[request_index]
                    .ok_or(ExecutionBenchmarkError::MissingCompletion(
                        request_ids[request_index],
                    ))?
                    .1,
            })
        })
        .collect::<Result<Vec<_>, ExecutionBenchmarkError>>()?;
    let outputs = request_ids
        .into_iter()
        .map(|request_id| {
            executor
                .result(request_id)
                .map(|result| result.token_ids.clone())
                .ok_or(ExecutionBenchmarkError::MissingResult(request_id))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        PrefillFairnessRun {
            chunk_size,
            iterations: execution.iterations,
            pending_segment_events,
            requests,
            execution,
        },
        outputs,
    ))
}

/// Failure produced while configuring or running an execution-backed probe.
#[derive(Debug, Eq, PartialEq)]
pub enum ExecutionBenchmarkError {
    /// The workload definition is invalid.
    Workload(WorkloadError),
    /// The runtime executor rejected configuration or execution.
    Executor(ExecutorError),
    /// A model session failed while processing a request.
    RequestFailed(String),
    /// Unconstrained benchmark work was unexpectedly deferred or rejected.
    UnexpectedDeferral(&'static str),
    /// The executor reported idle while benchmark work remained.
    UnexpectedIdle,
    /// A completed request did not retain its execution result.
    MissingResult(RequestId),
    /// Strategies produced different generated token sequences.
    OutputMismatch,
    /// A repeated benchmark must retain at least one paired trial.
    NoMeasuredTrials,
    /// A resumable prefill segment must contain at least one token.
    InvalidChunkSize(usize),
    /// A chunk-size sweep must contain at least one candidate size.
    NoChunkSizes,
    /// The loaded architecture only implements complete-prompt prefill.
    ChunkedPrefillUnsupported,
    /// A request never emitted its first generated token during a run.
    MissingFirstToken(RequestId),
    /// A request never reached its configured generation limit.
    MissingCompletion(RequestId),
    /// Measured timing/shape data cannot fit the calibration model.
    CalibrationUnavailable(&'static str),
    /// The logical iteration timestamp exceeded its representation.
    TimestampOverflow,
    /// The safety bound was exceeded or could not be represented.
    IterationLimit,
}

impl fmt::Display for ExecutionBenchmarkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workload(error) => error.fmt(formatter),
            Self::Executor(error) => error.fmt(formatter),
            Self::RequestFailed(message) => {
                write!(formatter, "benchmark request failed: {message}")
            }
            Self::UnexpectedDeferral(kind) => write!(formatter, "unexpected {kind} deferral"),
            Self::UnexpectedIdle => {
                formatter.write_str("executor became idle before benchmark completion")
            }
            Self::MissingResult(request_id) => write!(
                formatter,
                "completed request `{request_id}` has no retained result"
            ),
            Self::OutputMismatch => {
                formatter.write_str("prefill strategies produced different token sequences")
            }
            Self::NoMeasuredTrials => {
                formatter.write_str("prefill benchmark requires at least one measured trial")
            }
            Self::InvalidChunkSize(size) => {
                write!(formatter, "prefill chunk size must be positive, got {size}")
            }
            Self::NoChunkSizes => {
                formatter.write_str("prefill fairness sweep requires a chunk size")
            }
            Self::ChunkedPrefillUnsupported => {
                formatter.write_str("loaded model does not support resumable prefill")
            }
            Self::MissingFirstToken(request_id) => write!(
                formatter,
                "request `{request_id}` did not emit its first token in the fairness run"
            ),
            Self::MissingCompletion(request_id) => write!(
                formatter,
                "request `{request_id}` did not complete in the fairness run"
            ),
            Self::CalibrationUnavailable(reason) => {
                write!(formatter, "prefill cost calibration unavailable: {reason}")
            }
            Self::TimestampOverflow => {
                formatter.write_str("benchmark iteration timestamp overflowed")
            }
            Self::IterationLimit => {
                formatter.write_str("benchmark exceeded its executor iteration safety bound")
            }
        }
    }
}

impl Error for ExecutionBenchmarkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Workload(error) => Some(error),
            Self::Executor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WorkloadError> for ExecutionBenchmarkError {
    fn from(value: WorkloadError) -> Self {
        Self::Workload(value)
    }
}

impl From<ExecutorError> for ExecutionBenchmarkError {
    fn from(value: ExecutorError) -> Self {
        Self::Executor(value)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
/// Nearest-rank latency distribution summary in microseconds.
pub struct LatencySummary {
    /// Number of observed durations.
    pub samples: usize,
    /// Minimum duration.
    pub min_micros: u64,
    /// 50th percentile duration.
    pub p50_micros: u64,
    /// 95th percentile duration.
    pub p95_micros: u64,
    /// 99th percentile duration.
    pub p99_micros: u64,
    /// Maximum duration.
    pub max_micros: u64,
}

/// Summarizes durations using sorted nearest-rank percentiles.
///
/// An empty slice returns an all-zero summary with zero samples. Durations that
/// exceed `u64` microseconds saturate to `u64::MAX`.
pub fn summarize_latencies(samples: &[Duration]) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary::default();
    }
    let mut micros = samples
        .iter()
        .map(|sample| u64::try_from(sample.as_micros()).unwrap_or(u64::MAX))
        .collect::<Vec<_>>();
    micros.sort_unstable();

    LatencySummary {
        samples: micros.len(),
        min_micros: micros[0],
        p50_micros: percentile(&micros, 50),
        p95_micros: percentile(&micros, 95),
        p99_micros: percentile(&micros, 99),
        max_micros: micros[micros.len() - 1],
    }
}

fn summarize_fairness_latency(runs: &[PrefillFairnessRun]) -> Vec<PrefillFairnessLatency> {
    let Some(first) = runs.first() else {
        return Vec::new();
    };
    first
        .requests
        .iter()
        .map(|request| {
            let samples = runs
                .iter()
                .filter_map(|run| run.requests.get(request.request_index));
            let mut ttft = Vec::with_capacity(runs.len());
            let mut completion = Vec::with_capacity(runs.len());
            for sample in samples {
                ttft.push(sample.ttft);
                completion.push(sample.completion_latency);
            }
            PrefillFairnessLatency {
                request_index: request.request_index,
                ttft: summarize_latencies(&ttft),
                completion: summarize_latencies(&completion),
            }
        })
        .collect()
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Invalid workload definition.
pub enum WorkloadError {
    /// Generator was configured with zero arrivals.
    NoRequests,
    /// Generator was configured with a zero output-token limit.
    NoOutputTokens,
    /// Explicit trace arrival times decrease.
    EventsOutOfOrder,
    /// Arrival time cannot be represented by `Timestamp`.
    TimestampOverflow,
    /// A prefill-shape trace must include at least one prompt.
    NoPromptLengths,
    /// Model prefill does not accept a zero-token prompt.
    EmptyPrompt,
    /// Prompt shape arithmetic exceeded the platform token-count range.
    TokenCountOverflow,
}

impl fmt::Display for WorkloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NoRequests => "workload must contain at least one request",
            Self::NoOutputTokens => "workload requests must generate at least one token",
            Self::EventsOutOfOrder => "workload events must be ordered by arrival time",
            Self::TimestampOverflow => "workload timestamps exceed the supported range",
            Self::NoPromptLengths => "prefill workload must contain at least one prompt",
            Self::EmptyPrompt => "prefill workload prompts must contain at least one token",
            Self::TokenCountOverflow => "prefill workload token counts overflowed",
        };
        formatter.write_str(message)
    }
}

impl Error for WorkloadError {}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use ola_core::ModelDescriptor;
    use ola_model::{
        BatchCapabilities, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
        ModelExecutionError, PrefillOutcome, SessionMetrics, Tokenizer,
    };

    use super::*;

    #[test]
    fn output_check_lengths_tolerates_flipped_tokens_but_not_different_work() {
        let expected = vec![vec![7, 9], vec![3]];
        let flipped = vec![vec![7, 8], vec![4]];
        let shorter = vec![vec![7], vec![3]];

        assert!(OutputCheck::Exact.accepts(&expected, &expected));
        assert!(!OutputCheck::Exact.accepts(&expected, &flipped));
        assert!(OutputCheck::Lengths.accepts(&expected, &flipped));
        assert!(!OutputCheck::Lengths.accepts(&expected, &shorter));
        assert!(!OutputCheck::Lengths.accepts(&expected, &expected[..1]));
    }

    #[test]
    fn fixed_workload_replays_deterministically() {
        let workload = FixedWorkload {
            model: "tiny".into(),
            requests: 3,
            prompt_tokens: 4,
            output_tokens: 2,
            interarrival: Duration::from_micros(10),
        }
        .build()
        .unwrap();
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "tiny-decoder"))
            .unwrap();

        let ids = workload.replay(&mut runtime).unwrap();

        assert_eq!(
            ids.iter().map(|id| id.get()).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(workload.events()[2].at, Timestamp::from_micros(20));
    }

    #[test]
    fn summarizes_tail_latency_with_nearest_rank_percentiles() {
        let samples = [1, 2, 3, 4, 100].map(Duration::from_micros);

        let summary = summarize_latencies(&samples);

        assert_eq!(summary.p50_micros, 3);
        assert_eq!(summary.p95_micros, 100);
        assert_eq!(summary.p99_micros, 100);
    }

    #[test]
    fn compares_prefill_padding_without_claiming_elapsed_performance() {
        let comparison = compare_prefill_batching(&[5, 9, 7]).unwrap();

        assert_eq!(
            comparison.unbucketed,
            PrefillBatchSummary {
                requests: 3,
                invocations: 1,
                tensor_invocations: 1,
                individual_invocations: 0,
                input_tokens: 21,
                padding_tokens: 6,
            }
        );
        assert_eq!(
            comparison.power_of_two,
            PrefillBatchSummary {
                requests: 3,
                invocations: 2,
                tensor_invocations: 1,
                individual_invocations: 1,
                input_tokens: 21,
                padding_tokens: 2,
            }
        );
        assert_eq!(comparison.padding_tokens_saved(), 4);
        assert_eq!(comparison.padding_reduction(), Some(2.0 / 3.0));
        assert_eq!(
            comparison.power_of_two.input_token_efficiency(),
            Some(21.0 / 23.0)
        );
    }

    #[test]
    fn rejects_empty_or_zero_length_prefill_traces() {
        assert_eq!(
            compare_prefill_batching(&[]),
            Err(WorkloadError::NoPromptLengths)
        );
        assert_eq!(
            compare_prefill_batching(&[4, 0]),
            Err(WorkloadError::EmptyPrompt)
        );
    }

    #[test]
    fn analyzes_the_cost_aware_policy_without_elapsed_time_claims() {
        let diverse = analyze_cost_aware_prefill(&[5, 7, 9, 63], 32).unwrap();
        let clustered = analyze_cost_aware_prefill(&[5, 6, 7, 8], 32).unwrap();

        assert_eq!(
            diverse.decision.selected,
            ola_runtime::PrefillGroupingStrategy::PowerOfTwo
        );
        assert_eq!(diverse.decision.unbucketed.estimated_cost, 284);
        assert_eq!(diverse.decision.power_of_two.estimated_cost, 182);
        assert_eq!(diverse.selected.invocations, 3);
        assert_eq!(diverse.selected.padding_tokens, 2);
        assert_eq!(
            clustered.decision.selected,
            ola_runtime::PrefillGroupingStrategy::Unbucketed
        );
        assert_eq!(clustered.selected.invocations, 1);
        assert_eq!(clustered.selected.padding_tokens, 6);
    }

    #[test]
    fn execution_probe_runs_both_strategies_and_preserves_outputs() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let comparison = benchmark_prefill_execution(
            &model,
            &PrefillExecutionConfig {
                prompts: vec![vec![1, 2], vec![1, 2, 3], vec![1, 2, 3, 4, 5], vec![1; 7]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
        )
        .unwrap();

        assert_eq!(comparison.unbucketed.requests, 4);
        assert_eq!(comparison.unbucketed.output_tokens, 8);
        assert_eq!(comparison.unbucketed.prefill_invocations, 1);
        assert_eq!(comparison.unbucketed.tensor_prefill_invocations, 1);
        assert_eq!(comparison.unbucketed.prefill_input_tokens, 17);
        assert_eq!(comparison.unbucketed.prefill_padding_tokens, 11);
        assert_eq!(comparison.power_of_two.prefill_invocations, 3);
        assert_eq!(comparison.power_of_two.tensor_prefill_invocations, 1);
        assert_eq!(comparison.power_of_two.individual_prefill_invocations, 2);
        assert_eq!(comparison.power_of_two.prefill_input_tokens, 17);
        assert_eq!(comparison.power_of_two.prefill_padding_tokens, 2);
        assert!(comparison.unbucketed.peak_kv_cache_bytes > 0);
        assert_eq!(
            comparison.power_of_two.prefill_token_efficiency(),
            Some(17.0 / 19.0)
        );
    }

    #[test]
    fn chunked_execution_probe_preserves_outputs_and_records_segments() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let comparison = benchmark_chunked_prefill_execution(
            &model,
            &PrefillExecutionConfig {
                prompts: vec![vec![1, 2, 3, 4, 5], vec![5, 6, 7, 8]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
            2,
        )
        .unwrap();

        assert_eq!(comparison.chunk_size, 2);
        assert_eq!(comparison.unchunked.prefill_invocations, 1);
        assert_eq!(comparison.chunked.prefill_invocations, 3);
        assert_eq!(comparison.chunked.tensor_prefill_invocations, 2);
        assert_eq!(comparison.chunked.individual_prefill_invocations, 1);
        assert_eq!(comparison.chunked.prefill_input_tokens, 9);
        assert_eq!(comparison.chunked.prefill_padding_tokens, 0);
        assert_eq!(comparison.unchunked.output_tokens, 4);
        assert_eq!(comparison.chunked.output_tokens, 4);
        assert!(comparison.chunked.iterations > comparison.unchunked.iterations);
    }

    #[test]
    fn fairness_sweep_tracks_first_token_and_completion_by_request() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let comparison = benchmark_prefill_fairness(
            &model,
            &PrefillExecutionConfig {
                prompts: vec![vec![1], vec![1; 9], vec![1; 2]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
            2,
            &[2, 4],
        )
        .unwrap();

        assert_eq!(comparison.chunked.len(), 2);
        assert_eq!(comparison.chunked[0].chunk_size, Some(2));
        assert_eq!(comparison.chunked[1].chunk_size, Some(4));
        assert_eq!(comparison.unchunked.requests.len(), 3);
        assert!(comparison.unchunked.pending_segment_events == 0);
        assert!(comparison.chunked[0].pending_segment_events > 0);
        assert!(comparison.chunked[0].execution.tensor_prefill_invocations > 0);
        assert_eq!(
            comparison.chunked[0].requests[0].first_token_iteration,
            comparison.unchunked.requests[0].first_token_iteration
        );
        assert!(comparison.chunked[0].iterations > comparison.unchunked.iterations);
        assert!(
            comparison.chunked[0]
                .requests
                .iter()
                .all(|request| { request.first_token_iteration < request.completion_iteration })
        );
        assert!(
            comparison
                .unchunked
                .requests
                .iter()
                .all(|request| { request.ttft <= request.completion_latency })
        );

        let trials = benchmark_prefill_fairness_trials(
            &model,
            &PrefillExecutionConfig {
                prompts: vec![vec![1], vec![1; 9], vec![1; 2]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
            2,
            &[2, 4],
            1,
            2,
        )
        .unwrap();
        assert_eq!(trials.unchunked.len(), 2);
        assert_eq!(trials.chunked.len(), 2);
        assert_eq!(trials.strategy_order_by_trial.len(), 2);
        assert_eq!(trials.unchunked_latency()[0].ttft.samples, 2);
        assert_eq!(trials.chunked_latency()[1][2].completion.samples, 2);
    }

    #[test]
    fn execution_probe_validates_workload_before_running_a_model() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let error = benchmark_prefill_execution(
            &model,
            &PrefillExecutionConfig {
                prompts: vec![Vec::new()],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            ExecutionBenchmarkError::Workload(WorkloadError::EmptyPrompt)
        );
    }

    #[test]
    fn repeated_probe_discards_warmups_and_alternates_measured_order() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let comparison = benchmark_prefill_trials(
            &model,
            &PrefillTrialConfig {
                execution: PrefillExecutionConfig {
                    prompts: vec![vec![1, 2], vec![1, 2, 3], vec![1; 5], vec![1; 7]],
                    max_new_tokens: 2,
                    output_check: OutputCheck::Exact,
                },
                warmup_trials: 1,
                measured_trials: 4,
            },
        )
        .unwrap();

        assert_eq!(comparison.warmup_trials, 1);
        assert_eq!(
            comparison.first_strategy_by_trial,
            vec![
                PrefillStrategy::Unbucketed,
                PrefillStrategy::PowerOfTwo,
                PrefillStrategy::Unbucketed,
                PrefillStrategy::PowerOfTwo,
            ]
        );
        assert_eq!(comparison.unbucketed.samples.len(), 4);
        assert_eq!(comparison.power_of_two.samples.len(), 4);
        assert_eq!(comparison.unbucketed.total_latency.samples, 4);
        assert_eq!(comparison.power_of_two.prefill_iteration_latency.samples, 4);
        assert!(
            comparison
                .unbucketed
                .samples
                .iter()
                .all(|sample| sample.prefill_padding_tokens == 11)
        );
        assert!(
            comparison
                .power_of_two
                .samples
                .iter()
                .all(|sample| sample.prefill_padding_tokens == 2)
        );
        assert!(comparison.unbucketed.output_tokens_per_second().is_some());
        assert!(comparison.unbucketed.peak_kv_cache_bytes() > 0);
    }

    #[test]
    fn repeated_probe_rejects_an_empty_measurement_set() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let error = benchmark_prefill_trials(
            &model,
            &PrefillTrialConfig {
                execution: PrefillExecutionConfig {
                    prompts: vec![vec![1]],
                    max_new_tokens: 1,
                    output_check: OutputCheck::Exact,
                },
                warmup_trials: 1,
                measured_trials: 0,
            },
        )
        .unwrap_err();

        assert_eq!(error, ExecutionBenchmarkError::NoMeasuredTrials);
    }

    #[test]
    fn calibrates_token_equivalent_invocation_overhead_from_p50_times() {
        let comparison = PrefillTrialComparison {
            warmup_trials: 0,
            first_strategy_by_trial: vec![PrefillStrategy::Unbucketed],
            unbucketed: synthetic_trials(28, 1, 380),
            power_of_two: synthetic_trials(19, 3, 490),
        };

        let calibration = calibrate_prefill_invocation_overhead(&comparison).unwrap();

        assert_eq!(calibration.invocation_overhead_tokens, 10);
        assert_eq!(calibration.unbucketed_dense_tokens, 28);
        assert_eq!(calibration.power_of_two_dense_tokens, 19);
        assert_eq!(calibration.unbucketed_p50_micros, 380);
        assert_eq!(calibration.power_of_two_p50_micros, 490);
    }

    #[test]
    fn three_way_probe_rotates_order_and_measures_the_adaptive_strategy() {
        let model = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let comparison = benchmark_three_way_prefill_trials(
            &model,
            &PrefillThreeWayTrialConfig {
                execution: PrefillExecutionConfig {
                    prompts: vec![vec![1, 2], vec![1, 2, 3], vec![1; 5], vec![1; 7]],
                    max_new_tokens: 2,
                    output_check: OutputCheck::Exact,
                },
                warmup_trials: 1,
                measured_trials: 3,
                adaptive_invocation_overhead_tokens: 4,
            },
        )
        .unwrap();

        assert_eq!(
            comparison.strategy_order_by_trial,
            vec![
                [
                    PrefillStrategy::Unbucketed,
                    PrefillStrategy::PowerOfTwo,
                    PrefillStrategy::CostAware,
                ],
                [
                    PrefillStrategy::PowerOfTwo,
                    PrefillStrategy::CostAware,
                    PrefillStrategy::Unbucketed,
                ],
                [
                    PrefillStrategy::CostAware,
                    PrefillStrategy::Unbucketed,
                    PrefillStrategy::PowerOfTwo,
                ],
            ]
        );
        assert_eq!(comparison.cost_aware.samples.len(), 3);
        assert!(
            comparison
                .cost_aware
                .samples
                .iter()
                .all(|sample| sample.prefill_invocations == 3
                    && sample.prefill_padding_tokens == 2)
        );
    }

    fn synthetic_trials(
        dense_tokens: usize,
        invocations: usize,
        prefill_micros: u64,
    ) -> PrefillStrategyTrials {
        let input_tokens = 17;
        let sample = PrefillExecutionSummary {
            requests: 4,
            iterations: 2,
            output_tokens: 8,
            prefill_invocations: invocations,
            tensor_prefill_invocations: 1,
            individual_prefill_invocations: invocations.saturating_sub(1),
            prefill_input_tokens: input_tokens,
            prefill_padding_tokens: dense_tokens - input_tokens,
            prefill_iteration_elapsed: Duration::from_micros(prefill_micros),
            total_elapsed: Duration::from_micros(prefill_micros + 100),
            peak_kv_cache_bytes: 128,
            peak_reserved_kv_bytes: 128,
            kernel: None,
        };
        PrefillStrategyTrials {
            samples: vec![sample],
            prefill_iteration_latency: summarize_latencies(&[Duration::from_micros(
                prefill_micros,
            )]),
            total_latency: summarize_latencies(&[Duration::from_micros(prefill_micros + 100)]),
        }
    }

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
                    .map(|(session, input)| session.decode(input))
                    .collect(),
            )
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
                is_end_of_sequence: false,
            })
        }

        fn metrics(&self) -> SessionMetrics {
            self.metrics
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }
}
