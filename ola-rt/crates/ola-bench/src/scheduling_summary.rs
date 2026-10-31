//! Scheduling metrics and the cross-policy comparison protocol.
//!
//! See `SCHEDULING_BASELINES.md` in `ola-runtime`. All times are virtual
//! microseconds from [`run_scheduling_trial`], so summaries are exactly
//! reproducible.
//!
//! Definitions:
//!
//! - **queue wait**: start − arrival; for requests that never started, the
//!   trial's end − arrival.
//! - **isolated time**: a request's end-to-end time when it runs alone, under
//!   the same configuration and the default policy with memory-aware
//!   admission. It is independent of the policy being compared.
//! - **slowdown**: end-to-end time ÷ isolated time, for completed requests.
//! - **fairness**: Jain's index over slowdowns, `(Σx)² / (n·Σx²)`; 1.0 means
//!   every request was slowed equally.
//! - **starved**: a request that was not rejected or failed and whose queue
//!   wait exceeds `starvation_factor ×` its isolated time.
//! - **deadline missed**: a request with a deadline that did not complete by
//!   it, including requests that never completed.

use std::{error::Error, fmt, time::Duration};

use ola_core::Timestamp;

use ola_runtime::{
    AdaptiveIntegerDomain, AdaptiveObjectiveMetrics, AdaptiveObjectiveWeights,
    AdaptiveOptimizationError, ComposedPolicy, ExecutionPolicy, ExecutorIteration,
    FiniteCandidateOptimizer, MemoryAwareAdmission, PolicyError, RuntimeState,
};

use crate::{
    LatencySummary, SchedulingRun, SchedulingTrace, SimulationConfig, SimulationError,
    TraceParameters, TrialOutcome, run_observed_trial, run_scheduling_trial, summarize_latencies,
};

/// Default multiple of isolated time a queue wait may reach before the
/// request counts as starved.
pub const DEFAULT_STARVATION_FACTOR: u64 = 10;

/// Prompts at least this long belong to the `long` size class.
pub const LONG_PROMPT_TOKENS: usize = 64;

/// Error returned when a virtual-time adaptive candidate cannot be evaluated.
#[derive(Debug, Eq, PartialEq)]
pub enum VirtualTimeOptimizationError {
    /// Candidate policy construction failed.
    Policy(String),
    /// The virtual-time trial failed.
    Simulation(SimulationError),
    /// The optimizer found no feasible candidate or rejected its model.
    Optimization(AdaptiveOptimizationError),
}

impl fmt::Display for VirtualTimeOptimizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(error) => write!(formatter, "candidate policy failed: {error}"),
            Self::Simulation(error) => error.fmt(formatter),
            Self::Optimization(error) => error.fmt(formatter),
        }
    }
}

impl Error for VirtualTimeOptimizationError {}

/// Result of selecting one policy configuration by replaying virtual time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualTimeCandidateEvaluation {
    /// Selected adaptive parameter vector.
    pub theta: Vec<(&'static str, String)>,
    /// Objective score assigned to the selected replay.
    pub objective: i64,
    /// Full virtual-time replay for the selected candidate.
    pub run: SchedulingRun,
}

/// Paired objective scores for one held-out seeded trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualTimeHeldOutScore {
    /// Seed identifying the held-out trace.
    pub seed: u64,
    /// Objective score for the selected adaptive θ.
    pub adaptive_objective: i64,
    /// Objective score for the fixed baseline.
    pub baseline_objective: i64,
}

/// Train-on-one-trace, evaluate-on-held-out-traces result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VirtualTimeHeldOutStudy {
    /// θ selected using only the training trace.
    pub theta: Vec<(&'static str, String)>,
    /// Training objective used for selection.
    pub training_objective: i64,
    /// Paired held-out scores, in caller-provided trace order.
    pub held_out: Vec<VirtualTimeHeldOutScore>,
}

/// Evaluates bounded integer candidates by replaying each one in virtual time.
///
/// The finite optimizer supplies deterministic candidate ordering; this
/// function supplies the expensive objective evaluation from actual simulator
/// outcomes. It is intentionally benchmark-side and does not run during
/// production `plan` calls.
pub fn optimize_virtual_time_candidate<F>(
    trace: &SchedulingTrace,
    config: &SimulationConfig,
    domains: Vec<AdaptiveIntegerDomain>,
    current: Vec<(&'static str, String)>,
    factory: F,
    weights: AdaptiveObjectiveWeights,
) -> Result<VirtualTimeCandidateEvaluation, VirtualTimeOptimizationError>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
{
    let factory_ref = &factory;
    let mut optimizer = FiniteCandidateOptimizer::new(
        domains,
        move |candidate: &[(&'static str, String)],
              _: &[ola_runtime::AdaptiveRequestObservation],
              _: ola_runtime::AdaptiveConstraints|
              -> Result<Option<i64>, AdaptiveOptimizationError> {
            let policy = match (factory_ref)(candidate) {
                Ok(policy) => policy,
                Err(_) => return Ok(None),
            };
            let run = run_scheduling_trial(trace, policy, config)
                .map_err(|error| AdaptiveOptimizationError::Infeasible(error.to_string()))?;
            let observations = run_observations(&run);
            Ok(AdaptiveObjectiveMetrics::from_observations(&observations).score(weights))
        },
    )
    .map_err(VirtualTimeOptimizationError::Optimization)?;
    let theta = optimizer
        .propose(&current, &[], ola_runtime::AdaptiveConstraints::default())
        .map_err(VirtualTimeOptimizationError::Optimization)?;
    let policy =
        factory(&theta).map_err(|error| VirtualTimeOptimizationError::Policy(error.to_string()))?;
    let run = run_scheduling_trial(trace, policy, config)
        .map_err(VirtualTimeOptimizationError::Simulation)?;
    let observations = run_observations(&run);
    let objective = AdaptiveObjectiveMetrics::from_observations(&observations)
        .score(weights)
        .ok_or(VirtualTimeOptimizationError::Optimization(
            AdaptiveOptimizationError::NoCandidate,
        ))?;
    Ok(VirtualTimeCandidateEvaluation {
        theta,
        objective,
        run,
    })
}

/// Selects θ on one trace and evaluates that fixed θ against a baseline on
/// independent held-out traces. No held-out result influences selection.
pub fn evaluate_virtual_time_held_out<F, G>(
    training_trace: &SchedulingTrace,
    held_out_traces: &[SchedulingTrace],
    config: &SimulationConfig,
    domains: Vec<AdaptiveIntegerDomain>,
    current: Vec<(&'static str, String)>,
    adaptive_factory: F,
    baseline_factory: G,
    weights: AdaptiveObjectiveWeights,
) -> Result<VirtualTimeHeldOutStudy, VirtualTimeOptimizationError>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
    G: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
{
    let selected = optimize_virtual_time_candidate(
        training_trace,
        config,
        domains,
        current,
        &adaptive_factory,
        weights,
    )?;
    let held_out = held_out_traces
        .iter()
        .map(|trace| {
            let adaptive = adaptive_factory(&selected.theta)
                .map_err(|error| VirtualTimeOptimizationError::Policy(error.to_string()))?;
            let baseline = baseline_factory(&[])
                .map_err(|error| VirtualTimeOptimizationError::Policy(error.to_string()))?;
            let adaptive_run = run_scheduling_trial(trace, adaptive, config)
                .map_err(VirtualTimeOptimizationError::Simulation)?;
            let baseline_run = run_scheduling_trial(trace, baseline, config)
                .map_err(VirtualTimeOptimizationError::Simulation)?;
            let adaptive_objective =
                AdaptiveObjectiveMetrics::from_observations(&run_observations(&adaptive_run))
                    .score(weights)
                    .ok_or(VirtualTimeOptimizationError::Optimization(
                        AdaptiveOptimizationError::NoCandidate,
                    ))?;
            let baseline_objective =
                AdaptiveObjectiveMetrics::from_observations(&run_observations(&baseline_run))
                    .score(weights)
                    .ok_or(VirtualTimeOptimizationError::Optimization(
                        AdaptiveOptimizationError::NoCandidate,
                    ))?;
            Ok(VirtualTimeHeldOutScore {
                seed: trace.parameters.seed,
                adaptive_objective,
                baseline_objective,
            })
        })
        .collect::<Result<Vec<_>, VirtualTimeOptimizationError>>()?;
    Ok(VirtualTimeHeldOutStudy {
        theta: selected.theta,
        training_objective: selected.objective,
        held_out,
    })
}

pub(crate) fn run_observations(
    run: &SchedulingRun,
) -> Vec<ola_runtime::AdaptiveRequestObservation> {
    run.requests
        .iter()
        .map(|request| {
            let arrived_at = Timestamp::from_micros(request.arrival_micros);
            let first_token_at = request.first_token_micros.map(Timestamp::from_micros);
            let completed_at = request.completed_micros.map(Timestamp::from_micros);
            ola_runtime::AdaptiveRequestObservation {
                request_id: request.request_id,
                arrived_at,
                first_token_at,
                completed_at,
                time_to_first_token_micros: first_token_at
                    .map(|at| at.as_micros().saturating_sub(arrived_at.as_micros())),
                completion_latency_micros: completed_at
                    .map(|at| at.as_micros().saturating_sub(arrived_at.as_micros())),
                deadline_missed: request.deadline_micros.is_some_and(|deadline| {
                    completed_at.is_none_or(|at| at.as_micros() > deadline)
                }),
                capability_fallback: false,
                prompt_tokens: None,
                max_new_tokens: None,
                generated_tokens: None,
            }
        })
        .collect()
}

/// Distribution of slowdowns and their fairness.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlowdownSummary {
    /// Requests with a slowdown (completed with a known isolated time).
    pub samples: usize,
    /// Median slowdown.
    pub p50: f64,
    /// 95th-percentile slowdown.
    pub p95: f64,
    /// Worst slowdown.
    pub max: f64,
    /// Mean slowdown.
    pub mean: f64,
    /// Jain's fairness index over slowdowns; 1.0 when there are no samples.
    pub jain_fairness: f64,
}

/// Metrics for one subset of a trace's requests.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassSummary {
    /// Class label, such as `priority=5` or `size=long`.
    pub class: String,
    /// Requests in the class.
    pub requests: usize,
    /// Completed requests.
    pub completed: usize,
    /// End-to-end latency of completed requests.
    pub completion: LatencySummary,
    /// Slowdowns of completed requests.
    pub slowdown: SlowdownSummary,
    /// Starved requests.
    pub starved: usize,
    /// Missed deadlines.
    pub deadline_missed: usize,
}

/// Everything the protocol reports for one policy on one trace.
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulingSummary {
    /// Stable policy descriptor.
    pub policy: String,
    /// Trace generator name.
    pub trace: &'static str,
    /// Trace seed.
    pub seed: u64,
    /// Requests in the trace.
    pub requests: usize,
    /// Requests completed.
    pub completed: usize,
    /// Requests rejected as impossible.
    pub rejected: usize,
    /// Requests that failed.
    pub failed: usize,
    /// Requests still queued or running when the trial stopped.
    pub unfinished: usize,
    /// Queue wait of requests that started.
    pub queue: LatencySummary,
    /// Time to first token of requests that produced one.
    pub time_to_first_token: LatencySummary,
    /// End-to-end latency of completed requests.
    pub completion: LatencySummary,
    /// Prompt and decode input positions the model computed.
    pub input_tokens: usize,
    /// Tokens generated by completed requests.
    pub output_tokens: usize,
    /// Virtual time when the trial stopped.
    pub makespan_micros: u64,
    /// Iterations executed.
    pub iterations: usize,
    /// Highest physical KV use after any iteration.
    pub peak_kv_cache_bytes: u64,
    /// Highest KV reservation after any iteration.
    pub peak_reserved_kv_bytes: u64,
    /// Sum over iterations of KV bytes live sessions occupy.
    pub kv_used_byte_iterations: u64,
    /// Sum over iterations of whole-page KV bytes held for them.
    pub kv_allocated_byte_iterations: u64,
    /// Prompt tokens reused from the prefix cache during the trial.
    pub prefix_hit_tokens: u64,
    /// Admission deferrals across iterations.
    pub admission_deferrals: usize,
    /// Token-budget deferrals across iterations.
    pub budget_deferrals: usize,
    /// Slot deferrals across iterations.
    pub slot_deferrals: usize,
    /// Backfill skips across iterations.
    pub backfill_skips: usize,
    /// Live requests returned to the queue to free KV blocks.
    pub preemptions: usize,
    /// Capability fallbacks across iterations.
    pub capability_fallbacks: usize,
    /// Slowdowns and fairness.
    pub slowdown: SlowdownSummary,
    /// Starved requests.
    pub starved: usize,
    /// Longest queue wait.
    pub worst_queue_wait_micros: u64,
    /// Requests carrying a deadline.
    pub with_deadline: usize,
    /// Requests that missed their deadline.
    pub deadline_missed: usize,
    /// Per-priority classes (highest first), then `size=short` and
    /// `size=long`; classes with no requests are omitted.
    pub classes: Vec<ClassSummary>,
}

impl SchedulingSummary {
    /// Share of held KV page bytes that live sessions did not occupy,
    /// averaged over iterations: internal fragmentation from partly filled
    /// last pages. Meaningful only without prefix sharing, where a shared
    /// page counts once in pages but once per session in occupancy.
    pub fn kv_internal_fragmentation(&self) -> f64 {
        if self.kv_allocated_byte_iterations == 0 {
            return 0.0;
        }
        1.0 - self.kv_used_byte_iterations as f64 / self.kv_allocated_byte_iterations as f64
    }

    /// Returns generated tokens per virtual second, if any time elapsed.
    pub fn output_tokens_per_second(&self) -> Option<f64> {
        (self.makespan_micros > 0)
            .then(|| to_f64(self.output_tokens) * 1_000_000.0 / to_f64(self.makespan_micros))
    }
}

/// One trace replayed under several policies with a shared baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulingComparison {
    /// Trace generator name.
    pub trace: &'static str,
    /// Trace parameters, including the seed.
    pub parameters: TraceParameters,
    /// Configuration shared by every run.
    pub config: SimulationConfig,
    /// Starvation threshold used.
    pub starvation_factor: u64,
    /// Isolated end-to-end time per trace request; `None` if it cannot
    /// complete alone.
    pub isolated_micros: Vec<Option<u64>>,
    /// One summary per policy, in the order given.
    pub summaries: Vec<SchedulingSummary>,
}

/// Measures each request's end-to-end time when run alone.
///
/// # Errors
///
/// Returns [`SimulationError`] if a single-request trial fails to run.
pub fn isolated_end_to_end(
    trace: &SchedulingTrace,
    config: &SimulationConfig,
) -> Result<Vec<Option<u64>>, SimulationError> {
    trace
        .requests
        .iter()
        .map(|request| {
            let alone = SchedulingTrace {
                name: trace.name,
                parameters: trace.parameters,
                requests: vec![crate::TraceRequest {
                    arrival_micros: 0,
                    ..request.clone()
                }],
            };
            let run = run_scheduling_trial(
                &alone,
                ComposedPolicy::default().with_admission(MemoryAwareAdmission),
                config,
            )?;
            let result = &run.requests[0];
            Ok((result.outcome == TrialOutcome::Completed)
                .then_some(result.completed_micros)
                .flatten())
        })
        .collect()
}

/// Replays `trace` under each policy and summarizes every run against the
/// same isolated baseline.
///
/// # Errors
///
/// Returns [`SimulationError`] if any trial fails to run.
pub fn compare_schedulers(
    trace: &SchedulingTrace,
    config: &SimulationConfig,
    policies: Vec<Box<dyn ExecutionPolicy>>,
    starvation_factor: u64,
) -> Result<SchedulingComparison, SimulationError> {
    compare_schedulers_observed(
        trace,
        config,
        policies,
        starvation_factor,
        &mut |_, _, _| {},
    )
}

/// Like [`compare_schedulers`], calling `observe` with each policy's index,
/// every iteration of its trial, and the state after it. The isolated
/// single-request baselines are not observed.
///
/// # Errors
///
/// Returns [`SimulationError`] if any trial fails to run.
pub fn compare_schedulers_observed(
    trace: &SchedulingTrace,
    config: &SimulationConfig,
    policies: Vec<Box<dyn ExecutionPolicy>>,
    starvation_factor: u64,
    observe: &mut dyn FnMut(usize, &ExecutorIteration, &RuntimeState),
) -> Result<SchedulingComparison, SimulationError> {
    let isolated_micros = isolated_end_to_end(trace, config)?;
    let summaries = policies
        .into_iter()
        .enumerate()
        .map(|(index, policy)| {
            let run = run_observed_trial(trace, policy, config, &[], &mut |iteration, state| {
                observe(index, iteration, state);
            })?;
            Ok(summarize_run(&run, &isolated_micros, starvation_factor))
        })
        .collect::<Result<Vec<_>, SimulationError>>()?;
    Ok(SchedulingComparison {
        trace: trace.name,
        parameters: trace.parameters,
        config: *config,
        starvation_factor,
        isolated_micros,
        summaries,
    })
}

/// Summarizes one run. `isolated_micros` is indexed by trace position.
pub fn summarize_run(
    run: &SchedulingRun,
    isolated_micros: &[Option<u64>],
    starvation_factor: u64,
) -> SchedulingSummary {
    let all = run.requests.iter().collect::<Vec<_>>();
    let class = |requests: Vec<&crate::TrialRequest>, label: String| {
        let metrics = RequestMetrics::new(&requests, run, isolated_micros, starvation_factor);
        ClassSummary {
            class: label,
            requests: requests.len(),
            completed: metrics.completed,
            completion: metrics.completion,
            slowdown: metrics.slowdown,
            starved: metrics.starved,
            deadline_missed: metrics.deadline_missed,
        }
    };

    let mut priorities = run
        .requests
        .iter()
        .map(|request| request.priority)
        .collect::<Vec<_>>();
    priorities.sort_unstable_by(|left, right| right.cmp(left));
    priorities.dedup();
    let mut classes = priorities
        .into_iter()
        .map(|priority| {
            class(
                all.iter()
                    .copied()
                    .filter(|request| request.priority == priority)
                    .collect(),
                format!("priority={priority}"),
            )
        })
        .collect::<Vec<_>>();
    for (label, long) in [("size=short", false), ("size=long", true)] {
        let members = all
            .iter()
            .copied()
            .filter(|request| (request.prompt_tokens >= LONG_PROMPT_TOKENS) == long)
            .collect::<Vec<_>>();
        if !members.is_empty() {
            classes.push(class(members, label.to_owned()));
        }
    }

    let metrics = RequestMetrics::new(&all, run, isolated_micros, starvation_factor);
    let count = |outcome| {
        run.requests
            .iter()
            .filter(|request| request.outcome == outcome)
            .count()
    };
    let started = run
        .requests
        .iter()
        .filter_map(|request| {
            request
                .started_micros
                .map(|started| started - request.arrival_micros)
        })
        .collect::<Vec<_>>();
    let first_tokens = run
        .requests
        .iter()
        .filter_map(|request| {
            request
                .first_token_micros
                .map(|first| first - request.arrival_micros)
        })
        .collect::<Vec<_>>();
    let total = |field: fn(&crate::TrialIteration) -> usize| -> usize {
        run.iterations.iter().map(field).sum()
    };

    SchedulingSummary {
        policy: run.policy.to_string(),
        trace: run.trace,
        seed: run.seed,
        requests: run.requests.len(),
        completed: metrics.completed,
        rejected: count(TrialOutcome::Rejected),
        failed: count(TrialOutcome::Failed),
        unfinished: count(TrialOutcome::Unfinished),
        queue: latencies(&started),
        time_to_first_token: latencies(&first_tokens),
        completion: metrics.completion,
        output_tokens: run
            .requests
            .iter()
            .filter(|request| request.outcome == TrialOutcome::Completed)
            .map(|request| request.output_tokens.len())
            .sum(),
        makespan_micros: run.makespan_micros,
        iterations: run.iterations.len(),
        peak_kv_cache_bytes: run
            .iterations
            .iter()
            .map(|iteration| iteration.kv_cache_bytes)
            .max()
            .unwrap_or(0),
        peak_reserved_kv_bytes: run
            .iterations
            .iter()
            .map(|iteration| iteration.reserved_kv_bytes)
            .max()
            .unwrap_or(0),
        admission_deferrals: total(|iteration| iteration.admission_deferrals),
        budget_deferrals: total(|iteration| iteration.budget_deferrals),
        slot_deferrals: total(|iteration| iteration.slot_deferrals),
        backfill_skips: total(|iteration| iteration.backfill_skips),
        preemptions: total(|iteration| iteration.preemptions),
        kv_used_byte_iterations: run
            .iterations
            .iter()
            .filter(|iteration| iteration.kv_allocated_bytes > 0)
            .map(|iteration| iteration.kv_cache_bytes)
            .sum(),
        kv_allocated_byte_iterations: run
            .iterations
            .iter()
            .map(|iteration| iteration.kv_allocated_bytes)
            .sum(),
        prefix_hit_tokens: run
            .iterations
            .last()
            .map_or(0, |iteration| iteration.prefix_hit_tokens),
        input_tokens: total(|iteration| iteration.input_tokens),
        capability_fallbacks: total(|iteration| iteration.capability_fallbacks),
        slowdown: metrics.slowdown,
        starved: metrics.starved,
        worst_queue_wait_micros: metrics.worst_queue_wait_micros,
        with_deadline: run
            .requests
            .iter()
            .filter(|request| request.deadline_micros.is_some())
            .count(),
        deadline_missed: metrics.deadline_missed,
        classes,
    }
}

/// Metrics shared by whole-run and per-class summaries.
struct RequestMetrics {
    completed: usize,
    completion: LatencySummary,
    slowdown: SlowdownSummary,
    starved: usize,
    worst_queue_wait_micros: u64,
    deadline_missed: usize,
}

impl RequestMetrics {
    fn new(
        requests: &[&crate::TrialRequest],
        run: &SchedulingRun,
        isolated_micros: &[Option<u64>],
        starvation_factor: u64,
    ) -> Self {
        let mut completion = Vec::new();
        let mut slowdowns = Vec::new();
        let mut starved = 0;
        let mut worst_queue_wait_micros = 0;
        let mut deadline_missed = 0;
        for request in requests {
            let isolated = isolated_micros
                .get(request.trace_index)
                .copied()
                .flatten()
                .filter(|isolated| *isolated > 0);
            let completed_at = (request.outcome == TrialOutcome::Completed)
                .then_some(request.completed_micros)
                .flatten();
            if let Some(completed_at) = completed_at {
                let end_to_end = completed_at - request.arrival_micros;
                completion.push(end_to_end);
                if let Some(isolated) = isolated {
                    slowdowns.push(to_f64(end_to_end) / to_f64(isolated));
                }
            }
            if !matches!(
                request.outcome,
                TrialOutcome::Rejected | TrialOutcome::Failed
            ) {
                let wait = request
                    .started_micros
                    .unwrap_or(run.makespan_micros)
                    .saturating_sub(request.arrival_micros);
                worst_queue_wait_micros = worst_queue_wait_micros.max(wait);
                if isolated
                    .is_some_and(|isolated| wait > isolated.saturating_mul(starvation_factor))
                {
                    starved += 1;
                }
            }
            if let Some(deadline) = request.deadline_micros
                && completed_at.is_none_or(|completed_at| completed_at > deadline)
            {
                deadline_missed += 1;
            }
        }
        Self {
            completed: completion.len(),
            completion: latencies(&completion),
            slowdown: summarize_slowdowns(slowdowns),
            starved,
            worst_queue_wait_micros,
            deadline_missed,
        }
    }
}

fn latencies(micros: &[u64]) -> LatencySummary {
    summarize_latencies(
        &micros
            .iter()
            .map(|micros| Duration::from_micros(*micros))
            .collect::<Vec<_>>(),
    )
}

fn summarize_slowdowns(mut slowdowns: Vec<f64>) -> SlowdownSummary {
    if slowdowns.is_empty() {
        return SlowdownSummary {
            jain_fairness: 1.0,
            ..SlowdownSummary::default()
        };
    }
    slowdowns.sort_unstable_by(f64::total_cmp);
    let count = to_f64(slowdowns.len());
    let sum = slowdowns.iter().sum::<f64>();
    let squares = slowdowns.iter().map(|value| value * value).sum::<f64>();
    let rank = |percent: usize| {
        let index = (slowdowns.len() * percent).div_ceil(100).saturating_sub(1);
        slowdowns[index.min(slowdowns.len() - 1)]
    };
    SlowdownSummary {
        samples: slowdowns.len(),
        p50: rank(50),
        p95: rank(95),
        max: slowdowns[slowdowns.len() - 1],
        mean: sum / count,
        jain_fairness: if squares > 0.0 {
            sum * sum / (count * squares)
        } else {
            1.0
        },
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "metric values are far below 2^52"
)]
fn to_f64(value: impl TryInto<u64>) -> f64 {
    value.try_into().map_or(f64::MAX, |value: u64| value as f64)
}

#[cfg(test)]
mod tests {
    use ola_runtime::{
        AdaptiveConstraints, AdaptiveIntegerDomain, AdaptiveObjectiveMetrics, AdaptivePolicy,
        EarliestDeadlineFirst, FifoTokenBudgetPolicy, HeadOfLine, JobCost, MilpCandidateOptimizer,
        PriorityScheduler, ShortestJobFirst,
    };

    use super::*;
    use crate::KvGeometry;

    // These traces and loads were chosen by sweeping load: each test pins a
    // regime where the policy's intended effect is real, and every number
    // comes from a deterministic virtual-time run.
    fn parameters(mean_interarrival_micros: u64) -> TraceParameters {
        TraceParameters {
            seed: 3,
            requests: 120,
            mean_interarrival_micros,
        }
    }

    fn fifo() -> ComposedPolicy {
        ComposedPolicy::default().with_admission(MemoryAwareAdmission)
    }

    fn compare(
        trace: &SchedulingTrace,
        config: &SimulationConfig,
        policies: Vec<Box<dyn ExecutionPolicy>>,
    ) -> SchedulingComparison {
        compare_schedulers(trace, config, policies, DEFAULT_STARVATION_FACTOR).unwrap()
    }

    fn class<'a>(summary: &'a SchedulingSummary, label: &str) -> &'a ClassSummary {
        summary
            .classes
            .iter()
            .find(|class| class.class == label)
            .unwrap()
    }

    #[test]
    fn jain_fairness_follows_its_definition() {
        assert!((summarize_slowdowns(vec![1.0, 3.0]).jain_fairness - 0.8).abs() < 1e-12);
        assert!((summarize_slowdowns(vec![2.0, 2.0, 2.0]).jain_fairness - 1.0).abs() < 1e-12);
        let empty = summarize_slowdowns(Vec::new());
        assert_eq!((empty.samples, empty.jain_fairness), (0, 1.0));
        let spread = summarize_slowdowns(vec![4.0, 1.0, 2.0, 3.0]);
        assert_eq!((spread.p50, spread.max, spread.mean), (2.0, 4.0, 2.5));
    }

    #[test]
    fn comparisons_are_exactly_reproducible() {
        let trace = SchedulingTrace::overload(TraceParameters {
            requests: 60,
            ..parameters(2_000)
        });
        let policies = || -> Vec<Box<dyn ExecutionPolicy>> {
            vec![
                Box::new(fifo()),
                Box::new(fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt))),
            ]
        };
        let config = SimulationConfig::default();

        assert_eq!(
            compare(&trace, &config, policies()),
            compare(&trace, &config, policies())
        );
    }

    #[test]
    fn offline_comparison_runs_an_adaptive_policy_deterministically() {
        let trace = SchedulingTrace::overload(TraceParameters {
            requests: 30,
            ..parameters(2_000)
        });
        let config = SimulationConfig::default();
        let adaptive_factory = || {
            AdaptivePolicy::new(fifo(), vec![("mode", "fixed".into())])
                .with_fallback(fifo())
                .with_update(|_, observations, theta| {
                    if let Some((_, value)) = theta.iter_mut().find(|(key, _)| *key == "mode") {
                        *value = observations.len().to_string();
                    }
                })
                .with_seed(11)
        };
        let policies = || {
            vec![
                Box::new(adaptive_factory()) as Box<dyn ExecutionPolicy>,
                Box::new(fifo()) as Box<dyn ExecutionPolicy>,
            ]
        };
        let first = compare(&trace, &config, policies());
        let second = compare(&trace, &config, policies());

        assert_eq!(first, second);
        assert!(first.summaries[0].policy.starts_with("adaptive{"));
        assert_eq!(first.summaries[0].completed, first.summaries[1].completed);
    }

    #[test]
    fn virtual_time_harness_runs_milp_adaptive_policy_against_a_fixed_baseline() {
        let trace = SchedulingTrace::overload(TraceParameters {
            requests: 30,
            ..parameters(2_000)
        });
        let config = SimulationConfig::default();
        let adaptive_factory = || {
            AdaptivePolicy::new(
                fifo().with_batch(FifoTokenBudgetPolicy::new(4).unwrap()),
                vec![("max_tokens", "4".into())],
            )
            .with_fallback(fifo())
            .with_constraints(AdaptiveConstraints {
                max_active_requests: Some(config.max_active_requests),
                prefill_chunk_size: config.prefill_chunk_size,
                available_memory_bytes: None,
            })
            .with_optimizer(
                MilpCandidateOptimizer::new(
                    vec![AdaptiveIntegerDomain {
                        name: "max_tokens",
                        min: 1,
                        max: 16,
                        step: 1,
                    }],
                    vec![("max_tokens", 1)],
                )
                .unwrap()
                .with_objective_builder(|observations, _| {
                    let metrics = AdaptiveObjectiveMetrics::from_observations(observations);
                    if metrics.starved_requests > 0 {
                        vec![("max_tokens", -1)]
                    } else {
                        vec![("max_tokens", 1)]
                    }
                }),
            )
            .with_factory(|theta: &[(&'static str, String)]| {
                let max_tokens = theta
                    .iter()
                    .find(|(name, _)| *name == "max_tokens")
                    .and_then(|(_, value)| value.parse::<usize>().ok())
                    .ok_or_else(|| {
                        ola_runtime::PolicyError::InvalidCandidate("missing max_tokens".into())
                    })?;
                let batch = FifoTokenBudgetPolicy::new(max_tokens).ok_or_else(|| {
                    ola_runtime::PolicyError::InvalidCandidate("max_tokens must be positive".into())
                })?;
                Ok(Box::new(fifo().with_batch(batch)) as Box<dyn ExecutionPolicy>)
            })
        };
        let policies = || {
            vec![
                Box::new(adaptive_factory()) as Box<dyn ExecutionPolicy>,
                Box::new(fifo()) as Box<dyn ExecutionPolicy>,
            ]
        };

        let first = compare(&trace, &config, policies());
        let second = compare(&trace, &config, policies());

        assert_eq!(first, second);
        assert!(first.summaries[0].policy.starts_with("adaptive{"));
        assert_eq!(first.summaries[0].completed, first.summaries[1].completed);
    }

    #[test]
    fn virtual_time_candidate_search_ranks_replayed_policy_outcomes() {
        let trace = SchedulingTrace::overload(TraceParameters {
            requests: 12,
            ..parameters(2_000)
        });
        let config = SimulationConfig::default();
        let factory = |theta: &[(&'static str, String)]| {
            let max_tokens = theta
                .iter()
                .find(|(name, _)| *name == "max_tokens")
                .and_then(|(_, value)| value.parse::<usize>().ok())
                .ok_or_else(|| PolicyError::InvalidCandidate("missing max_tokens".into()))?;
            let batch = FifoTokenBudgetPolicy::new(max_tokens).ok_or_else(|| {
                PolicyError::InvalidCandidate("max_tokens must be positive".into())
            })?;
            Ok(Box::new(fifo().with_batch(batch)) as Box<dyn ExecutionPolicy>)
        };
        let weights = AdaptiveObjectiveWeights {
            ttft: 1,
            completion: 1,
            ..AdaptiveObjectiveWeights::default()
        };
        let first = optimize_virtual_time_candidate(
            &trace,
            &config,
            vec![AdaptiveIntegerDomain {
                name: "max_tokens",
                min: 1,
                max: 4,
                step: 1,
            }],
            vec![("max_tokens", "1".into())],
            factory,
            weights,
        )
        .unwrap();
        let second = optimize_virtual_time_candidate(
            &trace,
            &config,
            vec![AdaptiveIntegerDomain {
                name: "max_tokens",
                min: 1,
                max: 4,
                step: 1,
            }],
            vec![("max_tokens", "1".into())],
            factory,
            weights,
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.run.requests.len(), trace.requests.len());
        assert_eq!(first.run.final_live_requests, 0);
        assert!(first.objective > 0);
    }

    #[test]
    fn held_out_study_does_not_use_evaluation_traces_for_selection() {
        let base = parameters(2_000);
        let training = SchedulingTrace::overload(TraceParameters { seed: 1, ..base });
        let held_out = vec![
            SchedulingTrace::overload(TraceParameters { seed: 2, ..base }),
            SchedulingTrace::overload(TraceParameters { seed: 3, ..base }),
        ];
        let config = SimulationConfig::default();
        let adaptive_factory = |theta: &[(&'static str, String)]| {
            let max_tokens = theta
                .iter()
                .find(|(name, _)| *name == "max_tokens")
                .and_then(|(_, value)| value.parse::<usize>().ok())
                .ok_or_else(|| PolicyError::InvalidCandidate("missing max_tokens".into()))?;
            let batch = FifoTokenBudgetPolicy::new(max_tokens).ok_or_else(|| {
                PolicyError::InvalidCandidate("max_tokens must be positive".into())
            })?;
            Ok(Box::new(fifo().with_batch(batch)) as Box<dyn ExecutionPolicy>)
        };
        let baseline_factory =
            |_: &[(&'static str, String)]| Ok(Box::new(fifo()) as Box<dyn ExecutionPolicy>);

        let study = evaluate_virtual_time_held_out(
            &training,
            &held_out,
            &config,
            vec![AdaptiveIntegerDomain {
                name: "max_tokens",
                min: 1,
                max: 4,
                step: 1,
            }],
            vec![("max_tokens", "1".into())],
            adaptive_factory,
            baseline_factory,
            AdaptiveObjectiveWeights {
                ttft: 1,
                completion: 1,
                ..AdaptiveObjectiveWeights::default()
            },
        )
        .unwrap();

        assert_eq!(
            study
                .held_out
                .iter()
                .map(|score| score.seed)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(study.theta.len(), 1);
        assert!(
            study
                .held_out
                .iter()
                .all(|score| { score.adaptive_objective > 0 && score.baseline_objective > 0 })
        );
    }

    #[test]
    fn priority_scheduling_protects_urgent_requests_at_the_others_expense() {
        let trace = SchedulingTrace::priority_classes(parameters(300));
        let comparison = compare(
            &trace,
            &SimulationConfig::default(),
            vec![
                Box::new(fifo()),
                Box::new(fifo().with_scheduler(PriorityScheduler::new())),
            ],
        );
        let (fifo_run, priority_run) = (&comparison.summaries[0], &comparison.summaries[1]);

        let urgent = |summary| class(summary, "priority=5");
        let normal = |summary| class(summary, "priority=0");
        assert!(
            urgent(priority_run).completion.p95_micros < urgent(fifo_run).completion.p95_micros / 4
        );
        assert_eq!(urgent(priority_run).starved, 0);
        assert!(normal(priority_run).slowdown.mean > normal(fifo_run).slowdown.mean);
    }

    #[test]
    fn shortest_job_first_trades_long_request_latency_for_short_ones() {
        let trace = SchedulingTrace::overload(parameters(2_000));
        let comparison = compare(
            &trace,
            &SimulationConfig::default(),
            vec![
                Box::new(fifo()),
                Box::new(fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt))),
            ],
        );
        let (fifo_run, sjf_run) = (&comparison.summaries[0], &comparison.summaries[1]);

        let p50 = |summary, label| class(summary, label).completion.p50_micros;
        assert!(p50(sjf_run, "size=short") < p50(fifo_run, "size=short"));
        assert!(p50(sjf_run, "size=long") > p50(fifo_run, "size=long"));
        // Short jobs no longer wait behind long ones, so slowdowns are more
        // even: about 0.55 against FIFO's 0.32 on this trace. Before P6 the
        // simulator stamped tokens at iteration start, which dropped
        // one-token requests from slowdown (zero isolated time) and
        // understated every isolated time, and this comparison reversed.
        assert!(sjf_run.slowdown.jain_fairness > fifo_run.slowdown.jain_fairness);
    }

    #[test]
    fn backfill_helps_when_the_kv_pool_binds() {
        let config = SimulationConfig {
            kv: KvGeometry {
                capacity_bytes: 512 * 64,
                ..KvGeometry::default()
            },
            ..SimulationConfig::default()
        };
        let trace = SchedulingTrace::mixed_short_long(parameters(500));
        let comparison = compare(
            &trace,
            &config,
            vec![
                Box::new(fifo()),
                Box::new(fifo().with_head_of_line(HeadOfLine::Backfill { max_skipped: 4 })),
            ],
        );
        let (blocking, backfill) = (&comparison.summaries[0], &comparison.summaries[1]);

        assert!(blocking.admission_deferrals > 0);
        assert_eq!(blocking.backfill_skips, 0);
        assert!(backfill.backfill_skips > 0);
        assert!(backfill.completion.p50_micros < blocking.completion.p50_micros);
        assert!(backfill.starved < blocking.starved);
    }

    #[test]
    fn earliest_deadline_first_misses_fewer_deadlines_under_moderate_contention() {
        let config = SimulationConfig {
            max_active_requests: 2,
            ..SimulationConfig::default()
        };
        let trace = SchedulingTrace::deadlines(parameters(6_000), 504);
        let comparison = compare(
            &trace,
            &config,
            vec![
                Box::new(fifo()),
                Box::new(fifo().with_scheduler(EarliestDeadlineFirst)),
            ],
        );
        let (fifo_run, edf_run) = (&comparison.summaries[0], &comparison.summaries[1]);

        assert_eq!(fifo_run.with_deadline, 120);
        assert!(edf_run.deadline_missed < fifo_run.deadline_missed);
    }

    #[test]
    fn impossible_requests_have_no_isolated_time_and_never_count_as_starved() {
        let config = SimulationConfig::default();
        let trace = SchedulingTrace::impossible(parameters(500), config.kv.capacity_tokens());
        let comparison = compare(&trace, &config, vec![Box::new(fifo())]);

        for (traced, isolated) in trace.requests.iter().zip(&comparison.isolated_micros) {
            assert_eq!(
                isolated.is_none(),
                traced.prompt_tokens > config.kv.capacity_tokens()
            );
        }
        let summary = &comparison.summaries[0];
        assert!(summary.rejected > 0);
        assert_eq!(summary.completed + summary.rejected, summary.requests);
    }

    #[test]
    fn adaptive_budget_recovers_throughput_when_overhead_dominates() {
        // With the default cost model each iteration's fixed and invocation
        // overhead dwarfs token work, so a small fixed budget forces many
        // more iterations under load; growing the budget with the queue
        // recovers most of the lost throughput.
        let trace = SchedulingTrace::mixed_short_long(parameters(1_000));
        let comparison = compare(
            &trace,
            &SimulationConfig::default(),
            vec![
                Box::new(fifo().with_batch(ola_runtime::FifoTokenBudgetPolicy::new(32).unwrap())),
                Box::new(
                    fifo().with_batch(
                        ola_runtime::QueueAdaptiveTokenBudget::new(32, 256, 16).unwrap(),
                    ),
                ),
            ],
        );
        let (fixed, adaptive) = (&comparison.summaries[0], &comparison.summaries[1]);

        let throughput = |summary: &SchedulingSummary| summary.output_tokens_per_second().unwrap();
        assert!(throughput(adaptive) > throughput(fixed) * 1.5);
        assert!(adaptive.completion.p50_micros < fixed.completion.p50_micros);
    }

    #[test]
    fn capping_prompt_work_helps_when_token_work_dominates() {
        // When per-token cost dominates, an iteration carrying a long prompt
        // stalls every live decode; capping prompt work per iteration lowers
        // completion latency against an unbounded batch.
        let config = SimulationConfig {
            cost: crate::CostModel {
                fixed_micros: 50,
                per_invocation_micros: 50,
                per_dense_token_micros: 40,
            },
            ..SimulationConfig::default()
        };
        for trace in [
            SchedulingTrace::prefill_decode_competition(parameters(2_000)),
            SchedulingTrace::mixed_short_long(parameters(1_000)),
        ] {
            let comparison = compare(
                &trace,
                &config,
                vec![
                    Box::new(fifo()),
                    Box::new(
                        fifo()
                            .with_batch(ola_runtime::MixedTokenBudgetPolicy::new(256, 32).unwrap()),
                    ),
                ],
            );
            let (unbounded, capped) = (&comparison.summaries[0], &comparison.summaries[1]);

            assert!(
                capped.completion.p50_micros < unbounded.completion.p50_micros,
                "{}",
                trace.name
            );
        }
    }
}
