//! Online parameter optimization by replaying recent work in virtual time.
//!
//! [`ReplayOptimizer`] answers the question an online optimizer cannot answer
//! from live metrics alone: how would a parameter vector it is *not* running
//! have done? It remembers the recent workload from adaptive observations,
//! rebuilds it as a [`SchedulingTrace`], and replays that trace through the
//! virtual-time simulator once per candidate. The candidate with the lowest
//! objective wins, but only if it beats the current vector's own replay by a
//! margin, so noise does not make the policy oscillate.
//!
//! Replays are deterministic and use the [`CostModel`] in the optimizer's
//! [`SimulationConfig`]. Their value depends on that model matching the real
//! executor: calibrate it (`cost-calibration`, [`crate::CostFit`]) before
//! trusting proposals on real hardware.
//!
//! [`CostModel`]: crate::CostModel

use std::collections::BTreeMap;

use ola_core::{RequestId, Timestamp};
use ola_runtime::{
    AdaptiveConstraints, AdaptiveIntegerDomain, AdaptiveObjectiveMetrics, AdaptiveObjectiveWeights,
    AdaptiveOptimizationError, AdaptiveOptimizer, AdaptivePolicy, AdaptiveRequestObservation,
    ComposedPolicy, ExecutionPolicy, FiniteCandidateOptimizer, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, PolicyError, SwitchingCost,
};

use crate::{
    CostModel, KvGeometry, SchedulingTrace, SimulationConfig, TraceParameters, TraceRequest,
    run_scheduling_trial, scheduling_summary::run_observations,
};

/// Token-budget grid shared by the adaptive studies and `ola serve
/// --adaptive`: 2^4 to 2^10 tokens per iteration, of which 25% to 100% may
/// be prompt work. The top of the range admits a whole iteration's work at
/// typical sizes, so unbounded batching is always (nearly) a candidate.
pub const TOKEN_BUDGET_DOMAINS: [AdaptiveIntegerDomain; 2] = [
    AdaptiveIntegerDomain {
        name: "log2_max_tokens",
        min: 4,
        max: 10,
        step: 1,
    },
    AdaptiveIntegerDomain {
        name: "prefill_share_percent",
        min: 25,
        max: 100,
        step: 25,
    },
];

/// The default objective: p95 completion plus p95 time to first token.
pub const LATENCY_OBJECTIVE: AdaptiveObjectiveWeights = AdaptiveObjectiveWeights {
    ttft: 1,
    completion: 1,
    throughput: 0,
    starvation: 0,
    deadline: 0,
    fallback: 0,
};

/// The grid's smallest budget, 16 tokens with half for prompts.
pub fn smallest_token_budget() -> Vec<(&'static str, String)> {
    vec![
        ("log2_max_tokens", "4".into()),
        ("prefill_share_percent", "50".into()),
    ]
}

/// The grid's largest budget, 1024 tokens with no prompt cap.
pub fn largest_token_budget() -> Vec<(&'static str, String)> {
    vec![
        ("log2_max_tokens", "10".into()),
        ("prefill_share_percent", "100".into()),
    ]
}

/// The `(max_tokens, max_prefill_tokens)` budget a grid vector names.
///
/// # Errors
///
/// Fails if a parameter is missing or out of range.
pub fn token_budget(theta: &[(&'static str, String)]) -> Result<(usize, usize), PolicyError> {
    let value = |name: &str| {
        theta
            .iter()
            .find(|(key, _)| *key == name)
            .and_then(|(_, value)| value.parse::<u32>().ok())
            .ok_or_else(|| PolicyError::InvalidCandidate(format!("missing {name}")))
    };
    let log2 = value("log2_max_tokens")?;
    let share = value("prefill_share_percent")?;
    if log2 > 20 || !(1..=100).contains(&share) {
        return Err(PolicyError::InvalidCandidate(
            "budget parameters out of range".into(),
        ));
    }
    let max_tokens = 1_usize << log2;
    let max_prefill_tokens = (max_tokens * share as usize / 100).max(1);
    Ok((max_tokens, max_prefill_tokens))
}

/// FIFO with memory-aware admission and the mixed token budget `theta`
/// names: the policy family the adaptive studies and `ola serve --adaptive`
/// tune.
///
/// # Errors
///
/// Fails if `theta` does not name a valid budget.
pub fn token_budget_policy(
    theta: &[(&'static str, String)],
) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let (max_tokens, max_prefill_tokens) = token_budget(theta)?;
    let batch = MixedTokenBudgetPolicy::new(max_tokens, max_prefill_tokens).ok_or_else(|| {
        PolicyError::InvalidCandidate(
            "max_prefill_tokens must be positive and no greater than max_tokens".into(),
        )
    })?;
    Ok(Box::new(
        ComposedPolicy::default()
            .with_admission(MemoryAwareAdmission)
            .with_batch(batch),
    ))
}

/// How often a serving adaptive policy re-plans, and how cautiously.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnlineTuning {
    /// Executor iterations between proposals.
    pub interval: u64,
    /// Completed requests replayed per proposal.
    pub window: usize,
    /// Replay improvement, in percent, required to switch.
    pub min_improvement_percent: u64,
}

impl Default for OnlineTuning {
    /// The values `online-adaptive-policy-study` was evaluated with.
    fn default() -> Self {
        Self {
            interval: 40,
            window: 64,
            min_improvement_percent: 5,
        }
    }
}

/// An adaptive token-budget policy for a live executor.
///
/// It starts at the largest budget, which behaves like unbounded batching,
/// and only moves when replaying recent work under `cost` predicts an
/// improvement. `cost` must describe the executor's hardware: fit it with
/// the `cost-calibration` experiment. Replays model `max_active_requests`
/// slots and no prefill chunking, with a KV pool large enough never to
/// bind; live admission still enforces real memory.
///
/// # Errors
///
/// Fails if `tuning` is unusable.
pub fn serving_adaptive_policy(
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    adaptive_token_budget_policy(largest_token_budget(), cost, max_active_requests, tuning)
}

/// [`serving_adaptive_policy`] starting from `theta`.
fn adaptive_token_budget_policy(
    theta: Vec<(&'static str, String)>,
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    let geometry = KvGeometry::default();
    let config = SimulationConfig {
        max_active_requests,
        prefill_chunk_size: None,
        kv: KvGeometry {
            capacity_bytes: geometry.bytes_per_token.saturating_mul(1 << 20),
            ..geometry
        },
        cost,
        ..SimulationConfig::default()
    };
    let optimizer = ReplayOptimizer::new(
        ReplaySettings {
            domains: TOKEN_BUDGET_DOMAINS.to_vec(),
            weights: LATENCY_OBJECTIVE,
            config,
            window: tuning.window,
            min_requests: tuning.window.min(8),
            min_improvement_percent: tuning.min_improvement_percent,
            switching: SwitchingCost::default(),
        },
        token_budget_policy,
    )?;
    let inner = token_budget_policy(&theta)
        .map_err(|error| AdaptiveOptimizationError::InvalidDomain(error.to_string()))?;
    Ok(AdaptivePolicy::new(inner, theta)
        .with_fallback(ComposedPolicy::default().with_admission(MemoryAwareAdmission))
        .with_constraints(AdaptiveConstraints {
            max_active_requests: Some(max_active_requests),
            prefill_chunk_size: None,
            available_memory_bytes: None,
        })
        .with_optimization_interval(tuning.interval.max(1))
        .with_factory(token_budget_policy)
        .with_optimizer(optimizer))
}

/// What the optimizer has learned about one request from observations.
#[derive(Clone, Copy, Debug)]
struct Seen {
    arrived_at: Timestamp,
    prompt_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    generated_tokens: usize,
    completed: bool,
}

/// How a [`ReplayOptimizer`] searches and when it switches.
#[derive(Clone, Debug)]
pub struct ReplaySettings {
    /// Candidate grid.
    pub domains: Vec<AdaptiveIntegerDomain>,
    /// Objective to minimize over each replay.
    pub weights: AdaptiveObjectiveWeights,
    /// Simulated executor. Its slot limit and prefill chunk are replaced by
    /// the live executor's when the runtime publishes them.
    pub config: SimulationConfig,
    /// Most recent completed requests replayed.
    pub window: usize,
    /// Completed requests needed before the first proposal; until then the
    /// current vector is kept.
    pub min_requests: usize,
    /// Minimum objective improvement, in percent of the current vector's
    /// replay score, required to switch. It guards against replay noise
    /// (the window is a small sample), so it is relative.
    pub min_improvement_percent: u64,
    /// Explicit cost, in objective microseconds, of changing each knob. It
    /// models the transition itself, so it is absolute and enters the
    /// argmin; the default charges nothing.
    pub switching: SwitchingCost,
}

/// How one proposal chose, with the objective exactly as it was compared.
///
/// `J(θ) = replay_score(θ) + switch_cost(θ_prev, θ)`; the candidate is the
/// argmin of `J` over the grid, and it replaces `θ_prev` only if
/// `J(candidate) < replay_score(θ_prev) − margin`, with
/// `margin = min_improvement_percent · |replay_score(θ_prev)| / 100`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaySelection {
    /// Proposal number, from 1.
    pub proposal: u64,
    /// Replayed requests.
    pub window_requests: usize,
    /// `replay_score(θ_prev)`; `None` when it cannot be scored.
    pub current_score: Option<i64>,
    /// `argmin J`.
    pub candidate: Vec<(&'static str, String)>,
    /// `replay_score(candidate)`.
    pub candidate_score: Option<i64>,
    /// `switch_cost(θ_prev, candidate)`.
    pub switching_cost: i64,
    /// `J(candidate)`.
    pub candidate_objective: Option<i64>,
    /// Noise margin subtracted from the current score.
    pub margin: i64,
    /// Whether the candidate was installed.
    pub switched: bool,
    /// Simulated replays this proposal ran.
    pub replays: usize,
}

/// An [`AdaptiveOptimizer`] that replays recent work for each candidate.
///
/// `factory` builds the policy a parameter vector describes; use the same
/// factory the [`ola_runtime::AdaptivePolicy`] installs proposals with.
pub struct ReplayOptimizer<F> {
    settings: ReplaySettings,
    factory: F,
    seen: BTreeMap<RequestId, Seen>,
    proposals: u64,
    switches: u64,
    replays: u64,
    last: Option<ReplaySelection>,
}

impl<F> ReplayOptimizer<F>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
{
    /// Creates an optimizer.
    ///
    /// # Errors
    ///
    /// Fails if a domain is malformed or the window is empty.
    pub fn new(settings: ReplaySettings, factory: F) -> Result<Self, AdaptiveOptimizationError> {
        if settings.window == 0 {
            return Err(AdaptiveOptimizationError::InvalidDomain(
                "the replay window must hold at least one request".into(),
            ));
        }
        // Validates the domains once, up front.
        FiniteCandidateOptimizer::new(
            settings.domains.clone(),
            |_: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints|
             -> Result<Option<i64>, AdaptiveOptimizationError> { Ok(None) },
        )?;
        Ok(Self {
            settings,
            factory,
            seen: BTreeMap::new(),
            proposals: 0,
            switches: 0,
            replays: 0,
            last: None,
        })
    }

    /// Simulated replays run by every proposal so far: the controller's
    /// deterministic overhead, independent of host speed.
    pub const fn replays(&self) -> u64 {
        self.replays
    }

    /// The most recent proposal that had enough history to replay.
    pub const fn last_selection(&self) -> Option<&ReplaySelection> {
        self.last.as_ref()
    }

    /// Proposals made with enough history to replay.
    pub const fn proposals(&self) -> u64 {
        self.proposals
    }

    /// Proposals that changed the parameter vector.
    pub const fn switches(&self) -> u64 {
        self.switches
    }

    fn ingest(&mut self, observations: &[AdaptiveRequestObservation]) {
        for observation in observations {
            let seen = self.seen.entry(observation.request_id).or_insert(Seen {
                arrived_at: observation.arrived_at,
                prompt_tokens: None,
                max_new_tokens: None,
                generated_tokens: 0,
                completed: false,
            });
            seen.prompt_tokens = seen.prompt_tokens.or(observation.prompt_tokens);
            seen.max_new_tokens = seen.max_new_tokens.or(observation.max_new_tokens);
            if let Some(generated) = observation.generated_tokens {
                seen.generated_tokens = seen.generated_tokens.max(generated);
            }
            seen.completed |= observation.completed_at.is_some();
        }
        // Keep the window's worth of completed requests and anything live.
        let mut completed = self
            .seen
            .iter()
            .filter(|(_, seen)| seen.completed)
            .map(|(id, seen)| (seen.arrived_at, *id))
            .collect::<Vec<_>>();
        if completed.len() > self.settings.window {
            completed.sort_unstable();
            let excess = completed.len() - self.settings.window;
            for (_, id) in &completed[..excess] {
                self.seen.remove(id);
            }
        }
    }

    /// The recent completed workload, arrivals relative to the earliest.
    fn window_trace(&self) -> Option<SchedulingTrace> {
        let mut requests = self
            .seen
            .values()
            .filter(|seen| seen.completed)
            .filter_map(|seen| {
                let prompt_tokens = seen.prompt_tokens?;
                let max_new_tokens = seen.max_new_tokens?;
                // A request that stopped short of its limit hit EOS there.
                let eos_after = (seen.generated_tokens > 0
                    && seen.generated_tokens < max_new_tokens)
                    .then_some(seen.generated_tokens);
                Some((seen.arrived_at, prompt_tokens, max_new_tokens, eos_after))
            })
            .collect::<Vec<_>>();
        if requests.len() < self.settings.min_requests.max(1) {
            return None;
        }
        requests.sort_by_key(|(arrived_at, ..)| *arrived_at);
        let start = requests[0].0.as_micros();
        let end = requests[requests.len() - 1].0.as_micros();
        let count = requests.len();
        Some(SchedulingTrace {
            name: "online-window",
            parameters: TraceParameters {
                seed: 0,
                requests: count,
                mean_interarrival_micros: (end - start) / (count as u64).max(1),
            },
            requests: requests
                .into_iter()
                .map(
                    |(arrived_at, prompt_tokens, max_new_tokens, eos_after)| TraceRequest {
                        arrival_micros: arrived_at.as_micros() - start,
                        prompt_tokens,
                        max_new_tokens,
                        priority: 0,
                        deadline_micros: None,
                        eos_after,
                        shared_prefix: None,
                    },
                )
                .collect(),
        })
    }

    fn config(&self, constraints: AdaptiveConstraints) -> SimulationConfig {
        let mut config = self.settings.config;
        if let Some(slots) = constraints.max_active_requests {
            config.max_active_requests = slots;
        }
        if constraints.prefill_chunk_size.is_some() {
            config.prefill_chunk_size = constraints.prefill_chunk_size;
        }
        config
    }
}

/// Replays `trace` under the policy `theta` describes and scores the run;
/// `None` for a vector the factory rejects or a run the objective cannot
/// score.
fn replay_score<F>(
    factory: &F,
    theta: &[(&'static str, String)],
    trace: &SchedulingTrace,
    config: &SimulationConfig,
    weights: AdaptiveObjectiveWeights,
) -> Result<Option<i64>, AdaptiveOptimizationError>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>,
{
    let Ok(policy) = factory(theta) else {
        return Ok(None);
    };
    let run = run_scheduling_trial(trace, policy, config)
        .map_err(|error| AdaptiveOptimizationError::Infeasible(error.to_string()))?;
    Ok(AdaptiveObjectiveMetrics::from_observations(&run_observations(&run)).score(weights))
}

impl<F> AdaptiveOptimizer for ReplayOptimizer<F>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
{
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        self.ingest(observations);
        let Some(trace) = self.window_trace() else {
            return Ok(current.to_vec());
        };
        self.proposals += 1;
        let config = self.config(constraints);
        let weights = self.settings.weights;
        let switching = &self.settings.switching;
        let factory = &self.factory;
        let replays = std::sync::atomic::AtomicUsize::new(0);
        let score = |theta: &[(&'static str, String)]| {
            replays.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            replay_score(factory, theta, &trace, &config, weights)
        };
        let current_score = score(current)?;
        let mut search = FiniteCandidateOptimizer::new(
            self.settings.domains.clone(),
            |candidate: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints| {
                Ok(score(candidate)?
                    .map(|operating| operating.saturating_add(switching.cost(current, candidate))))
            },
        )?;
        // No feasible candidate keeps the current vector, and says so.
        let (best, best_score) = match search.propose(current, &[], constraints) {
            Ok(best) => {
                let best_score = score(&best)?;
                (best, best_score)
            }
            Err(AdaptiveOptimizationError::NoCandidate) => (current.to_vec(), None),
            Err(error) => return Err(error),
        };
        let switching_cost = switching.cost(current, &best);
        let best_objective = best_score.map(|operating| operating.saturating_add(switching_cost));
        let margin = current_score.map_or(0, |current| {
            i64::try_from(
                i128::from(current.unsigned_abs())
                    * i128::from(self.settings.min_improvement_percent)
                    / 100,
            )
            .unwrap_or(i64::MAX)
        });
        let switch = match (current_score, best_objective) {
            // The current vector cannot even be scored; take any that can.
            (None, Some(_)) => true,
            (Some(current), Some(best)) => {
                i128::from(best) < i128::from(current) - i128::from(margin)
            }
            (_, None) => false,
        };
        let switched = switch && best != current;
        let replays = replays.into_inner();
        self.replays += replays as u64;
        self.last = Some(ReplaySelection {
            proposal: self.proposals,
            window_requests: trace.requests.len(),
            current_score,
            candidate: best.clone(),
            candidate_score: best_score,
            switching_cost,
            candidate_objective: best_objective,
            margin,
            switched,
            replays,
        });
        if switched {
            self.switches += 1;
            Ok(best)
        } else {
            Ok(current.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ola_runtime::{
        AdaptivePolicy, ComposedPolicy, MemoryAwareAdmission, MixedTokenBudgetPolicy,
    };

    use super::*;

    fn budget(theta: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
        let log2 = theta
            .iter()
            .find(|(name, _)| *name == "log2_max_tokens")
            .and_then(|(_, value)| value.parse::<u32>().ok())
            .ok_or_else(|| PolicyError::InvalidCandidate("missing log2_max_tokens".into()))?;
        let max_tokens = 1_usize << log2;
        let batch = MixedTokenBudgetPolicy::new(max_tokens, max_tokens)
            .ok_or_else(|| PolicyError::InvalidCandidate("bad budget".into()))?;
        Ok(Box::new(
            ComposedPolicy::default()
                .with_admission(MemoryAwareAdmission)
                .with_batch(batch),
        ))
    }

    fn settings() -> ReplaySettings {
        ReplaySettings {
            domains: vec![AdaptiveIntegerDomain {
                name: "log2_max_tokens",
                min: 4,
                max: 10,
                step: 1,
            }],
            weights: AdaptiveObjectiveWeights {
                ttft: 1,
                completion: 1,
                ..AdaptiveObjectiveWeights::default()
            },
            config: SimulationConfig::default(),
            window: 64,
            min_requests: 8,
            min_improvement_percent: 5,
            switching: SwitchingCost::default(),
        }
    }

    type Factory = fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>;
    type Shared = Arc<Mutex<ReplayOptimizer<Factory>>>;

    /// Runs an overload trace under an adaptive policy starting at
    /// `2^log2` tokens, returning the optimizer and the final run.
    fn adapt(log2: u32) -> (Shared, crate::SchedulingRun) {
        adapt_with(log2, settings())
    }

    fn adapt_with(log2: u32, settings: ReplaySettings) -> (Shared, crate::SchedulingRun) {
        let factory: Factory = budget;
        let optimizer = Arc::new(Mutex::new(ReplayOptimizer::new(settings, factory).unwrap()));
        let shared = optimizer.clone();
        let theta = vec![("log2_max_tokens", log2.to_string())];
        let policy = AdaptivePolicy::new(budget(&theta).unwrap(), theta)
            .with_optimization_interval(40)
            .with_factory(budget)
            .with_optimizer(
                move |current: &[(&'static str, String)],
                      observations: &[AdaptiveRequestObservation],
                      constraints: AdaptiveConstraints| {
                    shared
                        .lock()
                        .unwrap()
                        .propose(current, observations, constraints)
                },
            );
        let trace = SchedulingTrace::overload(TraceParameters {
            seed: 3,
            requests: 80,
            mean_interarrival_micros: 2_000,
        });
        let run = run_scheduling_trial(&trace, policy, &SimulationConfig::default()).unwrap();
        (optimizer, run)
    }

    /// Serves `requests` on a synthetic executor under `policy`, returning
    /// each request's tokens and the policy's final descriptor.
    fn serve(
        policy: impl ExecutionPolicy + 'static,
        requests: &[(usize, usize, Option<usize>)],
    ) -> (Vec<Vec<u32>>, String) {
        use ola_core::{Input, ModelDescriptor, RequestSpec};
        use ola_runtime::{Executor, Runtime};

        let mut executor = Executor::with_max_active_requests(Runtime::default(), 4)
            .unwrap()
            .with_policy(policy);
        executor
            .register_model(
                ModelDescriptor::language("synthetic", "synthetic"),
                crate::synthetic_artifact(KvGeometry {
                    capacity_bytes: 64 * 8 * 1024,
                    ..KvGeometry::default()
                }),
            )
            .unwrap();
        let ids = requests
            .iter()
            .enumerate()
            .map(|(index, &(prompt, max_new_tokens, eos))| {
                let mut spec = RequestSpec::text("synthetic", "");
                spec.inputs = vec![Input::Tokens(crate::synthetic_prompt(prompt, eos))];
                spec.generation.max_new_tokens = max_new_tokens;
                executor
                    .submit(spec, Timestamp::from_micros(index as u64 * 50))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut step = 0;
        while executor.state().queued_len() > 0 || executor.has_active_request() {
            step += 1;
            executor
                .step_iteration(Timestamp::from_micros(step * 1_000))
                .unwrap();
        }
        let outputs = ids
            .into_iter()
            .map(|id| executor.take_result(id).unwrap().token_ids)
            .collect();
        (outputs, executor.policy_descriptor().to_string())
    }

    #[test]
    fn the_serving_policy_re_plans_without_changing_outputs() {
        let requests = (0..40)
            .map(|index| {
                let long = index % 5 == 0;
                (
                    if long { 120 } else { 6 + index % 7 },
                    if long { 24 } else { 6 },
                    (index % 3 == 0).then_some(3),
                )
            })
            .collect::<Vec<_>>();
        // From the smallest budget, replays should move it.
        let adaptive = adaptive_token_budget_policy(
            smallest_token_budget(),
            CostModel::default(),
            4,
            OnlineTuning {
                interval: 5,
                window: 8,
                min_improvement_percent: 0,
            },
        )
        .unwrap();
        let (adapted, descriptor) = serve(adaptive, &requests);
        let (baseline, _) = serve(
            ComposedPolicy::default().with_admission(MemoryAwareAdmission),
            &requests,
        );
        // Budgets reorder work; greedy outputs stay the same.
        assert_eq!(adapted, baseline);
        assert!(!descriptor.contains("log2_max_tokens=4,"), "{descriptor}");
    }

    #[test]
    fn keeps_the_current_vector_until_enough_work_has_completed() {
        let mut optimizer = ReplayOptimizer::new(settings(), budget).unwrap();
        let current = vec![("log2_max_tokens", "4".to_owned())];
        let proposed = optimizer
            .propose(&current, &[], AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, current);
        assert_eq!(optimizer.proposals(), 0);
    }

    #[test]
    fn replays_observed_work_and_leaves_a_poor_budget() {
        let (optimizer, run) = adapt(4);
        let optimizer = optimizer.lock().unwrap();
        // Live observations carried enough to rebuild the workload.
        assert!(optimizer.proposals() > 0);
        assert!(optimizer.switches() > 0);
        let fixed = run_scheduling_trial(
            &SchedulingTrace::overload(TraceParameters {
                seed: 3,
                requests: 80,
                mean_interarrival_micros: 2_000,
            }),
            budget(&[("log2_max_tokens", "4".into())]).unwrap(),
            &SimulationConfig::default(),
        )
        .unwrap();
        let score = |run: &crate::SchedulingRun| {
            AdaptiveObjectiveMetrics::from_observations(&run_observations(run))
                .score(settings().weights)
                .unwrap()
        };
        assert!(score(&run) < score(&fixed), "adapting did not help");
    }

    #[test]
    fn keeps_a_budget_that_replays_cannot_beat() {
        let (optimizer, _) = adapt(10);
        let optimizer = optimizer.lock().unwrap();
        assert!(optimizer.proposals() > 0);
        assert_eq!(optimizer.switches(), 0);
    }

    fn with_switching(cost: i64) -> ReplaySettings {
        ReplaySettings {
            min_improvement_percent: 0,
            switching: SwitchingCost::uniform(&["log2_max_tokens"], cost),
            ..settings()
        }
    }

    #[test]
    fn a_dominant_switching_cost_keeps_the_current_budget() {
        let (optimizer, _) = adapt_with(4, with_switching(1_000_000_000));
        let optimizer = optimizer.lock().unwrap();
        assert!(optimizer.proposals() > 0);
        assert_eq!(optimizer.switches(), 0);
        let last = optimizer.last_selection().unwrap();
        assert!(!last.switched);
        // Staying put costs nothing, so it is the argmin of J.
        assert_eq!(last.candidate, vec![("log2_max_tokens", "4".to_owned())]);
        assert_eq!(last.switching_cost, 0);
    }

    #[test]
    fn a_large_enough_improvement_pays_for_the_switch() {
        let (optimizer, _) = adapt_with(4, with_switching(1_000));
        let optimizer = optimizer.lock().unwrap();
        assert!(optimizer.switches() > 0);
        assert!(optimizer.replays() >= optimizer.proposals() * 8);
    }

    #[test]
    fn the_recorded_objective_is_the_one_compared() {
        let (optimizer, _) = adapt_with(4, with_switching(1_000));
        let optimizer = optimizer.lock().unwrap();
        let last = optimizer.last_selection().unwrap();
        assert_eq!(
            last.candidate_objective,
            last.candidate_score
                .map(|score| score + last.switching_cost)
        );
        if last.switched {
            assert!(last.candidate_objective.unwrap() < last.current_score.unwrap() - last.margin);
        }
    }

    #[test]
    fn unscorable_candidates_never_replace_the_current_vector() {
        fn reject(_: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
            Err(PolicyError::InvalidCandidate("rejected".into()))
        }
        let mut optimizer = ReplayOptimizer::new(with_switching(0), reject).unwrap();
        let observations = (0..8)
            .map(|index| AdaptiveRequestObservation {
                request_id: RequestId::new(index),
                arrived_at: Timestamp::from_micros(index * 100),
                first_token_at: Some(Timestamp::from_micros(index * 100 + 50)),
                completed_at: Some(Timestamp::from_micros(index * 100 + 90)),
                time_to_first_token_micros: Some(50),
                completion_latency_micros: Some(90),
                deadline_missed: false,
                capability_fallback: false,
                prompt_tokens: Some(8),
                max_new_tokens: Some(4),
                generated_tokens: Some(4),
            })
            .collect::<Vec<_>>();
        let current = vec![("log2_max_tokens", "6".to_owned())];
        let proposed = optimizer
            .propose(&current, &observations, AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, current);
        let last = optimizer.last_selection().unwrap();
        assert_eq!((last.current_score, last.candidate_objective), (None, None));
        assert!(!last.switched);
    }
}
