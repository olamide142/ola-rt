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
    AdaptiveOptimizationError, AdaptiveOptimizer, AdaptiveRequestObservation, ExecutionPolicy,
    FiniteCandidateOptimizer, PolicyError,
};

use crate::{
    SchedulingTrace, SimulationConfig, TraceParameters, TraceRequest, run_scheduling_trial,
    scheduling_summary::run_observations,
};

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
    /// replay score, required to switch.
    pub min_improvement_percent: u64,
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
        })
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
        let factory = &self.factory;
        let current_score = replay_score(factory, current, &trace, &config, weights)?;
        let mut search = FiniteCandidateOptimizer::new(
            self.settings.domains.clone(),
            |candidate: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints| {
                replay_score(factory, candidate, &trace, &config, weights)
            },
        )?;
        let best = search.propose(current, &[], constraints)?;
        let best_score = replay_score(factory, &best, &trace, &config, weights)?;
        let switch = match (current_score, best_score) {
            // The current vector cannot even be scored; take any that can.
            (None, Some(_)) => true,
            (Some(current), Some(best)) => {
                let margin = i128::from(current.unsigned_abs())
                    * i128::from(self.settings.min_improvement_percent)
                    / 100;
                i128::from(best) < i128::from(current) - margin
            }
            (_, None) => false,
        };
        if switch && best != current {
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
        }
    }

    type Factory = fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>;
    type Shared = Arc<Mutex<ReplayOptimizer<Factory>>>;

    /// Runs an overload trace under an adaptive policy starting at
    /// `2^log2` tokens, returning the optimizer and the final run.
    fn adapt(log2: u32) -> (Shared, crate::SchedulingRun) {
        let factory: Factory = budget;
        let optimizer = Arc::new(Mutex::new(
            ReplayOptimizer::new(settings(), factory).unwrap(),
        ));
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
}
