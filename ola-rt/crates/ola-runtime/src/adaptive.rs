//! A stateful wrapper for policy configuration experiments.

use std::collections::BTreeMap;

use ola_core::{RequestId, Timestamp};

use crate::{
    ExecutionPolicy, ExecutorEvent, ExecutorIteration, PlanningSnapshot, PolicyDecision,
    PolicyDescriptor, PolicyError,
};

/// Timing observations derived from one committed request iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRequestObservation {
    /// Request represented by this observation.
    pub request_id: RequestId,
    /// Timestamp at which the request entered the runtime queue.
    pub arrived_at: Timestamp,
    /// Timestamp of the first emitted token, if one was emitted.
    pub first_token_at: Option<Timestamp>,
    /// Timestamp at which the request completed, if it completed.
    pub completed_at: Option<Timestamp>,
    /// Time from arrival to first token, in microseconds.
    pub time_to_first_token_micros: Option<u64>,
    /// Time from arrival to completion, in microseconds.
    pub completion_latency_micros: Option<u64>,
}

impl ExecutorIteration {
    /// Derives timing observations using arrival timestamps retained in the
    /// iteration's decision record and token events committed by execution.
    pub fn adaptive_observations(&self) -> Vec<AdaptiveRequestObservation> {
        let mut timings = self
            .decision
            .arrival_times
            .iter()
            .map(|(request_id, arrived_at)| (*request_id, (*arrived_at, None, None)))
            .collect::<BTreeMap<_, _>>();

        for event in &self.events {
            match event {
                ExecutorEvent::Token {
                    request_id,
                    at,
                    finished,
                    ..
                } => {
                    if let Some((_, first_token_at, completed_at)) = timings.get_mut(request_id) {
                        *first_token_at = Some(first_token_at.unwrap_or(*at));
                        if *finished {
                            *completed_at = Some(*at);
                        }
                    }
                }
                ExecutorEvent::Failed { request_id, .. } => {
                    if let Some((_, _, completed_at)) = timings.get_mut(request_id) {
                        *completed_at = Some(self.at);
                    }
                }
                _ => {}
            }
        }

        timings
            .into_iter()
            .map(
                |(
                    request_id,
                    (arrived_at, first_token_at, completed_at),
                )| AdaptiveRequestObservation {
                    request_id,
                    arrived_at,
                    time_to_first_token_micros: first_token_at
                        .map(|at| at.as_micros().saturating_sub(arrived_at.as_micros())),
                    completion_latency_micros: completed_at
                        .map(|at| at.as_micros().saturating_sub(arrived_at.as_micros())),
                    first_token_at,
                    completed_at,
                },
            )
            .collect()
    }
}

/// Wraps a policy with a stable adaptive configuration and update counter.
///
/// The wrapper does not change planning yet: it delegates plans to `inner`.
/// Its descriptor is copied into every decision record, making the active
/// parameter vector and observation count part of the execution audit trail.
pub struct AdaptivePolicy<P> {
    inner: P,
    fallback: Option<Box<dyn ExecutionPolicy>>,
    validator: Option<Box<dyn Fn(&PlanningSnapshot, &PolicyDecision) -> bool + Send + Sync>>,
    updater: Option<
        Box<dyn FnMut(&ExecutorIteration, &[AdaptiveRequestObservation], &mut Vec<(&'static str, String)>)
            + Send>,
    >,
    theta: Vec<(&'static str, String)>,
    update_count: u64,
    seed: u64,
    random_state: u64,
}

impl<P: ExecutionPolicy> AdaptivePolicy<P> {
    /// Creates a wrapper around `inner` with the supplied adaptive parameters.
    ///
    /// Parameter keys are sorted once so equivalent configurations produce the
    /// same descriptor regardless of construction order.
    pub fn new(inner: P, mut theta: Vec<(&'static str, String)>) -> Self {
        theta.sort_by_key(|(key, _)| *key);
        Self {
            inner,
            fallback: None,
            validator: None,
            updater: None,
            theta,
            update_count: 0,
            seed: 0,
            random_state: 0,
        }
    }

    /// Installs a known-safe policy used when the candidate returns an error.
    #[must_use]
    pub fn with_fallback(mut self, fallback: impl ExecutionPolicy + 'static) -> Self {
        self.fallback = Some(Box::new(fallback));
        self
    }

    /// Installs a pure candidate-plan validator used before commit.
    #[must_use]
    pub fn with_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&PlanningSnapshot, &PolicyDecision) -> bool + Send + Sync + 'static,
    {
        self.validator = Some(Box::new(validator));
        self
    }

    /// Installs an update hook invoked only after a committed iteration.
    ///
    /// The hook receives derived timing observations and may update θ for the
    /// next planning boundary. It cannot affect the plan currently committed.
    #[must_use]
    pub fn with_update<F>(mut self, updater: F) -> Self
    where
        F: FnMut(&ExecutorIteration, &[AdaptiveRequestObservation], &mut Vec<(&'static str, String)>)
            + Send
            + 'static,
    {
        self.updater = Some(Box::new(updater));
        self
    }

    /// Sets the deterministic seed reserved for observation-time updates.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.random_state = seed;
        self
    }

    /// Returns the wrapped policy.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Returns the current adaptive parameter vector in descriptor order.
    pub fn theta(&self) -> &[(&'static str, String)] {
        &self.theta
    }

    /// Returns the number of committed observations received by the wrapper.
    pub const fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Returns the current observation-time random state for diagnostics.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    fn adaptive_descriptor(&self) -> PolicyDescriptor {
        let mut descriptor = PolicyDescriptor::new("adaptive")
            .with_config("update_count", self.update_count)
            .with_config("seed", self.seed)
            .with_config("random_state", self.random_state)
            .with_component(self.inner.descriptor());
        if let Some(fallback) = &self.fallback {
            descriptor = descriptor.with_component(fallback.descriptor());
        }
        for (key, value) in &self.theta {
            descriptor = descriptor.with_config(key, value);
        }
        descriptor
    }
}

fn checked_plan(
    policy: &dyn ExecutionPolicy,
    snapshot: &PlanningSnapshot,
) -> Result<PolicyDecision, PolicyError> {
    let decision = policy.plan(snapshot)?;
    decision
        .plan
        .validate(snapshot, policy.enforces_model_kv_capacity())
        .map_err(|error| PolicyError::InvalidCandidate(error.to_string()))?;
    Ok(decision)
}

impl<P> ExecutionPolicy for AdaptivePolicy<P>
where
    P: ExecutionPolicy,
{

    fn descriptor(&self) -> PolicyDescriptor {
        self.adaptive_descriptor()
    }

    fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
        let mut decision = match checked_plan(&self.inner, snapshot) {
            Ok(decision) => decision,
            Err(error) => match &self.fallback {
                Some(fallback) => checked_plan(fallback.as_ref(), snapshot)?,
                None => return Err(error),
            },
        };
        if self
            .validator
            .as_ref()
            .is_some_and(|validator| !validator(snapshot, &decision))
        {
            decision = match &self.fallback {
                Some(fallback) => checked_plan(fallback.as_ref(), snapshot)?,
                None => {
                    return Err(PolicyError::InvalidCandidate(
                        "validator returned false".into(),
                    ));
                }
            };
        }
        decision.record.policy = self.adaptive_descriptor();
        Ok(decision)
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        self.inner.enforces_model_kv_capacity()
            && self
                .fallback
                .as_ref()
                .is_none_or(|fallback| fallback.enforces_model_kv_capacity())
    }

    fn observe(&mut self, iteration: &ExecutorIteration) {
        self.inner.observe(iteration);
        if let Some(fallback) = &mut self.fallback {
            fallback.observe(iteration);
        }
        let observations = iteration.adaptive_observations();
        if let Some(updater) = &mut self.updater {
            updater(iteration, &observations, &mut self.theta);
            self.theta.sort_by_key(|(key, _)| *key);
        }
        self.random_state = splitmix64(self.random_state);
        self.update_count = self.update_count.saturating_add(1);
    }

    fn recovers_kv_exhaustion(&self) -> bool {
        self.inner.recovers_kv_exhaustion()
            || self
                .fallback
                .as_ref()
                .is_some_and(|fallback| fallback.recovers_kv_exhaustion())
    }
}

fn splitmix64(state: u64) -> u64 {
    let mut value = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::{ComposedPolicy, DecisionRecord, ExecutionPlan};
    use ola_core::{ModelId, Timestamp};
    use ola_model::GeneratedToken;

    struct RejectingPolicy;

    impl ExecutionPolicy for RejectingPolicy {
        fn descriptor(&self) -> PolicyDescriptor {
            PolicyDescriptor::new("rejecting")
        }

        fn plan(&self, _snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
            Err(PolicyError::KernelSelection {
                model_id: ModelId::new("candidate"),
                message: "candidate rejected".into(),
            })
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            true
        }
    }

    struct AcceptingPolicy;

    impl ExecutionPolicy for AcceptingPolicy {
        fn descriptor(&self) -> PolicyDescriptor {
            PolicyDescriptor::new("fallback")
        }

        fn plan(&self, _snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
            Ok(PolicyDecision {
                plan: ExecutionPlan::default(),
                record: DecisionRecord::default(),
            })
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            true
        }
    }

    struct StalePolicy;

    impl ExecutionPolicy for StalePolicy {
        fn descriptor(&self) -> PolicyDescriptor {
            PolicyDescriptor::new("stale")
        }

        fn plan(&self, _snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
            Ok(PolicyDecision {
                plan: ExecutionPlan {
                    generation: 99,
                    ..ExecutionPlan::default()
                },
                record: DecisionRecord::default(),
            })
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            true
        }
    }

    #[test]
    fn descriptor_is_stable_and_contains_theta_and_counter() {
        let policy = AdaptivePolicy::new(
            ComposedPolicy::default(),
            vec![("learning_rate", "0.1".into()), ("exploration", "7".into())],
        );

        assert_eq!(
            policy.descriptor().to_string(),
            "adaptive{update_count=0,seed=0,random_state=0,exploration=7,learning_rate=0.1}[composed{head_of_line=block}[fifo,unconstrained-admission,unbounded-batch,unbucketed-prefill]]"
        );
    }

    #[test]
    fn observation_randomness_is_seeded_and_consumed_only_on_observe() {
        let mut first = AdaptivePolicy::new(AcceptingPolicy, vec![]).with_seed(7);
        let mut second = AdaptivePolicy::new(AcceptingPolicy, vec![]).with_seed(7);
        assert_eq!(first.random_state(), 7);

        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        first.observe(&iteration);
        second.observe(&iteration);

        assert_eq!(first.update_count(), 1);
        assert_eq!(first.random_state(), second.random_state());
        assert_ne!(first.random_state(), 7);
    }

    #[test]
    fn update_hook_changes_theta_only_after_observation() {
        let mut policy = AdaptivePolicy::new(AcceptingPolicy, vec![("value", "0".into())])
            .with_update(|_, observations, theta| {
                theta.push(("observed", observations.len().to_string()));
            });
        assert!(!policy.descriptor().to_string().contains("observed="));

        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        policy.observe(&iteration);

        assert!(policy.descriptor().to_string().contains("observed=0"));
    }

    #[test]
    fn derives_ttft_and_completion_latency_from_committed_events() {
        let mut decision = DecisionRecord::default();
        decision
            .arrival_times
            .insert(ola_core::RequestId::new(4), Timestamp::from_micros(10));
        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::from_micros(30),
            events: vec![ExecutorEvent::Token {
                request_id: ola_core::RequestId::new(4),
                token: GeneratedToken {
                    id: 9,
                    is_end_of_sequence: true,
                },
                generated_tokens: 1,
                kv_cache_bytes: 0,
                phase: crate::ExecutionPhase::Prefill,
                finished: true,
                at: Timestamp::from_micros(25),
            }],
            execution_plan: ExecutionPlan::default(),
            decision,
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };

        assert_eq!(
            iteration.adaptive_observations(),
            vec![AdaptiveRequestObservation {
                request_id: ola_core::RequestId::new(4),
                arrived_at: Timestamp::from_micros(10),
                first_token_at: Some(Timestamp::from_micros(25)),
                completed_at: Some(Timestamp::from_micros(25)),
                time_to_first_token_micros: Some(15),
                completion_latency_micros: Some(15),
            }]
        );
    }

    #[test]
    fn candidate_errors_use_the_safe_fallback_and_keep_adaptive_identity() {
        let policy = AdaptivePolicy::new(RejectingPolicy, vec![]).with_fallback(AcceptingPolicy);
        let snapshot = PlanningSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            limits: crate::ExecutorLimits {
                max_active_requests: 1,
                prefill_chunk_size: None,
            },
            queued: Vec::new(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: crate::MemoryState::default(),
            device: None,
            measurements: crate::PolicyMeasurements::default(),
        };
        let decision = policy.plan(&snapshot).unwrap();

        assert_eq!(decision.record.policy.name, "adaptive");
        assert_eq!(decision.record.policy.components[1].name, "fallback");
    }

    #[test]
    fn rejected_candidates_use_the_safe_fallback() {
        let policy = AdaptivePolicy::new(AcceptingPolicy, vec![])
            .with_fallback(ComposedPolicy::default())
            .with_validator(|_, _| false);
        let snapshot = PlanningSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            limits: crate::ExecutorLimits {
                max_active_requests: 1,
                prefill_chunk_size: None,
            },
            queued: Vec::new(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: crate::MemoryState::default(),
            device: None,
            measurements: crate::PolicyMeasurements::default(),
        };

        let decision = policy.plan(&snapshot).unwrap();
        assert_eq!(decision.record.policy.components[1].name, "composed");
    }

    #[test]
    fn invalid_candidate_plans_are_replaced_before_returning() {
        let policy = AdaptivePolicy::new(StalePolicy, vec![]).with_fallback(ComposedPolicy::default());
        let snapshot = PlanningSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            limits: crate::ExecutorLimits {
                max_active_requests: 1,
                prefill_chunk_size: None,
            },
            queued: Vec::new(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: crate::MemoryState::default(),
            device: None,
            measurements: crate::PolicyMeasurements::default(),
        };

        let decision = policy.plan(&snapshot).unwrap();
        assert_eq!(decision.record.policy.components[1].name, "composed");
    }
}
