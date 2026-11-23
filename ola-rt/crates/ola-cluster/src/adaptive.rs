//! Slowly adapting cluster routing policy.

use std::collections::BTreeMap;

use ola_runtime::PolicyDescriptor;

use crate::{
    ClusterDecision, ClusterIteration, ClusterPolicy, ClusterPolicyError, ClusterSnapshot,
    KvHeadroomRouting, LeastLoadedRouting, RouteReason, RoutingRequest, RoutingWorker,
    optimize_batch_routing,
};

/// Weights for the cluster-level routing objective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusterWeights {
    /// Weight assigned to queue/load pressure.
    pub load: u64,
    /// Weight assigned to available KV headroom.
    pub kv_headroom: u64,
    /// Weight assigned to calibrated worker-link cost.
    pub network: u64,
}

impl Default for ClusterWeights {
    fn default() -> Self {
        Self {
            load: 1,
            kv_headroom: 1,
            network: 0,
        }
    }
}

/// Adaptive routing wrapper with transactional weight updates.
///
/// The policy deliberately keeps the candidate set and plan validation in the
/// existing baseline policies. Its slow loop may push new weights through
/// [`set_weights`](Self::set_weights); malformed or all-zero proposals are
/// rejected without changing the active policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveClusterPolicy {
    weights: ClusterWeights,
    observations: u64,
}

impl AdaptiveClusterPolicy {
    /// Creates an adaptive policy with the default balanced objective.
    pub fn new(weights: ClusterWeights) -> Result<Self, ClusterPolicyError> {
        if weights.load == 0 && weights.kv_headroom == 0 {
            return Err(ClusterPolicyError::InvalidSnapshot(
                "adaptive cluster weights cannot both be zero".into(),
            ));
        }
        Ok(Self {
            weights,
            observations: 0,
        })
    }

    /// Returns the currently committed weights.
    pub const fn weights(&self) -> ClusterWeights {
        self.weights
    }

    /// Installs a new objective atomically.
    pub fn set_weights(&mut self, weights: ClusterWeights) -> Result<(), ClusterPolicyError> {
        if weights.load == 0 && weights.kv_headroom == 0 {
            return Err(ClusterPolicyError::InvalidSnapshot(
                "adaptive cluster weights cannot both be zero".into(),
            ));
        }
        self.weights = weights;
        Ok(())
    }

    /// Number of committed observations consumed by the slow loop.
    pub const fn observations(&self) -> u64 {
        self.observations
    }
}

impl Default for AdaptiveClusterPolicy {
    fn default() -> Self {
        Self::new(ClusterWeights::default()).expect("default weights are valid")
    }
}

impl ClusterPolicy for AdaptiveClusterPolicy {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("adaptive-cluster-routing")
            .with_config("load_weight", self.weights.load)
            .with_config("kv_weight", self.weights.kv_headroom)
            .with_config("network_weight", self.weights.network)
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        let mut decision = if self.weights.kv_headroom > self.weights.load {
            KvHeadroomRouting.plan(snapshot)?
        } else {
            LeastLoadedRouting.plan(snapshot)?
        };
        let mut grouped = BTreeMap::new();
        for request in &snapshot.pending {
            grouped
                .entry(request.model.clone())
                .or_insert_with(Vec::new)
                .push(request);
        }
        for (model, requests) in grouped {
            let workers = snapshot
                .eligible_workers(&model)
                .map(|(worker, view)| RoutingWorker {
                    worker: worker.clone(),
                    capacity: view.summary.max_active_requests,
                    current_load: view.load(),
                    kv_headroom_tokens: view.kv_headroom_tokens(&model).unwrap_or(usize::MAX),
                    load_cost: (view.load() as u64)
                        .saturating_mul(self.weights.load.max(1))
                        .saturating_add(view.network_cost_us.saturating_mul(self.weights.network)),
                    network_cost_us: view.network_cost_us,
                })
                .collect::<Vec<_>>();
            if workers.is_empty() {
                continue;
            }
            let batch = requests
                .iter()
                .map(|request| RoutingRequest {
                    request: request.id,
                    kv_tokens: request.demand_tokens(),
                })
                .collect::<Vec<_>>();
            let Ok(solution) = optimize_batch_routing(&workers, &batch) else {
                continue;
            };
            for action in &mut decision.plan.actions {
                let crate::ClusterAction::Assign { request, worker } = action else {
                    continue;
                };
                let Some(target) = solution.assignments.get(request) else {
                    continue;
                };
                *worker = target.clone();
                if let Some(route) = decision.record.route_mut(*request) {
                    route.reason = RouteReason::BatchOptimizer;
                }
            }
        }
        Ok(decision)
    }

    fn observe(&mut self, _iteration: &ClusterIteration) {
        self.observations = self.observations.saturating_add(1);
    }

    fn update_weights(&mut self, weights: ClusterWeights) -> Result<(), ClusterPolicyError> {
        self.set_weights(weights)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ola_core::{ModelId, Priority, Timestamp, WorkerId};

    use super::*;
    use crate::{PendingRequest, WorkerHealth, WorkerRole, WorkerSummary, WorkerView};

    fn worker(id: &str, queued: usize) -> (WorkerId, WorkerView) {
        let model = ModelId::from("tiny");
        (
            WorkerId::new(id),
            WorkerView {
                summary: WorkerSummary {
                    worker: WorkerId::new(id),
                    generation: 0,
                    reported_at: Timestamp::ZERO,
                    health: WorkerHealth::Ready,
                    role: WorkerRole::Mixed,
                    models: BTreeSet::from([model]),
                    max_active_requests: 2,
                    queued_requests: queued,
                    live_requests: 0,
                    queued_tokens: queued,
                    admitted_requests: 0,
                    kv: BTreeMap::new(),
                },
                in_flight: 0,
                in_flight_tokens: 0,
                network_cost_us: 0,
            },
        )
    }

    #[test]
    fn adaptive_policy_marks_a_capacity_constrained_batch_assignment() {
        let model = ModelId::from("tiny");
        let snapshot = ClusterSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            workers: BTreeMap::from([worker("busy", 1), worker("idle", 0)]),
            pending: vec![
                PendingRequest {
                    id: ola_core::ClusterRequestId::new(1),
                    model: model.clone(),
                    priority: Priority::NORMAL,
                    arrival_at: Timestamp::ZERO,
                    prompt_tokens: Some(1),
                    prompt: vec![1],
                    max_new_tokens: 1,
                },
                PendingRequest {
                    id: ola_core::ClusterRequestId::new(2),
                    model,
                    priority: Priority::NORMAL,
                    arrival_at: Timestamp::ZERO,
                    prompt_tokens: Some(1),
                    prompt: vec![2],
                    max_new_tokens: 1,
                },
            ],
            routed_requests: 0,
        };
        let decision = AdaptiveClusterPolicy::default().plan(&snapshot).unwrap();
        assert!(
            decision
                .record
                .routes
                .iter()
                .all(|route| route.reason == RouteReason::BatchOptimizer)
        );
        assert_eq!(decision.plan.actions.len(), 2);
    }

    #[test]
    fn adaptive_policy_accounts_for_calibrated_network_cost() {
        let model = ModelId::from("tiny");
        let mut near = worker("near", 0).1;
        near.network_cost_us = 1;
        let mut far = worker("far", 0).1;
        far.network_cost_us = 100;
        let snapshot = ClusterSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            workers: BTreeMap::from([(WorkerId::new("far"), far), (WorkerId::new("near"), near)]),
            pending: vec![PendingRequest {
                id: ola_core::ClusterRequestId::new(1),
                model,
                priority: Priority::NORMAL,
                arrival_at: Timestamp::ZERO,
                prompt_tokens: Some(1),
                prompt: vec![1],
                max_new_tokens: 1,
            }],
            routed_requests: 0,
        };
        let policy = AdaptiveClusterPolicy::new(ClusterWeights {
            load: 1,
            kv_headroom: 1,
            network: 1,
        })
        .unwrap();
        let decision = policy.plan(&snapshot).unwrap();
        assert!(matches!(
            decision.plan.actions.as_slice(),
            [crate::ClusterAction::Assign { worker, .. }] if worker == &WorkerId::new("near")
        ));
    }
}
