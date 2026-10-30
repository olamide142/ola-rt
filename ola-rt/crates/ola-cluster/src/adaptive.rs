//! Slowly adapting cluster routing policy.

use ola_runtime::PolicyDescriptor;

use crate::{
    ClusterDecision, ClusterIteration, ClusterPolicy, ClusterPolicyError, ClusterSnapshot,
    KvHeadroomRouting, LeastLoadedRouting,
};

/// Weights for the cluster-level routing objective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusterWeights {
    /// Weight assigned to queue/load pressure.
    pub load: u64,
    /// Weight assigned to available KV headroom.
    pub kv_headroom: u64,
}

impl Default for ClusterWeights {
    fn default() -> Self {
        Self {
            load: 1,
            kv_headroom: 1,
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
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        if self.weights.kv_headroom > self.weights.load {
            KvHeadroomRouting.plan(snapshot)
        } else {
            LeastLoadedRouting.plan(snapshot)
        }
    }

    fn observe(&mut self, _iteration: &ClusterIteration) {
        self.observations = self.observations.saturating_add(1);
    }
}
