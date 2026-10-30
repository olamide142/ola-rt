//! Cluster policies: snapshot in, plan out.

use std::{error::Error, fmt};

use ola_core::{ClusterRequestId, WorkerId};
use ola_runtime::PolicyDescriptor;
use serde::{Deserialize, Serialize};

use crate::{ClusterIteration, ClusterPlan, ClusterSnapshot};

/// Turns an immutable cluster snapshot into a cluster plan.
///
/// `plan` takes `&self`, so identical snapshots and configuration produce
/// identical decisions, and a plan rejected by validation leaves the policy
/// unchanged. Stateful policies learn only in [`observe`](Self::observe).
pub trait ClusterPolicy: Send {
    /// Returns the policy's stable name, configuration, and components.
    fn descriptor(&self) -> PolicyDescriptor;

    /// Plans one cluster iteration.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterPolicyError`] when the policy cannot decide; the
    /// coordinator then changes nothing.
    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError>;

    /// Receives each committed cluster iteration; the only place a policy
    /// may learn.
    fn observe(&mut self, _iteration: &ClusterIteration) {}
}

impl<P: ClusterPolicy + ?Sized> ClusterPolicy for Box<P> {
    fn descriptor(&self) -> PolicyDescriptor {
        (**self).descriptor()
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        (**self).plan(snapshot)
    }

    fn observe(&mut self, iteration: &ClusterIteration) {
        (**self).observe(iteration);
    }
}

/// Output of one cluster planning pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterDecision {
    /// Plan to validate and commit.
    pub plan: ClusterPlan,
    /// Inputs and alternatives behind the plan.
    pub record: ClusterDecisionRecord,
}

/// What a cluster policy considered.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClusterDecisionRecord {
    /// Policy that produced the plan.
    pub policy: PolicyDescriptor,
    /// Alternatives for each request the policy routed or tried to route.
    pub routes: Vec<RouteRecord>,
}

impl ClusterDecisionRecord {
    /// The route record for `request`, if the policy kept one.
    pub fn route(&self, request: ClusterRequestId) -> Option<&RouteRecord> {
        self.routes.iter().find(|route| route.request == request)
    }
}

/// The workers a policy weighed for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteRecord {
    /// Request being routed.
    pub request: ClusterRequestId,
    /// Eligible workers in ID order, with the load the policy saw.
    pub candidates: Vec<RouteCandidate>,
    /// Why the policy chose its worker.
    pub reason: RouteReason,
}

/// Why a routing policy picked a worker.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum RouteReason {
    /// Next worker in turn.
    RoundRobin,
    /// Least load among all candidates.
    LeastLoaded,
    /// Less loaded of two sampled candidates.
    PowerOfTwo,
    /// Most KV headroom.
    KvHeadroom,
    /// Highest-ranked worker for the prompt prefix within the load bound;
    /// `rank` 0 is the prefix's home worker.
    PrefixAffinity {
        /// Position in the prefix's rendezvous ranking.
        rank: usize,
    },
    /// No prefix to route by, so least load.
    NoPrefix,
}

/// One eligible worker and its load when a request was routed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteCandidate {
    /// Worker considered.
    pub worker: WorkerId,
    /// Queued, live, and in-flight requests, including earlier assignments
    /// in the same plan.
    pub load: usize,
    /// KV tokens the worker could still give out after queued, in-flight,
    /// and earlier same-plan demand; absent when its pool reports nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_headroom_tokens: Option<usize>,
}

/// A cluster policy could not produce a plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClusterPolicyError {
    /// The policy found the snapshot unusable.
    InvalidSnapshot(String),
}

impl fmt::Display for ClusterPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSnapshot(message) => {
                write!(formatter, "cluster policy cannot plan: {message}")
            }
        }
    }
}

impl Error for ClusterPolicyError {}
