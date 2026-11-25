//! Cluster-level decisions and their validation.

use std::{collections::BTreeSet, error::Error, fmt};

use ola_core::{ClusterRequestId, ModelId, WorkerId};
use serde::{Deserialize, Serialize};

use crate::ClusterSnapshot;

/// Why a pending request stays pending.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum DeferReason {
    /// Workers host the model, but none of them is ready.
    NoReadyWorker {
        /// Model the request needs.
        model: ModelId,
    },
}

/// Why a pending request can never be routed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum RejectReason {
    /// No joined worker hosts the model.
    ModelNotServed {
        /// Model the request needs.
        model: ModelId,
    },
}

impl fmt::Display for RejectReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotServed { model } => {
                write!(formatter, "no worker serves model `{model}`")
            }
        }
    }
}

/// One decision about one pending request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ClusterAction {
    /// Make `worker` the request's owner.
    Assign {
        /// Request to route.
        request: ClusterRequestId,
        /// Worker that will own it.
        worker: WorkerId,
    },
    /// Leave the request pending until a later plan.
    Defer {
        /// Request left pending.
        request: ClusterRequestId,
        /// Why it was not routed.
        reason: DeferReason,
    },
    /// Fail the request.
    Reject {
        /// Request that fails.
        request: ClusterRequestId,
        /// Why it can never be routed.
        reason: RejectReason,
    },
}

impl ClusterAction {
    /// The request the action decides.
    pub const fn request(&self) -> ClusterRequestId {
        match self {
            Self::Assign { request, .. }
            | Self::Defer { request, .. }
            | Self::Reject { request, .. } => *request,
        }
    }
}

/// Every decision of one cluster planning pass.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ClusterPlan {
    /// Coordinator generation of the snapshot the plan was built from.
    pub generation: u64,
    /// One action per pending request, in the order they are committed.
    pub actions: Vec<ClusterAction>,
}

impl ClusterPlan {
    /// Checks the plan against the snapshot it claims to be built from.
    ///
    /// A valid plan decides every pending request exactly once, assigns only
    /// to ready workers that host the request's model, rejects only requests
    /// no worker hosts, and defers with a reason that holds.
    ///
    /// # Errors
    ///
    /// Returns the first violation found, in action order.
    pub fn validate(&self, snapshot: &ClusterSnapshot) -> Result<(), ClusterPlanError> {
        if self.generation != snapshot.generation {
            return Err(ClusterPlanError::Stale {
                plan: self.generation,
                snapshot: snapshot.generation,
            });
        }
        let mut decided = BTreeSet::new();
        for action in &self.actions {
            let request = action.request();
            let pending = snapshot
                .pending
                .iter()
                .find(|pending| pending.id == request)
                .ok_or(ClusterPlanError::UnknownRequest(request))?;
            if !decided.insert(request) {
                return Err(ClusterPlanError::DuplicateDecision(request));
            }
            let model = &pending.model;
            match action {
                ClusterAction::Assign { worker, .. } => {
                    let view = snapshot.workers.get(worker).ok_or_else(|| {
                        ClusterPlanError::UnknownWorker {
                            request,
                            worker: worker.clone(),
                        }
                    })?;
                    if !view.summary.models.contains(model) {
                        return Err(ClusterPlanError::ModelNotHosted {
                            request,
                            worker: worker.clone(),
                            model: model.clone(),
                        });
                    }
                    if !view.accepts(model) {
                        return Err(ClusterPlanError::WorkerNotReady {
                            request,
                            worker: worker.clone(),
                        });
                    }
                }
                ClusterAction::Defer {
                    reason: DeferReason::NoReadyWorker { model: claimed },
                    ..
                } => {
                    let holds = claimed == model
                        && snapshot.hosts_model(model)
                        && snapshot.eligible_workers(model).next().is_none();
                    if !holds {
                        return Err(ClusterPlanError::ReasonDoesNotHold(request));
                    }
                }
                ClusterAction::Reject {
                    reason: RejectReason::ModelNotServed { model: claimed },
                    ..
                } => {
                    if claimed != model || snapshot.hosts_model(model) {
                        return Err(ClusterPlanError::ReasonDoesNotHold(request));
                    }
                }
            }
        }
        if let Some(missing) = snapshot
            .pending
            .iter()
            .find(|pending| !decided.contains(&pending.id))
        {
            return Err(ClusterPlanError::MissingDecision(missing.id));
        }
        Ok(())
    }
}

/// A cluster plan that cannot be committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClusterPlanError {
    /// The plan was built from a different coordinator generation.
    Stale {
        /// Generation the plan carries.
        plan: u64,
        /// Generation of the snapshot it was checked against.
        snapshot: u64,
    },
    /// The plan decides a request that is not pending.
    UnknownRequest(ClusterRequestId),
    /// The plan decides a request twice.
    DuplicateDecision(ClusterRequestId),
    /// The plan leaves a pending request undecided.
    MissingDecision(ClusterRequestId),
    /// The plan assigns to a worker that has not joined.
    UnknownWorker {
        /// Request assigned.
        request: ClusterRequestId,
        /// Unknown worker.
        worker: WorkerId,
    },
    /// The plan assigns to a worker that does not host the model.
    ModelNotHosted {
        /// Request assigned.
        request: ClusterRequestId,
        /// Worker chosen.
        worker: WorkerId,
        /// Model the request needs.
        model: ModelId,
    },
    /// The plan assigns to a worker that is not accepting work.
    WorkerNotReady {
        /// Request assigned.
        request: ClusterRequestId,
        /// Worker chosen.
        worker: WorkerId,
    },
    /// A deferral or rejection gives a reason the snapshot contradicts.
    ReasonDoesNotHold(ClusterRequestId),
}

impl fmt::Display for ClusterPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stale { plan, snapshot } => write!(
                formatter,
                "plan built from generation {plan} but coordinator is at {snapshot}"
            ),
            Self::UnknownRequest(request) => {
                write!(
                    formatter,
                    "plan decides request `{request}`, which is not pending"
                )
            }
            Self::DuplicateDecision(request) => {
                write!(formatter, "plan decides request `{request}` twice")
            }
            Self::MissingDecision(request) => {
                write!(
                    formatter,
                    "plan leaves pending request `{request}` undecided"
                )
            }
            Self::UnknownWorker { request, worker } => write!(
                formatter,
                "plan assigns request `{request}` to unknown worker `{worker}`"
            ),
            Self::ModelNotHosted {
                request,
                worker,
                model,
            } => write!(
                formatter,
                "plan assigns request `{request}` to worker `{worker}`, which does not host `{model}`"
            ),
            Self::WorkerNotReady { request, worker } => write!(
                formatter,
                "plan assigns request `{request}` to worker `{worker}`, which is not ready"
            ),
            Self::ReasonDoesNotHold(request) => write!(
                formatter,
                "plan gives request `{request}` a deferral or rejection reason that does not hold"
            ),
        }
    }
}

impl Error for ClusterPlanError {}
