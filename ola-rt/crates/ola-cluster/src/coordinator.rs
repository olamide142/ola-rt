//! The sans-IO cluster coordinator.

use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fmt,
    time::Duration,
};

use ola_core::{ClusterRequestId, RequestError, RequestId, RequestSpec, Timestamp, WorkerId};
use ola_runtime::{ExecutionPhase, MigratedRequest, PolicyDescriptor};
use serde::{Deserialize, Serialize};

use crate::{
    ClusterAction, ClusterDecision, ClusterDecisionRecord, ClusterPlan, ClusterPlanError,
    ClusterPolicy, ClusterPolicyError, ClusterSnapshot, DeferReason, PendingRequest, RejectReason,
    RouteCandidate, RouteReason, WorkerHealth, WorkerSummary, WorkerView,
};

/// An instruction from the coordinator to one worker.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerCommand {
    /// Queue a request the worker now owns.
    Admit {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// The request as submitted.
        spec: RequestSpec,
        /// When the coordinator received it.
        arrival_at: Timestamp,
    },
    /// Cancel a request the worker owns.
    Cancel {
        /// Request to cancel.
        request: ClusterRequestId,
    },
    /// Stop accepting new work while allowing owned requests to finish.
    Drain,
    /// Export an unfinished request, releasing its local KV/session state.
    MigrateOut {
        /// Cluster-wide identity.
        request: ClusterRequestId,
    },
    /// Import a request exported by another worker.
    AdmitMigrated {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// Device-independent resumable state.
        state: MigratedRequest,
        /// Original coordinator arrival time.
        arrival_at: Timestamp,
    },
    /// Pushes a validated adaptive-routing parameter update to a worker.
    PolicyUpdate {
        /// Load-pressure weight.
        load_weight: u64,
        /// KV-headroom weight.
        kv_headroom_weight: u64,
    },
}

/// A report from one worker to the coordinator.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerMessage {
    /// The worker's current load.
    Summary(WorkerSummary),
    /// The worker queued an admitted request under a local ID.
    Accepted {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// The worker's runtime-local ID for it.
        local: RequestId,
    },
    /// The worker would not queue an admitted request.
    Refused {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// Worker diagnostic.
        message: String,
    },
    /// The request produced a token.
    Token {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// Selected token.
        token_id: u32,
        /// Output tokens produced so far, including this one.
        generated_tokens: usize,
        /// Model phase that produced it.
        phase: ExecutionPhase,
        /// When the worker produced it.
        at: Timestamp,
    },
    /// The request completed.
    Finished {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// Every generated token in order.
        token_ids: Vec<u32>,
        /// Tokenizer decoding of `token_ids`.
        text: String,
    },
    /// The request failed on the worker.
    Failed {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// Worker diagnostic.
        message: String,
    },
    /// The worker cancelled the request.
    Cancelled {
        /// Cluster-wide identity.
        request: ClusterRequestId,
    },
    /// An unfinished request was exported and can be admitted elsewhere.
    Migrated {
        /// Cluster-wide identity.
        request: ClusterRequestId,
        /// Device-independent resumable state.
        state: MigratedRequest,
    },
    /// Confirms a worker accepted an adaptive-routing parameter update.
    PolicyUpdated {
        /// Load-pressure weight.
        load_weight: u64,
        /// KV-headroom weight.
        kv_headroom_weight: u64,
    },
}

/// What a client sees about its request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientEvent {
    /// A token was produced.
    Token {
        /// Selected token.
        token_id: u32,
        /// Output tokens produced so far, including this one.
        generated_tokens: usize,
        /// Model phase that produced it.
        phase: ExecutionPhase,
        /// When the worker produced it.
        at: Timestamp,
    },
    /// The request completed.
    Finished {
        /// Every generated token in order.
        token_ids: Vec<u32>,
        /// Tokenizer decoding of `token_ids`.
        text: String,
    },
    /// The request failed.
    Failed {
        /// Diagnostic.
        message: String,
    },
    /// The request was cancelled.
    Cancelled,
}

/// Where a request is in the cluster.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClusterRequestStatus {
    /// Submitted; no worker owns it.
    Pending,
    /// Assigned to a worker that has not yet accepted it.
    Assigned {
        /// Owner.
        worker: WorkerId,
    },
    /// Queued or running on a worker under a local ID.
    Running {
        /// Owner.
        worker: WorkerId,
        /// The owner's runtime-local ID.
        local: RequestId,
    },
    /// Completed successfully.
    Completed,
    /// Failed; no further events follow.
    Failed(String),
    /// Cancelled; no further events follow.
    Cancelled,
}

impl ClusterRequestStatus {
    /// Whether the request can no longer change state.
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Cancelled)
    }

    /// The owning worker, if any.
    pub const fn owner(&self) -> Option<&WorkerId> {
        match self {
            Self::Assigned { worker } | Self::Running { worker, .. } => Some(worker),
            _ => None,
        }
    }
}

/// One cluster-level fact about one request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ClusterTraceRecord {
    /// Cluster request the record is about.
    pub request: u64,
    /// Cluster iteration that produced the record, when one did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration: Option<u64>,
    /// When the fact became true, in microseconds of the coordinator's epoch.
    pub at_us: u64,
    /// What happened.
    pub event: ClusterTraceEvent,
}

/// What a [`ClusterTraceRecord`] records.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClusterTraceEvent {
    /// The coordinator received the request.
    Submitted {
        /// Model it needs.
        model: String,
    },
    /// A plan assigned the request.
    Routed {
        /// New owner.
        worker: WorkerId,
        /// Eligible workers and the load the policy saw.
        candidates: Vec<RouteCandidate>,
        /// Why the policy chose `worker`, if it said.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<RouteReason>,
    },
    /// A plan left the request pending.
    Deferred {
        /// Why.
        reason: DeferReason,
    },
    /// A plan failed the request.
    Rejected {
        /// Why.
        reason: RejectReason,
    },
    /// The owner queued the request; `local_request` joins with the
    /// worker's own traces.
    Accepted {
        /// Owner.
        worker: WorkerId,
        /// The owner's runtime-local request ID.
        local_request: u64,
    },
    /// The owner would not queue the request.
    Refused {
        /// Worker that refused.
        worker: WorkerId,
        /// Its diagnostic.
        message: String,
    },
    /// The request completed.
    Completed {
        /// Owner.
        worker: WorkerId,
        /// Generated tokens.
        tokens: usize,
    },
    /// The request failed on its owner.
    Failed {
        /// Owner.
        worker: WorkerId,
        /// Diagnostic.
        message: String,
    },
    /// The request was cancelled.
    Cancelled {
        /// Owner at cancellation, if one had been assigned.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worker: Option<WorkerId>,
    },
}

/// Everything the caller must deliver after one coordinator call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoordinatorOutput {
    /// Commands for workers, in send order.
    pub commands: Vec<(WorkerId, WorkerCommand)>,
    /// Events for clients, in delivery order.
    pub client: Vec<(ClusterRequestId, ClientEvent)>,
    /// Trace records, in order.
    pub trace: Vec<ClusterTraceRecord>,
}

impl CoordinatorOutput {
    /// Appends `other` after this output.
    pub fn extend(&mut self, other: Self) {
        self.commands.extend(other.commands);
        self.client.extend(other.client);
        self.trace.extend(other.trace);
    }

    fn record(
        &mut self,
        request: ClusterRequestId,
        iteration: Option<u64>,
        at: Timestamp,
        event: ClusterTraceEvent,
    ) {
        self.trace.push(ClusterTraceRecord {
            request: request.get(),
            iteration,
            at_us: at.as_micros(),
            event,
        });
    }
}

/// One committed cluster planning pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterIteration {
    /// Sequential iteration number, starting at one.
    pub id: u64,
    /// Planning time.
    pub at: Timestamp,
    /// The committed plan.
    pub plan: ClusterPlan,
    /// What the policy considered.
    pub decision: ClusterDecisionRecord,
}

/// A coordinator call that could not be applied; nothing changed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoordinatorError {
    /// A worker tried to join with an empty ID.
    EmptyWorkerId,
    /// A worker tried to join twice.
    DuplicateWorker(WorkerId),
    /// A heartbeat came from an older worker process epoch.
    StaleWorkerEpoch {
        /// Worker whose heartbeat was rejected.
        worker: WorkerId,
        /// Older process epoch supplied by the worker.
        epoch: u64,
    },
    /// A message came from a worker that has not joined.
    UnknownWorker(WorkerId),
    /// A summary named a different worker than the one that sent it.
    MismatchedSummary {
        /// Sender.
        from: WorkerId,
        /// Worker the summary describes.
        summary: WorkerId,
    },
    /// The call named a request the coordinator does not know.
    UnknownRequest(ClusterRequestId),
    /// A submitted request failed validation.
    InvalidRequest(RequestError),
    /// A worker reported on a request it does not own, or out of order.
    UnexpectedMessage {
        /// Sender.
        from: WorkerId,
        /// Request named.
        request: ClusterRequestId,
    },
    /// The cluster policy could not plan.
    Policy(ClusterPolicyError),
    /// The cluster policy returned a plan that failed validation.
    InvalidPlan(ClusterPlanError),
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyWorkerId => formatter.write_str("worker ID cannot be empty"),
            Self::DuplicateWorker(worker) => write!(formatter, "worker `{worker}` already joined"),
            Self::StaleWorkerEpoch { worker, epoch } => {
                write!(formatter, "worker `{worker}` sent stale epoch {epoch}")
            }
            Self::UnknownWorker(worker) => write!(formatter, "worker `{worker}` has not joined"),
            Self::MismatchedSummary { from, summary } => write!(
                formatter,
                "worker `{from}` sent a summary for worker `{summary}`"
            ),
            Self::UnknownRequest(request) => write!(formatter, "unknown request `{request}`"),
            Self::InvalidRequest(error) => write!(formatter, "invalid request: {error:?}"),
            Self::UnexpectedMessage { from, request } => write!(
                formatter,
                "worker `{from}` reported on request `{request}`, which it does not own"
            ),
            Self::Policy(error) => error.fmt(formatter),
            Self::InvalidPlan(error) => write!(formatter, "invalid cluster plan: {error}"),
        }
    }
}

impl Error for CoordinatorError {}

struct WorkerEntry {
    summary: WorkerSummary,
    /// Process epoch that owns the current lease.
    epoch: u64,
    /// Coordinator-time heartbeat deadline source.
    last_heartbeat: Timestamp,
    /// Admit commands sent to the worker.
    assigned: u64,
    /// Worst-case KV tokens of the most recent assignments, oldest first;
    /// trimmed to those the latest summary does not yet count.
    in_flight_tokens: VecDeque<usize>,
}

impl WorkerEntry {
    fn in_flight(&self) -> usize {
        usize::try_from(self.assigned.saturating_sub(self.summary.admitted_requests))
            .unwrap_or(usize::MAX)
    }
}

struct RequestEntry {
    spec: RequestSpec,
    arrival_at: Timestamp,
    status: ClusterRequestStatus,
}

/// Owns cluster-level request state and routes work to workers.
///
/// The coordinator performs no I/O. Each call returns a
/// [`CoordinatorOutput`] the caller delivers: commands to workers, events to
/// clients, and trace records.
pub struct Coordinator {
    policy: Box<dyn ClusterPolicy>,
    workers: BTreeMap<WorkerId, WorkerEntry>,
    requests: BTreeMap<ClusterRequestId, RequestEntry>,
    /// Pending requests in arrival order.
    pending: Vec<ClusterRequestId>,
    next_request: u64,
    routed: u64,
    generation: u64,
    iterations: u64,
}

impl Coordinator {
    /// Creates a coordinator with no workers.
    pub fn new<P: ClusterPolicy + 'static>(policy: P) -> Self {
        Self {
            policy: Box::new(policy),
            workers: BTreeMap::new(),
            requests: BTreeMap::new(),
            pending: Vec::new(),
            next_request: 1,
            routed: 0,
            generation: 0,
            iterations: 0,
        }
    }

    /// The cluster policy's descriptor.
    pub fn policy_descriptor(&self) -> PolicyDescriptor {
        self.policy.descriptor()
    }

    /// Adds a worker, described by its first summary.
    ///
    /// # Errors
    ///
    /// Fails for an empty or already-joined worker ID.
    pub fn join(&mut self, summary: WorkerSummary) -> Result<(), CoordinatorError> {
        if summary.worker.is_empty() {
            return Err(CoordinatorError::EmptyWorkerId);
        }
        if self.workers.contains_key(&summary.worker) {
            return Err(CoordinatorError::DuplicateWorker(summary.worker));
        }
        self.workers.insert(
            summary.worker.clone(),
            WorkerEntry {
                last_heartbeat: summary.reported_at,
                summary,
                epoch: 0,
                assigned: 0,
                in_flight_tokens: VecDeque::new(),
            },
        );
        self.generation += 1;
        Ok(())
    }

    /// Records a worker heartbeat in the coordinator's clock domain.
    ///
    /// A larger epoch represents a worker restart and replaces the previous
    /// lease. Heartbeats from an older process are rejected so a delayed
    /// packet cannot revive a restarted worker.
    pub fn heartbeat(
        &mut self,
        worker: &WorkerId,
        epoch: u64,
        at: Timestamp,
    ) -> Result<(), CoordinatorError> {
        let entry = self
            .workers
            .get_mut(worker)
            .ok_or_else(|| CoordinatorError::UnknownWorker(worker.clone()))?;
        if epoch < entry.epoch {
            return Err(CoordinatorError::StaleWorkerEpoch {
                worker: worker.clone(),
                epoch,
            });
        }
        entry.epoch = epoch;
        entry.last_heartbeat = at;
        if entry.summary.health == WorkerHealth::Lost {
            entry.summary.health = WorkerHealth::Ready;
        }
        self.generation += 1;
        Ok(())
    }

    /// Expires workers whose lease has been silent for `timeout`.
    ///
    /// This only changes routing eligibility. Requests remain owned by the
    /// worker until recovery or an explicit migration handoff completes.
    pub fn expire_workers(&mut self, now: Timestamp, timeout: Duration) -> Vec<WorkerId> {
        let expired = self
            .workers
            .iter()
            .filter(|(_, entry)| {
                entry.summary.health != WorkerHealth::Lost
                    && now.saturating_duration_since(entry.last_heartbeat) >= timeout
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for worker in &expired {
            if let Some(entry) = self.workers.get_mut(worker) {
                entry.summary.health = WorkerHealth::Lost;
            }
        }
        if !expired.is_empty() {
            self.generation += 1;
        }
        expired
    }

    /// Requeues requests owned by workers whose leases have expired.
    ///
    /// The request specification remains in coordinator state, so a new plan
    /// can admit it on another worker. This is the recompute recovery path;
    /// callers that have a [`MigratedRequest`] may instead use the explicit
    /// worker migration commands to preserve generated tokens.
    pub fn recover_lost_requests(
        &mut self,
        now: Timestamp,
        timeout: Duration,
    ) -> Vec<ClusterRequestId> {
        let expired = self.expire_workers(now, timeout);
        if expired.is_empty() {
            return Vec::new();
        }
        let lost = expired.iter().collect::<std::collections::BTreeSet<_>>();
        let mut recovered = Vec::new();
        for (request, entry) in &mut self.requests {
            let owner = entry.status.owner();
            if owner.is_some_and(|worker| lost.contains(worker)) {
                entry.status = ClusterRequestStatus::Pending;
                self.pending.push(*request);
                recovered.push(*request);
            }
        }
        self.generation += recovered.len() as u64;
        recovered
    }

    /// Latest summary of each joined worker, in ID order.
    pub fn workers(&self) -> impl ExactSizeIterator<Item = &WorkerSummary> {
        self.workers.values().map(|entry| &entry.summary)
    }

    /// Validates and queues a request for routing.
    ///
    /// # Errors
    ///
    /// Fails if the request is invalid; nothing is queued.
    pub fn submit(
        &mut self,
        spec: RequestSpec,
        at: Timestamp,
    ) -> Result<(ClusterRequestId, CoordinatorOutput), CoordinatorError> {
        spec.validate().map_err(CoordinatorError::InvalidRequest)?;
        let id = ClusterRequestId::new(self.next_request);
        self.next_request += 1;
        let mut output = CoordinatorOutput::default();
        output.record(
            id,
            None,
            at,
            ClusterTraceEvent::Submitted {
                model: spec.model.to_string(),
            },
        );
        self.requests.insert(
            id,
            RequestEntry {
                spec,
                arrival_at: at,
                status: ClusterRequestStatus::Pending,
            },
        );
        self.pending.push(id);
        self.generation += 1;
        Ok((id, output))
    }

    /// Cancels a request.
    ///
    /// A pending request is cancelled at once. An owned request is cancelled
    /// by its worker, which reports back. Cancelling a terminal request does
    /// nothing.
    ///
    /// # Errors
    ///
    /// Fails for an unknown request.
    pub fn cancel(
        &mut self,
        id: ClusterRequestId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, CoordinatorError> {
        let entry = self
            .requests
            .get_mut(&id)
            .ok_or(CoordinatorError::UnknownRequest(id))?;
        let mut output = CoordinatorOutput::default();
        match &entry.status {
            ClusterRequestStatus::Pending => {
                entry.status = ClusterRequestStatus::Cancelled;
                self.pending.retain(|pending| *pending != id);
                self.generation += 1;
                output.client.push((id, ClientEvent::Cancelled));
                output.record(id, None, at, ClusterTraceEvent::Cancelled { worker: None });
            }
            ClusterRequestStatus::Assigned { worker }
            | ClusterRequestStatus::Running { worker, .. } => {
                output
                    .commands
                    .push((worker.clone(), WorkerCommand::Cancel { request: id }));
            }
            _ => {}
        }
        Ok(output)
    }

    /// Where a request is.
    pub fn status(&self, id: ClusterRequestId) -> Option<&ClusterRequestStatus> {
        self.requests.get(&id).map(|entry| &entry.status)
    }

    /// Unowned requests waiting for a plan.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Builds the snapshot the next plan would see.
    pub fn snapshot(&self, at: Timestamp) -> ClusterSnapshot {
        ClusterSnapshot {
            at,
            generation: self.generation,
            workers: self
                .workers
                .iter()
                .map(|(id, entry)| {
                    let in_flight = entry.in_flight();
                    let in_flight_tokens =
                        entry.in_flight_tokens.iter().rev().take(in_flight).sum();
                    (
                        id.clone(),
                        WorkerView {
                            summary: entry.summary.clone(),
                            in_flight,
                            in_flight_tokens,
                        },
                    )
                })
                .collect(),
            pending: self
                .pending
                .iter()
                .map(|id| {
                    let entry = &self.requests[id];
                    PendingRequest::new(*id, &entry.spec, entry.arrival_at)
                })
                .collect(),
            routed_requests: self.routed,
        }
    }

    /// Plans and commits one cluster iteration, if any request is pending.
    ///
    /// # Errors
    ///
    /// Fails if the policy cannot plan or its plan is invalid; nothing
    /// changes.
    pub fn plan(
        &mut self,
        at: Timestamp,
    ) -> Result<Option<(ClusterIteration, CoordinatorOutput)>, CoordinatorError> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let snapshot = self.snapshot(at);
        let ClusterDecision { plan, record } = self
            .policy
            .plan(&snapshot)
            .map_err(CoordinatorError::Policy)?;
        plan.validate(&snapshot)
            .map_err(CoordinatorError::InvalidPlan)?;

        let id = self.iterations + 1;
        let mut output = CoordinatorOutput::default();
        for action in &plan.actions {
            let request = action.request();
            match action {
                ClusterAction::Assign { worker, .. } => {
                    let entry = self
                        .requests
                        .get_mut(&request)
                        .expect("validated plans decide only pending requests");
                    entry.status = ClusterRequestStatus::Assigned {
                        worker: worker.clone(),
                    };
                    let target = self
                        .workers
                        .get_mut(worker)
                        .expect("validated plans assign only to joined workers");
                    target.assigned += 1;
                    target
                        .in_flight_tokens
                        .push_back(crate::token_demand(&entry.spec));
                    self.routed += 1;
                    output.commands.push((
                        worker.clone(),
                        WorkerCommand::Admit {
                            request,
                            spec: entry.spec.clone(),
                            arrival_at: entry.arrival_at,
                        },
                    ));
                    let route = record.route(request);
                    let candidates = route
                        .map(|route| route.candidates.clone())
                        .unwrap_or_default();
                    output.record(
                        request,
                        Some(id),
                        at,
                        ClusterTraceEvent::Routed {
                            worker: worker.clone(),
                            candidates,
                            reason: route.map(|route| route.reason),
                        },
                    );
                }
                ClusterAction::Defer { reason, .. } => {
                    output.record(
                        request,
                        Some(id),
                        at,
                        ClusterTraceEvent::Deferred {
                            reason: reason.clone(),
                        },
                    );
                }
                ClusterAction::Reject { reason, .. } => {
                    let entry = self
                        .requests
                        .get_mut(&request)
                        .expect("validated plans decide only pending requests");
                    entry.status = ClusterRequestStatus::Failed(reason.to_string());
                    output.client.push((
                        request,
                        ClientEvent::Failed {
                            message: reason.to_string(),
                        },
                    ));
                    output.record(
                        request,
                        Some(id),
                        at,
                        ClusterTraceEvent::Rejected {
                            reason: reason.clone(),
                        },
                    );
                }
            }
        }
        let requests = &self.requests;
        self.pending
            .retain(|pending| requests[pending].status == ClusterRequestStatus::Pending);
        self.iterations = id;
        self.generation += 1;
        let iteration = ClusterIteration {
            id,
            at,
            plan,
            decision: record,
        };
        self.policy.observe(&iteration);
        Ok(Some((iteration, output)))
    }

    /// Applies one message from a worker.
    ///
    /// # Errors
    ///
    /// Fails if the worker has not joined, or reports on a request it does
    /// not own or out of order; nothing changes.
    pub fn handle(
        &mut self,
        from: &WorkerId,
        message: WorkerMessage,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, CoordinatorError> {
        if !self.workers.contains_key(from) {
            return Err(CoordinatorError::UnknownWorker(from.clone()));
        }
        let mut output = CoordinatorOutput::default();
        let request = match message {
            WorkerMessage::Summary(summary) => {
                if &summary.worker != from {
                    return Err(CoordinatorError::MismatchedSummary {
                        from: from.clone(),
                        summary: summary.worker,
                    });
                }
                let entry = self.workers.get_mut(from).expect("checked above");
                entry.summary = summary;
                let in_flight = entry.in_flight();
                while entry.in_flight_tokens.len() > in_flight {
                    entry.in_flight_tokens.pop_front();
                }
                self.generation += 1;
                return Ok(output);
            }
            WorkerMessage::Accepted { request, local } => {
                let entry = self.owned(from, request, false)?;
                entry.status = ClusterRequestStatus::Running {
                    worker: from.clone(),
                    local,
                };
                output.record(
                    request,
                    None,
                    at,
                    ClusterTraceEvent::Accepted {
                        worker: from.clone(),
                        local_request: local.get(),
                    },
                );
                request
            }
            WorkerMessage::Refused { request, message } => {
                let entry = self.owned(from, request, false)?;
                entry.status = ClusterRequestStatus::Failed(message.clone());
                output.client.push((
                    request,
                    ClientEvent::Failed {
                        message: message.clone(),
                    },
                ));
                output.record(
                    request,
                    None,
                    at,
                    ClusterTraceEvent::Refused {
                        worker: from.clone(),
                        message,
                    },
                );
                request
            }
            WorkerMessage::Token {
                request,
                token_id,
                generated_tokens,
                phase,
                at: token_at,
            } => {
                self.owned(from, request, true)?;
                output.client.push((
                    request,
                    ClientEvent::Token {
                        token_id,
                        generated_tokens,
                        phase,
                        at: token_at,
                    },
                ));
                return Ok(output);
            }
            WorkerMessage::Finished {
                request,
                token_ids,
                text,
            } => {
                let entry = self.owned(from, request, true)?;
                entry.status = ClusterRequestStatus::Completed;
                output.record(
                    request,
                    None,
                    at,
                    ClusterTraceEvent::Completed {
                        worker: from.clone(),
                        tokens: token_ids.len(),
                    },
                );
                output
                    .client
                    .push((request, ClientEvent::Finished { token_ids, text }));
                request
            }
            WorkerMessage::Failed { request, message } => {
                let entry = self.owned(from, request, true)?;
                entry.status = ClusterRequestStatus::Failed(message.clone());
                output.client.push((
                    request,
                    ClientEvent::Failed {
                        message: message.clone(),
                    },
                ));
                output.record(
                    request,
                    None,
                    at,
                    ClusterTraceEvent::Failed {
                        worker: from.clone(),
                        message,
                    },
                );
                request
            }
            WorkerMessage::Cancelled { request } => {
                let entry = self.owned(from, request, false)?;
                entry.status = ClusterRequestStatus::Cancelled;
                output.client.push((request, ClientEvent::Cancelled));
                output.record(
                    request,
                    None,
                    at,
                    ClusterTraceEvent::Cancelled {
                        worker: Some(from.clone()),
                    },
                );
                request
            }
            WorkerMessage::Migrated { .. } => {
                // Migration payloads are consumed by the recovery controller;
                // the normal client stream has no terminal event yet.
                return Ok(output);
            }
            WorkerMessage::PolicyUpdated { .. } => return Ok(output),
        };
        debug_assert!(self.requests.contains_key(&request));
        self.generation += 1;
        Ok(output)
    }

    /// The entry for a request `from` owns. With `accepted`, the worker must
    /// already have accepted it.
    fn owned(
        &mut self,
        from: &WorkerId,
        request: ClusterRequestId,
        accepted: bool,
    ) -> Result<&mut RequestEntry, CoordinatorError> {
        let unexpected = || CoordinatorError::UnexpectedMessage {
            from: from.clone(),
            request,
        };
        let entry = self.requests.get_mut(&request).ok_or_else(unexpected)?;
        let owns = match &entry.status {
            ClusterRequestStatus::Assigned { worker } => !accepted && worker == from,
            ClusterRequestStatus::Running { worker, .. } => worker == from,
            _ => false,
        };
        if owns { Ok(entry) } else { Err(unexpected()) }
    }
}
