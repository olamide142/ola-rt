//! The sans-IO cluster coordinator.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
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
    RouteCandidate, RouteReason, RouteRecord, WorkerHealth, WorkerRole, WorkerSummary, WorkerView,
};

const REQUEST_OBSERVATION_WINDOW: usize = 256;

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
    /// Finalize a source-side migration after the target accepted it.
    CommitMigration {
        /// Cluster-wide identity.
        request: ClusterRequestId,
    },
    /// Restore a source-side migration after target admission failed.
    AbortMigration {
        /// Cluster-wide identity.
        request: ClusterRequestId,
    },
    /// Pushes a validated adaptive-routing parameter update to a worker.
    PolicyUpdate {
        /// Load-pressure weight.
        load_weight: u64,
        /// KV-headroom weight.
        kv_headroom_weight: u64,
        /// Network-cost weight.
        network_weight: u64,
    },
    /// Changes the worker's prefill/decode disaggregation role.
    SetRole {
        /// New execution role.
        role: WorkerRole,
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
        /// Text this token completes, decoded on the worker; may be empty
        /// while a multi-token character is incomplete. Concatenated deltas
        /// equal the final text.
        #[serde(default)]
        text: String,
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
        /// Prompt tokens the worker's model accepted.
        #[serde(default)]
        prompt_tokens: usize,
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
        /// Network-cost weight.
        network_weight: u64,
    },
    /// Confirms a worker role change.
    RoleUpdated {
        /// Role now active on the worker.
        role: WorkerRole,
    },
}

impl WorkerMessage {
    /// The cluster request the message reports on, if any.
    pub const fn request(&self) -> Option<ClusterRequestId> {
        match self {
            Self::Accepted { request, .. }
            | Self::Refused { request, .. }
            | Self::Token { request, .. }
            | Self::Finished { request, .. }
            | Self::Failed { request, .. }
            | Self::Cancelled { request }
            | Self::Migrated { request, .. } => Some(*request),
            Self::Summary(_) | Self::PolicyUpdated { .. } | Self::RoleUpdated { .. } => None,
        }
    }
}

/// What a client sees about its request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientEvent {
    /// A token was produced.
    Token {
        /// Selected token.
        token_id: u32,
        /// Text this token completes; see [`WorkerMessage::Token`].
        text: String,
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
        /// Prompt tokens the worker's model accepted.
        prompt_tokens: usize,
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
    /// A former owner reported on a request that recovery had moved away;
    /// the coordinator asked it to cancel its stale copy.
    Superseded {
        /// Former owner still running the request.
        worker: WorkerId,
    },
    /// A migration was accepted by its target worker.
    Migrated {
        /// Source owner.
        from: WorkerId,
        /// Target owner.
        to: WorkerId,
        /// Handoff mode selected by the source payload.
        mode: String,
        /// Architecture-specific session bytes transferred, if any.
        bytes: u64,
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
    /// Completed request measurements for downstream consumers.
    pub observations: Vec<ClusterRequestObservation>,
    /// Worker state snapshots received while producing this output.
    pub worker_observations: Vec<ClusterWorkerObservation>,
    /// Confirmed migration handoffs produced while handling this output.
    pub migration_observations: Vec<ClusterMigrationObservation>,
    /// Worker policy updates confirmed while handling this output.
    pub policy_update_observations: Vec<ClusterPolicyUpdateObservation>,
}

impl CoordinatorOutput {
    /// Appends `other` after this output.
    pub fn extend(&mut self, other: Self) {
        self.commands.extend(other.commands);
        self.client.extend(other.client);
        self.trace.extend(other.trace);
        self.observations.extend(other.observations);
        self.worker_observations.extend(other.worker_observations);
        self.migration_observations
            .extend(other.migration_observations);
        self.policy_update_observations
            .extend(other.policy_update_observations);
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

/// Terminal outcome measured by the coordinator for one cluster request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterRequestOutcome {
    /// The request produced its complete response.
    Completed,
    /// A worker refused admission or execution failed.
    Failed,
    /// The request was cancelled.
    Cancelled,
}

/// Coordinator-time measurements emitted once when a request becomes terminal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterRequestObservation {
    /// Cluster-wide request identity.
    pub request: ClusterRequestId,
    /// Worker that handled the final attempt, if any.
    pub worker: Option<WorkerId>,
    /// Routing decisions applied to this request, in order (including recovery reroutes).
    pub routes: Vec<ClusterRouteObservation>,
    /// Terminal result.
    pub outcome: ClusterRequestOutcome,
    /// Arrival to first worker acceptance, in microseconds.
    pub queue_micros: Option<u64>,
    /// Arrival to first generated token, in microseconds.
    pub time_to_first_token_micros: Option<u64>,
    /// Arrival to terminal outcome, in microseconds, including failures and cancellations.
    pub elapsed_micros: u64,
    /// Generated token count known at completion.
    pub generated_tokens: usize,
}

/// One recorded assignment and the candidates considered by its policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterRouteObservation {
    /// Worker selected by the plan.
    pub selected_worker: WorkerId,
    /// Candidate workers and the policy's stated reason.
    pub decision: RouteRecord,
}

/// A worker summary as observed by the coordinator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterWorkerObservation {
    /// Worker the summary describes.
    pub worker: WorkerId,
    /// Coordinator time at which the summary was received.
    pub received_at: Timestamp,
    /// Worker summary, including health, load, roles, models, and KV pools.
    pub summary: WorkerSummary,
}

/// A successfully committed request handoff between workers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterMigrationObservation {
    /// Request that moved.
    pub request: ClusterRequestId,
    /// Whether ownership moved or the handoff returned to its source.
    pub outcome: ClusterMigrationOutcome,
    /// Worker that exported the session.
    pub from: WorkerId,
    /// Worker that accepted the session.
    pub to: WorkerId,
    /// Whether the handoff transferred session state or recomputed it.
    pub mode: String,
    /// Session bytes sent, zero for recompute.
    pub bytes: u64,
    /// Coordinator time at commit.
    pub at: Option<Timestamp>,
}

/// Result of a staged migration attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterMigrationOutcome {
    /// Target accepted the session and ownership moved.
    Committed,
    /// Target refused or was lost, so the source resumed ownership.
    RolledBack,
}

/// A worker acknowledgement of a routing policy update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterPolicyUpdateObservation {
    /// Worker that accepted the update.
    pub worker: WorkerId,
    /// Coordinator time at acknowledgement.
    pub at: Timestamp,
    /// Accepted load pressure weight.
    pub load_weight: u64,
    /// Accepted KV headroom weight.
    pub kv_headroom_weight: u64,
    /// Accepted network cost weight.
    pub network_weight: u64,
}

/// Bounded distributed feedback available to a cluster adaptation controller.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClusterFeedbackSnapshot {
    /// Active routing policy and its current configuration.
    pub policy: PolicyDescriptor,
    /// Recent terminal request outcomes, oldest first.
    pub requests: Vec<ClusterRequestObservation>,
    /// Recent worker-state snapshots and coordinator state changes.
    pub workers: Vec<ClusterWorkerObservation>,
    /// Recent committed or rolled-back migration attempts.
    pub migrations: Vec<ClusterMigrationObservation>,
    /// Recent worker policy-update acknowledgements.
    pub policy_updates: Vec<ClusterPolicyUpdateObservation>,
}

impl ClusterRequestObservation {
    const fn with_generated_tokens(mut self, generated_tokens: usize) -> Self {
        self.generated_tokens = generated_tokens;
        self
    }
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
    /// Highest `generated_tokens` already delivered to the client. A
    /// recovered request restarts on a new worker; its tokens up to here are
    /// replays and are not delivered again.
    delivered_tokens: usize,
    /// Workers that owned the request before recovery moved it. Their late
    /// reports are stale rather than protocol errors.
    former_owners: Vec<WorkerId>,
    accepted_at: Option<Timestamp>,
    first_token_at: Option<Timestamp>,
    routes: Vec<ClusterRouteObservation>,
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
    migrations: BTreeMap<ClusterRequestId, StagedMigration>,
    network_costs_us: BTreeMap<WorkerId, u64>,
    /// Bounded coordinator history for recent adaptive observations.
    request_observations: VecDeque<ClusterRequestObservation>,
    worker_observations: VecDeque<ClusterWorkerObservation>,
    migration_observations: VecDeque<ClusterMigrationObservation>,
    policy_update_observations: VecDeque<ClusterPolicyUpdateObservation>,
    /// Commands produced outside a call that returns output, such as
    /// migration aborts during recovery; see [`Coordinator::take_commands`].
    outbox: Vec<(WorkerId, WorkerCommand)>,
}

/// A migration whose target has been asked to admit the request.
#[derive(Clone)]
struct StagedMigration {
    source: WorkerId,
    source_local: RequestId,
    target: WorkerId,
    mode: String,
    bytes: u64,
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
            migrations: BTreeMap::new(),
            network_costs_us: BTreeMap::new(),
            request_observations: VecDeque::new(),
            worker_observations: VecDeque::new(),
            migration_observations: VecDeque::new(),
            policy_update_observations: VecDeque::new(),
            outbox: Vec::new(),
        }
    }

    /// The cluster policy's descriptor.
    pub fn policy_descriptor(&self) -> PolicyDescriptor {
        self.policy.descriptor()
    }

    /// Validates and installs adaptive routing weights atomically, returning
    /// the descriptor of the configuration that is now active.
    ///
    /// The coordinator generation advances only after the policy accepts the
    /// proposal, so future snapshots reflect the new policy version.
    pub fn update_policy_weights(
        &mut self,
        weights: crate::ClusterWeights,
    ) -> Result<PolicyDescriptor, ClusterPolicyError> {
        self.policy.update_weights(weights)?;
        self.generation = self.generation.saturating_add(1);
        Ok(self.policy_descriptor())
    }

    /// Recent terminal request observations, oldest first, capped at 256.
    pub fn recent_request_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterRequestObservation> {
        self.request_observations.iter()
    }

    /// Recent worker summaries received by the coordinator, oldest first.
    pub fn recent_worker_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterWorkerObservation> {
        self.worker_observations.iter()
    }

    /// Recent committed migrations, oldest first.
    pub fn recent_migration_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterMigrationObservation> {
        self.migration_observations.iter()
    }

    /// Recent acknowledged worker policy updates, oldest first.
    pub fn recent_policy_update_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterPolicyUpdateObservation> {
        self.policy_update_observations.iter()
    }

    /// Clones all bounded observation histories into one controller input.
    pub fn feedback_snapshot(&self) -> ClusterFeedbackSnapshot {
        ClusterFeedbackSnapshot {
            policy: self.policy_descriptor(),
            requests: self.request_observations.iter().cloned().collect(),
            workers: self.worker_observations.iter().cloned().collect(),
            migrations: self.migration_observations.iter().cloned().collect(),
            policy_updates: self.policy_update_observations.iter().cloned().collect(),
        }
    }

    fn record_request_observation(
        &mut self,
        output: &mut CoordinatorOutput,
        observation: ClusterRequestObservation,
    ) {
        self.policy.observe_request(&observation);
        self.request_observations.push_back(observation.clone());
        if self.request_observations.len() > REQUEST_OBSERVATION_WINDOW {
            self.request_observations.pop_front();
        }
        output.observations.push(observation);
    }

    fn record_worker_observation(
        &mut self,
        output: &mut CoordinatorOutput,
        observation: ClusterWorkerObservation,
    ) {
        self.retain_worker_observation(observation.clone());
        output.worker_observations.push(observation);
    }

    fn retain_worker_observation(&mut self, observation: ClusterWorkerObservation) {
        self.policy.observe_worker(&observation);
        self.worker_observations.push_back(observation);
        if self.worker_observations.len() > REQUEST_OBSERVATION_WINDOW {
            self.worker_observations.pop_front();
        }
    }

    fn record_migration_observation(
        &mut self,
        output: &mut CoordinatorOutput,
        observation: ClusterMigrationObservation,
    ) {
        self.policy.observe_migration(&observation);
        self.migration_observations.push_back(observation.clone());
        if self.migration_observations.len() > REQUEST_OBSERVATION_WINDOW {
            self.migration_observations.pop_front();
        }
        output.migration_observations.push(observation);
    }

    fn record_policy_update_observation(
        &mut self,
        output: &mut CoordinatorOutput,
        observation: ClusterPolicyUpdateObservation,
    ) {
        self.policy.observe_policy_update(&observation);
        self.policy_update_observations
            .push_back(observation.clone());
        if self.policy_update_observations.len() > REQUEST_OBSERVATION_WINDOW {
            self.policy_update_observations.pop_front();
        }
        output.policy_update_observations.push(observation);
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
        let worker = summary.worker.clone();
        let received_at = summary.reported_at;
        self.workers.insert(
            worker.clone(),
            WorkerEntry {
                last_heartbeat: summary.reported_at,
                summary: summary.clone(),
                epoch: 0,
                assigned: 0,
                in_flight_tokens: VecDeque::new(),
            },
        );
        let observation = ClusterWorkerObservation {
            worker,
            received_at,
            summary: summary.clone(),
        };
        self.retain_worker_observation(observation);
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
        let was_lost = entry.summary.health == WorkerHealth::Lost;
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
        if was_lost {
            let summary = self.workers[worker].summary.clone();
            self.retain_worker_observation(ClusterWorkerObservation {
                worker: worker.clone(),
                received_at: at,
                summary,
            });
        }
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
        let mut observations = Vec::new();
        for worker in &expired {
            if let Some(entry) = self.workers.get_mut(worker) {
                entry.summary.health = WorkerHealth::Lost;
                observations.push(ClusterWorkerObservation {
                    worker: worker.clone(),
                    received_at: now,
                    summary: entry.summary.clone(),
                });
            }
        }
        for observation in observations {
            self.retain_worker_observation(observation);
        }
        if !expired.is_empty() {
            self.generation += 1;
        }
        expired
    }

    /// Requeues requests owned by workers whose leases have expired.
    ///
    /// The request specification remains in coordinator state, so a new plan
    /// can admit it on another worker. This is the recompute recovery path:
    /// the new owner regenerates from the prompt, and tokens the client
    /// already received are not delivered again. That replay matches only
    /// when generation is deterministic (greedy or seeded sampling). A staged
    /// migration whose target is lost is rolled back to its source instead;
    /// the resulting `AbortMigration` command is in [`take_commands`].
    ///
    /// [`take_commands`]: Self::take_commands
    pub fn recover_lost_requests(
        &mut self,
        now: Timestamp,
        timeout: Duration,
    ) -> Vec<ClusterRequestId> {
        let expired = self.expire_workers(now, timeout);
        if expired.is_empty() {
            return Vec::new();
        }
        self.requeue_owned_by(&expired.into_iter().collect(), now)
    }

    /// Marks one worker lost and requeues its owned requests immediately.
    ///
    /// Virtual-time simulations and connection loss use this to model a
    /// worker disappearing without waiting for lease expiry. Recovery
    /// follows [`recover_lost_requests`](Self::recover_lost_requests).
    pub fn recover_worker(
        &mut self,
        worker: &WorkerId,
        now: Timestamp,
    ) -> Result<Vec<ClusterRequestId>, CoordinatorError> {
        let entry = self
            .workers
            .get_mut(worker)
            .ok_or_else(|| CoordinatorError::UnknownWorker(worker.clone()))?;
        entry.summary.health = WorkerHealth::Lost;
        let observation = ClusterWorkerObservation {
            worker: worker.clone(),
            received_at: now,
            summary: entry.summary.clone(),
        };
        self.retain_worker_observation(observation);
        self.generation = self.generation.saturating_add(1);
        Ok(self.requeue_owned_by(&BTreeSet::from([worker.clone()]), now))
    }

    fn requeue_owned_by(
        &mut self,
        lost: &BTreeSet<WorkerId>,
        now: Timestamp,
    ) -> Vec<ClusterRequestId> {
        // A staged migration to a lost target goes back to a live source,
        // which still holds the exported state.
        let rolled_back = self
            .migrations
            .iter()
            .filter(|(_, staged)| lost.contains(&staged.target) && !lost.contains(&staged.source))
            .map(|(request, staged)| (*request, staged.source.clone()))
            .collect::<Vec<_>>();
        for (request, source) in rolled_back {
            if self
                .rollback_migration_at(request, &source, Some(now))
                .is_ok()
            {
                self.outbox
                    .push((source, WorkerCommand::AbortMigration { request }));
            }
        }
        let mut recovered = Vec::new();
        for (request, entry) in &mut self.requests {
            let Some(owner) = entry.status.owner().filter(|owner| lost.contains(*owner)) else {
                continue;
            };
            entry.former_owners.push(owner.clone());
            entry.status = ClusterRequestStatus::Pending;
            self.pending.push(*request);
            self.migrations.remove(request);
            recovered.push(*request);
        }
        let _ = now;
        self.generation = self.generation.saturating_add(recovered.len() as u64);
        recovered
    }

    /// Drains commands produced by recovery, which has no output of its own.
    /// Deliver them before the next plan's commands.
    pub fn take_commands(&mut self) -> Vec<(WorkerId, WorkerCommand)> {
        std::mem::take(&mut self.outbox)
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
                delivered_tokens: 0,
                former_owners: Vec::new(),
                accepted_at: None,
                first_token_at: None,
                routes: Vec::new(),
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
                let observation =
                    self.request_observation(id, None, ClusterRequestOutcome::Cancelled, at);
                self.record_request_observation(&mut output, observation);
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

    fn request_observation(
        &self,
        request: ClusterRequestId,
        worker: Option<WorkerId>,
        outcome: ClusterRequestOutcome,
        at: Timestamp,
    ) -> ClusterRequestObservation {
        let entry = &self.requests[&request];
        ClusterRequestObservation {
            request,
            worker,
            routes: entry.routes.clone(),
            outcome,
            queue_micros: entry.accepted_at.map(|accepted| {
                accepted
                    .saturating_duration_since(entry.arrival_at)
                    .as_micros() as u64
            }),
            time_to_first_token_micros: entry.first_token_at.map(|first| {
                first
                    .saturating_duration_since(entry.arrival_at)
                    .as_micros() as u64
            }),
            elapsed_micros: at.saturating_duration_since(entry.arrival_at).as_micros() as u64,
            generated_tokens: entry.delivered_tokens,
        }
    }

    /// Supplies a calibrated network cost for a worker to the next plan.
    pub fn set_network_cost(&mut self, worker: WorkerId, cost_us: u64) {
        self.network_costs_us.insert(worker, cost_us);
        self.generation = self.generation.saturating_add(1);
    }

    /// Commits the coordinator side of a migration after the source has
    /// exported the request. The source remains authoritative until this
    /// method is called; the returned command is the target admission step.
    pub fn stage_migration(
        &mut self,
        request: ClusterRequestId,
        source: &WorkerId,
        target: WorkerId,
        state: MigratedRequest,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, CoordinatorError> {
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or(CoordinatorError::UnknownRequest(request))?;
        if !matches!(
            &entry.status,
            ClusterRequestStatus::Running { worker, .. } if worker == source
        ) {
            return Err(CoordinatorError::UnexpectedMessage {
                from: source.clone(),
                request,
            });
        }
        let worker = self
            .workers
            .get_mut(&target)
            .ok_or_else(|| CoordinatorError::UnknownWorker(target.clone()))?;
        if !worker.summary.accepts_migrated(&entry.spec.model) {
            return Err(CoordinatorError::UnexpectedMessage {
                from: target,
                request,
            });
        }
        let source_local = match &entry.status {
            ClusterRequestStatus::Running { local, .. } => *local,
            _ => unreachable!("status checked above"),
        };
        let arrival_at = entry.arrival_at;
        let spec = entry.spec.clone();
        entry.status = ClusterRequestStatus::Assigned {
            worker: target.clone(),
        };
        let (mode, bytes) = state
            .session_state
            .as_ref()
            .map(|state| ("transfer".to_owned(), crate::session_state_bytes(state)))
            .unwrap_or_else(|| ("recompute".to_owned(), 0));
        self.migrations.insert(
            request,
            StagedMigration {
                source: source.clone(),
                source_local,
                target: target.clone(),
                mode,
                bytes,
            },
        );
        worker.assigned = worker.assigned.saturating_add(1);
        worker
            .in_flight_tokens
            .push_back(crate::token_demand(&spec));
        self.generation = self.generation.saturating_add(1);
        let mut output = CoordinatorOutput::default();
        output.commands.push((
            target,
            WorkerCommand::AdmitMigrated {
                request,
                state,
                arrival_at,
            },
        ));
        let _ = at;
        Ok(output)
    }

    /// Rolls a staged migration back to the source owner.
    ///
    /// # Errors
    ///
    /// Fails if no migration of `request` from `source` is staged.
    pub fn rollback_migration(
        &mut self,
        request: ClusterRequestId,
        source: &WorkerId,
    ) -> Result<(), CoordinatorError> {
        self.rollback_migration_at(request, source, None)
    }

    /// Rolls back a staged migration and records the time when known.
    pub fn rollback_migration_at(
        &mut self,
        request: ClusterRequestId,
        source: &WorkerId,
        at: Option<Timestamp>,
    ) -> Result<(), CoordinatorError> {
        let Some(staged) = self.migrations.get(&request).cloned() else {
            return Err(CoordinatorError::UnknownRequest(request));
        };
        if &staged.source != source {
            return Err(CoordinatorError::UnexpectedMessage {
                from: source.clone(),
                request,
            });
        }
        self.migrations.remove(&request);
        if let Some(worker) = self.workers.get_mut(&staged.target) {
            worker.assigned = worker.assigned.saturating_sub(1);
            worker.in_flight_tokens.pop_back();
        }
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or(CoordinatorError::UnknownRequest(request))?;
        entry.status = ClusterRequestStatus::Running {
            worker: staged.source.clone(),
            local: staged.source_local,
        };
        let mut output = CoordinatorOutput::default();
        self.record_migration_observation(
            &mut output,
            ClusterMigrationObservation {
                request,
                outcome: ClusterMigrationOutcome::RolledBack,
                from: staged.source,
                to: staged.target,
                mode: staged.mode,
                bytes: staged.bytes,
                at,
            },
        );
        self.generation = self.generation.saturating_add(1);
        Ok(())
    }

    /// Records a staged migration as committed once its target accepted it.
    ///
    /// The source's load accounting is left alone: it counted the request
    /// when it was admitted there, and the source's summary stops counting
    /// it once the export is committed.
    pub fn finish_migration(
        &mut self,
        request: ClusterRequestId,
        at: Timestamp,
    ) -> CoordinatorOutput {
        let mut output = CoordinatorOutput::default();
        if let Some(staged) = self.migrations.remove(&request) {
            self.record_migration_observation(
                &mut output,
                ClusterMigrationObservation {
                    request,
                    outcome: ClusterMigrationOutcome::Committed,
                    from: staged.source.clone(),
                    to: staged.target.clone(),
                    mode: staged.mode.clone(),
                    bytes: staged.bytes,
                    at: Some(at),
                },
            );
            output.record(
                request,
                None,
                at,
                ClusterTraceEvent::Migrated {
                    from: staged.source,
                    to: staged.target,
                    mode: staged.mode,
                    bytes: staged.bytes,
                },
            );
        }
        self.generation = self.generation.saturating_add(1);
        output
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
                            network_cost_us: self.network_costs_us.get(id).copied().unwrap_or(0),
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

    /// Builds a policy decision for pending requests without committing it.
    ///
    /// Replicated coordinators can propose this value to their consensus log,
    /// then call [`commit_plan`](Self::commit_plan) when it is committed. This
    /// keeps followers from independently executing potentially divergent
    /// policy decisions.
    ///
    /// # Errors
    ///
    /// Fails if the policy cannot plan or its plan is invalid; coordinator
    /// state is unchanged.
    pub fn plan_decision(
        &self,
        at: Timestamp,
    ) -> Result<Option<ClusterDecision>, CoordinatorError> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let snapshot = self.snapshot(at);
        let decision = self
            .policy
            .plan(&snapshot)
            .map_err(CoordinatorError::Policy)?;
        decision
            .plan
            .validate(&snapshot)
            .map_err(CoordinatorError::InvalidPlan)?;
        Ok(Some(decision))
    }

    /// Applies a previously computed, committed policy decision.
    ///
    /// Validation is repeated against current state so stale or invalid
    /// decisions have no effect. The returned effects are not delivered by
    /// this method; a replicated caller should publish them only on its leader.
    ///
    /// # Errors
    ///
    /// Fails if the decision no longer matches current state.
    pub fn commit_plan(
        &mut self,
        decision: ClusterDecision,
        at: Timestamp,
    ) -> Result<(ClusterIteration, CoordinatorOutput), CoordinatorError> {
        let ClusterDecision { plan, record } = decision;
        let snapshot = self.snapshot(at);
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
                    if let Some(route) = record.route(request) {
                        entry.routes.push(ClusterRouteObservation {
                            selected_worker: worker.clone(),
                            decision: route.clone(),
                        });
                    }
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
        Ok((iteration, output))
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
        let Some(decision) = self.plan_decision(at)? else {
            return Ok(None);
        };
        self.commit_plan(decision, at).map(Some)
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
        if let Some(request) = message.request()
            && self.is_stale(from, request)
        {
            // A worker that recovery gave up on came back still running the
            // request. Its reports are dropped, and a live copy is cancelled.
            if matches!(message, WorkerMessage::Accepted { .. }) {
                output.record(
                    request,
                    None,
                    at,
                    ClusterTraceEvent::Superseded {
                        worker: from.clone(),
                    },
                );
            }
            if matches!(
                message,
                WorkerMessage::Accepted { .. } | WorkerMessage::Token { .. }
            ) {
                output
                    .commands
                    .push((from.clone(), WorkerCommand::Cancel { request }));
            }
            return Ok(output);
        }
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
                let summary = entry.summary.clone();
                let in_flight = entry.in_flight();
                while entry.in_flight_tokens.len() > in_flight {
                    entry.in_flight_tokens.pop_front();
                }
                self.generation += 1;
                self.record_worker_observation(
                    &mut output,
                    ClusterWorkerObservation {
                        worker: from.clone(),
                        received_at: at,
                        summary,
                    },
                );
                return Ok(output);
            }
            WorkerMessage::Accepted { request, local } => {
                let entry = self.owned(from, request, false)?;
                entry.accepted_at.get_or_insert(at);
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
                let observation = self.request_observation(
                    request,
                    Some(from.clone()),
                    ClusterRequestOutcome::Failed,
                    at,
                );
                self.record_request_observation(&mut output, observation);
                request
            }
            WorkerMessage::Token {
                request,
                token_id,
                text,
                generated_tokens,
                phase,
                at: token_at,
            } => {
                let entry = self.owned(from, request, true)?;
                if generated_tokens <= entry.delivered_tokens {
                    // A recovered request replaying output the client has.
                    return Ok(output);
                }
                entry.delivered_tokens = generated_tokens;
                // `token_at` is worker-stamped for the client stream. The
                // coordinator's `at` is the common clock domain for policy
                // measurements across workers.
                entry.first_token_at.get_or_insert(at);
                output.client.push((
                    request,
                    ClientEvent::Token {
                        token_id,
                        text,
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
                prompt_tokens,
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
                let observation = self
                    .request_observation(
                        request,
                        Some(from.clone()),
                        ClusterRequestOutcome::Completed,
                        at,
                    )
                    .with_generated_tokens(token_ids.len());
                self.record_request_observation(&mut output, observation);
                output.client.push((
                    request,
                    ClientEvent::Finished {
                        token_ids,
                        text,
                        prompt_tokens,
                    },
                ));
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
                let observation = self.request_observation(
                    request,
                    Some(from.clone()),
                    ClusterRequestOutcome::Failed,
                    at,
                );
                self.record_request_observation(&mut output, observation);
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
                let observation = self.request_observation(
                    request,
                    Some(from.clone()),
                    ClusterRequestOutcome::Cancelled,
                    at,
                );
                self.record_request_observation(&mut output, observation);
                request
            }
            WorkerMessage::Migrated { .. } => {
                // Migration payloads are consumed by the recovery controller;
                // the normal client stream has no terminal event yet.
                return Ok(output);
            }
            WorkerMessage::PolicyUpdated {
                load_weight,
                kv_headroom_weight,
                network_weight,
            } => {
                self.record_policy_update_observation(
                    &mut output,
                    ClusterPolicyUpdateObservation {
                        worker: from.clone(),
                        at,
                        load_weight,
                        kv_headroom_weight,
                        network_weight,
                    },
                );
                return Ok(output);
            }
            WorkerMessage::RoleUpdated { role } => {
                let entry = self
                    .workers
                    .get_mut(from)
                    .ok_or_else(|| CoordinatorError::UnknownWorker(from.clone()))?;
                entry.summary.role = role;
                let summary = entry.summary.clone();
                self.generation = self.generation.saturating_add(1);
                self.record_worker_observation(
                    &mut output,
                    ClusterWorkerObservation {
                        worker: from.clone(),
                        received_at: at,
                        summary,
                    },
                );
                return Ok(output);
            }
        };
        debug_assert!(self.requests.contains_key(&request));
        self.generation += 1;
        Ok(output)
    }

    /// Whether `from` reports on a request it owned before recovery moved it.
    fn is_stale(&self, from: &WorkerId, request: ClusterRequestId) -> bool {
        self.requests.get(&request).is_some_and(|entry| {
            entry.status.owner() != Some(from) && entry.former_owners.contains(from)
        })
    }

    /// The source and target of a staged migration of `request`.
    pub fn staged_migration(&self, request: ClusterRequestId) -> Option<(&WorkerId, &WorkerId)> {
        self.migrations
            .get(&request)
            .map(|staged| (&staged.source, &staged.target))
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

#[cfg(test)]
mod feedback_tests {
    use super::*;
    use crate::{AdaptiveClusterPolicy, ClusterWeights};

    #[test]
    fn feedback_snapshot_contains_active_policy_and_recent_requests() {
        let mut coordinator = Coordinator::new(AdaptiveClusterPolicy::default());
        let observation = ClusterRequestObservation {
            request: ClusterRequestId::new(7),
            worker: None,
            routes: Vec::new(),
            outcome: ClusterRequestOutcome::Cancelled,
            queue_micros: None,
            time_to_first_token_micros: None,
            elapsed_micros: 12,
            generated_tokens: 0,
        };
        let mut output = CoordinatorOutput::default();
        coordinator.record_request_observation(&mut output, observation.clone());

        let initial = coordinator.feedback_snapshot();
        assert_eq!(initial.policy.name, "adaptive-cluster-routing");
        assert_eq!(initial.requests, vec![observation]);
        assert!(initial.workers.is_empty());
        assert!(initial.migrations.is_empty());
        assert!(initial.policy_updates.is_empty());

        let installed = coordinator
            .update_policy_weights(ClusterWeights {
                load: 0,
                kv_headroom: 0,
                network: 3,
            })
            .unwrap();
        assert_eq!(coordinator.feedback_snapshot().policy, installed);
    }
}
