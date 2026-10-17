use std::collections::{BTreeMap, BTreeSet, VecDeque};

use ola_core::{ModelId, Request, RequestId, Timestamp};

use crate::ModelRegistry;

#[derive(Clone, Debug, Eq, PartialEq)]
/// Authoritative lifecycle state for a request.
pub enum RequestStatus {
    /// Admitted and eligible for scheduler selection.
    Queued,
    /// Selected and owned by an executor.
    Running,
    /// Successfully reached its terminal output.
    Completed,
    /// Cancelled before or during execution.
    Cancelled,
    /// Terminated with a diagnostic message.
    Failed(String),
}

impl RequestStatus {
    /// Returns whether the request can no longer change state.
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed(_))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Monotonic lifecycle timestamps used to derive latency metrics.
pub struct RequestTiming {
    /// Time at which the request entered the runtime queue.
    pub queued_at: Timestamp,
    /// Time at which the scheduler moved the request to running.
    pub started_at: Option<Timestamp>,
    /// Time at which the first generated token became observable.
    pub first_token_at: Option<Timestamp>,
    /// Time at which the request entered any terminal state.
    pub completed_at: Option<Timestamp>,
}

#[derive(Clone, Debug, PartialEq)]
/// Request, lifecycle status, and measurements retained by the runtime.
pub struct RequestRecord {
    /// Validated admitted request.
    pub request: Request,
    /// Current lifecycle state.
    pub status: RequestStatus,
    /// Lifecycle timestamps.
    pub timing: RequestTiming,
    /// Times the request was preempted back to the queue and resumed.
    pub preemptions: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Device-memory accounting visible to admission and scheduling policies.
pub struct MemoryState {
    /// Total allocatable capacity.
    pub capacity_bytes: u64,
    /// Bytes occupied by resident model weights.
    pub weights_bytes: u64,
    /// Bytes occupied by live activations.
    pub activations_bytes: u64,
    /// Bytes occupied by KV-cache state.
    pub kv_cache_bytes: u64,
    /// Bytes reserved for backend workspaces and temporary buffers.
    pub workspace_bytes: u64,
}

impl MemoryState {
    /// Returns accounted memory, using saturation to avoid wraparound in
    /// telemetry paths.
    pub const fn used_bytes(self) -> u64 {
        self.weights_bytes
            .saturating_add(self.activations_bytes)
            .saturating_add(self.kv_cache_bytes)
            .saturating_add(self.workspace_bytes)
    }

    /// Returns unaccounted capacity, clamped to zero if accounting exceeds the
    /// configured capacity.
    pub const fn available_bytes(self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Execution lease produced by a successful scheduling transition.
pub struct InferenceSession {
    /// Request assigned to an executor.
    pub request_id: RequestId,
    /// Model required by the request.
    pub model_id: ModelId,
    /// Monotonic start time of this execution session.
    pub started_at: Timestamp,
}

#[derive(Clone, Debug, Default)]
/// Observable state supplied to schedulers and, later, execution policies.
///
/// Queue and active-set storage remain private so only `Runtime` can perform
/// lifecycle transitions. Read-only iterators expose policy inputs.
pub struct RuntimeState {
    pub(crate) models: ModelRegistry,
    pub(crate) requests: BTreeMap<RequestId, RequestRecord>,
    pub(crate) queue: VecDeque<RequestId>,
    pub(crate) active: BTreeSet<RequestId>,
    /// Current device-memory accounting snapshot.
    pub memory: MemoryState,
}

impl RuntimeState {
    /// Returns the model registry visible to this runtime.
    pub fn models(&self) -> &ModelRegistry {
        &self.models
    }

    /// Returns a retained request record.
    pub fn request(&self, id: RequestId) -> Option<&RequestRecord> {
        self.requests.get(&id)
    }

    /// Iterates queued identifiers in queue order.
    pub fn queued_ids(&self) -> impl ExactSizeIterator<Item = RequestId> + '_ {
        self.queue.iter().copied()
    }

    /// Iterates running identifiers in stable identifier order.
    pub fn active_ids(&self) -> impl ExactSizeIterator<Item = RequestId> + '_ {
        self.active.iter().copied()
    }

    /// Returns the current scheduler queue length.
    pub fn queued_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns the number of running requests.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }
}
