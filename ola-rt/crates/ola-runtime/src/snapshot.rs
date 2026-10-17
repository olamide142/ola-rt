//! Immutable runtime state that execution policies plan from.
//!
//! See `EXECUTION_POLICY.md` in this crate for the design. A snapshot is
//! plain data: policies read it, plans are validated against it, and nothing
//! in it can reach back into the runtime.

use std::collections::BTreeMap;

use ola_core::{ModelId, Priority, RequestId, Timestamp};
use ola_model::{BatchCapabilities, KvCacheProfile, KvPoolStatus};

use crate::{BatchPressure, DecodeView, MemoryState, PrefillView};

/// Executor configuration that bounds every plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutorLimits {
    /// Maximum number of live sessions.
    pub max_active_requests: usize,
    /// Maximum prompt positions per resumable prefill segment, if bounded.
    pub prefill_chunk_size: Option<usize>,
}

/// One queued request, in scheduler-visible queue order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedRequest {
    /// Request identifier.
    pub request_id: RequestId,
    /// Requested model.
    pub model_id: ModelId,
    /// Tokenized prompt length, or the preparation error to report.
    pub prompt_tokens: Result<usize, String>,
    /// Leading prompt tokens whose KV the model's prefix cache holds; the
    /// first prefill starts after them. Always leaves at least one prompt
    /// token to compute, and is zero without resumable prefill.
    pub cached_prefix_tokens: usize,
    /// Cached prefix blocks no live session holds, which admission takes
    /// out of the evictable pool.
    pub cached_prefix_evictable_blocks: usize,
    /// Caller output limit.
    pub max_new_tokens: usize,
    /// Caller-supplied priority.
    pub priority: Priority,
    /// Optional caller deadline.
    pub deadline: Option<Timestamp>,
    /// Time the request entered the queue.
    pub queued_at: Timestamp,
}

/// What policies may assume about one registered model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelSnapshot {
    /// Whether an executable is attached; requests for detached models fail.
    pub attached: bool,
    /// Tensor-batched hooks the executable declares.
    pub batch: BatchCapabilities,
    /// Whether sessions can process prompts in resumable segments.
    pub chunked_prefill: bool,
    /// Physical KV-pool geometry, when published.
    pub kv_cache: Option<KvCacheProfile>,
    /// Physical KV-pool occupancy at the iteration boundary, when the model
    /// allocates from a shared pool.
    pub kv_pool: Option<KvPoolStatus>,
}

impl ModelSnapshot {
    /// Snapshot of a registered model with no executable attached.
    pub const DETACHED: Self = Self {
        attached: false,
        batch: BatchCapabilities::NONE,
        chunked_prefill: false,
        kv_cache: None,
        kv_pool: None,
    };
}

/// Device state visible to policies. Not populated until device telemetry
/// exists (P6); policies must treat `None` as unknown.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeviceSnapshot {
    /// Human-readable device identity.
    pub label: String,
    /// Total device memory, when reported.
    pub memory_capacity_bytes: Option<u64>,
    /// Device memory in use, when reported.
    pub memory_used_bytes: Option<u64>,
}

/// Measured inputs a policy may use. Empty until calibration feeds the
/// runtime (P4).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PolicyMeasurements {
    /// Calibrated token-equivalent cost of one prefill invocation.
    pub prefill_invocation_overhead_tokens: Option<usize>,
}

/// Complete, immutable input to one planning pass.
///
/// Every live request appears in exactly one of `prefilling` or `decoding`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanningSnapshot {
    /// Iteration timestamp.
    pub at: Timestamp,
    /// Executor state generation; plans built from it carry this value.
    pub generation: u64,
    /// Executor configuration.
    pub limits: ExecutorLimits,
    /// Requests waiting for admission, in queue order.
    pub queued: Vec<QueuedRequest>,
    /// Live requests with unfinished prompts.
    pub prefilling: BTreeMap<RequestId, PrefillView>,
    /// Live requests ready to decode, keyed by request; `decode_order` gives
    /// their round-robin order.
    pub decoding: BTreeMap<RequestId, DecodeView>,
    /// Live decoding requests in round-robin order.
    pub decode_order: Vec<RequestId>,
    /// Live prefilling requests in round-robin order.
    pub prefill_order: Vec<RequestId>,
    /// Every registered model.
    pub models: BTreeMap<ModelId, ModelSnapshot>,
    /// Device accounting visible to admission.
    pub memory: MemoryState,
    /// Device telemetry, when available.
    pub device: Option<DeviceSnapshot>,
    /// Measured policy inputs.
    pub measurements: PolicyMeasurements,
}

impl PlanningSnapshot {
    /// Returns the number of live sessions.
    pub fn live_requests(&self) -> usize {
        self.prefilling.len().saturating_add(self.decoding.len())
    }

    /// Returns the load batch policies may adapt to.
    pub fn pressure(&self) -> BatchPressure {
        BatchPressure {
            queued_requests: self.queued.len(),
            live_requests: self.live_requests(),
            max_active_requests: self.limits.max_active_requests,
        }
    }

    /// Returns the model snapshot, treating unknown models as detached.
    pub fn model(&self, model_id: &ModelId) -> ModelSnapshot {
        self.models
            .get(model_id)
            .copied()
            .unwrap_or(ModelSnapshot::DETACHED)
    }

    /// Returns worst-case KV bytes reserved by every live request.
    pub fn reserved_kv_bytes(&self) -> u64 {
        self.live_reservations().map(|(_, bytes)| bytes).sum()
    }

    /// Returns worst-case KV bytes reserved by live requests of one model.
    pub fn model_reserved_kv_bytes(&self, model_id: &ModelId) -> u64 {
        self.live_reservations()
            .filter(|(model, _)| *model == model_id)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// Returns the first prefill segment length for a prompt on a model.
    ///
    /// Models without resumable prefill always process the whole prompt.
    pub fn initial_prefill_chunk(&self, model_id: &ModelId, prompt_tokens: usize) -> usize {
        match self.limits.prefill_chunk_size {
            Some(chunk) if self.model(model_id).chunked_prefill => prompt_tokens.min(chunk),
            _ => prompt_tokens,
        }
    }

    fn live_reservations(&self) -> impl Iterator<Item = (&ModelId, u64)> {
        self.prefilling
            .values()
            .map(|view| (&view.model_id, view.reserved_kv_bytes))
            .chain(
                self.decoding
                    .values()
                    .map(|view| (&view.model_id, view.reserved_kv_bytes)),
            )
    }
}
