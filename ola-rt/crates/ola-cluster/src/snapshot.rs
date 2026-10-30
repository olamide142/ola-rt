//! Immutable input to one cluster planning pass.

use std::collections::BTreeMap;

use ola_core::{ClusterRequestId, Input, ModelId, Priority, RequestSpec, Timestamp, WorkerId};

use crate::WorkerSummary;

/// Worst-case KV tokens a request needs: its pre-tokenized prompt plus its
/// output limit.
pub fn token_demand(spec: &RequestSpec) -> usize {
    spec.inputs
        .iter()
        .map(|input| match input {
            Input::Tokens(tokens) => tokens.len(),
            _ => 0,
        })
        .sum::<usize>()
        .saturating_add(spec.generation.max_new_tokens)
}

/// A submitted request no worker owns yet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingRequest {
    /// Cluster-wide identity.
    pub id: ClusterRequestId,
    /// Model the request needs.
    pub model: ModelId,
    /// Relative scheduling priority.
    pub priority: Priority,
    /// When the coordinator received the request.
    pub arrival_at: Timestamp,
    /// Prompt length, when the request carries pre-tokenized input.
    pub prompt_tokens: Option<usize>,
    /// Pre-tokenized prompt, empty for other inputs.
    pub prompt: Vec<u32>,
    /// Output-token limit.
    pub max_new_tokens: usize,
}

impl PendingRequest {
    pub(crate) fn new(id: ClusterRequestId, spec: &RequestSpec, arrival_at: Timestamp) -> Self {
        let mut prompt = Vec::new();
        let prompt_tokens = spec
            .inputs
            .iter()
            .map(|input| match input {
                Input::Tokens(tokens) => {
                    prompt.extend_from_slice(tokens);
                    Some(tokens.len())
                }
                _ => None,
            })
            .sum();
        Self {
            id,
            model: spec.model.clone(),
            priority: spec.priority,
            arrival_at,
            prompt_tokens,
            prompt,
            max_new_tokens: spec.generation.max_new_tokens,
        }
    }

    /// Worst-case KV tokens: prompt plus output limit.
    pub fn demand_tokens(&self) -> usize {
        self.prompt_tokens
            .unwrap_or(0)
            .saturating_add(self.max_new_tokens)
    }

    /// A platform-independent hash of the model and the first `tokens`
    /// prompt tokens (or the whole prompt, if shorter); `None` without a
    /// pre-tokenized prompt.
    pub fn prefix_fingerprint(&self, tokens: usize) -> Option<u64> {
        if self.prompt.is_empty() {
            return None;
        }
        let mut hash = fnv1a(FNV_OFFSET, self.model.as_str().as_bytes());
        for token in self.prompt.iter().take(tokens.max(1)) {
            hash = fnv1a(hash, &token.to_le_bytes());
        }
        Some(hash)
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a over `bytes`, continuing from `hash`.
pub(crate) fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// FNV-1a of `bytes` from the standard offset basis.
pub(crate) fn fnv1a_str(text: &str) -> u64 {
    fnv1a(FNV_OFFSET, text.as_bytes())
}

/// A worker as one planning pass sees it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerView {
    /// The worker's latest summary.
    pub summary: WorkerSummary,
    /// Requests assigned to the worker that its summary does not yet count.
    pub in_flight: usize,
    /// Worst-case KV tokens of those in-flight requests.
    pub in_flight_tokens: usize,
}

impl WorkerView {
    /// Load including in-flight assignments.
    pub const fn load(&self) -> usize {
        self.summary.load() + self.in_flight
    }

    /// Whether the worker would accept a new request for `model`.
    pub fn accepts(&self, model: &ModelId) -> bool {
        self.summary.accepts(model)
    }

    /// The summary's KV headroom for `model`, less in-flight demand.
    pub fn kv_headroom_tokens(&self, model: &ModelId) -> Option<usize> {
        self.summary
            .kv_headroom_tokens(model)
            .map(|free| free.saturating_sub(self.in_flight_tokens))
    }
}

/// Complete, immutable input to one cluster planning pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterSnapshot {
    /// Planning time.
    pub at: Timestamp,
    /// Coordinator state generation; plans built from it carry this value.
    pub generation: u64,
    /// Joined workers in ID order.
    pub workers: BTreeMap<WorkerId, WorkerView>,
    /// Unowned requests in arrival order.
    pub pending: Vec<PendingRequest>,
    /// Assignments committed so far; a stateless round-robin cursor.
    pub routed_requests: u64,
}

impl ClusterSnapshot {
    /// Ready workers hosting `model`, in ID order.
    pub fn eligible_workers<'a>(
        &'a self,
        model: &'a ModelId,
    ) -> impl Iterator<Item = (&'a WorkerId, &'a WorkerView)> + 'a {
        self.workers.iter().filter(|(_, view)| view.accepts(model))
    }

    /// Whether any joined worker hosts `model`, ready or not.
    pub fn hosts_model(&self, model: &ModelId) -> bool {
        self.workers
            .values()
            .any(|view| view.summary.models.contains(model))
    }
}
