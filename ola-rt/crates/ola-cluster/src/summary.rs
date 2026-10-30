//! What a worker tells the coordinator about itself.

use std::collections::{BTreeMap, BTreeSet};

use ola_core::{ModelId, Timestamp, WorkerId};
use ola_model::KvPoolStatus;
use ola_runtime::Executor;
use serde::{Deserialize, Serialize};

use crate::token_demand;

/// Whether a worker accepts new requests.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerHealth {
    /// Accepts new requests.
    #[default]
    Ready,
    /// Finishes owned requests but accepts no new ones.
    Draining,
    /// The coordinator has not heard from the worker before its lease expired.
    Lost,
}

/// A worker's view of its own load, as of one report.
///
/// Summaries are estimates: by the time the coordinator plans with one, the
/// worker may have moved on.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct WorkerSummary {
    /// Worker the summary describes.
    pub worker: WorkerId,
    /// The worker executor's state generation when the summary was taken.
    pub generation: u64,
    /// When the summary was taken.
    pub reported_at: Timestamp,
    /// Whether the worker accepts new requests.
    pub health: WorkerHealth,
    /// Models the worker can execute.
    pub models: BTreeSet<ModelId>,
    /// The worker executor's ceiling on concurrently live requests.
    pub max_active_requests: usize,
    /// Requests queued on the worker.
    pub queued_requests: usize,
    /// Requests holding a generation session on the worker.
    pub live_requests: usize,
    /// Worst-case KV tokens (prompt plus output limit) of queued requests,
    /// which hold no cache yet but will.
    pub queued_tokens: usize,
    /// Admit commands the worker has processed, accepted or refused.
    ///
    /// The coordinator compares this with the assignments it has sent to
    /// count work the summary does not yet include.
    pub admitted_requests: u64,
    /// KV-pool occupancy per model, for models that report one.
    pub kv: BTreeMap<ModelId, KvPoolStatus>,
}

impl WorkerSummary {
    /// Summarizes an executor without changing it.
    pub fn from_executor(
        worker: WorkerId,
        executor: &Executor,
        health: WorkerHealth,
        admitted_requests: u64,
        at: Timestamp,
    ) -> Self {
        let models = executor
            .loaded_model_ids()
            .cloned()
            .collect::<BTreeSet<_>>();
        let kv = models
            .iter()
            .filter_map(|id| {
                let status = executor.model(id)?.kv_pool_status()?;
                Some((id.clone(), status))
            })
            .collect();
        Self {
            worker,
            generation: executor.generation(),
            reported_at: at,
            health,
            models,
            max_active_requests: executor.max_active_requests(),
            queued_requests: executor.state().queued_len(),
            live_requests: executor.active_request_count(),
            queued_tokens: executor
                .state()
                .queued_ids()
                .filter_map(|id| executor.state().request(id))
                .map(|record| token_demand(&record.request.spec))
                .sum(),
            admitted_requests,
            kv,
        }
    }

    /// Queued plus live requests.
    pub const fn load(&self) -> usize {
        self.queued_requests + self.live_requests
    }

    /// Whether the worker would accept a new request for `model`.
    pub fn accepts(&self, model: &ModelId) -> bool {
        self.health == WorkerHealth::Ready && self.models.contains(model)
    }

    /// KV tokens `model`'s pool could still give out, counting evictable
    /// prefix-cache blocks as free and queued requests at their worst case;
    /// `None` when the model reports no pool.
    pub fn kv_headroom_tokens(&self, model: &ModelId) -> Option<usize> {
        let pool = self.kv.get(model)?;
        let free = pool
            .free_blocks
            .saturating_add(pool.cached_blocks)
            .saturating_mul(pool.tokens_per_block);
        Some(free.saturating_sub(self.queued_tokens))
    }
}
