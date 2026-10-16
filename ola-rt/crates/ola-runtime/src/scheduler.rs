use ola_core::RequestId;

use crate::{PolicyDescriptor, RuntimeState};

/// Selects the next queued request without mutating runtime state.
pub trait Scheduler {
    /// Selects one queued request or returns `None` when no work is eligible.
    ///
    /// Returning a non-queued ID is a policy error rejected by `Runtime`.
    fn select(&self, state: &RuntimeState) -> Option<RequestId>;

    /// Returns the scheduler's stable name and configuration.
    fn descriptor(&self) -> PolicyDescriptor;

    /// Ranks up to `limit` distinct queued requests in admission order.
    ///
    /// Planners use this to consider several requests before any of them
    /// starts. The default repeatedly calls [`select`](Self::select) on a
    /// private copy of the queue with earlier selections removed; schedulers
    /// with a cheaper ordering should override it.
    fn rank(&self, state: &RuntimeState, limit: usize) -> Vec<RequestId> {
        let mut remaining = state.clone();
        let mut ranked = Vec::new();
        while ranked.len() < limit {
            let Some(request_id) = self.select(&remaining) else {
                break;
            };
            ranked.push(request_id);
            let before = remaining.queue.len();
            remaining.queue.retain(|queued| *queued != request_id);
            if remaining.queue.len() == before {
                // Not queued: return it so `Runtime` reports the policy error.
                break;
            }
        }
        ranked
    }
}

/// First-in, first-out baseline scheduler.
#[derive(Clone, Copy, Debug, Default)]
pub struct FifoScheduler;

impl Scheduler for FifoScheduler {
    fn select(&self, state: &RuntimeState) -> Option<RequestId> {
        state.queued_ids().next()
    }

    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("fifo")
    }

    fn rank(&self, state: &RuntimeState, limit: usize) -> Vec<RequestId> {
        state.queued_ids().take(limit).collect()
    }
}
