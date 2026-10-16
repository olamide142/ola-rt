use ola_core::RequestId;

use crate::{PlanningSnapshot, PolicyDescriptor, RuntimeState};

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
    /// Execution policies call this with the snapshot they plan from, so a
    /// ranking can use prompt lengths, priorities, and deadlines without
    /// touching runtime state. Returning a request that is not queued, a
    /// duplicate, or more than `limit` requests is a policy error.
    fn rank(&self, snapshot: &PlanningSnapshot, limit: usize) -> Vec<RequestId>;
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

    fn rank(&self, snapshot: &PlanningSnapshot, limit: usize) -> Vec<RequestId> {
        snapshot
            .queued
            .iter()
            .take(limit)
            .map(|request| request.request_id)
            .collect()
    }
}
