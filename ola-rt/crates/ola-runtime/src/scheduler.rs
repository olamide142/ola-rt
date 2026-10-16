use ola_core::RequestId;

use crate::{PlanningSnapshot, PolicyDescriptor};

/// Orders queued requests for admission without mutating runtime state.
///
/// A scheduler is a component of an execution policy such as
/// `ComposedPolicy`; it never starts requests itself.
pub trait Scheduler {
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
