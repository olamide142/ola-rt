use ola_core::RequestId;

use crate::RuntimeState;

/// Selects the next queued request without mutating runtime state.
pub trait Scheduler {
    /// Selects one queued request or returns `None` when no work is eligible.
    ///
    /// Returning a non-queued ID is a policy error rejected by `Runtime`.
    fn select(&mut self, state: &RuntimeState) -> Option<RequestId>;
}

/// First-in, first-out baseline scheduler.
#[derive(Clone, Copy, Debug, Default)]
pub struct FifoScheduler;

impl Scheduler for FifoScheduler {
    fn select(&mut self, state: &RuntimeState) -> Option<RequestId> {
        state.queued_ids().next()
    }
}
