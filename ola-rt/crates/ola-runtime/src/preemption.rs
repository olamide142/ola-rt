//! Choosing which live requests to preempt when a KV pool runs short.

use ola_core::{ModelId, Priority, RequestId, Timestamp};

use crate::PolicyDescriptor;

/// A live request that could be preempted to free KV blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreemptionCandidate {
    /// Live request.
    pub request_id: RequestId,
    /// Model whose pool is short.
    pub model_id: ModelId,
    /// Caller-supplied priority.
    pub priority: Priority,
    /// Time the request first entered the queue.
    pub arrived_at: Timestamp,
    /// KV blocks the request holds now.
    pub held_blocks: usize,
    /// Blocks its next operation would add this iteration.
    pub needed_blocks: usize,
    /// Output tokens generated so far, which a resume must prefill again.
    pub generated_tokens: usize,
}

/// Orders preemption victims for one model whose pool cannot hold an
/// iteration's growth.
///
/// The planner preempts candidates in the returned order until the freed
/// and no-longer-needed blocks cover the shortfall. Returning nothing
/// disables preemption: the iteration proceeds and allocations that fail are
/// reported as KV-capacity failures. Implementations must be deterministic.
pub trait PreemptionPolicy: Send {
    /// Returns the policy's stable name and configuration.
    fn descriptor(&self) -> PolicyDescriptor;

    /// Returns candidates in the order they should be preempted.
    fn order(&self, candidates: &[PreemptionCandidate]) -> Vec<RequestId>;
}

/// Never preempts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoPreemption;

impl PreemptionPolicy for NoPreemption {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("no-preemption")
    }

    fn order(&self, _candidates: &[PreemptionCandidate]) -> Vec<RequestId> {
        Vec::new()
    }
}

/// Preempts the most recently arrived request first, so the oldest work
/// keeps making progress (the order vLLM uses).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NewestFirst;

impl PreemptionPolicy for NewestFirst {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("newest-first")
    }

    fn order(&self, candidates: &[PreemptionCandidate]) -> Vec<RequestId> {
        let mut ordered = candidates.to_vec();
        ordered.sort_by_key(|candidate| {
            std::cmp::Reverse((candidate.arrived_at, candidate.request_id))
        });
        ordered
            .into_iter()
            .map(|candidate| candidate.request_id)
            .collect()
    }
}

/// Preempts the lowest-priority request first, newest first among equals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LowestPriorityFirst;

impl PreemptionPolicy for LowestPriorityFirst {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("lowest-priority-first")
    }

    fn order(&self, candidates: &[PreemptionCandidate]) -> Vec<RequestId> {
        let mut ordered = candidates.to_vec();
        ordered.sort_by_key(|candidate| {
            (
                candidate.priority,
                std::cmp::Reverse((candidate.arrived_at, candidate.request_id)),
            )
        });
        ordered
            .into_iter()
            .map(|candidate| candidate.request_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: u64, priority: i32, arrived: u64) -> PreemptionCandidate {
        PreemptionCandidate {
            request_id: RequestId::new(id),
            model_id: "m".into(),
            priority: Priority::new(priority),
            arrived_at: Timestamp::from_micros(arrived),
            held_blocks: 1,
            needed_blocks: 1,
            generated_tokens: 0,
        }
    }

    #[test]
    fn policies_order_victims_deterministically() {
        let candidates = [
            candidate(1, 5, 10),
            candidate(2, 0, 20),
            candidate(3, 0, 30),
        ];
        let ids = |ids: Vec<RequestId>| ids.iter().map(|id| id.get()).collect::<Vec<_>>();

        assert!(NoPreemption.order(&candidates).is_empty());
        assert_eq!(ids(NewestFirst.order(&candidates)), [3, 2, 1]);
        assert_eq!(ids(LowestPriorityFirst.order(&candidates)), [3, 2, 1]);
        let urgent_newest = [candidate(1, 0, 10), candidate(2, 9, 30)];
        assert_eq!(ids(LowestPriorityFirst.order(&urgent_newest)), [1, 2]);
    }
}
