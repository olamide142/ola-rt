use std::{cmp::Reverse, num::NonZeroU64};

use ola_core::RequestId;

use crate::{PlanningSnapshot, PolicyDescriptor, QueuedRequest};

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

/// Highest priority first, with optional aging so waiting work rises.
///
/// Without aging, a steady stream of higher-priority work can starve lower
/// priorities indefinitely; that is intentional for the textbook baseline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PriorityScheduler {
    aging_micros: Option<NonZeroU64>,
}

impl PriorityScheduler {
    /// Creates a strict priority scheduler without aging.
    pub const fn new() -> Self {
        Self { aging_micros: None }
    }

    /// Raises effective priority by one for every `interval_micros` waited.
    #[must_use]
    pub const fn with_aging(mut self, interval_micros: NonZeroU64) -> Self {
        self.aging_micros = Some(interval_micros);
        self
    }
}

impl Scheduler for PriorityScheduler {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("priority").with_config("aging_micros", aging(self.aging_micros))
    }

    fn rank(&self, snapshot: &PlanningSnapshot, limit: usize) -> Vec<RequestId> {
        rank_by(snapshot, limit, |request| {
            let boost = aging_steps(snapshot, request, self.aging_micros);
            // Larger effective priority ranks first.
            Reverse(i128::from(request.priority.get()) + i128::from(boost))
        })
    }
}

/// Earliest deadline first; requests without a deadline follow in FIFO
/// order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EarliestDeadlineFirst;

impl Scheduler for EarliestDeadlineFirst {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("earliest-deadline-first")
    }

    fn rank(&self, snapshot: &PlanningSnapshot, limit: usize) -> Vec<RequestId> {
        rank_by(snapshot, limit, |request| {
            (request.deadline.is_none(), request.deadline)
        })
    }
}

/// Size estimate used by [`ShortestJobFirst`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JobCost {
    /// Prompt tokens only.
    #[default]
    Prompt,
    /// Prompt tokens plus the output limit, an upper bound on the request's
    /// total work that is known before it runs.
    PromptPlusOutputLimit,
}

impl JobCost {
    const fn name(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::PromptPlusOutputLimit => "prompt-plus-output-limit",
        }
    }
}

/// Smallest estimated job first, with optional aging so large work is
/// eventually served.
///
/// Requests whose preparation failed cost nothing, so their failure is
/// reported immediately.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShortestJobFirst {
    cost: JobCost,
    aging_micros: Option<NonZeroU64>,
}

impl ShortestJobFirst {
    /// Creates a scheduler using `cost` without aging.
    pub const fn new(cost: JobCost) -> Self {
        Self {
            cost,
            aging_micros: None,
        }
    }

    /// Lowers effective cost by one token for every `interval_micros` waited.
    #[must_use]
    pub const fn with_aging(mut self, interval_micros: NonZeroU64) -> Self {
        self.aging_micros = Some(interval_micros);
        self
    }
}

impl Scheduler for ShortestJobFirst {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("shortest-job-first")
            .with_config("cost", self.cost.name())
            .with_config("aging_micros", aging(self.aging_micros))
    }

    fn rank(&self, snapshot: &PlanningSnapshot, limit: usize) -> Vec<RequestId> {
        rank_by(snapshot, limit, |request| {
            let prompt = request.prompt_tokens.as_ref().copied().unwrap_or(0);
            let cost = match self.cost {
                JobCost::Prompt => prompt,
                JobCost::PromptPlusOutputLimit => prompt.saturating_add(request.max_new_tokens),
            };
            let discount = aging_steps(snapshot, request, self.aging_micros);
            u128::try_from(cost)
                .unwrap_or(u128::MAX)
                .saturating_sub(u128::from(discount))
        })
    }
}

/// Ranks queued requests by `key`, breaking ties by arrival then identifier.
fn rank_by<K: Ord>(
    snapshot: &PlanningSnapshot,
    limit: usize,
    key: impl Fn(&QueuedRequest) -> K,
) -> Vec<RequestId> {
    let mut ranked = snapshot
        .queued
        .iter()
        .map(|request| {
            (
                (key(request), request.queued_at, request.request_id),
                request.request_id,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.0.cmp(&right.0));
    ranked.into_iter().take(limit).map(|(_, id)| id).collect()
}

/// Whole aging intervals a queued request has waited at the snapshot time.
fn aging_steps(
    snapshot: &PlanningSnapshot,
    request: &QueuedRequest,
    interval_micros: Option<NonZeroU64>,
) -> u64 {
    interval_micros.map_or(0, |interval| {
        let waited = snapshot.at.saturating_duration_since(request.queued_at);
        u64::try_from(waited.as_micros()).unwrap_or(u64::MAX) / interval.get()
    })
}

fn aging(interval_micros: Option<NonZeroU64>) -> String {
    interval_micros.map_or_else(|| "off".to_owned(), |interval| interval.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ola_core::{ModelId, Priority, Timestamp};

    use super::*;
    use crate::{ExecutorLimits, MemoryState, PolicyMeasurements};

    struct Queued {
        id: u64,
        queued_at: u64,
        prompt: usize,
        max_new: usize,
        priority: i32,
        deadline: Option<u64>,
    }

    fn queued(id: u64, queued_at: u64) -> Queued {
        Queued {
            id,
            queued_at,
            prompt: 4,
            max_new: 4,
            priority: 0,
            deadline: None,
        }
    }

    fn snapshot(at: u64, requests: &[Queued]) -> PlanningSnapshot {
        PlanningSnapshot {
            at: Timestamp::from_micros(at),
            generation: 0,
            limits: ExecutorLimits {
                max_active_requests: 8,
                prefill_chunk_size: None,
            },
            queued: requests
                .iter()
                .map(|request| QueuedRequest {
                    request_id: RequestId::new(request.id),
                    model_id: ModelId::from("tiny"),
                    prompt_tokens: Ok(request.prompt),
                    max_new_tokens: request.max_new,
                    priority: Priority::new(request.priority),
                    deadline: request.deadline.map(Timestamp::from_micros),
                    queued_at: Timestamp::from_micros(request.queued_at),
                })
                .collect(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: MemoryState::default(),
            device: None,
            measurements: PolicyMeasurements::default(),
        }
    }

    fn ids(ranked: Vec<RequestId>) -> Vec<u64> {
        ranked.into_iter().map(RequestId::get).collect()
    }

    #[test]
    fn priority_ranks_highest_first_and_breaks_ties_by_arrival() {
        let snapshot = snapshot(
            100,
            &[
                Queued {
                    priority: 0,
                    ..queued(1, 0)
                },
                Queued {
                    priority: 5,
                    ..queued(2, 10)
                },
                Queued {
                    priority: 5,
                    ..queued(3, 5)
                },
                Queued {
                    priority: -1,
                    ..queued(4, 1)
                },
            ],
        );

        assert_eq!(
            ids(PriorityScheduler::new().rank(&snapshot, 4)),
            [3, 2, 1, 4]
        );
        assert_eq!(ids(PriorityScheduler::new().rank(&snapshot, 2)), [3, 2]);
    }

    #[test]
    fn priority_aging_lifts_long_waiting_requests() {
        // Request 1 has waited 1000 µs: ten 100 µs intervals lift it from 0
        // to an effective priority of 10, above the fresh priority-5 request.
        let snapshot = snapshot(
            1_000,
            &[
                Queued {
                    priority: 0,
                    ..queued(1, 0)
                },
                Queued {
                    priority: 5,
                    ..queued(2, 990)
                },
            ],
        );
        let aging = PriorityScheduler::new().with_aging(NonZeroU64::new(100).unwrap());

        assert_eq!(ids(PriorityScheduler::new().rank(&snapshot, 2)), [2, 1]);
        assert_eq!(ids(aging.rank(&snapshot, 2)), [1, 2]);
    }

    #[test]
    fn earliest_deadline_first_places_requests_without_deadlines_last() {
        let snapshot = snapshot(
            0,
            &[
                queued(1, 0),
                Queued {
                    deadline: Some(500),
                    ..queued(2, 1)
                },
                Queued {
                    deadline: Some(200),
                    ..queued(3, 2)
                },
                queued(4, 3),
            ],
        );

        assert_eq!(ids(EarliestDeadlineFirst.rank(&snapshot, 4)), [3, 2, 1, 4]);
    }

    #[test]
    fn shortest_job_first_uses_the_configured_cost() {
        let snapshot = snapshot(
            0,
            &[
                Queued {
                    prompt: 10,
                    max_new: 1,
                    ..queued(1, 0)
                },
                Queued {
                    prompt: 2,
                    max_new: 50,
                    ..queued(2, 1)
                },
                Queued {
                    prompt: 6,
                    max_new: 6,
                    ..queued(3, 2)
                },
            ],
        );

        assert_eq!(
            ids(ShortestJobFirst::new(JobCost::Prompt).rank(&snapshot, 3)),
            [2, 3, 1]
        );
        assert_eq!(
            ids(ShortestJobFirst::new(JobCost::PromptPlusOutputLimit).rank(&snapshot, 3)),
            [1, 3, 2]
        );
    }

    #[test]
    fn shortest_job_first_aging_eventually_serves_large_requests() {
        // The 50-token request has waited 4600 µs, discounting 46 tokens to
        // an effective 4, below the fresh 5-token request.
        let snapshot = snapshot(
            5_000,
            &[
                Queued {
                    prompt: 50,
                    ..queued(1, 400)
                },
                Queued {
                    prompt: 5,
                    ..queued(2, 4_990)
                },
            ],
        );
        let aging =
            ShortestJobFirst::new(JobCost::Prompt).with_aging(NonZeroU64::new(100).unwrap());

        assert_eq!(
            ids(ShortestJobFirst::new(JobCost::Prompt).rank(&snapshot, 2)),
            [2, 1]
        );
        assert_eq!(ids(aging.rank(&snapshot, 2)), [1, 2]);
    }

    #[test]
    fn shortest_job_first_surfaces_preparation_failures_first() {
        let mut snapshot = snapshot(0, &[queued(1, 0), queued(2, 1)]);
        snapshot.queued[1].prompt_tokens = Err("bad input".into());

        assert_eq!(
            ids(ShortestJobFirst::new(JobCost::Prompt).rank(&snapshot, 2)),
            [2, 1]
        );
    }

    #[test]
    fn scheduler_descriptors_are_stable() {
        let interval = NonZeroU64::new(250).unwrap();
        let names = [
            PriorityScheduler::new().descriptor(),
            PriorityScheduler::new().with_aging(interval).descriptor(),
            EarliestDeadlineFirst.descriptor(),
            ShortestJobFirst::new(JobCost::PromptPlusOutputLimit)
                .with_aging(interval)
                .descriptor(),
        ]
        .map(|descriptor| descriptor.to_string());

        assert_eq!(
            names,
            [
                "priority{aging_micros=off}",
                "priority{aging_micros=250}",
                "earliest-deadline-first",
                "shortest-job-first{cost=prompt-plus-output-limit,aging_micros=250}",
            ]
        );
    }
}
