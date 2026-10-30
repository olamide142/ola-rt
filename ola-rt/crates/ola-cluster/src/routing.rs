//! Baseline routing policies.

use std::collections::BTreeMap;

use ola_core::WorkerId;
use ola_runtime::PolicyDescriptor;

use crate::{
    ClusterAction, ClusterDecision, ClusterDecisionRecord, ClusterPlan, ClusterPolicy,
    ClusterPolicyError, ClusterSnapshot, DeferReason, PendingRequest, RejectReason, RouteCandidate,
    RouteReason, RouteRecord,
    snapshot::{fnv1a, fnv1a_str},
};

/// Assigns pending requests to eligible workers in turn.
///
/// The cursor is the snapshot's committed-assignment count, so the policy is
/// stateless and a rejected plan does not advance it. Load is recorded but
/// ignored.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoundRobinRouting;

impl ClusterPolicy for RoundRobinRouting {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("round-robin-routing")
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        let mut cursor = snapshot.routed_requests;
        Ok(route_each(snapshot, self.descriptor(), |_, candidates| {
            // The candidate count is small and the cursor only needs to
            // cycle, so truncating the 64-bit cursor is harmless.
            let index = (cursor % candidates.len() as u64) as usize;
            cursor += 1;
            (index, RouteReason::RoundRobin)
        }))
    }
}

/// Join-the-shortest-queue: assigns each pending request to the eligible
/// worker with the least load.
///
/// Load is queued plus live plus in-flight requests, and counts assignments
/// made earlier in the same plan. Ties go to the lowest worker ID.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeastLoadedRouting;

impl ClusterPolicy for LeastLoadedRouting {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("least-loaded-routing")
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        Ok(route_each(snapshot, self.descriptor(), |_, candidates| {
            (least_loaded(candidates), RouteReason::LeastLoaded)
        }))
    }
}

/// Power of two choices: samples two distinct eligible workers and takes
/// the less loaded one (the first sampled on a tie).
///
/// The sample is a hash of the seed and the cluster request ID, so the
/// policy is stateless and replays exactly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PowerOfTwoRouting {
    /// Sampling seed.
    pub seed: u64,
}

impl PowerOfTwoRouting {
    /// Creates the policy with a sampling seed.
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl ClusterPolicy for PowerOfTwoRouting {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("power-of-two-routing").with_config("seed", self.seed)
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        Ok(route_each(
            snapshot,
            self.descriptor(),
            |pending, candidates| {
                let count = candidates.len() as u64;
                if count == 1 {
                    return (0, RouteReason::PowerOfTwo);
                }
                let sample = mix(self.seed ^ mix(pending.id.get()));
                // Both indices are below the small candidate count.
                let first = (sample % count) as usize;
                let mut second = ((sample >> 32) % (count - 1)) as usize;
                if second >= first {
                    second += 1;
                }
                let index = if candidates[second].load < candidates[first].load {
                    second
                } else {
                    first
                };
                (index, RouteReason::PowerOfTwo)
            },
        ))
    }
}

/// Assigns each pending request to the eligible worker with the most KV
/// headroom, so long requests land where memory is free.
///
/// Headroom counts free and evictable blocks, less the worst-case demand of
/// queued, in-flight, and earlier same-plan requests. Workers whose pool
/// reports nothing count as having none. Ties go to the lesser load, then
/// the lowest worker ID.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvHeadroomRouting;

impl ClusterPolicy for KvHeadroomRouting {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("kv-headroom-routing")
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        Ok(route_each(snapshot, self.descriptor(), |_, candidates| {
            let index = candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| {
                    (
                        std::cmp::Reverse(candidate.kv_headroom_tokens.unwrap_or(0)),
                        candidate.load,
                    )
                })
                .map_or(0, |(index, _)| index);
            (index, RouteReason::KvHeadroom)
        }))
    }
}

/// Sends requests that share a prompt prefix to the same worker, so its
/// prefix cache serves them, unless that worker is overloaded.
///
/// Each prefix of `prefix_tokens` tokens ranks the eligible workers by
/// rendezvous hashing; the request goes to the highest-ranked worker whose
/// load is at most the least load plus `load_slack`. Adding or removing a
/// worker moves only the prefixes that ranked it first. Requests without a
/// pre-tokenized prompt go to the least-loaded worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixAffinityRouting {
    /// Prompt tokens that identify a prefix.
    pub prefix_tokens: usize,
    /// How much more loaded than the least-loaded worker a preferred worker
    /// may be.
    pub load_slack: usize,
}

impl PrefixAffinityRouting {
    /// Creates the policy.
    pub const fn new(prefix_tokens: usize, load_slack: usize) -> Self {
        Self {
            prefix_tokens,
            load_slack,
        }
    }
}

impl Default for PrefixAffinityRouting {
    fn default() -> Self {
        Self::new(32, 4)
    }
}

impl ClusterPolicy for PrefixAffinityRouting {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("prefix-affinity-routing")
            .with_config("prefix_tokens", self.prefix_tokens)
            .with_config("load_slack", self.load_slack)
    }

    fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, ClusterPolicyError> {
        Ok(route_each(
            snapshot,
            self.descriptor(),
            |pending, candidates| {
                let Some(prefix) = pending.prefix_fingerprint(self.prefix_tokens) else {
                    return (least_loaded(candidates), RouteReason::NoPrefix);
                };
                let bound = candidates
                    .iter()
                    .map(|candidate| candidate.load)
                    .min()
                    .unwrap_or(0)
                    .saturating_add(self.load_slack);
                let mut ranking = (0..candidates.len()).collect::<Vec<_>>();
                ranking.sort_by_key(|&index| {
                    let worker = fnv1a_str(candidates[index].worker.as_str());
                    std::cmp::Reverse(mix(fnv1a(prefix, &worker.to_le_bytes())))
                });
                let (rank, &index) = ranking
                    .iter()
                    .enumerate()
                    .find(|(_, index)| candidates[**index].load <= bound)
                    .expect("the least-loaded candidate is within the bound");
                (index, RouteReason::PrefixAffinity { rank })
            },
        ))
    }
}

fn least_loaded(candidates: &[RouteCandidate]) -> usize {
    candidates
        .iter()
        .enumerate()
        .min_by_key(|(_, candidate)| candidate.load)
        .map_or(0, |(index, _)| index)
}

/// SplitMix64 finalizer: a fixed, well-mixed bijection.
const fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Decides every pending request in arrival order, letting `choose` pick an
/// index into the non-empty eligible candidates and say why.
fn route_each(
    snapshot: &ClusterSnapshot,
    policy: PolicyDescriptor,
    mut choose: impl FnMut(&PendingRequest, &[RouteCandidate]) -> (usize, RouteReason),
) -> ClusterDecision {
    let mut assigned = BTreeMap::<&WorkerId, (usize, usize)>::new();
    let mut actions = Vec::with_capacity(snapshot.pending.len());
    let mut routes = Vec::new();
    for pending in &snapshot.pending {
        let candidates = snapshot
            .eligible_workers(&pending.model)
            .map(|(worker, view)| {
                let (requests, tokens) = assigned.get(worker).copied().unwrap_or_default();
                RouteCandidate {
                    worker: worker.clone(),
                    load: view.load() + requests,
                    kv_headroom_tokens: view
                        .kv_headroom_tokens(&pending.model)
                        .map(|free| free.saturating_sub(tokens)),
                }
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            let model = pending.model.clone();
            actions.push(if snapshot.hosts_model(&pending.model) {
                ClusterAction::Defer {
                    request: pending.id,
                    reason: DeferReason::NoReadyWorker { model },
                }
            } else {
                ClusterAction::Reject {
                    request: pending.id,
                    reason: RejectReason::ModelNotServed { model },
                }
            });
            continue;
        }
        let (index, reason) = choose(pending, &candidates);
        let chosen = candidates[index].worker.clone();
        let (key, _) = snapshot
            .workers
            .get_key_value(&chosen)
            .expect("candidates come from the snapshot's workers");
        let entry = assigned.entry(key).or_default();
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(pending.demand_tokens());
        actions.push(ClusterAction::Assign {
            request: pending.id,
            worker: chosen,
        });
        routes.push(RouteRecord {
            request: pending.id,
            candidates,
            reason,
        });
    }
    ClusterDecision {
        plan: ClusterPlan {
            generation: snapshot.generation,
            actions,
        },
        record: ClusterDecisionRecord { policy, routes },
    }
}
