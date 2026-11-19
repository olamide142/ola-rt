//! Conformance between the paper's (A Compositional Formalization of Inference Execution Policies in Ola-RT)
//! feasibility predicate Γ and
//! `ExecutionPlan::validate`, plus invariant checks along simulated
//! trajectories, including adaptive policies whose parameters change while
//! they run.
//!
//! Three checks:
//!
//! 1. **Agreement.** `gamma_spec` is written from the paper's definitions,
//!    not from the validator. On every recorded plan, and on mutated and
//!    randomly fuzzed variants, `gamma_spec(s, A) == validate(s, A).is_ok()`.
//! 2. **Targeted mutation.** Each operator breaks exactly one constraint of
//!    a valid plan; the validator must reject it with that constraint's
//!    typed error.
//! 3. **Trajectory invariants.** Along every committed iteration of many
//!    seeded runs, the invariants the theorems state hold.
//!
//! Set `OLA_CONFORMANCE_REPORT=<path>` to write a JSON summary.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use ola_bench::{
    KvGeometry, SchedulingTrace, SimulationConfig, TraceParameters, run_observed_trial,
};
use ola_core::{ModelId, RequestId};
use ola_runtime::{
    AdaptiveConstraints, AdaptiveOptimizationError, AdaptivePolicy, AdaptiveRequestObservation,
    BatchExecutionMode, BatchPhase, ComposedPolicy, EarliestDeadlineFirst, ExecutionPlan,
    ExecutionPolicy, ExecutorEvent, ExecutorIteration, FifoTokenBudgetPolicy, HeadOfLine, JobCost,
    LifecycleAction, MemoryAwareAdmission, NewestFirst, PlanError, PlannedBatch, PlannedDeferral,
    PlannedPhase, PlanningSnapshot, PolicyDecision, PolicyDescriptor, PolicyError,
    PreemptionReason, PriorityScheduler, ShortestJobFirst, UnconstrainedAdmission,
    WatermarkAdmission,
};

// ---------------------------------------------------------------------------
// Recording wrapper: captures (snapshot, plan, enforce flag) at every plan.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Sample {
    snapshot: PlanningSnapshot,
    plan: ExecutionPlan,
    enforce: bool,
}

struct Recording {
    inner: Box<dyn ExecutionPolicy>,
    store: Arc<Mutex<Vec<Sample>>>,
}

impl ExecutionPolicy for Recording {
    fn descriptor(&self) -> PolicyDescriptor {
        self.inner.descriptor()
    }
    fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
        let decision = self.inner.plan(snapshot)?;
        self.store.lock().unwrap().push(Sample {
            snapshot: snapshot.clone(),
            plan: decision.plan.clone(),
            enforce: self.inner.enforces_model_kv_capacity(),
        });
        Ok(decision)
    }
    fn enforces_model_kv_capacity(&self) -> bool {
        self.inner.enforces_model_kv_capacity()
    }
    fn observe(&mut self, iteration: &ExecutorIteration) {
        self.inner.observe(iteration);
    }
    fn recovers_kv_exhaustion(&self) -> bool {
        self.inner.recovers_kv_exhaustion()
    }
}

// ---------------------------------------------------------------------------
// Independent specification of Γ(S, A; C), written from the paper.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    Admit,
    Prefill,
    ContinuedPrefill,
    Decode,
    Preempt,
    Terminate,
    Defer,
}

/// Γ(S, A; C): true iff the plan is admissible. Each block is one
/// constraint family from the paper (freshness, roles and status, model
/// identity, phase positions and resumability, batch structure, kernels,
/// event order, releases, slots, KV reservations).
fn gamma_spec(s: &PlanningSnapshot, a: &ExecutionPlan, enforce_kv: bool) -> bool {
    // Freshness.
    if a.generation != s.generation {
        return false;
    }
    let queued: BTreeMap<RequestId, _> = s.queued.iter().map(|q| (q.request_id, q)).collect();
    let live_model = |r: &RequestId| -> Option<&ModelId> {
        s.prefilling
            .get(r)
            .map(|p| &p.model_id)
            .or_else(|| s.decoding.get(r).map(|d| &d.model_id))
    };

    // Roles: every request mentioned by the plan has exactly one role, or
    // exactly {Admit, Prefill}.
    let mut roles: BTreeMap<RequestId, Vec<Role>> = BTreeMap::new();
    let mut events: Vec<usize> = Vec::new();
    let mut preempted = BTreeSet::new();
    let mut admits: Vec<(RequestId, &ModelId, u64)> = Vec::new();
    for action in &a.lifecycle {
        match action {
            LifecycleAction::Admit {
                request_id,
                model_id,
                reserved_kv_bytes,
            } => {
                roles.entry(*request_id).or_default().push(Role::Admit);
                let Some(q) = queued.get(request_id) else {
                    return false;
                };
                if &q.model_id != model_id {
                    return false;
                }
                admits.push((*request_id, model_id, *reserved_kv_bytes));
            }
            LifecycleAction::Reject {
                request_id,
                event_index,
                ..
            }
            | LifecycleAction::Fail {
                request_id,
                event_index,
                ..
            } => {
                roles.entry(*request_id).or_default().push(Role::Terminate);
                if !queued.contains_key(request_id) {
                    return false;
                }
                events.push(*event_index);
            }
            LifecycleAction::Preempt {
                request_id,
                event_index,
                ..
            } => {
                roles.entry(*request_id).or_default().push(Role::Preempt);
                if live_model(request_id).is_none() {
                    return false;
                }
                preempted.insert(*request_id);
                events.push(*event_index);
            }
        }
    }

    let mut finishing = BTreeSet::new();
    for batch in &a.batches {
        // Batch structure.
        if batch.rows.is_empty() {
            return false;
        }
        if batch.mode == BatchExecutionMode::Individual && batch.rows.len() != 1 {
            return false;
        }
        let model = s.models.get(&batch.model_id);
        let resumable = model.is_some_and(|m| m.chunked_prefill);
        for row in &batch.rows {
            events.push(row.event_index);
            let r = row.request_id;
            let len = row.positions.end.saturating_sub(row.positions.start);
            match batch.phase {
                PlannedPhase::Decode => {
                    roles.entry(r).or_default().push(Role::Decode);
                    let Some(d) = s.decoding.get(&r) else {
                        return false;
                    };
                    if d.model_id != batch.model_id
                        || row.positions.start != d.position
                        || row.positions.end != d.position + 1
                        || row.final_prompt_chunk
                    {
                        return false;
                    }
                    if d.generated_tokens + 1 >= d.max_new_tokens {
                        finishing.insert(r);
                    }
                }
                PlannedPhase::Prefill | PlannedPhase::ContinuedPrefill => {
                    let (model_id, start, prompt, max_new) = if batch.phase == PlannedPhase::Prefill
                    {
                        roles.entry(r).or_default().push(Role::Prefill);
                        let Some(q) = queued.get(&r) else {
                            return false;
                        };
                        let Ok(prompt) = q.prompt_tokens.as_ref().map(|p| *p) else {
                            return false;
                        };
                        (&q.model_id, q.cached_prefix_tokens, prompt, usize::MAX)
                    } else {
                        roles.entry(r).or_default().push(Role::ContinuedPrefill);
                        let Some(p) = s.prefilling.get(&r) else {
                            return false;
                        };
                        (&p.model_id, p.cursor, p.prompt_tokens, p.max_new_tokens)
                    };
                    if model_id != &batch.model_id {
                        return false;
                    }
                    // Phase positions: the row continues exactly where the
                    // request stands, is non-empty, stays inside the prompt,
                    // and is final iff it reaches the prompt's end.
                    if row.positions.start != start
                        || len == 0
                        || row.positions.end > prompt
                        || row.final_prompt_chunk != (row.positions.end == prompt)
                    {
                        return false;
                    }
                    // Resumability and chunk ceiling κ.
                    if !row.final_prompt_chunk && !resumable {
                        return false;
                    }
                    if resumable
                        && let Some(kappa) = s.limits.prefill_chunk_size
                        && len > kappa
                    {
                        return false;
                    }
                    if batch.phase == PlannedPhase::ContinuedPrefill
                        && row.final_prompt_chunk
                        && max_new <= 1
                    {
                        finishing.insert(r);
                    }
                }
            }
        }
        // Kernel requirements.
        let profile = model.and_then(|m| m.paged_attention);
        match (batch.phase, profile, &batch.kernel) {
            (PlannedPhase::Prefill, _, Some(_)) => return false,
            (PlannedPhase::Prefill, _, None) | (_, None, None) => {}
            (_, Some(_), None) | (_, None, Some(_)) => return false,
            (_, Some(profile), Some(kernel)) => {
                let tokens = batch.rows.iter().map(|r| r.positions.end).max().unwrap();
                if profile.requirements(tokens).ok() != Some(kernel.requirements) {
                    return false;
                }
            }
        }
    }

    for deferral in &a.deferrals {
        let (r, ok) = match deferral {
            PlannedDeferral::Admission {
                request_id,
                event_index,
                ..
            } => {
                events.push(*event_index);
                (*request_id, queued.contains_key(request_id))
            }
            PlannedDeferral::Slots { request_id } => (*request_id, queued.contains_key(request_id)),
            PlannedDeferral::Budget {
                request_id,
                phase,
                event_index,
                ..
            } => {
                events.push(*event_index);
                let ok = match phase {
                    BatchPhase::Decode => s.decoding.contains_key(request_id),
                    BatchPhase::Prefill => {
                        queued.contains_key(request_id) || s.prefilling.contains_key(request_id)
                    }
                };
                (*request_id, ok)
            }
        };
        roles.entry(r).or_default().push(Role::Defer);
        if !ok {
            return false;
        }
    }

    for (_, mut rs) in roles {
        rs.sort();
        let paired = rs == [Role::Admit, Role::Prefill];
        if !(paired || (rs.len() == 1 && rs[0] != Role::Admit && rs[0] != Role::Prefill)) {
            return false;
        }
    }

    // Event order: indices are exactly 0..n-1.
    events.sort_unstable();
    if events.iter().enumerate().any(|(i, e)| i != *e) {
        return false;
    }

    // Releases: distinct, and each names a row that finishes its request.
    let releases: BTreeSet<_> = a.predicted_releases.iter().copied().collect();
    if releases.len() != a.predicted_releases.len() || !releases.is_subset(&finishing) {
        return false;
    }

    // Slots, eq. (slots): |L| - |R ∪ P| + |D| ≤ M.
    let freed: BTreeSet<_> = releases.union(&preempted).copied().collect();
    let live = s.prefilling.len() + s.decoding.len();
    if live - freed.len() + admits.len() > s.limits.max_active_requests {
        return false;
    }

    // KV, eq. (kv): only pools receiving an admission, only if enforced.
    if enforce_kv {
        let admitting: BTreeSet<&ModelId> = admits.iter().map(|(_, m, _)| *m).collect();
        for m in admitting {
            let Some(capacity) = s
                .models
                .get(m)
                .and_then(|x| x.kv_cache)
                .map(|p| p.capacity_bytes())
            else {
                continue;
            };
            let held: u64 = s
                .prefilling
                .iter()
                .map(|(r, p)| (r, &p.model_id, p.reserved_kv_bytes))
                .chain(
                    s.decoding
                        .iter()
                        .map(|(r, d)| (r, &d.model_id, d.reserved_kv_bytes)),
                )
                .filter(|(r, model, _)| *model == m && !freed.contains(r))
                .map(|(_, _, b)| b)
                .sum();
            let new: u64 = admits
                .iter()
                .filter(|(_, model, _)| *model == m)
                .map(|(_, _, b)| b)
                .sum();
            if held + new > capacity {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Mutation operators: each breaks one constraint of a valid plan.
// ---------------------------------------------------------------------------

fn error_name(error: &PlanError) -> &'static str {
    match error {
        PlanError::Stale { .. } => "Stale",
        PlanError::DuplicateRequest(_) => "DuplicateRequest",
        PlanError::InvalidStatus { .. } => "InvalidStatus",
        PlanError::UnpairedAdmission(_) => "UnpairedAdmission",
        PlanError::EmptyBatch(_) => "EmptyBatch",
        PlanError::IndividualBatchWidth(_) => "IndividualBatchWidth",
        PlanError::ModelMismatch { .. } => "ModelMismatch",
        PlanError::InvalidPositions(_) => "InvalidPositions",
        PlanError::UnsupportedPartialPrefill(_) => "UnsupportedPartialPrefill",
        PlanError::ChunkTooLong { .. } => "ChunkTooLong",
        PlanError::InvalidEventOrder => "InvalidEventOrder",
        PlanError::InvalidRelease(_) => "InvalidRelease",
        PlanError::SlotCapacity { .. } => "SlotCapacity",
        PlanError::KvCapacity { .. } => "KvCapacity",
        PlanError::MissingKernelSelection(_) => "MissingKernelSelection",
        PlanError::UnexpectedKernelSelection(_) => "UnexpectedKernelSelection",
        PlanError::KernelRequirementsMismatch(_) => "KernelRequirementsMismatch",
        #[allow(unreachable_patterns)]
        _ => "Other",
    }
}

fn event_count(a: &ExecutionPlan) -> usize {
    let lifecycle = a
        .lifecycle
        .iter()
        .filter(|x| !matches!(x, LifecycleAction::Admit { .. }))
        .count();
    let rows: usize = a.batches.iter().map(|b| b.rows.len()).sum();
    let deferrals = a
        .deferrals
        .iter()
        .filter(|d| !matches!(d, PlannedDeferral::Slots { .. }))
        .count();
    lifecycle + rows + deferrals
}

fn mentioned(a: &ExecutionPlan) -> BTreeSet<RequestId> {
    let mut set = BTreeSet::new();
    for x in &a.lifecycle {
        set.insert(x.request_id());
    }
    for b in &a.batches {
        for r in &b.rows {
            set.insert(r.request_id);
        }
    }
    for d in &a.deferrals {
        set.insert(d.request_id());
    }
    set
}

type Mutation = (&'static str, PlanningSnapshot, ExecutionPlan, bool);

/// Returns every applicable single-constraint mutation of a valid sample,
/// named by the error the validator should return.
fn mutations(sample: &Sample) -> Vec<Mutation> {
    let s = &sample.snapshot;
    let a = &sample.plan;
    let e = sample.enforce;
    let mut out: Vec<Mutation> = Vec::new();
    let n_events = event_count(a);
    let used = mentioned(a);

    // Freshness.
    let mut m = a.clone();
    m.generation = m.generation.wrapping_add(1);
    out.push(("Stale", s.clone(), m, e));

    // Uniqueness: one row duplicated inside its batch, numbered freshly.
    if let Some(batch) = a.batches.first() {
        let mut m = a.clone();
        let mut row = batch.rows[0].clone();
        row.event_index = n_events;
        m.batches[0].rows.push(row);
        if m.batches[0].mode == BatchExecutionMode::Individual {
            m.batches[0].mode = BatchExecutionMode::Tensor;
        }
        out.push(("DuplicateRequest", s.clone(), m, e));
    }

    // Status: preempt a queued request the plan does not otherwise use.
    if let Some(q) = s.queued.iter().find(|q| !used.contains(&q.request_id)) {
        let mut m = a.clone();
        m.lifecycle.push(LifecycleAction::Preempt {
            request_id: q.request_id,
            reason: PreemptionReason::AllocationFailed,
            event_index: n_events,
        });
        out.push(("InvalidStatus", s.clone(), m, e));
    }

    // Model identity of an admission.
    if let Some(index) = a
        .lifecycle
        .iter()
        .position(|x| matches!(x, LifecycleAction::Admit { .. }))
    {
        let mut m = a.clone();
        if let LifecycleAction::Admit { model_id, .. } = &mut m.lifecycle[index] {
            *model_id = ModelId::new("conformance-other-model");
        }
        out.push(("ModelMismatch", s.clone(), m, e));

        // Admission/prefill pairing: drop the admitted request's first row.
        let admitted = a.lifecycle[index].request_id();
        let mut m = a.clone();
        for batch in &mut m.batches {
            if batch.phase == PlannedPhase::Prefill {
                batch.rows.retain(|r| r.request_id != admitted);
            }
        }
        m.batches.retain(|b| !b.rows.is_empty());
        out.push(("UnpairedAdmission", s.clone(), m, e));

        // Slot ceiling: shrink M to one below the post-plan live count.
        let releases: BTreeSet<_> = a.predicted_releases.iter().copied().collect();
        let preempted = a
            .lifecycle
            .iter()
            .filter(|x| matches!(x, LifecycleAction::Preempt { .. }))
            .count();
        let admits = a
            .lifecycle
            .iter()
            .filter(|x| matches!(x, LifecycleAction::Admit { .. }))
            .count();
        let live = s.prefilling.len() + s.decoding.len();
        let after = live - releases.len() - preempted + admits;
        if after >= 1 {
            let mut t = s.clone();
            t.limits.max_active_requests = after - 1;
            out.push(("SlotCapacity", t, a.clone(), e));
        }

        // KV bound: inflate one admission past its pool.
        if e && let LifecycleAction::Admit { model_id, .. } = &a.lifecycle[index]
            && let Some(capacity) = s
                .models
                .get(model_id)
                .and_then(|x| x.kv_cache)
                .map(|p| p.capacity_bytes())
        {
            let mut m = a.clone();
            if let LifecycleAction::Admit {
                reserved_kv_bytes, ..
            } = &mut m.lifecycle[index]
            {
                *reserved_kv_bytes = capacity + 1;
            }
            out.push(("KvCapacity", s.clone(), m, e));
        }
    }

    // Phase positions: shift a decode row.
    if let Some((bi, ri)) = a
        .batches
        .iter()
        .enumerate()
        .find_map(|(bi, b)| (b.phase == PlannedPhase::Decode).then_some((bi, 0)))
    {
        let mut m = a.clone();
        let row = &mut m.batches[bi].rows[ri];
        row.positions = row.positions.start + 1..row.positions.end + 1;
        out.push(("InvalidPositions", s.clone(), m, e));
    }

    // Batch structure: an empty batch; a multi-row batch run individually.
    if let Some(batch) = a.batches.first() {
        let mut m = a.clone();
        m.batches.push(PlannedBatch {
            rows: Vec::new(),
            ..batch.clone()
        });
        out.push(("EmptyBatch", s.clone(), m, e));
    }
    if let Some(bi) = a.batches.iter().position(|b| b.rows.len() >= 2) {
        let mut m = a.clone();
        m.batches[bi].mode = BatchExecutionMode::Individual;
        out.push(("IndividualBatchWidth", s.clone(), m, e));
    }

    // Event order: open a gap at the end.
    if n_events > 0
        && let Some(bi) = a.batches.iter().position(|b| !b.rows.is_empty())
    {
        let mut m = a.clone();
        m.batches[bi].rows[0].event_index = n_events + 3;
        out.push(("InvalidEventOrder", s.clone(), m, e));
    }

    // Releases: name a decode row that does not finish its request.
    if let Some(r) = a
        .batches
        .iter()
        .filter(|b| b.phase == PlannedPhase::Decode)
        .flat_map(|b| &b.rows)
        .map(|r| r.request_id)
        .find(|r| {
            s.decoding
                .get(r)
                .is_some_and(|d| d.generated_tokens + 1 < d.max_new_tokens)
        })
    {
        let mut m = a.clone();
        m.predicted_releases.push(r);
        out.push(("InvalidRelease", s.clone(), m, e));
    }

    // Chunk ceiling and resumability on a prompt row of a resumable model.
    if let Some((model_id, len, non_final)) = a
        .batches
        .iter()
        .filter(|b| b.phase != PlannedPhase::Decode)
        .flat_map(|b| b.rows.iter().map(move |r| (b, r)))
        .find(|(b, r)| {
            s.models.get(&b.model_id).is_some_and(|m| m.chunked_prefill) && r.positions.len() >= 2
        })
        .map(|(b, r)| (b.model_id.clone(), r.positions.len(), !r.final_prompt_chunk))
    {
        let mut t = s.clone();
        t.limits.prefill_chunk_size = Some(len - 1);
        out.push(("ChunkTooLong", t, a.clone(), e));
        if non_final {
            let mut t = s.clone();
            if let Some(model) = t.models.get_mut(&model_id) {
                model.chunked_prefill = false;
            }
            out.push(("UnsupportedPartialPrefill", t, a.clone(), e));
        }
    }
    out
}

/// Random structural fuzzing: 1–3 random edits; only agreement is checked.
fn fuzz(sample: &Sample, rng: &mut u64) -> (PlanningSnapshot, ExecutionPlan) {
    let mut s = sample.snapshot.clone();
    let mut a = sample.plan.clone();
    let edits = 1 + (next(rng) % 3) as usize;
    for _ in 0..edits {
        match next(rng) % 9 {
            0 => a.generation = a.generation.wrapping_add(next(rng) % 2),
            1 if !a.batches.is_empty() => {
                let bi = (next(rng) as usize) % a.batches.len();
                let b = &mut a.batches[bi];
                if !b.rows.is_empty() {
                    let ri = (next(rng) as usize) % b.rows.len();
                    let r = &mut b.rows[ri];
                    let d = (next(rng) % 3) as usize;
                    r.positions = r.positions.start + d..r.positions.end + d;
                }
            }
            2 if !a.batches.is_empty() => {
                let bi = (next(rng) as usize) % a.batches.len();
                a.batches.remove(bi);
            }
            3 if !a.lifecycle.is_empty() => {
                let li = (next(rng) as usize) % a.lifecycle.len();
                a.lifecycle.remove(li);
            }
            4 if !a.deferrals.is_empty() => {
                let di = (next(rng) as usize) % a.deferrals.len();
                a.deferrals.remove(di);
            }
            5 => s.limits.max_active_requests = (next(rng) % 26) as usize,
            6 => s.limits.prefill_chunk_size = Some(1 + (next(rng) % 64) as usize),
            7 if !s.decoding.is_empty() => {
                let r = *s.decoding.keys().next().unwrap();
                a.predicted_releases.push(r);
            }
            8 if !a.batches.is_empty() => {
                let bi = (next(rng) as usize) % a.batches.len();
                a.batches[bi].mode = BatchExecutionMode::Individual;
            }
            _ => {}
        }
    }
    (s, a)
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Workloads, policies, adaptive controllers.
// ---------------------------------------------------------------------------

fn headroom_theta(log2: i64) -> Vec<(&'static str, String)> {
    vec![("log2_headroom", log2.to_string())]
}

/// θ selects worst-case (9) or watermark headroom 2^θ; 10 selects
/// unconstrained admission, which turns Γ's KV check off.
fn theta_policy(theta: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let log2: i64 = theta
        .iter()
        .find(|(k, _)| *k == "log2_headroom")
        .and_then(|(_, v)| v.parse().ok())
        .ok_or_else(|| PolicyError::InvalidCandidate("theta".into()))?;
    let base = ComposedPolicy::default().with_preemption(NewestFirst);
    Ok(match log2 {
        9 => Box::new(base.with_admission(MemoryAwareAdmission)),
        10 => Box::new(base.with_admission(UnconstrainedAdmission)),
        _ => Box::new(base.with_admission(WatermarkAdmission {
            headroom_tokens: 1 << log2,
        })),
    })
}

/// An adaptive policy whose optimizer picks a random θ from `domain` at
/// every committed iteration: the most adversarial θ trajectory we can
/// drive through the real wrapper.
fn random_adaptive(domain: Vec<i64>, seed: u64) -> AdaptivePolicy {
    let mut state = seed;
    let start = headroom_theta(9);
    AdaptivePolicy::new(theta_policy(&start).unwrap(), start)
        .with_fallback(
            ComposedPolicy::default()
                .with_admission(MemoryAwareAdmission)
                .with_preemption(NewestFirst),
        )
        .with_factory(theta_policy)
        .with_optimization_interval(1)
        .with_optimizer(
            move |_current: &[(&'static str, String)],
                  _obs: &[AdaptiveRequestObservation],
                  _c: AdaptiveConstraints|
                  -> Result<_, AdaptiveOptimizationError> {
                let pick = domain[(next(&mut state) as usize) % domain.len()];
                Ok(headroom_theta(pick))
            },
        )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reservation {
    /// Reservation covers the worst case for every request.
    WorstCase,
    /// Reservation may be below the worst case.
    Optimistic,
    /// No KV enforcement.
    Unchecked,
}

fn policies(seed: u64) -> Vec<(&'static str, Box<dyn ExecutionPolicy>, Reservation)> {
    let fifo = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    vec![
        ("fifo", Box::new(fifo()), Reservation::WorstCase),
        (
            "priority",
            Box::new(fifo().with_scheduler(PriorityScheduler::new())),
            Reservation::WorstCase,
        ),
        (
            "edf",
            Box::new(fifo().with_scheduler(EarliestDeadlineFirst)),
            Reservation::WorstCase,
        ),
        (
            "sjf",
            Box::new(fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt))),
            Reservation::WorstCase,
        ),
        (
            "backfill",
            Box::new(fifo().with_head_of_line(HeadOfLine::Backfill { max_skipped: 4 })),
            Reservation::WorstCase,
        ),
        (
            "token-budget-64",
            Box::new(fifo().with_batch(FifoTokenBudgetPolicy::new(64).unwrap())),
            Reservation::WorstCase,
        ),
        (
            "watermark-32",
            Box::new(
                ComposedPolicy::default()
                    .with_admission(WatermarkAdmission {
                        headroom_tokens: 32,
                    })
                    .with_preemption(NewestFirst),
            ),
            Reservation::Optimistic,
        ),
        (
            "unconstrained",
            Box::new(
                ComposedPolicy::default()
                    .with_admission(UnconstrainedAdmission)
                    .with_preemption(NewestFirst),
            ),
            Reservation::Unchecked,
        ),
        (
            "adaptive-random-theta",
            Box::new(random_adaptive(vec![4, 5, 6, 7, 8, 9], seed)),
            Reservation::Optimistic,
        ),
        (
            "adaptive-random-theta-with-unchecked",
            Box::new(random_adaptive(vec![4, 6, 9, 10], seed)),
            Reservation::Unchecked,
        ),
    ]
}

fn traces(seed: u64, requests: usize) -> Vec<(&'static str, SchedulingTrace, SimulationConfig)> {
    let p = TraceParameters {
        seed,
        requests,
        mean_interarrival_micros: 2_000,
    };
    let fast = TraceParameters {
        mean_interarrival_micros: 1_000,
        ..p
    };
    let small = SimulationConfig::default();
    let geometry = KvGeometry::default();
    let pressure = SimulationConfig {
        max_active_requests: 24,
        kv: KvGeometry {
            capacity_bytes: 2_048 * geometry.bytes_per_token,
            ..geometry
        },
        ..SimulationConfig::default()
    };
    vec![
        ("overload", SchedulingTrace::overload(p), small),
        (
            "priority_classes",
            SchedulingTrace::priority_classes(p),
            small,
        ),
        ("deadlines", SchedulingTrace::deadlines(p, 504), small),
        (
            "mixed_short_long",
            SchedulingTrace::mixed_short_long(p),
            small,
        ),
        (
            "early_stopping",
            SchedulingTrace::early_stopping(fast),
            pressure,
        ),
        (
            "run_to_limit",
            SchedulingTrace::run_to_limit(fast),
            pressure,
        ),
    ]
}

// ---------------------------------------------------------------------------
// The study.
// ---------------------------------------------------------------------------

#[derive(Default, serde::Serialize)]
struct MutationTally {
    applied: u64,
    expected_error: u64,
    other_error: u64,
    accepted: u64,
    spec_agrees: u64,
}

#[derive(Default, serde::Serialize)]
struct PolicyTally {
    runs: u64,
    iterations: u64,
    plans_recorded: u64,
    plans_spec_true: u64,
    plans_kv_enforced: u64,
    slot_violations: u64,
    generation_violations: u64,
    reservation_over_capacity_iterations: u64,
    physical_over_reservation_iterations: u64,
    allocation_failed_preemptions: u64,
    planned_preemptions: u64,
}

#[derive(Default, serde::Serialize)]
struct Report {
    seeds: u64,
    requests_per_trace: usize,
    traces: Vec<String>,
    policies: BTreeMap<String, PolicyTally>,
    mutations: BTreeMap<String, MutationTally>,
    fuzz_cases: u64,
    fuzz_validator_accepts: u64,
    fuzz_agreements: u64,
    disagreements: Vec<String>,
}

#[test]
fn conformance_study() {
    let seeds: u64 = std::env::var("OLA_CONFORMANCE_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let requests = 100;
    let mutate_every = 7; // mutate a deterministic subsample of plans
    let fuzz_per_plan = 2;
    let mut report = Report {
        seeds,
        requests_per_trace: requests,
        ..Report::default()
    };
    let mut rng = 0x0C0F_FEE0_u64;

    for seed in 1..=seeds {
        for (trace_name, trace, config) in traces(seed, requests) {
            if seed == 1 {
                report.traces.push(trace_name.to_owned());
            }
            let capacity = config.kv.capacity_bytes;
            let max_active = config.max_active_requests;
            for (label, policy, _reservation) in policies(seed) {
                let store = Arc::new(Mutex::new(Vec::new()));
                let recording = Recording {
                    inner: policy,
                    store: store.clone(),
                };
                let tally = report.policies.entry(label.to_owned()).or_default();
                tally.runs += 1;
                let mut last_generation: Option<u64> = None;
                {
                    let observe =
                        &mut |iteration: &ExecutorIteration, _state: &ola_runtime::RuntimeState| {
                            tally.iterations += 1;
                            if iteration.active_requests > max_active {
                                tally.slot_violations += 1;
                            }
                            let g = iteration.execution_plan.generation;
                            if last_generation.is_some_and(|prev| g <= prev) {
                                tally.generation_violations += 1;
                            }
                            last_generation = Some(g);
                            if iteration.reserved_kv_bytes > capacity {
                                tally.reservation_over_capacity_iterations += 1;
                            }
                            if iteration.kv_cache_bytes > iteration.reserved_kv_bytes {
                                tally.physical_over_reservation_iterations += 1;
                            }
                            for event in &iteration.events {
                                if let ExecutorEvent::Preempted { reason, .. } = event {
                                    match reason {
                                        PreemptionReason::AllocationFailed => {
                                            tally.allocation_failed_preemptions += 1;
                                        }
                                        PreemptionReason::KvShortfall { .. } => {
                                            tally.planned_preemptions += 1;
                                        }
                                    }
                                }
                            }
                        };
                    run_observed_trial(&trace, recording, &config, &[], observe)
                        .unwrap_or_else(|e| panic!("{label} on {trace_name}/{seed}: {e}"));
                }

                let samples = std::mem::take(&mut *store.lock().unwrap());
                for (index, sample) in samples.iter().enumerate() {
                    tally.plans_recorded += 1;
                    tally.plans_kv_enforced += u64::from(sample.enforce);
                    let spec = gamma_spec(&sample.snapshot, &sample.plan, sample.enforce);
                    let actual = sample.plan.validate(&sample.snapshot, sample.enforce);
                    if spec {
                        tally.plans_spec_true += 1;
                    }
                    if spec != actual.is_ok() {
                        report.disagreements.push(format!(
                            "committed {label}/{trace_name}/{seed}#{index}: spec={spec} validate={actual:?}"
                        ));
                    }
                    if index % mutate_every != 0 {
                        continue;
                    }
                    for (expected, s, a, enforce) in mutations(sample) {
                        let t = report.mutations.entry(expected.to_owned()).or_default();
                        t.applied += 1;
                        let result = a.validate(&s, enforce);
                        let spec = gamma_spec(&s, &a, enforce);
                        match &result {
                            Ok(()) => t.accepted += 1,
                            Err(error) if error_name(error) == expected => t.expected_error += 1,
                            Err(_) => t.other_error += 1,
                        }
                        if spec == result.is_ok() {
                            t.spec_agrees += 1;
                        } else {
                            report.disagreements.push(format!(
                                "mutation {expected} {label}/{trace_name}/{seed}#{index}: spec={spec} validate={result:?}"
                            ));
                        }
                    }
                    for _ in 0..fuzz_per_plan {
                        let (s, a) = fuzz(sample, &mut rng);
                        let result = a.validate(&s, sample.enforce);
                        let spec = gamma_spec(&s, &a, sample.enforce);
                        report.fuzz_cases += 1;
                        report.fuzz_validator_accepts += u64::from(result.is_ok());
                        if spec == result.is_ok() {
                            report.fuzz_agreements += 1;
                        } else {
                            report.disagreements.push(format!(
                                "fuzz {label}/{trace_name}/{seed}#{index}: spec={spec} validate={result:?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    report.disagreements.truncate(50);

    if let Ok(path) = std::env::var("OLA_CONFORMANCE_REPORT") {
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }

    // Assertions: the theorems' claims.
    assert!(
        report.disagreements.is_empty(),
        "spec/validator disagreements: {:#?}",
        report.disagreements
    );
    for (label, tally) in &report.policies {
        assert_eq!(
            tally.plans_spec_true, tally.plans_recorded,
            "{label}: committed plan failed Γ"
        );
        assert_eq!(tally.slot_violations, 0, "{label}: slot invariant");
        assert_eq!(tally.generation_violations, 0, "{label}: generation");
    }
    for label in [
        "fifo",
        "priority",
        "edf",
        "sjf",
        "backfill",
        "token-budget-64",
    ] {
        let tally = &report.policies[label];
        assert_eq!(
            tally.reservation_over_capacity_iterations, 0,
            "{label}: Σ reserved ≤ K"
        );
        assert_eq!(
            tally.physical_over_reservation_iterations, 0,
            "{label}: phys ≤ reserved"
        );
        assert_eq!(
            tally.allocation_failed_preemptions, 0,
            "{label}: no allocation failure"
        );
    }
    for (name, tally) in &report.mutations {
        assert_eq!(tally.accepted, 0, "mutation {name} accepted");
    }
}
