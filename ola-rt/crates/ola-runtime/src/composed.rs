//! Default execution policy composed from replaceable sub-policies.
//!
//! See `EXECUTION_POLICY.md` in this crate for the design.

use std::collections::{BTreeMap, BTreeSet};

use ola_core::{ModelId, RequestId};
use ola_kernel::{KernelRegistry, KernelTarget};

use crate::{
    AdmissionContext, AdmissionDecision, AdmissionOutcome, AdmissionPolicy, AdmissionReason,
    AdmissionRecord, BatchCandidate, BatchContext, BatchExecutionMode, BatchHook, BatchPhase,
    BatchPolicy, CapabilityFallback, DecisionRecord, ExecutionPlan, ExecutionPolicy, FifoScheduler,
    GroupingRecord, LifecycleAction, NoPreemption, PlannedBatch, PlannedDeferral, PlannedKernel,
    PlannedPhase, PlannedRow, PlanningSnapshot, PolicyDecision, PolicyDescriptor, PolicyError,
    PreemptionCandidate, PreemptionPolicy, PreemptionReason, PreemptionRecord, PrefillBatcher,
    RankingRecord, Scheduler, SelectionRecord, SelectionStage, UnboundedBatchPolicy,
    UnbucketedPrefillBatcher, UnconstrainedAdmission,
};

/// Execution policy built from a scheduler, an admission policy, a batch
/// policy, and a prefill-grouping rule, each independently replaceable.
///
/// The default (FIFO, unconstrained admission, unbounded batching, one
/// prefill group per model) reproduces the executor's original behaviour.
pub struct ComposedPolicy {
    scheduler: Box<dyn Scheduler + Send>,
    admission: Box<dyn AdmissionPolicy>,
    batch: Box<dyn BatchPolicy>,
    grouping: Box<dyn PrefillBatcher>,
    head_of_line: HeadOfLine,
    preemption: Box<dyn PreemptionPolicy>,
    kernels: KernelRegistry,
}

/// What admission does when the policy defers a ranked request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HeadOfLine {
    /// Stop admitting for this iteration, so ranked order is never bypassed.
    #[default]
    Block,
    /// Skip a deferred request and try the next ranked one, up to
    /// `max_skipped` times per iteration. Skipped requests keep their queue
    /// position but can starve if smaller work keeps arriving.
    Backfill {
        /// Admission deferrals tolerated before admission stops.
        max_skipped: usize,
    },
}

impl HeadOfLine {
    const fn max_skipped(self) -> usize {
        match self {
            Self::Block => 0,
            Self::Backfill { max_skipped } => max_skipped,
        }
    }
}

impl Default for ComposedPolicy {
    fn default() -> Self {
        Self {
            scheduler: Box::new(FifoScheduler),
            admission: Box::new(UnconstrainedAdmission),
            batch: Box::new(UnboundedBatchPolicy),
            grouping: Box::new(UnbucketedPrefillBatcher),
            head_of_line: HeadOfLine::Block,
            preemption: Box::new(NoPreemption),
            kernels: KernelRegistry::new(),
        }
    }
}

impl ComposedPolicy {
    /// Replaces the scheduler that ranks queued work.
    #[must_use]
    pub fn with_scheduler(mut self, scheduler: impl Scheduler + Send + 'static) -> Self {
        self.scheduler = Box::new(scheduler);
        self
    }

    /// Replaces the admission policy.
    #[must_use]
    pub fn with_admission(mut self, admission: impl AdmissionPolicy + 'static) -> Self {
        self.admission = Box::new(admission);
        self
    }

    /// Replaces the per-iteration batch policy.
    #[must_use]
    pub fn with_batch(mut self, batch: impl BatchPolicy + 'static) -> Self {
        self.batch = Box::new(batch);
        self
    }

    /// Replaces the prefill-grouping rule. It affects tensor shape only, not
    /// event order, and never groups different models.
    #[must_use]
    pub fn with_grouping(mut self, grouping: impl PrefillBatcher + 'static) -> Self {
        self.grouping = Box::new(grouping);
        self
    }

    /// Replaces how victims are chosen when a model's KV pool cannot hold
    /// an iteration's growth. The default never preempts.
    #[must_use]
    pub fn with_preemption(mut self, preemption: impl PreemptionPolicy + 'static) -> Self {
        self.preemption = Box::new(preemption);
        self
    }

    /// Replaces the kernel providers consulted for every model invocation.
    ///
    /// A registry always retains a portable provider; callers add optimized
    /// providers before installing it here.
    #[must_use]
    pub fn with_kernel_registry(mut self, kernels: KernelRegistry) -> Self {
        self.kernels = kernels;
        self
    }

    /// Chooses live requests to preempt so each model's pool can hold this
    /// iteration's growth, recording every short pool.
    ///
    /// Demand assumes every live operation runs: a decode writes one
    /// position, needing a new block when its position starts one; a
    /// continued prefill writes its next chunk. Free blocks include
    /// evictable cached blocks. A victim frees the blocks it holds and no
    /// longer needs its own growth.
    ///
    /// Also returns, per pooled model, the blocks left for admissions once
    /// the remaining live work has grown.
    fn choose_preemptions(
        &self,
        snapshot: &PlanningSnapshot,
        record: &mut DecisionRecord,
    ) -> (Vec<(RequestId, PreemptionReason)>, BTreeMap<ModelId, usize>) {
        let mut victims = Vec::new();
        let mut room = BTreeMap::new();
        for (model_id, model) in &snapshot.models {
            let Some(pool) = model.kv_pool else {
                continue;
            };
            let block = pool.tokens_per_block.max(1);
            let mut candidates = Vec::new();
            for (request_id, view) in &snapshot.decoding {
                if &view.model_id == model_id {
                    candidates.push(PreemptionCandidate {
                        request_id: *request_id,
                        model_id: model_id.clone(),
                        priority: view.priority,
                        arrived_at: view.arrived_at,
                        held_blocks: view.position.div_ceil(block),
                        needed_blocks: usize::from(view.position % block == 0),
                        generated_tokens: view.generated_tokens,
                    });
                }
            }
            for (request_id, view) in &snapshot.prefilling {
                if &view.model_id == model_id {
                    let chunk = snapshot.initial_prefill_chunk(
                        model_id,
                        view.prompt_tokens.saturating_sub(view.cursor),
                    );
                    let held = view.cursor.div_ceil(block);
                    candidates.push(PreemptionCandidate {
                        request_id: *request_id,
                        model_id: model_id.clone(),
                        priority: view.priority,
                        arrived_at: view.arrived_at,
                        held_blocks: held,
                        needed_blocks: (view.cursor + chunk).div_ceil(block) - held,
                        generated_tokens: 0,
                    });
                }
            }
            let demand = candidates
                .iter()
                .map(|candidate| candidate.needed_blocks)
                .sum::<usize>();
            let free = pool.free_blocks.saturating_add(pool.cached_blocks);
            if demand <= free {
                room.insert(model_id.clone(), free - demand);
                continue;
            }
            let shortfall = demand - free;
            let mut freed = 0;
            let mut preempted = Vec::new();
            for request_id in self.preemption.order(&candidates) {
                if freed >= shortfall {
                    break;
                }
                if preempted.contains(&request_id) {
                    continue;
                }
                if let Some(candidate) = candidates
                    .iter()
                    .find(|candidate| candidate.request_id == request_id)
                {
                    freed += candidate.held_blocks + candidate.needed_blocks;
                    preempted.push(request_id);
                }
            }
            room.insert(model_id.clone(), freed.saturating_sub(shortfall));
            victims.extend(preempted.iter().map(|request_id| {
                (
                    *request_id,
                    PreemptionReason::KvShortfall {
                        shortfall_blocks: shortfall,
                    },
                )
            }));
            record.preemptions.push(PreemptionRecord {
                model_id: model_id.clone(),
                free_blocks: free,
                demand_blocks: demand,
                candidates,
                preempted,
            });
        }
        (victims, room)
    }

    /// Replaces what admission does after a deferral.
    #[must_use]
    pub const fn with_head_of_line(mut self, head_of_line: HeadOfLine) -> Self {
        self.head_of_line = head_of_line;
        self
    }

    /// Ranks up to `limit` queued requests and rejects impossible rankings.
    fn rank(
        &self,
        snapshot: &PlanningSnapshot,
        limit: usize,
    ) -> Result<Vec<RequestId>, PolicyError> {
        let ranked = self.scheduler.rank(snapshot, limit);
        let mut seen = BTreeSet::new();
        for request_id in &ranked {
            let queued = snapshot
                .queued
                .iter()
                .any(|request| request.request_id == *request_id);
            if ranked.len() > limit || !queued || !seen.insert(*request_id) {
                return Err(PolicyError::InvalidRanking(*request_id));
            }
        }
        Ok(ranked)
    }

    /// Plans one complete iteration from an immutable snapshot.
    ///
    /// Event indices follow the historical order: decodes, continued
    /// prefills, then admissions. Slots and reservations of requests that
    /// must finish by reaching their output limit are reused for admission;
    /// slots freed by EOS or failure are refilled next iteration.
    fn plan_iteration(
        &self,
        snapshot: &PlanningSnapshot,
        scheduled: &mut Vec<BatchCandidate>,
        record: &mut DecisionRecord,
        room: &mut Option<BTreeMap<ModelId, usize>>,
    ) -> Result<ExecutionPlan, PolicyError> {
        let mut execution_plan = ExecutionPlan {
            generation: snapshot.generation,
            at: snapshot.at,
            ..ExecutionPlan::default()
        };
        let decode_candidates = snapshot
            .decode_order
            .iter()
            .map(|request_id| BatchCandidate {
                request_id: *request_id,
                phase: BatchPhase::Decode,
                token_cost: 1,
            })
            .collect::<Vec<_>>();
        let decode_count = decode_candidates.len();
        let selected = self.select_batch_candidates(
            snapshot,
            scheduled,
            &decode_candidates,
            SelectionStage::Decode,
            record,
        )?;
        let mut decode_rows = Vec::new();
        for (event_index, candidate) in decode_candidates.into_iter().enumerate() {
            if !selected.contains(&candidate.request_id) {
                execution_plan
                    .deferrals
                    .push(budget_deferral(candidate, event_index));
                continue;
            }
            scheduled.push(candidate);
            let decode = &snapshot.decoding[&candidate.request_id];
            if decode.reaches_limit() {
                execution_plan.predicted_releases.push(candidate.request_id);
            }
            decode_rows.push((
                decode.model_id.clone(),
                PlannedRow {
                    request_id: candidate.request_id,
                    event_index,
                    positions: decode.position..decode.position.saturating_add(1),
                    final_prompt_chunk: false,
                },
            ));
        }
        execution_plan
            .batches
            .extend(plan_decode_batches(snapshot, decode_rows, record));

        let prefill_candidates = snapshot
            .prefill_order
            .iter()
            .map(|request_id| {
                let pending = &snapshot.prefilling[request_id];
                BatchCandidate {
                    request_id: *request_id,
                    phase: BatchPhase::Prefill,
                    token_cost: snapshot.initial_prefill_chunk(
                        &pending.model_id,
                        pending.prompt_tokens.saturating_sub(pending.cursor),
                    ),
                }
            })
            .collect::<Vec<_>>();
        let prefill_count = prefill_candidates.len();
        let selected = self.select_batch_candidates(
            snapshot,
            scheduled,
            &prefill_candidates,
            SelectionStage::ContinuedPrefill,
            record,
        )?;
        let mut prefill_rows = Vec::new();
        for (index, candidate) in prefill_candidates.into_iter().enumerate() {
            let event_index = decode_count.saturating_add(index);
            if !selected.contains(&candidate.request_id) {
                execution_plan
                    .deferrals
                    .push(budget_deferral(candidate, event_index));
                continue;
            }
            scheduled.push(candidate);
            let pending = &snapshot.prefilling[&candidate.request_id];
            let end = pending.cursor.saturating_add(candidate.token_cost);
            let final_prompt_chunk = end == pending.prompt_tokens;
            if final_prompt_chunk && pending.max_new_tokens <= 1 {
                execution_plan.predicted_releases.push(candidate.request_id);
            }
            prefill_rows.push((
                pending.model_id.clone(),
                PlannedRow {
                    request_id: candidate.request_id,
                    event_index,
                    positions: pending.cursor..end,
                    final_prompt_chunk,
                },
            ));
        }
        execution_plan.batches.extend(self.plan_prefill_batches(
            snapshot,
            PlannedPhase::ContinuedPrefill,
            prefill_rows,
            record,
        ));

        self.plan_admissions(
            snapshot,
            scheduled,
            &mut execution_plan,
            decode_count.saturating_add(prefill_count),
            record,
            room,
        )?;
        self.plan_kernels(snapshot, &mut execution_plan.batches)?;
        Ok(execution_plan)
    }

    fn plan_kernels(
        &self,
        snapshot: &PlanningSnapshot,
        batches: &mut [PlannedBatch],
    ) -> Result<(), PolicyError> {
        for batch in batches {
            if batch.phase == PlannedPhase::Prefill {
                batch.kernel = None;
                continue;
            }
            let Some(profile) = snapshot.model(&batch.model_id).paged_attention else {
                batch.kernel = None;
                continue;
            };
            let sequence_tokens = batch
                .rows
                .iter()
                .map(|row| row.positions.end)
                .max()
                .expect("planned batches contain at least one row");
            let requirements = profile.requirements(sequence_tokens).map_err(|error| {
                PolicyError::KernelSelection {
                    model_id: batch.model_id.clone(),
                    message: error.to_string(),
                }
            })?;
            let selection = self.kernels.select(requirements).map_err(|error| {
                PolicyError::KernelSelection {
                    model_id: batch.model_id.clone(),
                    message: error.to_string(),
                }
            })?;
            batch.kernel = Some(PlannedKernel {
                requirements,
                selection,
            });
        }
        Ok(())
    }

    /// Ranks queued work and plans admissions into the remaining slots.
    ///
    /// Each ranked request consumes one admission attempt whether it is
    /// admitted, rejected, or fails preparation. Planning stops at the first
    /// admission or budget deferral, preserving FIFO head-of-line semantics.
    fn plan_admissions(
        &self,
        snapshot: &PlanningSnapshot,
        scheduled: &mut Vec<BatchCandidate>,
        execution_plan: &mut ExecutionPlan,
        first_event_index: usize,
        record: &mut DecisionRecord,
        room: &mut Option<BTreeMap<ModelId, usize>>,
    ) -> Result<(), PolicyError> {
        let mut released_total = 0_u64;
        let mut released_by_model = BTreeMap::<&ModelId, u64>::new();
        for request_id in &execution_plan.predicted_releases {
            let (model_id, bytes) = match snapshot.decoding.get(request_id) {
                Some(decode) => (&decode.model_id, decode.reserved_kv_bytes),
                None => {
                    let pending = &snapshot.prefilling[request_id];
                    (&pending.model_id, pending.reserved_kv_bytes)
                }
            };
            released_total = released_total.saturating_add(bytes);
            *released_by_model.entry(model_id).or_default() += bytes;
        }
        let live_after = snapshot
            .live_requests()
            .saturating_sub(execution_plan.predicted_releases.len());
        let admission_count = snapshot
            .limits
            .max_active_requests
            .saturating_sub(live_after)
            .min(snapshot.queued.len());
        // Skipped requests consume no slot, so rank enough to fill every
        // slot after the maximum number of skips, plus one to record the
        // first request left waiting for a slot.
        let max_skipped = self.head_of_line.max_skipped();
        let limit = admission_count
            .saturating_add(max_skipped)
            .saturating_add(1);
        let ranked = self.rank(snapshot, limit)?;
        record.ranking = RankingRecord {
            slots_available: admission_count,
            limit,
            ranked: ranked.clone(),
        };
        let queued = snapshot
            .queued
            .iter()
            .map(|request| (request.request_id, request))
            .collect::<BTreeMap<_, _>>();

        let mut rows = Vec::new();
        let mut pending_total = 0_u64;
        let mut pending_by_model = BTreeMap::<&ModelId, u64>::new();
        // Admissions, rejections, and preparation failures each consume one
        // attempt; backfill skips do not.
        let mut attempts = 0;
        for (offset, request_id) in ranked.into_iter().enumerate() {
            if attempts == admission_count {
                execution_plan
                    .deferrals
                    .push(PlannedDeferral::Slots { request_id });
                record.admissions.push(AdmissionRecord {
                    request_id,
                    outcome: AdmissionOutcome::NoSlot,
                });
                break;
            }
            let event_index = first_event_index.saturating_add(offset);
            let request = queued[&request_id];
            let model_id = &request.model_id;
            let model = snapshot.model(model_id);
            let prompt_tokens = match (&request.prompt_tokens, model.attached) {
                (_, false) => Err(format!("model `{model_id}` has no executable attached")),
                (Ok(tokens), true) => Ok(*tokens),
                (Err(message), true) => Err(message.clone()),
            };
            let prompt_tokens = match prompt_tokens {
                Ok(tokens) => tokens,
                Err(message) => {
                    record.admissions.push(AdmissionRecord {
                        request_id,
                        outcome: AdmissionOutcome::PreparationFailed {
                            message: message.clone(),
                        },
                    });
                    execution_plan.lifecycle.push(LifecycleAction::Fail {
                        request_id,
                        message,
                        event_index,
                    });
                    attempts += 1;
                    continue;
                }
            };
            let model_released = released_by_model.get(model_id).copied().unwrap_or(0);
            let model_pending = pending_by_model.get(model_id).copied().unwrap_or(0);
            let cached = request
                .cached_prefix_tokens
                .min(prompt_tokens.saturating_sub(1));
            let context = AdmissionContext {
                request_id,
                model_id: model_id.clone(),
                prompt_tokens,
                cached_prefix_tokens: cached,
                max_new_tokens: request.max_new_tokens,
                kv_cache: model.kv_cache,
                memory: snapshot.memory,
                reserved_kv_bytes: snapshot
                    .reserved_kv_bytes()
                    .saturating_sub(released_total)
                    .saturating_add(pending_total),
                model_reserved_kv_bytes: snapshot
                    .model_reserved_kv_bytes(model_id)
                    .saturating_sub(model_released)
                    .saturating_add(model_pending),
            };
            let mut decision = self.admission.decide(&context);
            // Physical blocks, when the planner tracks them: the first
            // prefill must fit what live growth and earlier admissions left.
            let mut first_prefill_blocks = None;
            if let (AdmissionDecision::Admit { .. }, Some(room), Some(pool)) =
                (&decision, room.as_ref(), model.kv_pool)
            {
                // Blocks for the computed tokens, plus cached blocks that
                // attaching takes off the evictable list.
                let block = pool.tokens_per_block.max(1);
                let computed = snapshot.initial_prefill_chunk(model_id, prompt_tokens - cached);
                let needed_blocks = (cached + computed).div_ceil(block) - cached / block
                    + request.cached_prefix_evictable_blocks;
                let available_blocks = room.get(model_id).copied().unwrap_or(0);
                if needed_blocks > available_blocks {
                    decision = AdmissionDecision::Defer(AdmissionReason::KvPoolFull {
                        needed_blocks,
                        available_blocks,
                    });
                } else {
                    first_prefill_blocks = Some(needed_blocks);
                }
            }
            let decided = |budget_selected| AdmissionRecord {
                request_id,
                outcome: AdmissionOutcome::Decided {
                    context: context.clone(),
                    decision: decision.clone(),
                    budget_selected,
                },
            };
            let reserved_kv_bytes = match decision.clone() {
                AdmissionDecision::Admit { reserved_kv_bytes } => reserved_kv_bytes,
                AdmissionDecision::Defer(reason) => {
                    record.admissions.push(decided(None));
                    execution_plan.deferrals.push(PlannedDeferral::Admission {
                        request_id,
                        reason,
                        event_index,
                    });
                    if record.backfill_skips.len() < max_skipped {
                        record.backfill_skips.push(request_id);
                        continue;
                    }
                    break;
                }
                AdmissionDecision::Reject(reason) => {
                    record.admissions.push(decided(None));
                    execution_plan.lifecycle.push(LifecycleAction::Reject {
                        request_id,
                        reason,
                        event_index,
                    });
                    attempts += 1;
                    continue;
                }
            };
            let candidate = BatchCandidate {
                request_id,
                phase: BatchPhase::Prefill,
                token_cost: snapshot.initial_prefill_chunk(model_id, prompt_tokens - cached),
            };
            let budget_selected = !self
                .select_batch_candidates(
                    snapshot,
                    scheduled,
                    &[candidate],
                    SelectionStage::Admission,
                    record,
                )?
                .is_empty();
            record.admissions.push(decided(Some(budget_selected)));
            if !budget_selected {
                execution_plan
                    .deferrals
                    .push(budget_deferral(candidate, event_index));
                break;
            }
            scheduled.push(candidate);
            attempts += 1;
            if let (Some(room), Some(blocks)) = (room.as_mut(), first_prefill_blocks)
                && let Some(left) = room.get_mut(model_id)
            {
                *left = left.saturating_sub(blocks);
            }
            pending_total = pending_total.saturating_add(reserved_kv_bytes);
            *pending_by_model.entry(model_id).or_default() += reserved_kv_bytes;
            execution_plan.lifecycle.push(LifecycleAction::Admit {
                request_id,
                model_id: model_id.clone(),
                reserved_kv_bytes,
            });
            rows.push((
                model_id.clone(),
                PlannedRow {
                    request_id,
                    event_index,
                    positions: cached..cached + candidate.token_cost,
                    final_prompt_chunk: cached + candidate.token_cost == prompt_tokens,
                },
            ));
        }
        execution_plan.batches.extend(self.plan_prefill_batches(
            snapshot,
            PlannedPhase::Prefill,
            rows,
            record,
        ));
        Ok(())
    }

    /// Groups prompt rows by loaded model and prefill-batcher bucket.
    ///
    /// The batcher sees each model's complete row set, so workload-aware
    /// policies make one decision per model. Groups keep first-appearance
    /// order, so bucketing never reorders events.
    fn plan_prefill_batches(
        &self,
        snapshot: &PlanningSnapshot,
        phase: PlannedPhase,
        rows: Vec<(ModelId, PlannedRow)>,
        record: &mut DecisionRecord,
    ) -> Vec<PlannedBatch> {
        let mut model_positions: Vec<(ModelId, Vec<usize>)> = Vec::new();
        for (position, (model_id, _)) in rows.iter().enumerate() {
            if let Some((_, positions)) = model_positions
                .iter_mut()
                .find(|(existing, _)| existing == model_id)
            {
                positions.push(position);
            } else {
                model_positions.push((model_id.clone(), vec![position]));
            }
        }
        let mut assigned_buckets = vec![0; rows.len()];
        for (model_id, positions) in model_positions {
            let lengths = positions
                .iter()
                .map(|position| rows[*position].1.input_tokens())
                .collect::<Vec<_>>();
            let decision = self.grouping.group(&lengths);
            for (position, bucket) in positions.into_iter().zip(&decision.buckets) {
                assigned_buckets[position] = *bucket;
            }
            record.grouping.push(GroupingRecord {
                phase,
                model_id,
                prompt_tokens: lengths,
                decision,
            });
        }

        let mut groups: Vec<(usize, PlannedBatch)> = Vec::new();
        for ((model_id, row), bucket) in rows.into_iter().zip(assigned_buckets) {
            if let Some((_, group)) = groups
                .iter_mut()
                .find(|(key, group)| group.model_id == model_id && *key == bucket)
            {
                group.rows.push(row);
            } else {
                groups.push((
                    bucket,
                    PlannedBatch {
                        phase,
                        model_id,
                        mode: BatchExecutionMode::Individual,
                        kernel: None,
                        rows: vec![row],
                    },
                ));
            }
        }
        assign_batch_modes(
            snapshot,
            groups.into_iter().map(|(_, group)| group).collect(),
            record,
        )
    }

    fn select_batch_candidates(
        &self,
        snapshot: &PlanningSnapshot,
        scheduled: &[BatchCandidate],
        candidates: &[BatchCandidate],
        stage: SelectionStage,
        record: &mut DecisionRecord,
    ) -> Result<BTreeSet<RequestId>, PolicyError> {
        let context = BatchContext {
            scheduled: scheduled.to_vec(),
            candidates: candidates.to_vec(),
            pressure: snapshot.pressure(),
        };
        let returned = self.batch.select(&context);
        let mut selected = BTreeSet::new();
        for &request_id in &returned {
            let is_candidate = candidates
                .iter()
                .any(|candidate| candidate.request_id == request_id);
            if !is_candidate || !selected.insert(request_id) {
                return Err(PolicyError::InvalidBatchSelection(request_id));
            }
        }
        record.selections.push(SelectionRecord {
            stage,
            context,
            selected: returned,
        });
        Ok(selected)
    }
}

impl ExecutionPolicy for ComposedPolicy {
    fn descriptor(&self) -> PolicyDescriptor {
        let head_of_line = match self.head_of_line {
            HeadOfLine::Block => "block".to_owned(),
            HeadOfLine::Backfill { max_skipped } => format!("backfill-{max_skipped}"),
        };
        let mut descriptor = PolicyDescriptor::new("composed")
            .with_config("head_of_line", head_of_line)
            .with_component(self.scheduler.descriptor())
            .with_component(self.admission.descriptor())
            .with_component(self.batch.descriptor())
            .with_component(self.grouping.descriptor());
        let optimized = self
            .kernels
            .capabilities()
            .into_iter()
            .filter(|capability| capability.descriptor().target() != KernelTarget::Portable)
            .map(|capability| capability.descriptor().name())
            .collect::<Vec<_>>();
        if !optimized.is_empty() {
            descriptor = descriptor.with_component(
                PolicyDescriptor::new("kernel-registry")
                    .with_config("optimized", optimized.join("|")),
            );
        }
        // Listed only when enabled, so policies without preemption keep the
        // descriptors their reports already carry.
        let preemption = self.preemption.descriptor();
        if preemption.name == "no-preemption" {
            descriptor
        } else {
            descriptor.with_component(preemption)
        }
    }

    fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
        let mut scheduled = Vec::new();
        let mut record = DecisionRecord {
            policy: self.descriptor(),
            ..DecisionRecord::default()
        };
        // With preemption enabled, admissions must also fit the pool's free
        // blocks, or a resumed request would fail its own re-prefill.
        let (victims, room) = if self.preemption.descriptor().name == "no-preemption" {
            (Vec::new(), None)
        } else {
            let (victims, room) = self.choose_preemptions(snapshot, &mut record);
            (victims, Some(room))
        };
        let mut room = room;
        if victims.is_empty() {
            let plan = self.plan_iteration(snapshot, &mut scheduled, &mut record, &mut room)?;
            return Ok(PolicyDecision { plan, record });
        }
        // The rest of the iteration is planned as if the victims had already
        // left, and nothing new is admitted to a pool being relieved, so the
        // freed blocks go to the work that stays.
        let pressured = record
            .preemptions
            .iter()
            .filter(|pool| !pool.preempted.is_empty())
            .map(|pool| pool.model_id.clone())
            .collect::<BTreeSet<_>>();
        let mut remaining = snapshot.clone();
        for (request_id, _) in &victims {
            remaining.decoding.remove(request_id);
            remaining.prefilling.remove(request_id);
        }
        remaining
            .decode_order
            .retain(|request_id| remaining.decoding.contains_key(request_id));
        remaining
            .prefill_order
            .retain(|request_id| remaining.prefilling.contains_key(request_id));
        remaining
            .queued
            .retain(|queued| !pressured.contains(&queued.model_id));
        let mut plan = self.plan_iteration(&remaining, &mut scheduled, &mut record, &mut room)?;
        shift_event_indices(&mut plan, victims.len());
        plan.lifecycle.splice(
            0..0,
            victims
                .into_iter()
                .enumerate()
                .map(
                    |(event_index, (request_id, reason))| LifecycleAction::Preempt {
                        request_id,
                        reason,
                        event_index,
                    },
                ),
        );
        Ok(PolicyDecision { plan, record })
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        self.admission.enforces_model_kv_capacity()
    }

    /// Recovery requeues work just as planned preemption does, so it is on
    /// exactly when preemption is.
    fn recovers_kv_exhaustion(&self) -> bool {
        self.preemption.descriptor().name != "no-preemption"
    }
}

/// Makes room for `count` events at the start of an iteration.
fn shift_event_indices(plan: &mut ExecutionPlan, count: usize) {
    for row in plan.batches.iter_mut().flat_map(|batch| &mut batch.rows) {
        row.event_index += count;
    }
    for action in &mut plan.lifecycle {
        match action {
            LifecycleAction::Reject { event_index, .. }
            | LifecycleAction::Fail { event_index, .. }
            | LifecycleAction::Preempt { event_index, .. } => *event_index += count,
            LifecycleAction::Admit { .. } => {}
        }
    }
    for deferral in &mut plan.deferrals {
        match deferral {
            PlannedDeferral::Admission { event_index, .. }
            | PlannedDeferral::Budget { event_index, .. } => *event_index += count,
            PlannedDeferral::Slots { .. } => {}
        }
    }
}

/// Groups decode rows by loaded model in first-appearance order.
fn plan_decode_batches(
    snapshot: &PlanningSnapshot,
    rows: Vec<(ModelId, PlannedRow)>,
    record: &mut DecisionRecord,
) -> Vec<PlannedBatch> {
    let mut groups: Vec<PlannedBatch> = Vec::new();
    for (model_id, row) in rows {
        if let Some(group) = groups.iter_mut().find(|group| group.model_id == model_id) {
            group.rows.push(row);
        } else {
            groups.push(PlannedBatch {
                phase: PlannedPhase::Decode,
                model_id,
                mode: BatchExecutionMode::Individual,
                kernel: None,
                rows: vec![row],
            });
        }
    }
    assign_batch_modes(snapshot, groups, record)
}

/// Marks compatible groups as tensor invocations when the model declares
/// the matching hook, and splits the rest into one-row invocations.
fn assign_batch_modes(
    snapshot: &PlanningSnapshot,
    groups: Vec<PlannedBatch>,
    record: &mut DecisionRecord,
) -> Vec<PlannedBatch> {
    let mut batches = Vec::with_capacity(groups.len());
    for group in groups {
        let capabilities = snapshot.model(&group.model_id).batch;
        let (supported, hook) = match group.phase {
            PlannedPhase::Decode => (capabilities.decode, BatchHook::Decode),
            PlannedPhase::Prefill | PlannedPhase::ContinuedPrefill => {
                if group.rows.iter().all(is_complete_prompt) {
                    (capabilities.prefill, BatchHook::Prefill)
                } else {
                    (capabilities.chunked_prefill, BatchHook::ChunkedPrefill)
                }
            }
        };
        if !supported && group.rows.len() > 1 {
            record.capability_fallbacks.push(CapabilityFallback {
                phase: group.phase,
                model_id: group.model_id.clone(),
                rows: group.rows.len(),
                missing: hook,
            });
        }
        if supported && group.rows.len() > 1 {
            batches.push(PlannedBatch {
                mode: BatchExecutionMode::Tensor,
                ..group
            });
            continue;
        }
        let PlannedBatch {
            phase,
            model_id,
            rows,
            ..
        } = group;
        batches.extend(rows.into_iter().map(|row| PlannedBatch {
            phase,
            model_id: model_id.clone(),
            mode: BatchExecutionMode::Individual,
            kernel: None,
            rows: vec![row],
        }));
    }
    batches
}

const fn budget_deferral(candidate: BatchCandidate, event_index: usize) -> PlannedDeferral {
    PlannedDeferral::Budget {
        request_id: candidate.request_id,
        phase: candidate.phase,
        token_cost: candidate.token_cost,
        event_index,
    }
}

pub(crate) fn is_complete_prompt(row: &PlannedRow) -> bool {
    row.positions.start == 0 && row.final_prompt_chunk
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ola_core::{Priority, Timestamp};
    use ola_kernel::{
        KernelCapability, KernelDType, KernelDescriptor, KernelOperation, PagedAttentionKernel,
        PagedAttentionProfile, PagedAttentionProvider, PortablePagedAttention,
    };
    use ola_model::{BatchCapabilities, KvCacheProfile};

    use super::*;
    use crate::{
        AdmissionOutcome, AdmissionReason, CostAwarePrefillBatcher, DecodeView, ExecutorLimits,
        FifoTokenBudgetPolicy, MemoryAwareAdmission, MemoryState, ModelSnapshot,
        PolicyMeasurements, PrefillView, QueuedRequest,
    };

    fn id(value: u64) -> RequestId {
        RequestId::new(value)
    }

    fn model() -> ModelId {
        "tiny".into()
    }

    /// Pages hold four tokens at eight bytes each, so every page is 32 bytes.
    fn snapshot(
        max_active_requests: usize,
        kv_capacity_bytes: u64,
        batch: BatchCapabilities,
        queued: &[(u64, usize, usize)],
    ) -> PlanningSnapshot {
        PlanningSnapshot {
            at: Timestamp::from_micros(100),
            generation: 3,
            limits: ExecutorLimits {
                max_active_requests,
                prefill_chunk_size: None,
            },
            queued: queued
                .iter()
                .map(|&(request, prompt_tokens, max_new_tokens)| QueuedRequest {
                    request_id: id(request),
                    cached_prefix_tokens: 0,
                    cached_prefix_evictable_blocks: 0,
                    model_id: model(),
                    prompt_tokens: Ok(prompt_tokens),
                    max_new_tokens,
                    priority: Priority::default(),
                    deadline: None,
                    queued_at: Timestamp::ZERO,
                })
                .collect(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::from([(
                model(),
                ModelSnapshot {
                    attached: true,
                    batch,
                    chunked_prefill: true,
                    kv_cache: KvCacheProfile::new(8, 4, kv_capacity_bytes),
                    kv_pool: None,
                    paged_attention: None,
                },
            )]),
            memory: MemoryState::default(),
            device: None,
            measurements: PolicyMeasurements::default(),
        }
    }

    fn add_decode(
        snapshot: &mut PlanningSnapshot,
        request: u64,
        generated: usize,
        max: usize,
        reserved: u64,
    ) {
        snapshot.decoding.insert(
            id(request),
            DecodeView {
                model_id: model(),
                position: 4,
                generated_tokens: generated,
                max_new_tokens: max,
                reserved_kv_bytes: reserved,
                priority: ola_core::Priority::NORMAL,
                arrived_at: Timestamp::ZERO,
            },
        );
        snapshot.decode_order.push(id(request));
    }

    fn plan(policy: &ComposedPolicy, snapshot: &PlanningSnapshot) -> PolicyDecision {
        let decision = policy.plan(snapshot).unwrap();
        decision
            .plan
            .validate(snapshot, policy.enforces_model_kv_capacity())
            .unwrap();
        decision
    }

    struct CpuTestProvider {
        max_pages: usize,
    }

    impl PagedAttentionProvider for CpuTestProvider {
        fn capability(&self) -> KernelCapability {
            KernelCapability::new(
                KernelDescriptor::new(
                    "cpu-test-paged-attention",
                    KernelOperation::PagedAttentionDecode,
                    KernelTarget::Cpu,
                ),
                [KernelTarget::Cpu],
                [KernelDType::F32],
            )
            .with_head_dimensions([128])
            .with_gqa_ratios([4])
            .with_max_pages(self.max_pages)
        }

        fn priority(&self) -> u16 {
            10
        }

        fn create(&self) -> Box<dyn PagedAttentionKernel> {
            Box::new(PortablePagedAttention::new())
        }
    }

    fn kernel_policy(max_pages: usize) -> ComposedPolicy {
        let mut registry = KernelRegistry::new();
        registry.register(CpuTestProvider { max_pages }).unwrap();
        ComposedPolicy::default().with_kernel_registry(registry)
    }

    fn enable_kernel_profile(snapshot: &mut PlanningSnapshot) {
        snapshot.models.get_mut(&model()).unwrap().paged_attention = Some(
            PagedAttentionProfile::new(KernelTarget::Cpu, KernelDType::F32, 32, 8, 128, 4).unwrap(),
        );
    }

    #[test]
    fn dense_initial_prefill_has_no_paged_decode_selection() {
        let mut snapshot = snapshot(4, 1_024, BatchCapabilities::ALL, &[(5, 3, 2)]);
        enable_kernel_profile(&mut snapshot);

        let decision = plan(&kernel_policy(8), &snapshot);

        assert_eq!(decision.plan.batches.len(), 1);
        assert_eq!(decision.plan.batches[0].phase, PlannedPhase::Prefill);
        assert!(decision.plan.batches[0].kernel.is_none());
    }

    #[test]
    fn kernel_selection_runs_after_batch_policy_and_uses_only_selected_rows() {
        let mut snapshot = snapshot(4, 1_024, BatchCapabilities::ALL, &[]);
        add_decode(&mut snapshot, 1, 1, 8, 32);
        snapshot.decoding.get_mut(&id(1)).unwrap().position = 3;
        add_decode(&mut snapshot, 2, 1, 8, 32);
        snapshot.decoding.get_mut(&id(2)).unwrap().position = 8;
        enable_kernel_profile(&mut snapshot);

        let wide = plan(&kernel_policy(1), &snapshot);
        let wide_kernel = wide.plan.batches[0].kernel.as_ref().unwrap();
        assert_eq!(wide_kernel.requirements.page_count(), 3);
        assert!(wide_kernel.selection.is_fallback());

        let budgeted = kernel_policy(1).with_batch(FifoTokenBudgetPolicy::new(1).unwrap());
        let budgeted = plan(&budgeted, &snapshot);
        let kernel = budgeted.plan.batches[0].kernel.as_ref().unwrap();
        assert_eq!(kernel.requirements.page_count(), 1);
        assert_eq!(
            kernel.selection.descriptor().name(),
            "cpu-test-paged-attention"
        );
        assert!(!kernel.selection.is_fallback());
        assert!(matches!(
            budgeted.plan.deferrals.as_slice(),
            [PlannedDeferral::Budget { request_id, .. }] if *request_id == id(2)
        ));
    }

    #[test]
    fn deferred_memory_admission_does_not_change_live_kernel_selection() {
        let mut snapshot = snapshot(2, 128, BatchCapabilities::ALL, &[(5, 5, 4)]);
        add_decode(&mut snapshot, 1, 1, 8, 96);
        snapshot.decoding.get_mut(&id(1)).unwrap().position = 3;
        enable_kernel_profile(&mut snapshot);
        let policy = kernel_policy(1).with_admission(MemoryAwareAdmission);

        let decision = plan(&policy, &snapshot);

        assert!(decision.plan.lifecycle.is_empty());
        assert!(matches!(
            decision.plan.deferrals.as_slice(),
            [PlannedDeferral::Admission { request_id, .. }] if *request_id == id(5)
        ));
        let kernel = decision.plan.batches[0].kernel.as_ref().unwrap();
        assert_eq!(
            kernel.selection.descriptor().name(),
            "cpu-test-paged-attention"
        );
        assert_eq!(kernel.requirements.page_count(), 1);
    }

    #[test]
    fn identical_snapshots_produce_identical_decisions() {
        let mut snapshot = snapshot(
            4,
            1_024,
            BatchCapabilities::ALL,
            &[(5, 3, 2), (6, 9, 2), (7, 1, 2)],
        );
        add_decode(&mut snapshot, 1, 1, 8, 32);
        snapshot.prefilling.insert(
            id(2),
            PrefillView {
                model_id: model(),
                cursor: 2,
                prompt_tokens: 6,
                max_new_tokens: 3,
                reserved_kv_bytes: 32,
                priority: ola_core::Priority::NORMAL,
                arrived_at: Timestamp::ZERO,
            },
        );
        snapshot.prefill_order.push(id(2));
        snapshot.limits.prefill_chunk_size = Some(2);
        let build = || {
            ComposedPolicy::default()
                .with_admission(MemoryAwareAdmission)
                .with_batch(FifoTokenBudgetPolicy::new(5).unwrap())
                .with_grouping(CostAwarePrefillBatcher::new(4))
        };

        let first = plan(&build(), &snapshot);
        let second = plan(&build(), &snapshot.clone());

        assert_eq!(first, second);
        assert!(!first.record.selections.is_empty());
        assert!(!first.record.grouping.is_empty());
    }

    #[test]
    fn an_admission_deferral_blocks_later_requests_that_would_fit() {
        // A live decode holds one of two pages; the head needs two pages and
        // waits, so FIFO never considers the one-page request behind it.
        let mut snapshot = snapshot(4, 64, BatchCapabilities::ALL, &[(5, 5, 4), (6, 1, 4)]);
        add_decode(&mut snapshot, 1, 1, 8, 32);
        let policy = ComposedPolicy::default().with_admission(MemoryAwareAdmission);

        let decision = plan(&policy, &snapshot);

        assert!(decision.plan.lifecycle.is_empty());
        assert_eq!(
            decision.plan.deferrals,
            vec![PlannedDeferral::Admission {
                request_id: id(5),
                reason: AdmissionReason::ModelKvCapacity {
                    required_bytes: 64,
                    available_bytes: 32,
                },
                event_index: 1,
            }]
        );
        assert_eq!(decision.record.ranking.ranked, vec![id(5), id(6)]);
        assert_eq!(decision.record.admissions.len(), 1);
    }

    #[test]
    fn backfill_admits_smaller_work_past_a_memory_deferred_head() {
        // Same state as the head-of-line test: one page free, the head needs
        // two. Backfill skips it and admits the one-page request behind it.
        let mut snapshot = snapshot(4, 64, BatchCapabilities::ALL, &[(5, 5, 4), (6, 1, 4)]);
        add_decode(&mut snapshot, 1, 1, 8, 32);
        let policy = ComposedPolicy::default()
            .with_admission(MemoryAwareAdmission)
            .with_head_of_line(HeadOfLine::Backfill { max_skipped: 1 });

        let decision = plan(&policy, &snapshot);

        assert_eq!(decision.record.backfill_skips, vec![id(5)]);
        assert!(matches!(
            decision.plan.deferrals.as_slice(),
            [PlannedDeferral::Admission { request_id, event_index: 1, .. }] if *request_id == id(5)
        ));
        assert!(matches!(
            decision.plan.lifecycle.as_slice(),
            [LifecycleAction::Admit { request_id, .. }] if *request_id == id(6)
        ));
        assert_eq!(decision.plan.batches.last().unwrap().rows[0].event_index, 2);
        assert_eq!(
            policy.descriptor().to_string(),
            "composed{head_of_line=backfill-1}[fifo,memory-aware-admission,\
             unbounded-batch,unbucketed-prefill]"
        );
    }

    #[test]
    fn backfill_stops_after_its_skip_limit() {
        // Two large requests both defer; with one skip allowed admission stops
        // at the second, so the small request behind them still waits.
        let mut snapshot = snapshot(
            4,
            64,
            BatchCapabilities::ALL,
            &[(5, 5, 4), (7, 5, 4), (6, 1, 4)],
        );
        add_decode(&mut snapshot, 1, 1, 8, 32);
        let policy = ComposedPolicy::default()
            .with_admission(MemoryAwareAdmission)
            .with_head_of_line(HeadOfLine::Backfill { max_skipped: 1 });

        let decision = plan(&policy, &snapshot);

        assert_eq!(decision.record.backfill_skips, vec![id(5)]);
        assert!(decision.plan.lifecycle.is_empty());
        assert_eq!(decision.plan.deferrals.len(), 2);
    }

    #[test]
    fn backfill_skips_do_not_consume_slots() {
        // One free slot: the skipped head must not use it up.
        let mut snapshot = snapshot(
            2,
            64,
            BatchCapabilities::ALL,
            &[(5, 5, 4), (6, 1, 4), (8, 1, 4)],
        );
        add_decode(&mut snapshot, 1, 1, 8, 32);
        let policy = ComposedPolicy::default()
            .with_admission(MemoryAwareAdmission)
            .with_head_of_line(HeadOfLine::Backfill { max_skipped: 1 });

        let decision = plan(&policy, &snapshot);

        assert_eq!(decision.record.ranking.limit, 3);
        assert!(matches!(
            decision.plan.lifecycle.as_slice(),
            [LifecycleAction::Admit { request_id, .. }] if *request_id == id(6)
        ));
        assert_eq!(
            decision.plan.deferrals.last(),
            Some(&PlannedDeferral::Slots { request_id: id(8) })
        );
    }

    #[test]
    fn an_oversized_head_prefill_runs_alone_and_defers_the_rest() {
        let snapshot = snapshot(4, 1_024, BatchCapabilities::ALL, &[(5, 5, 2), (6, 1, 2)]);
        let policy = ComposedPolicy::default().with_batch(FifoTokenBudgetPolicy::new(2).unwrap());

        let decision = plan(&policy, &snapshot);

        assert_eq!(decision.plan.batches.len(), 1);
        assert_eq!(decision.plan.batches[0].rows[0].request_id, id(5));
        assert_eq!(decision.plan.batches[0].rows[0].positions, 0..5);
        assert_eq!(
            decision.plan.deferrals,
            vec![PlannedDeferral::Budget {
                request_id: id(6),
                phase: BatchPhase::Prefill,
                token_cost: 1,
                event_index: 1,
            }]
        );
        let budget = decision
            .record
            .admissions
            .iter()
            .map(|record| match &record.outcome {
                AdmissionOutcome::Decided {
                    budget_selected, ..
                } => *budget_selected,
                other => panic!("unexpected outcome {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(budget, vec![Some(true), Some(false)]);
    }

    #[test]
    fn predicted_releases_free_memory_and_slots_for_admission() {
        // The only slot and the whole pool belong to a decode that must end
        // at its output limit this iteration.
        let mut snapshot = snapshot(1, 32, BatchCapabilities::ALL, &[(5, 1, 4)]);
        add_decode(&mut snapshot, 1, 3, 4, 32);
        let policy = ComposedPolicy::default().with_admission(MemoryAwareAdmission);

        let decision = plan(&policy, &snapshot);

        assert_eq!(decision.plan.predicted_releases, vec![id(1)]);
        assert!(matches!(
            decision.plan.lifecycle.as_slice(),
            [LifecycleAction::Admit { request_id, reserved_kv_bytes: 32, .. }] if *request_id == id(5)
        ));
        let AdmissionOutcome::Decided { context, .. } = &decision.record.admissions[0].outcome
        else {
            panic!("admission should have run");
        };
        assert_eq!(context.model_reserved_kv_bytes, 0);

        // Without the prediction the same request would wait for a slot.
        let mut unpredictable = snapshot.clone();
        unpredictable
            .decoding
            .get_mut(&id(1))
            .unwrap()
            .max_new_tokens = 8;
        let waiting = plan(&policy, &unpredictable);
        assert_eq!(
            waiting.plan.deferrals,
            vec![PlannedDeferral::Slots { request_id: id(5) }]
        );
    }

    #[test]
    fn missing_batch_hooks_split_groups_and_are_recorded() {
        let snapshot = snapshot(4, 1_024, BatchCapabilities::NONE, &[(5, 2, 2), (6, 3, 2)]);

        let decision = plan(&ComposedPolicy::default(), &snapshot);

        assert_eq!(decision.plan.batches.len(), 2);
        assert!(
            decision
                .plan
                .batches
                .iter()
                .all(|batch| batch.mode == BatchExecutionMode::Individual)
        );
        assert_eq!(
            decision.record.capability_fallbacks,
            vec![CapabilityFallback {
                phase: PlannedPhase::Prefill,
                model_id: model(),
                rows: 2,
                missing: BatchHook::Prefill,
            }]
        );
    }

    #[test]
    fn kv_validation_catches_admission_that_breaks_its_own_guarantee() {
        /// Claims to respect the KV pool but always reserves 1 KiB.
        struct OverAdmit;

        impl AdmissionPolicy for OverAdmit {
            fn decide(&self, _context: &AdmissionContext) -> AdmissionDecision {
                AdmissionDecision::Admit {
                    reserved_kv_bytes: 1_024,
                }
            }

            fn descriptor(&self) -> PolicyDescriptor {
                PolicyDescriptor::new("over-admit")
            }

            fn enforces_model_kv_capacity(&self) -> bool {
                true
            }
        }

        let snapshot = snapshot(4, 64, BatchCapabilities::ALL, &[(5, 1, 2)]);
        let policy = ComposedPolicy::default().with_admission(OverAdmit);
        let decision = policy.plan(&snapshot).unwrap();

        assert_eq!(
            decision
                .plan
                .validate(&snapshot, policy.enforces_model_kv_capacity()),
            Err(crate::PlanError::KvCapacity {
                model_id: model(),
                reserved_bytes: 1_024,
                capacity_bytes: 64,
            })
        );
        // A permissive policy makes no such promise, so the same plan passes.
        assert_eq!(decision.plan.validate(&snapshot, false), Ok(()));
    }
}
