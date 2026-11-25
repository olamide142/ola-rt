//! Virtual-time scheduling trials.
//!
//! See `SCHEDULING_BASELINES.md` in `ola-runtime`. A trial replays a
//! [`SchedulingTrace`] through a real `Executor` and execution policy, with
//! the deterministic synthetic model standing in for weights. A
//! [`CostModel`] turns each iteration's actual model invocations into virtual
//! microseconds, so every timestamp, and therefore every latency, is exactly
//! reproducible and independent of the host.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use ola_core::Input;
use ola_core::{ModelDescriptor, Priority, RequestId, RequestSpec, Timestamp};
use ola_runtime::{
    BatchPhase, ExecutionBatch, ExecutionPolicy, Executor, ExecutorError, ExecutorEvent,
    ExecutorIteration, PlannedDeferral, PolicyDescriptor, RequestStatus, Runtime, RuntimeState,
};

use crate::{
    KvGeometry, SchedulingTrace, synthetic_artifact, synthetic_artifact_with_prefix_cache,
    synthetic_prompt,
};

pub(crate) const MODEL_ID: &str = "synthetic";

/// Virtual cost of one executor iteration.
///
/// `iteration = fixed + Σ over invocations (per_invocation + per_dense_token × dense tokens)`,
/// where dense tokens include right padding. The defaults are illustrative,
/// chosen so that invocation overhead and token work are both visible; P4
/// calibration can replace them with measured values.
///
/// Two optional refinements leave that formula unchanged when unset:
/// `per_padding_token_micros` prices padded positions separately from real
/// ones, and `decode` prices decode invocations by rows instead of tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostModel {
    /// Cost of an iteration boundary, independent of work.
    pub fixed_micros: u64,
    /// Cost of launching one model invocation.
    pub per_invocation_micros: u64,
    /// Cost of one dense input position.
    pub per_dense_token_micros: u64,
    /// Cost of one padded position, when it differs from a real one.
    pub per_padding_token_micros: Option<u64>,
    /// Cost of decode invocations, when it differs from prefill pricing.
    pub decode: Option<DecodeCost>,
}

/// Cost of one decode invocation: `per_invocation + per_row × rows`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeCost {
    /// Cost of launching one decode invocation.
    pub per_invocation_micros: u64,
    /// Cost of one decoded row.
    pub per_row_micros: u64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            fixed_micros: 200,
            per_invocation_micros: 300,
            per_dense_token_micros: 4,
            per_padding_token_micros: None,
            decode: None,
        }
    }
}

impl CostModel {
    /// Returns when each batch finishes, in microseconds after its
    /// iteration starts: the fixed boundary cost is paid first, then
    /// invocations run in order. The last offset equals
    /// [`iteration_micros`](Self::iteration_micros).
    pub fn batch_finish_offsets(&self, batches: &[ExecutionBatch]) -> Vec<u64> {
        batches
            .iter()
            .scan(self.fixed_micros, |elapsed, batch| {
                *elapsed = self
                    .iteration_micros(std::slice::from_ref(batch))
                    .saturating_sub(self.fixed_micros)
                    .saturating_add(*elapsed);
                Some(*elapsed)
            })
            .collect()
    }

    /// Returns the virtual duration of an iteration that ran `batches`.
    pub fn iteration_micros(&self, batches: &[ExecutionBatch]) -> u64 {
        batches.iter().fold(self.fixed_micros, |total, batch| {
            total.saturating_add(self.invocation_micros(batch))
        })
    }

    /// Returns the virtual duration of one invocation, without the
    /// iteration's fixed cost.
    pub fn invocation_micros(&self, batch: &ExecutionBatch) -> u64 {
        let count = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
        if let (BatchPhase::Decode, Some(decode)) = (batch.phase, self.decode) {
            return decode.per_invocation_micros.saturating_add(
                decode
                    .per_row_micros
                    .saturating_mul(count(batch.request_ids.len())),
            );
        }
        let tokens = match self.per_padding_token_micros {
            None => self
                .per_dense_token_micros
                .saturating_mul(count(batch.dense_input_tokens())),
            Some(per_padding) => self
                .per_dense_token_micros
                .saturating_mul(count(batch.input_tokens))
                .saturating_add(per_padding.saturating_mul(count(batch.padding_tokens))),
        };
        self.per_invocation_micros.saturating_add(tokens)
    }
}

/// Executor and cost configuration shared by every policy in a comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationConfig {
    /// Live-session limit.
    pub max_active_requests: usize,
    /// Resumable prefill segment length, if bounded.
    pub prefill_chunk_size: Option<usize>,
    /// Synthetic model KV pool.
    pub kv: KvGeometry,
    /// Virtual iteration cost.
    pub cost: CostModel,
    /// Iterations after which the trial stops; remaining work is unfinished.
    pub max_iterations: usize,
    /// Whether the synthetic model's pool keeps a prefix cache.
    pub prefix_cache: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            max_active_requests: 8,
            prefill_chunk_size: Some(64),
            kv: KvGeometry::default(),
            cost: CostModel::default(),
            max_iterations: 100_000,
            prefix_cache: false,
        }
    }
}

/// Terminal state of one trace request at the end of a trial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrialOutcome {
    /// Generated to EOS or its output limit.
    Completed,
    /// Admission rejected it as impossible.
    Rejected,
    /// Preparation or execution failed.
    Failed,
    /// Cancelled by the caller.
    Cancelled,
    /// Still queued or running when the trial stopped.
    Unfinished,
}

/// Lifecycle of one trace request in virtual microseconds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialRequest {
    /// Position in the trace.
    pub trace_index: usize,
    /// Runtime identifier.
    pub request_id: RequestId,
    /// Arrival time.
    pub arrival_micros: u64,
    /// Admission time.
    pub started_micros: Option<u64>,
    /// First-token time.
    pub first_token_micros: Option<u64>,
    /// Terminal time.
    pub completed_micros: Option<u64>,
    /// Terminal state.
    pub outcome: TrialOutcome,
    /// Generated tokens, for completed requests.
    pub output_tokens: Vec<u32>,
    /// Trace priority.
    pub priority: i32,
    /// Trace deadline.
    pub deadline_micros: Option<u64>,
    /// Trace prompt length.
    pub prompt_tokens: usize,
    /// Trace output limit.
    pub max_new_tokens: usize,
}

/// Work, queues, memory, and policy decisions of one iteration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrialIteration {
    /// Iteration start time.
    pub at_micros: u64,
    /// Virtual duration from the cost model.
    pub duration_micros: u64,
    /// Real input positions processed.
    pub input_tokens: usize,
    /// Right-padding positions processed.
    pub padding_tokens: usize,
    /// Model invocations.
    pub invocations: usize,
    /// Requests deferred by admission.
    pub admission_deferrals: usize,
    /// Operations deferred by the token budget.
    pub budget_deferrals: usize,
    /// Requests waiting for a slot.
    pub slot_deferrals: usize,
    /// Admission deferrals backfill skipped past.
    pub backfill_skips: usize,
    /// Live requests returned to the queue, planned or after a failed
    /// allocation.
    pub preemptions: usize,
    /// Multi-row groups planned individually for lack of a batch hook.
    pub capability_fallbacks: usize,
    /// Queued requests after the iteration.
    pub queued_requests: usize,
    /// Live sessions after the iteration.
    pub live_requests: usize,
    /// Physical KV bytes in use after the iteration.
    pub kv_cache_bytes: u64,
    /// KV bytes reserved after the iteration.
    pub reserved_kv_bytes: u64,
    /// Bytes of pool blocks held by live sessions after the iteration,
    /// whole pages, each shared block counted once.
    pub kv_allocated_bytes: u64,
    /// Prompt tokens reused from the prefix cache since the trial began.
    pub prefix_hit_tokens: u64,
}

/// Everything one policy did with one trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulingRun {
    /// Policy that planned every iteration.
    pub policy: PolicyDescriptor,
    /// Trace generator name.
    pub trace: &'static str,
    /// Trace seed.
    pub seed: u64,
    /// Per-request lifecycles in trace order.
    pub requests: Vec<TrialRequest>,
    /// Per-iteration counters in order.
    pub iterations: Vec<TrialIteration>,
    /// Virtual time when the trial stopped.
    pub makespan_micros: u64,
    /// Outcomes of requests injected by perturbations, in injection order.
    pub injected: Vec<TrialOutcome>,
    /// Live sessions when the trial stopped.
    pub final_live_requests: usize,
    /// KV bytes still reserved when the trial stopped.
    pub final_reserved_kv_bytes: u64,
    /// Physical KV bytes still in use when the trial stopped.
    pub final_kv_cache_bytes: u64,
}

/// A change applied while a trial runs, for isolation experiments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Perturbation {
    /// Cancel a trace request at a virtual time if it is queued or running
    /// then.
    Cancel {
        /// Trace position of the request.
        trace_index: usize,
        /// Virtual time of the cancellation.
        at_micros: u64,
    },
    /// Submit a request the synthetic model cannot prepare (text input).
    SubmitInvalid {
        /// Virtual arrival time.
        at_micros: u64,
    },
}

impl Perturbation {
    const fn at_micros(self) -> u64 {
        match self {
            Self::Cancel { at_micros, .. } | Self::SubmitInvalid { at_micros } => at_micros,
        }
    }
}

/// Failure to run a scheduling trial.
#[derive(Debug, Eq, PartialEq)]
pub enum SimulationError {
    /// The executor rejected configuration or returned an error.
    Executor(ExecutorError),
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "scheduling trial failed: {error}"),
        }
    }
}

impl Error for SimulationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Executor(error) => Some(error),
        }
    }
}

impl From<ExecutorError> for SimulationError {
    fn from(value: ExecutorError) -> Self {
        Self::Executor(value)
    }
}

/// Builds the request a trace entry submits.
pub fn trace_request_spec(request: &crate::TraceRequest) -> RequestSpec {
    let mut spec = RequestSpec::text(MODEL_ID, "");
    spec.inputs = vec![Input::Tokens(match request.shared_prefix {
        Some(shared) => crate::synthetic::synthetic_group_prompt(
            request.prompt_tokens,
            shared.tokens,
            shared.group,
            shared.seed,
        ),
        None => synthetic_prompt(request.prompt_tokens, request.eos_after),
    })];
    spec.generation.max_new_tokens = request.max_new_tokens;
    spec.generation.temperature = 0.0;
    spec.priority = Priority::new(request.priority);
    spec.deadline = request.deadline_micros.map(Timestamp::from_micros);
    spec
}

/// Replays `trace` under `policy` on a virtual clock.
///
/// Arrivals up to the current time are submitted with their trace
/// timestamps; an idle executor jumps to the next arrival; otherwise the
/// clock advances by the cost of the iteration's actual invocations. An
/// iteration that runs nothing waits for the next arrival, or ends the trial
/// when none remain, leaving blocked requests unfinished.
///
/// # Errors
///
/// Returns [`SimulationError`] if the executor rejects the configuration or
/// a plan.
pub fn run_scheduling_trial<P>(
    trace: &SchedulingTrace,
    policy: P,
    config: &SimulationConfig,
) -> Result<SchedulingRun, SimulationError>
where
    P: ExecutionPolicy + 'static,
{
    run_perturbed_trial(trace, policy, config, &[])
}

/// Like [`run_scheduling_trial`], applying each perturbation at the first
/// iteration boundary at or after its time. A perturbation whose time is
/// after the trial ends is not applied.
///
/// # Errors
///
/// Returns [`SimulationError`] if the executor rejects the configuration, a
/// plan, or a perturbation.
pub fn run_perturbed_trial<P>(
    trace: &SchedulingTrace,
    policy: P,
    config: &SimulationConfig,
    perturbations: &[Perturbation],
) -> Result<SchedulingRun, SimulationError>
where
    P: ExecutionPolicy + 'static,
{
    run_observed_trial(trace, policy, config, perturbations, &mut |_, _| {})
}

/// Like [`run_perturbed_trial`], calling `observe` with every iteration and
/// the runtime state right after it, for tracing.
///
/// Iteration times are virtual. Executor times stay at each iteration's
/// start, because the simulated executor has no clock; the summary's
/// corrected virtual times are not reflected in what `observe` sees.
/// Cancellation perturbations are not iterations and are not observed.
///
/// # Errors
///
/// Returns [`SimulationError`] as [`run_perturbed_trial`] does.
pub fn run_observed_trial<P>(
    trace: &SchedulingTrace,
    policy: P,
    config: &SimulationConfig,
    perturbations: &[Perturbation],
    observe: &mut dyn FnMut(&ExecutorIteration, &RuntimeState),
) -> Result<SchedulingRun, SimulationError>
where
    P: ExecutionPolicy + 'static,
{
    run_charged_trial(
        trace,
        policy,
        config,
        perturbations,
        &mut |iteration, state| {
            observe(iteration, state);
            0
        },
    )
}

/// Like [`run_observed_trial`], where `charge` also returns host time, in
/// virtual microseconds, that the executor spent after the iteration and
/// before the next one could start: work a live executor does on its own
/// thread between iterations, such as an adaptive policy's optimizer call
/// in `observe` or reconfiguring after a switch. The clock advances by it,
/// so arrivals during it queue and every request in flight waits.
///
/// # Errors
///
/// Returns [`SimulationError`] as [`run_perturbed_trial`] does.
pub fn run_charged_trial<P>(
    trace: &SchedulingTrace,
    policy: P,
    config: &SimulationConfig,
    perturbations: &[Perturbation],
    charge: &mut dyn FnMut(&ExecutorIteration, &RuntimeState) -> u64,
) -> Result<SchedulingRun, SimulationError>
where
    P: ExecutionPolicy + 'static,
{
    let mut pending = perturbations.to_vec();
    pending.sort_by_key(|perturbation| perturbation.at_micros());
    let mut pending = pending.into_iter().peekable();
    let mut injected_ids = Vec::new();
    let descriptor = policy.descriptor();
    let mut executor = simulation_executor(policy, config)?;

    let mut ids = Vec::with_capacity(trace.requests.len());
    let mut rejected = BTreeSet::new();
    // Virtual times at which work became observable. The executor has no
    // clock here, so it stamps everything at the iteration's start; these
    // record when the producing invocation finished instead, as a live
    // executor clock would.
    let mut first_token_at = BTreeMap::new();
    let mut completed_at = BTreeMap::new();
    let mut iterations = Vec::new();
    let mut clock = 0_u64;
    while iterations.len() < config.max_iterations {
        while let Some(request) = trace.requests.get(ids.len())
            && request.arrival_micros <= clock
        {
            ids.push(executor.submit(
                trace_request_spec(request),
                Timestamp::from_micros(request.arrival_micros),
            )?);
        }
        while let Some(perturbation) = pending.next_if(|next| next.at_micros() <= clock) {
            match perturbation {
                Perturbation::Cancel { trace_index, .. } => {
                    if let Some(&request_id) = ids.get(trace_index) {
                        let live = executor.state().request(request_id).is_some_and(|record| {
                            matches!(
                                record.status,
                                RequestStatus::Queued | RequestStatus::Running
                            )
                        });
                        if live {
                            executor.cancel(request_id, Timestamp::from_micros(clock))?;
                        }
                    }
                }
                Perturbation::SubmitInvalid { .. } => {
                    injected_ids.push(executor.submit(
                        RequestSpec::text(MODEL_ID, "not tokens"),
                        Timestamp::from_micros(clock),
                    )?);
                }
            }
        }
        let next_arrival = trace
            .requests
            .get(ids.len())
            .map(|request| request.arrival_micros);
        if executor.active_request_count() == 0 && executor.state().queued_len() == 0 {
            match next_arrival {
                Some(arrival) => {
                    clock = arrival;
                    continue;
                }
                None => break,
            }
        }

        let iteration = executor.step_iteration(Timestamp::from_micros(clock))?;
        let stall = charge(&iteration, executor.state());
        let finishes = config
            .cost
            .batch_finish_offsets(&iteration.execution_batches);
        let finished_at = |request_id: RequestId| {
            iteration
                .execution_batches
                .iter()
                .zip(&finishes)
                .find(|(batch, _)| batch.request_ids.contains(&request_id))
                .map_or(clock, |(_, offset)| clock.saturating_add(*offset))
        };
        for event in &iteration.events {
            match event {
                ExecutorEvent::Rejected { request_id, .. } => {
                    rejected.insert(*request_id);
                }
                ExecutorEvent::Token {
                    request_id,
                    generated_tokens,
                    finished,
                    ..
                } => {
                    let at = finished_at(*request_id);
                    if *generated_tokens == 1 {
                        first_token_at.insert(*request_id, at);
                    }
                    if *finished {
                        completed_at.insert(*request_id, at);
                    }
                }
                ExecutorEvent::Failed { request_id, .. } => {
                    completed_at.insert(*request_id, finished_at(*request_id));
                }
                _ => {}
            }
        }
        let duration = config.cost.iteration_micros(&iteration.execution_batches);
        iterations.push(trial_iteration(&iteration, &executor, config, clock));

        if iteration.execution_batches.is_empty() {
            // Nothing could run (for example, every queued request is
            // deferred). Only a new arrival can change that.
            match next_arrival {
                Some(arrival) => clock = clock.saturating_add(stall).max(arrival),
                None => break,
            }
        } else {
            clock = clock.saturating_add(duration.max(1)).saturating_add(stall);
        }
    }

    let requests = trace
        .requests
        .iter()
        .enumerate()
        .map(|(trace_index, request)| {
            let micros = |time: Option<Timestamp>| time.map(Timestamp::as_micros);
            let Some(&request_id) = ids.get(trace_index) else {
                return TrialRequest {
                    trace_index,
                    request_id: RequestId::new(0),
                    arrival_micros: request.arrival_micros,
                    started_micros: None,
                    first_token_micros: None,
                    completed_micros: None,
                    outcome: TrialOutcome::Unfinished,
                    output_tokens: Vec::new(),
                    priority: request.priority,
                    deadline_micros: request.deadline_micros,
                    prompt_tokens: request.prompt_tokens,
                    max_new_tokens: request.max_new_tokens,
                };
            };
            let record = executor
                .state()
                .request(request_id)
                .expect("submitted requests retain their records");
            let outcome = match &record.status {
                RequestStatus::Completed => TrialOutcome::Completed,
                RequestStatus::Failed(_) if rejected.contains(&request_id) => {
                    TrialOutcome::Rejected
                }
                RequestStatus::Failed(_) => TrialOutcome::Failed,
                RequestStatus::Cancelled => TrialOutcome::Cancelled,
                RequestStatus::Queued | RequestStatus::Running => TrialOutcome::Unfinished,
            };
            TrialRequest {
                trace_index,
                request_id,
                arrival_micros: request.arrival_micros,
                started_micros: micros(record.timing.started_at),
                first_token_micros: first_token_at
                    .get(&request_id)
                    .copied()
                    .or(micros(record.timing.first_token_at)),
                // Cancellations and rejections happen at a planning
                // boundary, which the runtime record already holds.
                completed_micros: completed_at
                    .get(&request_id)
                    .copied()
                    .or(micros(record.timing.completed_at)),
                outcome,
                output_tokens: executor
                    .result(request_id)
                    .map(|result| result.token_ids.clone())
                    .unwrap_or_default(),
                priority: request.priority,
                deadline_micros: request.deadline_micros,
                prompt_tokens: request.prompt_tokens,
                max_new_tokens: request.max_new_tokens,
            }
        })
        .collect();

    let injected = injected_ids
        .iter()
        .map(|request_id| {
            match executor
                .state()
                .request(*request_id)
                .map(|record| &record.status)
            {
                Some(RequestStatus::Completed) => TrialOutcome::Completed,
                Some(RequestStatus::Failed(_)) => TrialOutcome::Failed,
                Some(RequestStatus::Cancelled) => TrialOutcome::Cancelled,
                _ => TrialOutcome::Unfinished,
            }
        })
        .collect();

    Ok(SchedulingRun {
        policy: descriptor,
        trace: trace.name,
        seed: trace.parameters.seed,
        requests,
        iterations,
        makespan_micros: clock,
        injected,
        final_live_requests: executor.active_request_count(),
        final_reserved_kv_bytes: executor.reserved_kv_bytes(),
        final_kv_cache_bytes: executor.state().memory.kv_cache_bytes,
    })
}

/// Builds an executor running `policy` with the synthetic model configured
/// as `config` describes.
pub(crate) fn simulation_executor<P>(
    policy: P,
    config: &SimulationConfig,
) -> Result<Executor, ExecutorError>
where
    P: ExecutionPolicy + 'static,
{
    let mut executor =
        Executor::with_max_active_requests(Runtime::default(), config.max_active_requests)?
            .with_policy(policy);
    if let Some(chunk) = config.prefill_chunk_size {
        executor = executor.with_prefill_chunk_size(chunk)?;
    }
    executor.register_model(
        ModelDescriptor::language(MODEL_ID, "synthetic"),
        if config.prefix_cache {
            synthetic_artifact_with_prefix_cache(config.kv)
        } else {
            synthetic_artifact(config.kv)
        },
    )?;
    Ok(executor)
}

/// Counters for one executor iteration that started at `clock`, with pool
/// state read from `executor` right after it.
pub(crate) fn trial_iteration(
    iteration: &ExecutorIteration,
    executor: &Executor,
    config: &SimulationConfig,
    clock: u64,
) -> TrialIteration {
    let estimates = iteration.execution_plan.estimates();
    let count = |kind: fn(&PlannedDeferral) -> bool| {
        iteration
            .execution_plan
            .deferrals
            .iter()
            .filter(|deferral| kind(deferral))
            .count()
    };
    let pool = executor
        .model(&MODEL_ID.into())
        .and_then(ola_model::CausalLanguageModel::kv_pool_status);
    TrialIteration {
        at_micros: clock,
        duration_micros: config.cost.iteration_micros(&iteration.execution_batches),
        input_tokens: iteration
            .execution_batches
            .iter()
            .map(|batch| batch.input_tokens)
            .sum(),
        padding_tokens: iteration
            .execution_batches
            .iter()
            .map(|batch| batch.padding_tokens)
            .sum(),
        invocations: iteration.execution_batches.len(),
        admission_deferrals: count(|deferral| {
            matches!(deferral, PlannedDeferral::Admission { .. })
        }),
        budget_deferrals: estimates.budget_deferrals,
        slot_deferrals: count(|deferral| matches!(deferral, PlannedDeferral::Slots { .. })),
        backfill_skips: iteration.decision.backfill_skips.len(),
        preemptions: iteration
            .events
            .iter()
            .filter(|event| matches!(event, ExecutorEvent::Preempted { .. }))
            .count(),
        capability_fallbacks: iteration.decision.capability_fallbacks.len(),
        queued_requests: iteration.queued_requests,
        live_requests: iteration.active_requests,
        kv_cache_bytes: iteration.kv_cache_bytes,
        reserved_kv_bytes: iteration.reserved_kv_bytes,
        kv_allocated_bytes: pool.map_or(0, |pool| {
            (pool.allocated_blocks as u64)
                .saturating_mul(config.kv.tokens_per_block as u64)
                .saturating_mul(config.kv.bytes_per_token)
        }),
        prefix_hit_tokens: pool.map_or(0, |pool| pool.prefix_hit_tokens),
    }
}

#[cfg(test)]
mod tests {
    use ola_runtime::{ComposedPolicy, JobCost, MemoryAwareAdmission, ShortestJobFirst};

    use super::*;
    use crate::{TraceParameters, synthetic_token};

    const PARAMETERS: TraceParameters = TraceParameters {
        seed: 11,
        requests: 60,
        mean_interarrival_micros: 2_000,
    };

    fn fifo() -> ComposedPolicy {
        ComposedPolicy::default().with_admission(MemoryAwareAdmission)
    }

    #[test]
    fn cost_model_charges_fixed_invocation_and_dense_token_costs() {
        let batch = |input_tokens, padding_tokens| ExecutionBatch {
            started_at: Timestamp::ZERO,
            finished_at: Timestamp::ZERO,
            phase: ola_runtime::BatchPhase::Prefill,
            request_ids: Vec::new(),
            mode: ola_runtime::BatchExecutionMode::Tensor,
            input_tokens,
            padding_tokens,
            kernel: None,
        };
        let cost = CostModel::default();

        assert_eq!(cost.iteration_micros(&[]), 200);
        assert_eq!(
            cost.iteration_micros(&[batch(10, 2), batch(1, 0)]),
            200 + 300 + 48 + 300 + 4
        );
    }

    #[test]
    fn padding_and_decode_costs_refine_pricing_only_when_set() {
        let batch = |phase, rows: u64, input_tokens, padding_tokens| ExecutionBatch {
            started_at: Timestamp::ZERO,
            finished_at: Timestamp::ZERO,
            phase,
            request_ids: (1..=rows).map(ola_core::RequestId::new).collect(),
            mode: ola_runtime::BatchExecutionMode::Tensor,
            input_tokens,
            padding_tokens,
            kernel: None,
        };
        let prefill = batch(ola_runtime::BatchPhase::Prefill, 2, 10, 6);
        let decode = batch(ola_runtime::BatchPhase::Decode, 3, 3, 0);
        let refined = CostModel {
            per_padding_token_micros: Some(1),
            decode: Some(DecodeCost {
                per_invocation_micros: 500,
                per_row_micros: 70,
            }),
            ..CostModel::default()
        };

        // Unset, padding costs like a real token and decode like prefill.
        let plain = CostModel::default();
        assert_eq!(plain.invocation_micros(&prefill), 300 + 4 * 16);
        assert_eq!(plain.invocation_micros(&decode), 300 + 4 * 3);
        assert_eq!(refined.invocation_micros(&prefill), 300 + 4 * 10 + 6);
        assert_eq!(refined.invocation_micros(&decode), 500 + 70 * 3);
        assert_eq!(
            refined.iteration_micros(&[prefill, decode]),
            200 + 300 + 46 + 500 + 210
        );
    }

    #[test]
    fn trials_complete_every_request_with_deterministic_outputs() {
        let trace = SchedulingTrace::mixed_short_long(PARAMETERS);
        let run = run_scheduling_trial(&trace, fifo(), &SimulationConfig::default()).unwrap();

        assert_eq!(run.requests.len(), PARAMETERS.requests);
        for (request, traced) in run.requests.iter().zip(&trace.requests) {
            assert_eq!(request.outcome, TrialOutcome::Completed);
            let expected = traced.eos_after.unwrap_or(traced.max_new_tokens);
            assert_eq!(request.output_tokens.len(), expected);
            assert_eq!(
                request.output_tokens.last(),
                Some(&synthetic_token(traced.prompt_tokens, expected))
            );
            let started = request.started_micros.unwrap();
            let first = request.first_token_micros.unwrap();
            let completed = request.completed_micros.unwrap();
            assert!(request.arrival_micros <= started && started <= first && first <= completed);
        }
        assert!(
            run.iterations
                .windows(2)
                .all(|pair| pair[0].at_micros < pair[1].at_micros)
        );
    }

    #[test]
    fn trials_are_exactly_reproducible() {
        let trace = SchedulingTrace::overload(PARAMETERS);
        let config = SimulationConfig::default();

        let first = run_scheduling_trial(&trace, fifo(), &config).unwrap();
        let second = run_scheduling_trial(&trace, fifo(), &config).unwrap();

        assert_eq!(first, second);
        assert!(
            first
                .iterations
                .iter()
                .any(|iteration| iteration.queued_requests > 0)
        );
    }

    #[test]
    fn a_charged_stall_delays_everything_after_it_and_nothing_before() {
        let mut trace = SchedulingTrace::mixed_short_long(PARAMETERS);
        // Everything arrives at once, so a stall cannot change the schedule.
        for request in &mut trace.requests {
            request.arrival_micros = 0;
        }
        let config = SimulationConfig::default();
        let free = run_scheduling_trial(&trace, fifo(), &config).unwrap();
        let uncharged = run_charged_trial(&trace, fifo(), &config, &[], &mut |_, _| 0).unwrap();
        assert_eq!(uncharged, free);

        let mut first = true;
        let charged = run_charged_trial(&trace, fifo(), &config, &[], &mut |_, _| {
            if std::mem::take(&mut first) { 500 } else { 0 }
        })
        .unwrap();
        let first_end = free.iterations[1].at_micros;
        assert_eq!(charged.makespan_micros, free.makespan_micros + 500);
        for (charged, free) in charged.requests.iter().zip(&free.requests) {
            assert_eq!(charged.output_tokens, free.output_tokens);
            let (done, was) = (
                charged.completed_micros.unwrap(),
                free.completed_micros.unwrap(),
            );
            assert_eq!(done, if was <= first_end { was } else { was + 500 });
        }
    }

    #[test]
    fn scheduling_changes_timing_but_never_outputs() {
        // Under light load the queue rarely holds two requests, so ranking
        // cannot matter; overload builds queues where it must.
        let trace = SchedulingTrace::overload(PARAMETERS);
        let config = SimulationConfig::default();

        let fifo_run = run_scheduling_trial(&trace, fifo(), &config).unwrap();
        let sjf_run = run_scheduling_trial(
            &trace,
            fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt)),
            &config,
        )
        .unwrap();

        let outputs = |run: &SchedulingRun| {
            run.requests
                .iter()
                .map(|request| request.output_tokens.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(outputs(&fifo_run), outputs(&sjf_run));
        assert_ne!(
            fifo_run
                .requests
                .iter()
                .map(|request| request.completed_micros)
                .collect::<Vec<_>>(),
            sjf_run
                .requests
                .iter()
                .map(|request| request.completed_micros)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn impossible_requests_are_rejected_without_blocking_the_rest() {
        let config = SimulationConfig::default();
        let trace = SchedulingTrace::impossible(PARAMETERS, config.kv.capacity_tokens());

        let run = run_scheduling_trial(&trace, fifo(), &config).unwrap();

        for (request, traced) in run.requests.iter().zip(&trace.requests) {
            let expected = if traced.prompt_tokens > config.kv.capacity_tokens() {
                TrialOutcome::Rejected
            } else {
                TrialOutcome::Completed
            };
            assert_eq!(
                request.outcome, expected,
                "trace index {}",
                request.trace_index
            );
        }
        assert!(
            run.requests
                .iter()
                .any(|request| request.outcome == TrialOutcome::Rejected)
        );
    }

    #[test]
    fn trials_stop_at_the_iteration_limit_and_report_unfinished_work() {
        let trace = SchedulingTrace::overload(PARAMETERS);
        let config = SimulationConfig {
            max_iterations: 5,
            ..SimulationConfig::default()
        };

        let run = run_scheduling_trial(&trace, fifo(), &config).unwrap();

        assert_eq!(run.iterations.len(), 5);
        assert!(
            run.requests
                .iter()
                .any(|request| request.outcome == TrialOutcome::Unfinished)
        );
    }

    /// A named constructor, so each scenario gets a fresh policy.
    type NamedPolicy = (&'static str, Box<dyn Fn() -> ComposedPolicy>);

    fn scheduler_matrix() -> Vec<NamedPolicy> {
        vec![
            ("fifo", Box::new(fifo)),
            (
                "priority",
                Box::new(|| fifo().with_scheduler(ola_runtime::PriorityScheduler::new())),
            ),
            (
                "edf",
                Box::new(|| fifo().with_scheduler(ola_runtime::EarliestDeadlineFirst)),
            ),
            (
                "sjf",
                Box::new(|| fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt))),
            ),
            (
                "backfill",
                Box::new(|| {
                    fifo().with_head_of_line(ola_runtime::HeadOfLine::Backfill { max_skipped: 4 })
                }),
            ),
        ]
    }

    fn assert_no_leaks(run: &SchedulingRun, label: &str) {
        assert_eq!(
            (
                run.final_live_requests,
                run.final_reserved_kv_bytes,
                run.final_kv_cache_bytes
            ),
            (0, 0, 0),
            "{label} leaked a slot or KV memory"
        );
    }

    #[test]
    fn cancellation_and_failure_are_isolated_under_every_scheduler() {
        let trace = SchedulingTrace::overload(TraceParameters {
            requests: 40,
            ..PARAMETERS
        });
        let config = SimulationConfig::default();
        for (name, policy) in scheduler_matrix() {
            let baseline = run_scheduling_trial(&trace, policy(), &config).unwrap();
            assert_no_leaks(&baseline, name);
            // Cancel one request mid-flight and one while it still waits.
            let running = baseline
                .requests
                .iter()
                .find(|request| {
                    request.completed_micros > request.started_micros.map(|start| start + 1)
                })
                .unwrap();
            let queued = baseline
                .requests
                .iter()
                .find(|request| {
                    request.trace_index != running.trace_index
                        && request.started_micros.unwrap() > request.arrival_micros + 1
                })
                .unwrap();
            let perturbations = [
                Perturbation::Cancel {
                    trace_index: running.trace_index,
                    at_micros: running.started_micros.unwrap() + 1,
                },
                Perturbation::Cancel {
                    trace_index: queued.trace_index,
                    at_micros: queued.arrival_micros + 1,
                },
                Perturbation::SubmitInvalid {
                    at_micros: baseline.makespan_micros / 2,
                },
            ];

            let perturbed = run_perturbed_trial(&trace, policy(), &config, &perturbations).unwrap();

            assert_no_leaks(&perturbed, name);
            assert_eq!(perturbed.injected, vec![TrialOutcome::Failed], "{name}");
            for (request, unperturbed) in perturbed.requests.iter().zip(&baseline.requests) {
                if [running.trace_index, queued.trace_index].contains(&request.trace_index) {
                    assert_eq!(request.outcome, TrialOutcome::Cancelled, "{name}");
                } else {
                    assert_eq!(request.outcome, TrialOutcome::Completed, "{name}");
                    assert_eq!(request.output_tokens, unperturbed.output_tokens, "{name}");
                }
            }
        }
    }

    #[test]
    fn impossible_requests_are_isolated_under_every_scheduler() {
        let config = SimulationConfig::default();
        let trace = SchedulingTrace::impossible(PARAMETERS, config.kv.capacity_tokens());
        for (name, policy) in scheduler_matrix() {
            let run = run_scheduling_trial(&trace, policy(), &config).unwrap();

            assert_no_leaks(&run, name);
            for (request, traced) in run.requests.iter().zip(&trace.requests) {
                let impossible = traced.prompt_tokens > config.kv.capacity_tokens();
                let expected = if impossible {
                    TrialOutcome::Rejected
                } else {
                    TrialOutcome::Completed
                };
                assert_eq!(
                    request.outcome, expected,
                    "{name}, trace index {}",
                    request.trace_index
                );
            }
        }
    }
}
