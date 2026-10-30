//! Virtual-time cluster trials.
//!
//! See `DISTRIBUTED.md` (P9 D1). A cluster trial replays a
//! [`SchedulingTrace`] through the sans-IO [`Coordinator`] and several
//! [`WorkerHost`]s, each wrapping a real `Executor` with the synthetic
//! model. Workers advance on their own virtual clocks: an iteration occupies
//! its worker for the [`CostModel`](crate::CostModel) duration of its
//! invocations, and every message between the coordinator and a worker
//! crosses a link described by a [`NetworkModel`]. Summaries therefore
//! arrive late, as they would on a real network, and routing policies see
//! the same staleness they would in a live cluster.
//!
//! Request latencies are measured where a client would see them: arrival is
//! the trace time at the coordinator, and first token and completion are
//! when the coordinator receives them.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    error::Error,
    fmt,
};

use ola_cluster::{
    ClientEvent, ClusterPolicy, ClusterTraceEvent, ClusterTraceRecord, Coordinator,
    CoordinatorError, CoordinatorOutput, WorkerCommand, WorkerHost, WorkerMessage,
};
use ola_core::{ClusterRequestId, RequestId, Timestamp, WorkerId};
use ola_runtime::{
    ComposedPolicy, ExecutorError, ExecutorEvent, MemoryAwareAdmission, PolicyDescriptor,
    RequestStatus,
};

use crate::{
    SchedulingSummary, SchedulingTrace, SimulationConfig, TrialIteration, TrialOutcome,
    TrialRequest, simulation::simulation_executor, simulation::trial_iteration, summarize_run,
    trace_request_spec,
};

/// Fixed bytes of every message: framing, IDs, and timestamps.
pub const MESSAGE_HEADER_BYTES: u64 = 64;
/// Bytes per token carried in a message (prompts and results).
pub const TOKEN_BYTES: u64 = 4;
/// Bytes of a worker summary beyond the header.
pub const SUMMARY_BYTES: u64 = 192;

/// One direction of a coordinator–worker link.
///
/// A message of `b` bytes sent at `t` starts once earlier messages on the
/// same link have been serialized, takes `b / bytes_per_micro` to serialize,
/// and arrives `latency_micros` later. Messages on one link therefore arrive
/// in send order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NetworkModel {
    /// One-way propagation delay.
    pub latency_micros: u64,
    /// Serialization rate; zero means unlimited.
    pub bytes_per_micro: u64,
}

impl NetworkModel {
    /// A network that delivers instantly.
    pub const INSTANT: Self = Self {
        latency_micros: 0,
        bytes_per_micro: 0,
    };

    /// Time to serialize `bytes`.
    pub const fn transmit_micros(&self, bytes: u64) -> u64 {
        if self.bytes_per_micro == 0 {
            0
        } else {
            bytes.div_ceil(self.bytes_per_micro)
        }
    }
}

impl Default for NetworkModel {
    /// 50 µs one way, 1250 bytes/µs (10 Gbit/s): a datacenter link.
    fn default() -> Self {
        Self {
            latency_micros: 50,
            bytes_per_micro: 1_250,
        }
    }
}

/// Topology, worker, and network configuration shared by every routing
/// policy in a comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterSimulationConfig {
    /// One executor configuration per worker; worker `i` is named `w{i}`.
    pub workers: Vec<SimulationConfig>,
    /// Every coordinator–worker link, in both directions.
    pub network: NetworkModel,
    /// Simulation events after which the trial stops; remaining work is
    /// unfinished.
    pub max_events: usize,
}

impl ClusterSimulationConfig {
    /// `count` identical workers on `network`.
    pub fn uniform(count: usize, worker: SimulationConfig, network: NetworkModel) -> Self {
        Self {
            workers: vec![worker; count],
            network,
            max_events: 2_000_000,
        }
    }

    /// The worker IDs, in index order.
    pub fn worker_ids(&self) -> Vec<WorkerId> {
        (0..self.workers.len())
            .map(|index| WorkerId::new(format!("w{index}")))
            .collect()
    }
}

/// What one worker did during a cluster trial.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerRun {
    /// Worker identity.
    pub worker: WorkerId,
    /// Requests the coordinator routed to it.
    pub routed: usize,
    /// Per-iteration counters on the worker's clock, which is shared
    /// virtual time.
    pub iterations: Vec<TrialIteration>,
    /// Virtual time spent running iterations.
    pub busy_micros: u64,
    /// Prompt tokens served from its prefix cache.
    pub prefix_hit_tokens: u64,
}

/// Everything one routing policy did with one trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterRun {
    /// Cluster policy that routed every request.
    pub policy: PolicyDescriptor,
    /// Trace generator name.
    pub trace: &'static str,
    /// Trace seed.
    pub seed: u64,
    /// Per-request lifecycles in trace order, in coordinator time.
    /// `request_id` holds the cluster request ID.
    pub requests: Vec<TrialRequest>,
    /// Per-worker activity in worker-ID order.
    pub workers: Vec<WorkerRun>,
    /// Messages sent in either direction.
    pub messages: u64,
    /// Bytes sent in either direction.
    pub network_bytes: u64,
    /// Virtual time when the trial stopped.
    pub makespan_micros: u64,
    /// Every cluster trace record, in order.
    pub cluster_trace: Vec<ClusterTraceRecord>,
}

/// Failure to run a cluster trial.
#[derive(Debug)]
pub enum ClusterSimulationError {
    /// The configuration has no workers.
    NoWorkers,
    /// A worker's executor rejected configuration or failed.
    Executor(ExecutorError),
    /// The coordinator rejected a call or a plan.
    Coordinator(CoordinatorError),
}

impl fmt::Display for ClusterSimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoWorkers => formatter.write_str("cluster trial needs at least one worker"),
            Self::Executor(error) => write!(formatter, "cluster trial failed: {error}"),
            Self::Coordinator(error) => write!(formatter, "cluster trial failed: {error}"),
        }
    }
}

impl Error for ClusterSimulationError {}

impl From<ExecutorError> for ClusterSimulationError {
    fn from(value: ExecutorError) -> Self {
        Self::Executor(value)
    }
}

impl From<CoordinatorError> for ClusterSimulationError {
    fn from(value: CoordinatorError) -> Self {
        Self::Coordinator(value)
    }
}

enum Event {
    Arrival(usize),
    ToWorker(usize, WorkerCommand),
    ToCoordinator(usize, WorkerMessage),
    /// A worker finished its iteration and may step again.
    WorkerFree,
}

struct SimWorker {
    host: WorkerHost,
    config: SimulationConfig,
    busy_until: u64,
    /// Ran an iteration that executed nothing; waits for a command.
    blocked: bool,
    /// Local IDs admission rejected.
    rejected: BTreeSet<RequestId>,
    run: WorkerRun,
    /// Coordinator → worker link is free from this time.
    downlink_free: u64,
    /// Worker → coordinator link is free from this time.
    uplink_free: u64,
}

/// Discrete-event queue ordered by time, then by insertion.
struct Events {
    heap: BinaryHeap<Reverse<(u64, u64)>>,
    payloads: BTreeMap<u64, Event>,
    next: u64,
}

impl Events {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            payloads: BTreeMap::new(),
            next: 0,
        }
    }

    fn push(&mut self, at: u64, event: Event) {
        self.heap.push(Reverse((at, self.next)));
        self.payloads.insert(self.next, event);
        self.next += 1;
    }

    fn peek_time(&self) -> Option<u64> {
        self.heap.peek().map(|Reverse((at, _))| *at)
    }

    fn pop(&mut self) -> Option<(u64, Event)> {
        let Reverse((at, sequence)) = self.heap.pop()?;
        let event = self.payloads.remove(&sequence).expect("pushed together");
        Some((at, event))
    }
}

fn command_bytes(command: &WorkerCommand) -> u64 {
    match command {
        WorkerCommand::Admit { spec, .. } => MESSAGE_HEADER_BYTES
            .saturating_add(TOKEN_BYTES.saturating_mul(ola_cluster::token_demand(spec) as u64)),
        WorkerCommand::Cancel { .. } => MESSAGE_HEADER_BYTES,
        WorkerCommand::Drain => MESSAGE_HEADER_BYTES,
        WorkerCommand::MigrateOut { .. } => MESSAGE_HEADER_BYTES,
        WorkerCommand::AdmitMigrated { state, .. } => MESSAGE_HEADER_BYTES
            .saturating_add(TOKEN_BYTES.saturating_mul(
                (state.prompt_tokens.len() + state.generated_tokens.len()) as u64,
            )),
    }
}

fn message_bytes(message: &WorkerMessage) -> u64 {
    match message {
        WorkerMessage::Summary(_) => MESSAGE_HEADER_BYTES + SUMMARY_BYTES,
        WorkerMessage::Finished { token_ids, .. } => {
            MESSAGE_HEADER_BYTES.saturating_add(TOKEN_BYTES.saturating_mul(token_ids.len() as u64))
        }
        WorkerMessage::Refused { message, .. } | WorkerMessage::Failed { message, .. } => {
            MESSAGE_HEADER_BYTES.saturating_add(message.len() as u64)
        }
        WorkerMessage::Accepted { .. }
        | WorkerMessage::Token { .. }
        | WorkerMessage::Cancelled { .. } => MESSAGE_HEADER_BYTES,
        WorkerMessage::Migrated { state, .. } => MESSAGE_HEADER_BYTES
            .saturating_add(TOKEN_BYTES.saturating_mul(
                (state.prompt_tokens.len() + state.generated_tokens.len()) as u64,
            )),
    }
}

struct Simulation {
    coordinator: Coordinator,
    workers: Vec<SimWorker>,
    index: BTreeMap<WorkerId, usize>,
    events: Events,
    network: NetworkModel,
    messages: u64,
    network_bytes: u64,
    trace: Vec<ClusterTraceRecord>,
    client: Vec<(u64, ClusterRequestId, ClientEvent)>,
}

impl Simulation {
    fn send_commands(&mut self, output: CoordinatorOutput, at: u64) {
        self.trace.extend(output.trace);
        self.client
            .extend(output.client.into_iter().map(|(id, event)| (at, id, event)));
        for (worker, command) in output.commands {
            let index = self.index[&worker];
            let bytes = command_bytes(&command);
            let link = &mut self.workers[index].downlink_free;
            let deliver = self.network.deliver(link, at, bytes);
            self.messages += 1;
            self.network_bytes += bytes;
            self.events.push(deliver, Event::ToWorker(index, command));
        }
    }

    fn send_message(&mut self, index: usize, message: WorkerMessage, at: u64) {
        let bytes = message_bytes(&message);
        let link = &mut self.workers[index].uplink_free;
        let deliver = self.network.deliver(link, at, bytes);
        self.messages += 1;
        self.network_bytes += bytes;
        self.events
            .push(deliver, Event::ToCoordinator(index, message));
    }

    fn send_summary(&mut self, index: usize, at: u64) {
        let summary = self.workers[index].host.summary(Timestamp::from_micros(at));
        self.send_message(index, WorkerMessage::Summary(summary), at);
    }

    /// Runs one iteration on an idle worker with work, scheduling its
    /// messages at the times their invocations finish.
    fn step_worker(&mut self, index: usize, at: u64) -> Result<(), ClusterSimulationError> {
        let worker = &mut self.workers[index];
        let locals = worker.host.owned().collect::<BTreeMap<_, _>>();
        let Some((iteration, messages)) = worker.host.step(Timestamp::from_micros(at))? else {
            return Ok(());
        };
        for event in &iteration.events {
            if let ExecutorEvent::Rejected { request_id, .. } = event {
                worker.rejected.insert(*request_id);
            }
        }
        let record = trial_iteration(&iteration, worker.host.executor(), &worker.config, at);
        let duration = record.duration_micros;
        worker.run.iterations.push(record);
        let finishes = worker
            .config
            .cost
            .batch_finish_offsets(&iteration.execution_batches);
        let finished_at = |request: ClusterRequestId| {
            locals
                .get(&request)
                .and_then(|local| {
                    iteration
                        .execution_batches
                        .iter()
                        .zip(&finishes)
                        .find(|(batch, _)| batch.request_ids.contains(local))
                })
                .map_or(at, |(_, offset)| at.saturating_add(*offset))
        };
        let timed = messages
            .into_iter()
            .map(|message| {
                let sent = match &message {
                    WorkerMessage::Token { request, .. }
                    | WorkerMessage::Finished { request, .. }
                    | WorkerMessage::Failed { request, .. } => finished_at(*request),
                    _ => at,
                };
                (sent, message)
            })
            .collect::<Vec<_>>();
        if iteration.execution_batches.is_empty() {
            worker.blocked = true;
        } else {
            worker.busy_until = at.saturating_add(duration.max(1));
            worker.run.busy_micros = worker.run.busy_micros.saturating_add(duration);
            self.events.push(worker.busy_until, Event::WorkerFree);
        }
        let end = worker.busy_until.max(at);
        for (sent, message) in timed {
            self.send_message(index, message, sent);
        }
        self.send_summary(index, end);
        Ok(())
    }
}

impl NetworkModel {
    /// Delivery time of `bytes` sent at `at` on a link free from `*link`,
    /// advancing the link.
    fn deliver(&self, link: &mut u64, at: u64, bytes: u64) -> u64 {
        let start = (*link).max(at);
        *link = start.saturating_add(self.transmit_micros(bytes));
        link.saturating_add(self.latency_micros)
    }
}

/// Replays `trace` through a coordinator running `policy` and one worker per
/// entry of `config.workers`, on a shared virtual clock.
///
/// Every worker runs the memory-aware FIFO execution policy, so only
/// routing differs between cluster policies. Arrivals are submitted at their
/// trace times and planned as soon as every event at that time has been
/// handled; a worker steps whenever it is idle and has work.
///
/// # Errors
///
/// Returns [`ClusterSimulationError`] if the configuration has no workers,
/// an executor rejects its configuration, or the coordinator rejects a plan
/// or message.
pub fn run_cluster_trial<P>(
    trace: &SchedulingTrace,
    policy: P,
    config: &ClusterSimulationConfig,
) -> Result<ClusterRun, ClusterSimulationError>
where
    P: ClusterPolicy + 'static,
{
    if config.workers.is_empty() {
        return Err(ClusterSimulationError::NoWorkers);
    }
    let descriptor = policy.descriptor();
    let mut simulation = Simulation {
        coordinator: Coordinator::new(policy),
        workers: Vec::new(),
        index: BTreeMap::new(),
        events: Events::new(),
        network: config.network,
        messages: 0,
        network_bytes: 0,
        trace: Vec::new(),
        client: Vec::new(),
    };
    for (index, (id, worker)) in config
        .worker_ids()
        .into_iter()
        .zip(&config.workers)
        .enumerate()
    {
        let executor = simulation_executor(
            ComposedPolicy::default().with_admission(MemoryAwareAdmission),
            worker,
        )?;
        let host = WorkerHost::new(id.clone(), executor);
        simulation
            .coordinator
            .join(host.summary(Timestamp::from_micros(0)))?;
        simulation.index.insert(id.clone(), index);
        simulation.workers.push(SimWorker {
            host,
            config: *worker,
            busy_until: 0,
            blocked: false,
            rejected: BTreeSet::new(),
            run: WorkerRun {
                worker: id,
                routed: 0,
                iterations: Vec::new(),
                busy_micros: 0,
                prefix_hit_tokens: 0,
            },
            downlink_free: 0,
            uplink_free: 0,
        });
    }
    for (index, request) in trace.requests.iter().enumerate() {
        simulation
            .events
            .push(request.arrival_micros, Event::Arrival(index));
    }

    let mut ids = BTreeMap::<ClusterRequestId, usize>::new();
    let mut processed = 0_usize;
    let mut clock = 0_u64;
    while let Some(now) = simulation.events.peek_time() {
        if processed >= config.max_events {
            break;
        }
        clock = now;
        while simulation.events.peek_time() == Some(now) {
            let (_, event) = simulation.events.pop().expect("peeked");
            processed += 1;
            match event {
                Event::Arrival(index) => {
                    let (id, output) = simulation.coordinator.submit(
                        trace_request_spec(&trace.requests[index]),
                        Timestamp::from_micros(now),
                    )?;
                    ids.insert(id, index);
                    simulation.send_commands(output, now);
                }
                Event::ToWorker(index, command) => {
                    let worker = &mut simulation.workers[index];
                    worker.blocked = false;
                    let admit = matches!(command, WorkerCommand::Admit { .. });
                    let replies = worker.host.handle(command, Timestamp::from_micros(now))?;
                    for reply in replies {
                        simulation.send_message(index, reply, now);
                    }
                    if admit && simulation.workers[index].busy_until <= now {
                        simulation.send_summary(index, now);
                    }
                }
                Event::ToCoordinator(index, message) => {
                    let from = simulation.workers[index].run.worker.clone();
                    let output = simulation.coordinator.handle(
                        &from,
                        message,
                        Timestamp::from_micros(now),
                    )?;
                    simulation.send_commands(output, now);
                }
                Event::WorkerFree => {}
            }
        }
        if let Some((_, output)) = simulation.coordinator.plan(Timestamp::from_micros(now))? {
            simulation.send_commands(output, now);
        }
        for index in 0..simulation.workers.len() {
            let worker = &simulation.workers[index];
            if worker.busy_until <= now && !worker.blocked && worker.host.has_work() {
                simulation.step_worker(index, now)?;
            }
        }
    }

    let Simulation {
        workers,
        messages,
        network_bytes,
        trace: cluster_trace,
        client,
        ..
    } = simulation;
    let requests = trial_requests(trace, &ids, &workers, &cluster_trace, &client);
    let workers = workers
        .into_iter()
        .map(|worker| {
            let pool = worker
                .host
                .executor()
                .model(&crate::simulation::MODEL_ID.into())
                .and_then(ola_model::CausalLanguageModel::kv_pool_status);
            WorkerRun {
                routed: cluster_trace
                    .iter()
                    .filter(|record| {
                        matches!(&record.event, ClusterTraceEvent::Routed { worker: routed, .. }
                            if *routed == worker.run.worker)
                    })
                    .count(),
                prefix_hit_tokens: pool.map_or(0, |pool| pool.prefix_hit_tokens),
                ..worker.run
            }
        })
        .collect();
    Ok(ClusterRun {
        policy: descriptor,
        trace: trace.name,
        seed: trace.parameters.seed,
        requests,
        workers,
        messages,
        network_bytes,
        makespan_micros: clock,
        cluster_trace,
    })
}

/// Builds per-request lifecycles from client events (latencies) and the
/// owning worker's records (start time, rejection).
fn trial_requests(
    trace: &SchedulingTrace,
    ids: &BTreeMap<ClusterRequestId, usize>,
    workers: &[SimWorker],
    cluster_trace: &[ClusterTraceRecord],
    client: &[(u64, ClusterRequestId, ClientEvent)],
) -> Vec<TrialRequest> {
    let mut requests = trace
        .requests
        .iter()
        .enumerate()
        .map(|(trace_index, request)| TrialRequest {
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
        })
        .collect::<Vec<_>>();
    for (id, &index) in ids {
        requests[index].request_id = RequestId::new(id.get());
    }
    let owners = cluster_trace
        .iter()
        .filter_map(|record| match &record.event {
            ClusterTraceEvent::Accepted {
                worker,
                local_request,
            } => Some((record.request, (worker.clone(), *local_request))),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    for (at, id, event) in client {
        let Some(&index) = ids.get(id) else { continue };
        let request = &mut requests[index];
        match event {
            ClientEvent::Token {
                generated_tokens: 1,
                ..
            } => request.first_token_micros = Some(*at),
            ClientEvent::Token { .. } => {}
            ClientEvent::Finished { token_ids, .. } => {
                request.outcome = TrialOutcome::Completed;
                request.completed_micros = Some(*at);
                request.output_tokens.clone_from(token_ids);
            }
            ClientEvent::Failed { .. } => {
                request.outcome = TrialOutcome::Failed;
                request.completed_micros = Some(*at);
            }
            ClientEvent::Cancelled => {
                request.outcome = TrialOutcome::Cancelled;
                request.completed_micros = Some(*at);
            }
        }
    }
    for (id, &index) in ids {
        let Some((worker, local)) = owners.get(&id.get()) else {
            continue;
        };
        let worker = workers
            .iter()
            .find(|candidate| candidate.run.worker == *worker)
            .expect("accepted by a joined worker");
        let local = RequestId::new(*local);
        let request = &mut requests[index];
        if let Some(record) = worker.host.executor().state().request(local) {
            request.started_micros = record.timing.started_at.map(Timestamp::as_micros);
            if matches!(record.status, RequestStatus::Failed(_)) && worker.rejected.contains(&local)
            {
                request.outcome = TrialOutcome::Rejected;
            }
        }
    }
    requests
}

/// Request-level metrics of a cluster run plus what only a cluster has.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterSummary {
    /// Latency, throughput, fairness, and deadline metrics over every
    /// request, computed exactly as for one executor. Iteration totals and
    /// peaks span every worker.
    pub scheduling: SchedulingSummary,
    /// Prompt tokens served from prefix caches, summed over workers.
    pub prefix_hit_tokens: u64,
    /// Requests routed to each worker, in worker-ID order.
    pub routed: Vec<usize>,
    /// Busiest worker's busy time over the mean busy time; 1 is perfectly
    /// even.
    pub busy_imbalance: f64,
    /// Messages sent.
    pub messages: u64,
    /// Bytes sent.
    pub network_bytes: u64,
}

/// Summarizes a cluster run against isolated per-request times (see
/// [`isolated_end_to_end`](crate::isolated_end_to_end)).
pub fn summarize_cluster_run(
    run: &ClusterRun,
    isolated_micros: &[Option<u64>],
    starvation_factor: u64,
) -> ClusterSummary {
    let mut iterations = run
        .workers
        .iter()
        .flat_map(|worker| worker.iterations.iter().copied())
        .collect::<Vec<_>>();
    iterations.sort_by_key(|iteration| iteration.at_micros);
    let prefix_hit_tokens = run
        .workers
        .iter()
        .map(|worker| worker.prefix_hit_tokens)
        .sum();
    let flat = crate::SchedulingRun {
        policy: run.policy.clone(),
        trace: run.trace,
        seed: run.seed,
        requests: run.requests.clone(),
        iterations,
        makespan_micros: run.makespan_micros,
        injected: Vec::new(),
        final_live_requests: 0,
        final_reserved_kv_bytes: 0,
        final_kv_cache_bytes: 0,
    };
    let mut scheduling = summarize_run(&flat, isolated_micros, starvation_factor);
    scheduling.prefix_hit_tokens = prefix_hit_tokens;
    let busy = run
        .workers
        .iter()
        .map(|worker| worker.busy_micros)
        .collect::<Vec<_>>();
    let total = busy.iter().sum::<u64>();
    #[allow(
        clippy::cast_precision_loss,
        reason = "virtual times are far below 2^52 microseconds"
    )]
    let busy_imbalance = if total == 0 {
        1.0
    } else {
        busy.iter().copied().max().unwrap_or(0) as f64 * busy.len() as f64 / total as f64
    };
    ClusterSummary {
        scheduling,
        prefix_hit_tokens,
        routed: run.workers.iter().map(|worker| worker.routed).collect(),
        busy_imbalance,
        messages: run.messages,
        network_bytes: run.network_bytes,
    }
}
