//! Online parameter optimization by replaying recent work in virtual time.
//!
//! [`ReplayOptimizer`] answers the question an online optimizer cannot answer
//! from live metrics alone: how would a parameter vector it is *not* running
//! have done? It remembers the recent workload from adaptive observations,
//! rebuilds it as a [`SchedulingTrace`], and replays that trace through the
//! virtual-time simulator once per candidate. The candidate with the lowest
//! objective wins, but only if it beats the current vector's own replay by a
//! margin, so noise does not make the policy oscillate.
//!
//! Replays are deterministic and use the [`CostModel`] in the optimizer's
//! [`SimulationConfig`]. Their value depends on that model matching the real
//! executor: calibrate it (`cost-calibration`, [`crate::CostFit`]) before
//! trusting proposals on real hardware.
//!
//! [`CostModel`]: crate::CostModel

use std::{collections::BTreeMap, time::Instant};

use serde::Serialize;

use ola_core::{RequestId, Timestamp};
use ola_runtime::{
    AdaptiveConstraints, AdaptiveIntegerDomain, AdaptiveObjectiveMetrics, AdaptiveObjectiveWeights,
    AdaptiveOptimizationError, AdaptiveOptimizer, AdaptivePolicy, AdaptiveRequestObservation,
    ComposedPolicy, ExecutionPolicy, FiniteCandidateOptimizer, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, PolicyError, SwitchingCost,
};

use crate::{
    ArrivalWindows, CostModel, GaussianHmm, KvGeometry, RegimeFilter, SchedulingTrace,
    SimulationConfig, SizeWindows, TraceParameters, TraceRequest, run_scheduling_trial,
    scheduling_summary::run_observations,
};

/// Token-budget grid shared by the adaptive studies and `ola serve
/// --adaptive`: 2^4 to 2^10 tokens per iteration, of which 25% to 100% may
/// be prompt work. The top of the range admits a whole iteration's work at
/// typical sizes, so unbounded batching is always (nearly) a candidate.
pub const TOKEN_BUDGET_DOMAINS: [AdaptiveIntegerDomain; 2] = [
    AdaptiveIntegerDomain {
        name: "log2_max_tokens",
        min: 4,
        max: 10,
        step: 1,
    },
    AdaptiveIntegerDomain {
        name: "prefill_share_percent",
        min: 25,
        max: 100,
        step: 25,
    },
];

/// The default objective: p95 completion plus p95 time to first token.
pub const LATENCY_OBJECTIVE: AdaptiveObjectiveWeights = AdaptiveObjectiveWeights {
    ttft: 1,
    completion: 1,
    throughput: 0,
    starvation: 0,
    deadline: 0,
    fallback: 0,
};

/// The grid's smallest budget, 16 tokens with half for prompts.
pub fn smallest_token_budget() -> Vec<(&'static str, String)> {
    vec![
        ("log2_max_tokens", "4".into()),
        ("prefill_share_percent", "50".into()),
    ]
}

/// The grid's largest budget, 1024 tokens with no prompt cap.
pub fn largest_token_budget() -> Vec<(&'static str, String)> {
    vec![
        ("log2_max_tokens", "10".into()),
        ("prefill_share_percent", "100".into()),
    ]
}

/// The `(max_tokens, max_prefill_tokens)` budget a grid vector names.
///
/// # Errors
///
/// Fails if a parameter is missing or out of range.
pub fn token_budget(theta: &[(&'static str, String)]) -> Result<(usize, usize), PolicyError> {
    let value = |name: &str| {
        theta
            .iter()
            .find(|(key, _)| *key == name)
            .and_then(|(_, value)| value.parse::<u32>().ok())
            .ok_or_else(|| PolicyError::InvalidCandidate(format!("missing {name}")))
    };
    let log2 = value("log2_max_tokens")?;
    let share = value("prefill_share_percent")?;
    if log2 > 20 || !(1..=100).contains(&share) {
        return Err(PolicyError::InvalidCandidate(
            "budget parameters out of range".into(),
        ));
    }
    let max_tokens = 1_usize << log2;
    let max_prefill_tokens = (max_tokens * share as usize / 100).max(1);
    Ok((max_tokens, max_prefill_tokens))
}

/// FIFO with memory-aware admission and the mixed token budget `theta`
/// names: the policy family the adaptive studies and `ola serve --adaptive`
/// tune.
///
/// # Errors
///
/// Fails if `theta` does not name a valid budget.
pub fn token_budget_policy(
    theta: &[(&'static str, String)],
) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let (max_tokens, max_prefill_tokens) = token_budget(theta)?;
    let batch = MixedTokenBudgetPolicy::new(max_tokens, max_prefill_tokens).ok_or_else(|| {
        PolicyError::InvalidCandidate(
            "max_prefill_tokens must be positive and no greater than max_tokens".into(),
        )
    })?;
    Ok(Box::new(
        ComposedPolicy::default()
            .with_admission(MemoryAwareAdmission)
            .with_batch(batch),
    ))
}

/// How often a serving adaptive policy re-plans, and how cautiously.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnlineTuning {
    /// Executor iterations between proposals.
    pub interval: u64,
    /// Completed requests replayed per proposal.
    pub window: usize,
    /// Replay improvement, in percent, required to switch.
    pub min_improvement_percent: u64,
    /// What each candidate is replayed against.
    pub horizon: ReplayHorizon,
}

impl Default for OnlineTuning {
    /// The values `online-adaptive-policy-study` was evaluated with.
    fn default() -> Self {
        Self {
            interval: 40,
            window: 64,
            min_improvement_percent: 5,
            horizon: ReplayHorizon::Past,
        }
    }
}

/// An adaptive token-budget policy for a live executor.
///
/// It starts at the largest budget, which behaves like unbounded batching,
/// and only moves when replaying recent work under `cost` predicts an
/// improvement. `cost` must describe the executor's hardware: fit it with
/// the `cost-calibration` experiment. Replays model `max_active_requests`
/// slots and no prefill chunking, with a KV pool large enough never to
/// bind; live admission still enforces real memory.
///
/// # Errors
///
/// Fails if `tuning` is unusable.
pub fn serving_adaptive_policy(
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    adaptive_token_budget_policy(
        largest_token_budget(),
        cost,
        max_active_requests,
        tuning,
        None,
    )
}

/// [`serving_adaptive_policy`] that also passes a [`ProposalRecord`] for
/// every optimizer call to `log`, on the executor's thread. Proposals are
/// unchanged; `log` should be quick, since the executor waits on it.
///
/// # Errors
///
/// Fails if `tuning` is unusable.
pub fn serving_adaptive_policy_logged(
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
    log: impl FnMut(&ProposalRecord) + Send + 'static,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    adaptive_token_budget_policy(
        largest_token_budget(),
        cost,
        max_active_requests,
        tuning,
        Some(Box::new(log)),
    )
}

/// Callback that receives each optimizer call's record.
type ProposalLog = Box<dyn FnMut(&ProposalRecord) + Send>;

/// [`serving_adaptive_policy`] starting from `theta`, optionally logged.
fn adaptive_token_budget_policy(
    theta: Vec<(&'static str, String)>,
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
    log: Option<ProposalLog>,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    adaptive_token_budget_policy_with(
        theta,
        cost,
        max_active_requests,
        tuning,
        SwitchingCost::default(),
        None,
        log,
    )
}

/// [`adaptive_token_budget_policy`] with an explicit switching cost and an
/// optional regime trigger.
fn adaptive_token_budget_policy_with(
    theta: Vec<(&'static str, String)>,
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
    switching: SwitchingCost,
    trigger: Option<RegimeTrigger>,
    log: Option<ProposalLog>,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    adaptive_policy_over(
        theta,
        TOKEN_BUDGET_DOMAINS.to_vec(),
        token_budget_policy,
        cost,
        max_active_requests,
        tuning,
        switching,
        trigger,
        log,
    )
}

/// Builds the policy a parameter vector describes.
pub type PolicyFactory =
    fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>;

/// A serving adaptive policy over any grid: `factory` builds each point of
/// `domains`, both live and in replays, starting from `theta`.
#[allow(clippy::too_many_arguments)]
fn adaptive_policy_over(
    theta: Vec<(&'static str, String)>,
    domains: Vec<AdaptiveIntegerDomain>,
    factory: PolicyFactory,
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
    switching: SwitchingCost,
    trigger: Option<RegimeTrigger>,
    log: Option<ProposalLog>,
) -> Result<AdaptivePolicy, AdaptiveOptimizationError> {
    let geometry = KvGeometry::default();
    let config = SimulationConfig {
        max_active_requests,
        prefill_chunk_size: None,
        kv: KvGeometry {
            capacity_bytes: geometry.bytes_per_token.saturating_mul(1 << 20),
            ..geometry
        },
        cost,
        ..SimulationConfig::default()
    };
    let optimizer = ReplayOptimizer::new(
        ReplaySettings {
            domains,
            weights: LATENCY_OBJECTIVE,
            config,
            window: tuning.window,
            min_requests: tuning.window.min(8),
            min_improvement_percent: tuning.min_improvement_percent,
            switching,
            horizon: tuning.horizon,
            trigger,
        },
        factory,
    )?;
    let inner = factory(&theta)
        .map_err(|error| AdaptiveOptimizationError::InvalidDomain(error.to_string()))?;
    let policy = AdaptivePolicy::new(inner, theta)
        .with_fallback(ComposedPolicy::default().with_admission(MemoryAwareAdmission))
        .with_constraints(AdaptiveConstraints {
            max_active_requests: Some(max_active_requests),
            prefill_chunk_size: None,
            available_memory_bytes: None,
        })
        .with_optimization_interval(tuning.interval.max(1))
        .with_factory(factory);
    Ok(match log {
        Some(log) => policy.with_optimizer(LoggedOptimizer {
            inner: optimizer,
            log,
            calls: 0,
        }),
        None => policy.with_optimizer(optimizer),
    })
}

/// What the optimizer has learned about one request from observations.
#[derive(Clone, Copy, Debug)]
struct Seen {
    arrived_at: Timestamp,
    prompt_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    generated_tokens: usize,
    completed: bool,
    /// The last `propose` call whose observations mentioned the request.
    /// Every iteration reports every queued and running request, so one
    /// missing from the latest call has left (for example, cancelled).
    last_call: u64,
}

/// How a [`ReplayOptimizer`] searches and when it switches.
#[derive(Clone, Debug)]
pub struct ReplaySettings {
    /// Candidate grid.
    pub domains: Vec<AdaptiveIntegerDomain>,
    /// Objective to minimize over each replay.
    pub weights: AdaptiveObjectiveWeights,
    /// Simulated executor. Its slot limit and prefill chunk are replaced by
    /// the live executor's when the runtime publishes them.
    pub config: SimulationConfig,
    /// Most recent completed requests replayed.
    pub window: usize,
    /// Completed requests needed before the first proposal; until then the
    /// current vector is kept.
    pub min_requests: usize,
    /// Minimum objective improvement, in percent of the current vector's
    /// replay score, required to switch. It guards against replay noise
    /// (the window is a small sample), so it is relative.
    pub min_improvement_percent: u64,
    /// Explicit cost, in objective microseconds, of changing each knob. It
    /// models the transition itself, so it is absolute and enters the
    /// argmin; the default charges nothing.
    pub switching: SwitchingCost,
    /// What each candidate is replayed against.
    pub horizon: ReplayHorizon,
    /// When set, a call replays only when the workload regime changes (or a
    /// refresh is due); otherwise it keeps the current vector at no cost.
    /// Pair it with an optimization interval of one, so every iteration's
    /// arrivals are seen as they happen.
    pub trigger: Option<RegimeTrigger>,
}

/// Replays only on a change of workload regime, estimated by a
/// [`GaussianHmm`] over arrival windows (see `REGIME_DESIGN.md`).
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeTrigger {
    /// Fitted regime model over `features` windows.
    pub model: GaussianHmm,
    /// What the model's emissions describe; must match the fit.
    pub features: RegimeFeatures,
    /// Arrival gaps, or sized requests, per feature window; must match the
    /// fit.
    pub window_arrivals: usize,
    /// Posterior probability the new most probable regime must reach.
    pub confidence: f64,
    /// Replay anyway after this many calls without a replay; `None` never.
    pub refresh_calls: Option<u64>,
}

/// What a [`RegimeTrigger`]'s regime model observes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RegimeFeatures {
    /// [`ArrivalWindows`]: when requests arrive.
    #[default]
    Arrivals,
    /// [`SizeWindows`]: what arrives, as each request's size is reported
    /// (at admission).
    Sizes,
}

/// A [`RegimeTrigger`]'s running state.
#[derive(Clone, Debug)]
struct TriggerState {
    settings: RegimeTrigger,
    filter: RegimeFilter,
    windows: ArrivalWindows,
    sizes: SizeWindows,
    /// Regime the last replay was triggered for.
    regime: Option<usize>,
    /// A change seen and not yet replayed for.
    pending: bool,
    calls_since_replay: u64,
}

impl TriggerState {
    fn new(settings: RegimeTrigger) -> Self {
        Self {
            filter: RegimeFilter::new(settings.model.clone()),
            windows: ArrivalWindows::new(settings.window_arrivals),
            sizes: SizeWindows::new(settings.window_arrivals),
            regime: None,
            pending: false,
            calls_since_replay: 0,
            settings,
        }
    }

    /// Folds in new arrivals, in time order, and newly reported sizes
    /// `(prompt, output limit)`, in arrival order; the model's features
    /// pick which it reads.
    fn observe(
        &mut self,
        arrivals: impl IntoIterator<Item = u64>,
        sizes: impl IntoIterator<Item = (usize, usize)>,
    ) {
        let windows = match self.settings.features {
            RegimeFeatures::Arrivals => arrivals
                .into_iter()
                .filter_map(|arrival| self.windows.push(arrival))
                .collect::<Vec<_>>(),
            RegimeFeatures::Sizes => sizes
                .into_iter()
                .filter_map(|(prompt, output)| self.sizes.push(prompt, output))
                .collect(),
        };
        for features in windows {
            self.filter.update(&features);
            if let Some((state, probability)) = self.filter.most_probable()
                && probability >= self.settings.confidence
                && self.regime != Some(state)
            {
                self.regime = Some(state);
                self.pending = true;
            }
        }
    }

    /// Whether this call should replay.
    fn due(&mut self) -> bool {
        self.calls_since_replay += 1;
        self.pending
            || self
                .settings
                .refresh_calls
                .is_some_and(|refresh| self.calls_since_replay >= refresh)
    }

    fn replayed(&mut self) {
        self.pending = false;
        self.calls_since_replay = 0;
    }
}

/// What a [`ReplayOptimizer`] replays each candidate against.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReplayHorizon {
    /// The last `window` completed requests, from an empty executor: how
    /// the candidate would have done on the recent past. Requests still
    /// queued or running are left out, so under load the window describes
    /// arrivals from about one queueing delay ago.
    #[default]
    Past,
    /// The executor's present and near future: every request still running
    /// or queued, followed by projected arrivals, so a candidate is scored
    /// on the backlog it would actually inherit.
    ///
    /// The simulator cannot start with requests in flight, so each is
    /// approximated as arriving at time zero:
    ///
    /// - a request already decoding keeps its remaining output limit and a
    ///   one-token prompt (the cost model does not price context length,
    ///   so its decodes cost the same; the stand-in prefill is one batched
    ///   invocation for all of them);
    /// - an admitted request still prefilling keeps its prompt and limit;
    /// - a queued request, whose size is not reported until admission,
    ///   borrows the size and early stop of a completed window request.
    ///
    /// Running requests' remaining output is unknown, so it is taken to be
    /// their limit: pessimistic. Projected arrivals repeat the gaps between
    /// the most recent `window` arrivals, including those still queued, so
    /// the rate is current; their sizes cycle through the completed window.
    FromState,
}

/// How one proposal chose, with the objective exactly as it was compared.
///
/// `J(θ) = replay_score(θ) + switch_cost(θ_prev, θ)`; the candidate is the
/// argmin of `J` over the grid, and it replaces `θ_prev` only if
/// `J(candidate) < replay_score(θ_prev) − margin`, with
/// `margin = min_improvement_percent · |replay_score(θ_prev)| / 100`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaySelection {
    /// Proposal number, from 1.
    pub proposal: u64,
    /// Replayed requests.
    pub window_requests: usize,
    /// `replay_score(θ_prev)`; `None` when it cannot be scored.
    pub current_score: Option<i64>,
    /// `argmin J`.
    pub candidate: Vec<(&'static str, String)>,
    /// `replay_score(candidate)`.
    pub candidate_score: Option<i64>,
    /// `switch_cost(θ_prev, candidate)`.
    pub switching_cost: i64,
    /// `J(candidate)`.
    pub candidate_objective: Option<i64>,
    /// Noise margin subtracted from the current score.
    pub margin: i64,
    /// Whether the candidate was installed.
    pub switched: bool,
    /// Simulated replays this proposal ran.
    pub replays: usize,
}

/// An [`AdaptiveOptimizer`] that replays recent work for each candidate.
///
/// `factory` builds the policy a parameter vector describes; use the same
/// factory the [`ola_runtime::AdaptivePolicy`] installs proposals with.
pub struct ReplayOptimizer<F> {
    settings: ReplaySettings,
    factory: F,
    seen: BTreeMap<RequestId, Seen>,
    /// `propose` calls so far, with or without a replay.
    calls: u64,
    proposals: u64,
    switches: u64,
    replays: u64,
    last: Option<ReplaySelection>,
    trigger: Option<TriggerState>,
}

impl<F> ReplayOptimizer<F>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
{
    /// Creates an optimizer.
    ///
    /// # Errors
    ///
    /// Fails if a domain is malformed or the window is empty.
    pub fn new(settings: ReplaySettings, factory: F) -> Result<Self, AdaptiveOptimizationError> {
        if settings.window == 0 {
            return Err(AdaptiveOptimizationError::InvalidDomain(
                "the replay window must hold at least one request".into(),
            ));
        }
        // Validates the domains once, up front.
        FiniteCandidateOptimizer::new(
            settings.domains.clone(),
            |_: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints|
             -> Result<Option<i64>, AdaptiveOptimizationError> { Ok(None) },
        )?;
        Ok(Self {
            factory,
            seen: BTreeMap::new(),
            calls: 0,
            proposals: 0,
            switches: 0,
            replays: 0,
            last: None,
            trigger: settings.trigger.clone().map(TriggerState::new),
            settings,
        })
    }

    /// Simulated replays run by every proposal so far: the controller's
    /// deterministic overhead, independent of host speed.
    pub const fn replays(&self) -> u64 {
        self.replays
    }

    /// The most recent proposal that had enough history to replay.
    pub const fn last_selection(&self) -> Option<&ReplaySelection> {
        self.last.as_ref()
    }

    /// Proposals made with enough history to replay.
    pub const fn proposals(&self) -> u64 {
        self.proposals
    }

    /// The trigger's most probable workload regime and its probability,
    /// once it has seen a window; `None` without a trigger.
    pub fn regime(&self) -> Option<(usize, f64)> {
        self.trigger
            .as_ref()
            .and_then(|trigger| trigger.filter.most_probable())
    }

    /// Proposals that changed the parameter vector.
    pub const fn switches(&self) -> u64 {
        self.switches
    }

    fn ingest(&mut self, observations: &[AdaptiveRequestObservation]) {
        self.calls += 1;
        let mut arrivals = Vec::new();
        let mut sized = Vec::new();
        for observation in observations {
            if !self.seen.contains_key(&observation.request_id) {
                arrivals.push(observation.arrived_at.as_micros());
            }
            let seen = self.seen.entry(observation.request_id).or_insert(Seen {
                arrived_at: observation.arrived_at,
                prompt_tokens: None,
                max_new_tokens: None,
                generated_tokens: 0,
                completed: false,
                last_call: 0,
            });
            seen.last_call = self.calls;
            let was_sized = seen.prompt_tokens.is_some() && seen.max_new_tokens.is_some();
            seen.prompt_tokens = seen.prompt_tokens.or(observation.prompt_tokens);
            seen.max_new_tokens = seen.max_new_tokens.or(observation.max_new_tokens);
            if let (false, Some(prompt), Some(limit)) =
                (was_sized, seen.prompt_tokens, seen.max_new_tokens)
            {
                sized.push((seen.arrived_at, prompt, limit));
            }
            if let Some(generated) = observation.generated_tokens {
                seen.generated_tokens = seen.generated_tokens.max(generated);
            }
            seen.completed |= observation.completed_at.is_some();
        }
        if let Some(trigger) = &mut self.trigger {
            arrivals.sort_unstable();
            sized.sort_unstable();
            trigger.observe(
                arrivals,
                sized.into_iter().map(|(_, prompt, limit)| (prompt, limit)),
            );
        }
        // Keep the window's worth of completed requests and anything live.
        let mut completed = self
            .seen
            .iter()
            .filter(|(_, seen)| seen.completed)
            .map(|(id, seen)| (seen.arrived_at, *id))
            .collect::<Vec<_>>();
        if completed.len() > self.settings.window {
            completed.sort_unstable();
            let excess = completed.len() - self.settings.window;
            for (_, id) in &completed[..excess] {
                self.seen.remove(id);
            }
        }
    }

    /// The workload each candidate is replayed against, or `None` before
    /// `min_requests` requests have completed.
    fn window_trace(&self) -> Option<SchedulingTrace> {
        match self.settings.horizon {
            ReplayHorizon::Past => self.past_trace(),
            ReplayHorizon::FromState => self.state_trace(),
        }
    }

    /// The window's completed requests as `(arrived_at, prompt, limit,
    /// eos_after)`, by arrival.
    fn completed_window(&self) -> Vec<(Timestamp, usize, usize, Option<usize>)> {
        let mut requests = self
            .seen
            .values()
            .filter(|seen| seen.completed)
            .filter_map(|seen| {
                let prompt_tokens = seen.prompt_tokens?;
                let max_new_tokens = seen.max_new_tokens?;
                // A request that stopped short of its limit hit EOS there.
                let eos_after = (seen.generated_tokens > 0
                    && seen.generated_tokens < max_new_tokens)
                    .then_some(seen.generated_tokens);
                Some((seen.arrived_at, prompt_tokens, max_new_tokens, eos_after))
            })
            .collect::<Vec<_>>();
        requests.sort_by_key(|(arrived_at, ..)| *arrived_at);
        requests
    }

    /// [`ReplayHorizon::FromState`]: live requests at time zero, then
    /// projected arrivals.
    fn state_trace(&self) -> Option<SchedulingTrace> {
        let window = self.completed_window();
        if window.len() < self.settings.min_requests.max(1) {
            return None;
        }
        let request = |arrival_micros, prompt_tokens, max_new_tokens, eos_after| TraceRequest {
            arrival_micros,
            prompt_tokens,
            max_new_tokens,
            priority: 0,
            deadline_micros: None,
            eos_after,
            shared_prefix: None,
        };
        let mut live = self
            .seen
            .values()
            .filter(|seen| !seen.completed && seen.last_call == self.calls)
            .collect::<Vec<_>>();
        live.sort_by_key(|seen| seen.arrived_at);
        let mut borrowed = window.iter().cycle();
        let mut requests = Vec::with_capacity(live.len() + window.len());
        // Running first, then prefilling, then queued, as FIFO would hold them.
        for seen in &live {
            if let (Some(_), Some(limit)) = (seen.prompt_tokens, seen.max_new_tokens)
                && seen.generated_tokens > 0
            {
                let remaining = limit.saturating_sub(seen.generated_tokens).max(1);
                requests.push(request(0, 1, remaining, None));
            }
        }
        for seen in &live {
            if let (Some(prompt), Some(limit)) = (seen.prompt_tokens, seen.max_new_tokens)
                && seen.generated_tokens == 0
            {
                requests.push(request(0, prompt, limit, None));
            }
        }
        for seen in &live {
            if seen.prompt_tokens.is_none() {
                let &(_, prompt, limit, eos_after) = borrowed.next()?;
                requests.push(request(0, prompt, limit, eos_after));
            }
        }
        // Projected arrivals at the most recent gaps.
        let mut arrivals = self
            .seen
            .values()
            .map(|seen| seen.arrived_at.as_micros())
            .collect::<Vec<_>>();
        arrivals.sort_unstable();
        let recent = &arrivals[arrivals.len().saturating_sub(self.settings.window + 1)..];
        let gaps = recent
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();
        let mut at = 0_u64;
        for (index, &(_, prompt, limit, eos_after)) in window.iter().enumerate() {
            at = at.saturating_add(gaps.get(index % gaps.len().max(1)).copied().unwrap_or(0));
            requests.push(request(at, prompt, limit, eos_after));
        }
        let count = requests.len();
        Some(SchedulingTrace {
            name: "online-state",
            parameters: TraceParameters {
                seed: 0,
                requests: count,
                mean_interarrival_micros: at / (count as u64).max(1),
            },
            requests,
        })
    }

    /// [`ReplayHorizon::Past`]: the recent completed workload, arrivals
    /// relative to the earliest.
    fn past_trace(&self) -> Option<SchedulingTrace> {
        let requests = self.completed_window();
        if requests.len() < self.settings.min_requests.max(1) {
            return None;
        }
        let start = requests[0].0.as_micros();
        let end = requests[requests.len() - 1].0.as_micros();
        let count = requests.len();
        Some(SchedulingTrace {
            name: "online-window",
            parameters: TraceParameters {
                seed: 0,
                requests: count,
                mean_interarrival_micros: (end - start) / (count as u64).max(1),
            },
            requests: requests
                .into_iter()
                .map(
                    |(arrived_at, prompt_tokens, max_new_tokens, eos_after)| TraceRequest {
                        arrival_micros: arrived_at.as_micros() - start,
                        prompt_tokens,
                        max_new_tokens,
                        priority: 0,
                        deadline_micros: None,
                        eos_after,
                        shared_prefix: None,
                    },
                )
                .collect(),
        })
    }

    fn config(&self, constraints: AdaptiveConstraints) -> SimulationConfig {
        let mut config = self.settings.config;
        if let Some(slots) = constraints.max_active_requests {
            config.max_active_requests = slots;
        }
        if constraints.prefill_chunk_size.is_some() {
            config.prefill_chunk_size = constraints.prefill_chunk_size;
        }
        config
    }
}

/// Replays `trace` under the policy `theta` describes and scores the run;
/// `None` for a vector the factory rejects or a run the objective cannot
/// score.
fn replay_score<F>(
    factory: &F,
    theta: &[(&'static str, String)],
    trace: &SchedulingTrace,
    config: &SimulationConfig,
    weights: AdaptiveObjectiveWeights,
) -> Result<Option<i64>, AdaptiveOptimizationError>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>,
{
    let Ok(policy) = factory(theta) else {
        return Ok(None);
    };
    let run = run_scheduling_trial(trace, policy, config)
        .map_err(|error| AdaptiveOptimizationError::Infeasible(error.to_string()))?;
    Ok(AdaptiveObjectiveMetrics::from_observations(&run_observations(&run)).score(weights))
}

impl<F> AdaptiveOptimizer for ReplayOptimizer<F>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
{
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        self.ingest(observations);
        if self.trigger.as_mut().is_some_and(|trigger| !trigger.due()) {
            return Ok(current.to_vec());
        }
        let Some(trace) = self.window_trace() else {
            return Ok(current.to_vec());
        };
        if let Some(trigger) = &mut self.trigger {
            trigger.replayed();
        }
        self.proposals += 1;
        let config = self.config(constraints);
        let weights = self.settings.weights;
        let switching = &self.settings.switching;
        let factory = &self.factory;
        let replays = std::sync::atomic::AtomicUsize::new(0);
        let score = |theta: &[(&'static str, String)]| {
            replays.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            replay_score(factory, theta, &trace, &config, weights)
        };
        let current_score = score(current)?;
        let mut search = FiniteCandidateOptimizer::new(
            self.settings.domains.clone(),
            |candidate: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints| {
                Ok(score(candidate)?
                    .map(|operating| operating.saturating_add(switching.cost(current, candidate))))
            },
        )?;
        // No feasible candidate keeps the current vector, and says so.
        let (best, best_score) = match search.propose(current, &[], constraints) {
            Ok(best) => {
                let best_score = score(&best)?;
                (best, best_score)
            }
            Err(AdaptiveOptimizationError::NoCandidate) => (current.to_vec(), None),
            Err(error) => return Err(error),
        };
        let switching_cost = switching.cost(current, &best);
        let best_objective = best_score.map(|operating| operating.saturating_add(switching_cost));
        let margin = current_score.map_or(0, |current| {
            i64::try_from(
                i128::from(current.unsigned_abs())
                    * i128::from(self.settings.min_improvement_percent)
                    / 100,
            )
            .unwrap_or(i64::MAX)
        });
        let switch = match (current_score, best_objective) {
            // The current vector cannot even be scored; take any that can.
            (None, Some(_)) => true,
            (Some(current), Some(best)) => {
                i128::from(best) < i128::from(current) - i128::from(margin)
            }
            (_, None) => false,
        };
        let switched = switch && best != current;
        let replays = replays.into_inner();
        self.replays += replays as u64;
        self.last = Some(ReplaySelection {
            proposal: self.proposals,
            window_requests: trace.requests.len(),
            current_score,
            candidate: best.clone(),
            candidate_score: best_score,
            switching_cost,
            candidate_objective: best_objective,
            margin,
            switched,
            replays,
        });
        if switched {
            self.switches += 1;
            Ok(best)
        } else {
            Ok(current.to_vec())
        }
    }
}

/// One `propose` call of a serving adaptive policy, as
/// [`serving_adaptive_policy_logged`] reports it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProposalRecord {
    /// Call number, from 1.
    pub call: u64,
    /// Host wall time inside the call, in microseconds: how long the
    /// executor's observation stalled for it.
    pub host_micros: u64,
    /// Whether there was enough history to replay; without it the call
    /// keeps the current vector at once.
    pub replayed: bool,
    /// Requests in the replayed window.
    pub window_requests: usize,
    /// Simulated replays the call ran.
    pub replays: usize,
    /// `replay_score` of the vector in force.
    pub current_score: Option<i64>,
    /// `replay_score` of the best candidate.
    pub candidate_score: Option<i64>,
    /// Whether the call installed a different vector.
    pub switched: bool,
    /// Parameter vector in force before the call.
    pub from: BTreeMap<String, String>,
    /// Parameter vector after the call.
    pub to: BTreeMap<String, String>,
    /// Error, when the call failed and the policy kept its vector.
    pub error: Option<String>,
    /// The trigger's most probable workload regime after the call, when the
    /// optimizer has a trigger and has seen a window.
    pub regime: Option<usize>,
}

/// Host time an adaptive policy costs the executor it runs on, which the
/// virtual-time simulator otherwise treats as free.
///
/// A serving adaptive policy runs its optimizer inside `observe`, on the
/// executor's thread, so every replaying call stalls the next iteration:
/// `per_call_micros + per_replayed_request_micros × replays × window`.
/// A call without enough history returns at once and costs nothing. An
/// installed switch may cost more on hardware (for example, first use of
/// new kernel shapes); `per_switch_micros` charges that, once per switch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdaptationCost {
    /// Fixed stall of a replaying call.
    pub per_call_micros: u64,
    /// Stall per replayed request per replay.
    pub per_replayed_request_micros: u64,
    /// Extra stall after a call that installed a different vector.
    pub per_switch_micros: u64,
}

impl AdaptationCost {
    /// The stall measured on the RTX 4090 host of the second GPU round
    /// (vast.ai machine 26487, 2026-11-20, E4 runs): a least-squares fit of
    /// 50 logged calls gives 6,476 µs plus 59.4 µs per replayed request per
    /// replay, so about 120 ms for the full 30-replay, 64-request call. The
    /// switch cost there was not measured and is zero.
    pub const RTX_4090_E4: Self = Self {
        per_call_micros: 6_476,
        per_replayed_request_micros: 59,
        per_switch_micros: 0,
    };

    /// The stall `record`'s call costs under this model.
    pub fn stall_micros(&self, record: &ProposalRecord) -> u64 {
        if !record.replayed {
            return 0;
        }
        let replayed = (record.replays as u64).saturating_mul(record.window_requests as u64);
        let switch = if record.switched {
            self.per_switch_micros
        } else {
            0
        };
        self.per_call_micros
            .saturating_add(self.per_replayed_request_micros.saturating_mul(replayed))
            .saturating_add(switch)
    }

    /// The [`SwitchingCost`] that charges this model's switch stall to the
    /// latency objective: a stall delays every request in flight by the
    /// same amount, so it moves both p95 completion and p95 TTFT by about
    /// `per_switch_micros`. A switch's cost is shared by `knobs` changing
    /// together, so each knob carries all of it.
    pub fn latency_switching_cost(&self, knobs: &[&'static str]) -> SwitchingCost {
        SwitchingCost::uniform(
            knobs,
            i64::try_from(self.per_switch_micros.saturating_mul(2)).unwrap_or(i64::MAX),
        )
    }
}

/// Accumulates the stall an [`AdaptationCost`] assigns to each logged call,
/// for the simulator to charge after the iteration that made it.
#[derive(Clone, Debug, Default)]
pub struct AdaptationMeter {
    inner: std::sync::Arc<std::sync::Mutex<MeterState>>,
}

#[derive(Debug, Default)]
struct MeterState {
    pending_micros: u64,
    total_micros: u64,
    calls: u64,
    replaying_calls: u64,
    switches: u64,
}

impl AdaptationMeter {
    fn record(&self, cost: AdaptationCost, record: &ProposalRecord) {
        let stall = cost.stall_micros(record);
        let mut state = self.inner.lock().expect("adaptation meter");
        state.pending_micros = state.pending_micros.saturating_add(stall);
        state.total_micros = state.total_micros.saturating_add(stall);
        state.calls += 1;
        state.replaying_calls += u64::from(record.replayed);
        state.switches += u64::from(record.switched);
    }

    /// Stall charged since the last call, which it clears; pass it as the
    /// charge of [`crate::run_charged_trial`].
    pub fn take_micros(&self) -> u64 {
        std::mem::take(&mut self.inner.lock().expect("adaptation meter").pending_micros)
    }

    /// Stall charged over the meter's life.
    pub fn total_micros(&self) -> u64 {
        self.inner.lock().expect("adaptation meter").total_micros
    }

    /// Optimizer calls, calls that replayed, and switches.
    pub fn counts(&self) -> (u64, u64, u64) {
        let state = self.inner.lock().expect("adaptation meter");
        (state.calls, state.replaying_calls, state.switches)
    }
}

/// [`serving_adaptive_policy`] whose calls are priced by `adaptation` into
/// the returned meter, for virtual-time studies that charge the stall with
/// [`crate::run_charged_trial`]. `switching` and `trigger` act as
/// [`ReplaySettings::switching`] and [`ReplaySettings::trigger`] do; with a
/// trigger, set `tuning.interval` to one.
///
/// # Errors
///
/// Fails if `tuning` is unusable.
pub fn charged_adaptive_policy(
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
    adaptation: AdaptationCost,
    switching: SwitchingCost,
    trigger: Option<RegimeTrigger>,
) -> Result<(AdaptivePolicy, AdaptationMeter), AdaptiveOptimizationError> {
    charged_adaptive_policy_over(
        largest_token_budget(),
        TOKEN_BUDGET_DOMAINS.to_vec(),
        token_budget_policy,
        cost,
        max_active_requests,
        tuning,
        adaptation,
        switching,
        trigger,
    )
}

/// [`charged_adaptive_policy`] over any grid: `factory` builds each point
/// of `domains`, live and in replays, starting from `theta`. Use it to
/// adapt knobs other than the token budget, such as the scheduler.
///
/// # Errors
///
/// Fails if `tuning` or `domains` is unusable, or `factory` rejects `theta`.
#[allow(clippy::too_many_arguments)]
pub fn charged_adaptive_policy_over(
    theta: Vec<(&'static str, String)>,
    domains: Vec<AdaptiveIntegerDomain>,
    factory: PolicyFactory,
    cost: CostModel,
    max_active_requests: usize,
    tuning: OnlineTuning,
    adaptation: AdaptationCost,
    switching: SwitchingCost,
    trigger: Option<RegimeTrigger>,
) -> Result<(AdaptivePolicy, AdaptationMeter), AdaptiveOptimizationError> {
    let meter = AdaptationMeter::default();
    let sink = meter.clone();
    let policy = adaptive_policy_over(
        theta,
        domains,
        factory,
        cost,
        max_active_requests,
        tuning,
        switching,
        trigger,
        Some(Box::new(move |record: &ProposalRecord| {
            sink.record(adaptation, record);
        })),
    )?;
    Ok((policy, meter))
}

/// A [`ReplayOptimizer`] that reports every call to `log` without changing
/// what it proposes.
struct LoggedOptimizer<F, L> {
    inner: ReplayOptimizer<F>,
    log: L,
    calls: u64,
}

impl<F, L> AdaptiveOptimizer for LoggedOptimizer<F, L>
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send + Sync,
    L: FnMut(&ProposalRecord) + Send,
{
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        let proposals = self.inner.proposals();
        let started = Instant::now();
        let result = self.inner.propose(current, observations, constraints);
        let host_micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.calls += 1;
        let selection = self
            .inner
            .last_selection()
            .filter(|_| self.inner.proposals() > proposals);
        let vector = |theta: &[(&'static str, String)]| {
            theta
                .iter()
                .map(|(name, value)| ((*name).to_owned(), value.clone()))
                .collect::<BTreeMap<_, _>>()
        };
        (self.log)(&ProposalRecord {
            call: self.calls,
            host_micros,
            replayed: selection.is_some(),
            window_requests: selection.map_or(0, |s| s.window_requests),
            replays: selection.map_or(0, |s| s.replays),
            current_score: selection.and_then(|s| s.current_score),
            candidate_score: selection.and_then(|s| s.candidate_score),
            switched: selection.is_some_and(|s| s.switched),
            from: vector(current),
            to: result.as_deref().map_or_else(|_| vector(current), vector),
            error: result.as_ref().err().map(ToString::to_string),
            regime: self.inner.regime().map(|(state, _)| state),
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ola_runtime::{
        AdaptivePolicy, ComposedPolicy, MemoryAwareAdmission, MixedTokenBudgetPolicy,
    };

    use super::*;

    fn budget(theta: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
        let log2 = theta
            .iter()
            .find(|(name, _)| *name == "log2_max_tokens")
            .and_then(|(_, value)| value.parse::<u32>().ok())
            .ok_or_else(|| PolicyError::InvalidCandidate("missing log2_max_tokens".into()))?;
        let max_tokens = 1_usize << log2;
        let batch = MixedTokenBudgetPolicy::new(max_tokens, max_tokens)
            .ok_or_else(|| PolicyError::InvalidCandidate("bad budget".into()))?;
        Ok(Box::new(
            ComposedPolicy::default()
                .with_admission(MemoryAwareAdmission)
                .with_batch(batch),
        ))
    }

    fn settings() -> ReplaySettings {
        ReplaySettings {
            domains: vec![AdaptiveIntegerDomain {
                name: "log2_max_tokens",
                min: 4,
                max: 10,
                step: 1,
            }],
            weights: AdaptiveObjectiveWeights {
                ttft: 1,
                completion: 1,
                ..AdaptiveObjectiveWeights::default()
            },
            config: SimulationConfig::default(),
            window: 64,
            min_requests: 8,
            horizon: ReplayHorizon::Past,
            trigger: None,
            min_improvement_percent: 5,
            switching: SwitchingCost::default(),
        }
    }

    type Factory = fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>;
    type Shared = Arc<Mutex<ReplayOptimizer<Factory>>>;

    /// Runs an overload trace under an adaptive policy starting at
    /// `2^log2` tokens, returning the optimizer and the final run.
    fn adapt(log2: u32) -> (Shared, crate::SchedulingRun) {
        adapt_with(log2, settings())
    }

    fn adapt_with(log2: u32, settings: ReplaySettings) -> (Shared, crate::SchedulingRun) {
        let factory: Factory = budget;
        let optimizer = Arc::new(Mutex::new(ReplayOptimizer::new(settings, factory).unwrap()));
        let shared = optimizer.clone();
        let theta = vec![("log2_max_tokens", log2.to_string())];
        let policy = AdaptivePolicy::new(budget(&theta).unwrap(), theta)
            .with_optimization_interval(40)
            .with_factory(budget)
            .with_optimizer(
                move |current: &[(&'static str, String)],
                      observations: &[AdaptiveRequestObservation],
                      constraints: AdaptiveConstraints| {
                    shared
                        .lock()
                        .unwrap()
                        .propose(current, observations, constraints)
                },
            );
        let trace = SchedulingTrace::overload(TraceParameters {
            seed: 3,
            requests: 80,
            mean_interarrival_micros: 2_000,
        });
        let run = run_scheduling_trial(&trace, policy, &SimulationConfig::default()).unwrap();
        (optimizer, run)
    }

    /// Serves `requests` on a synthetic executor under `policy`, returning
    /// each request's tokens and the policy's final descriptor.
    fn serve(
        policy: impl ExecutionPolicy + 'static,
        requests: &[(usize, usize, Option<usize>)],
    ) -> (Vec<Vec<u32>>, String) {
        use ola_core::{Input, ModelDescriptor, RequestSpec};
        use ola_runtime::{Executor, Runtime};

        let mut executor = Executor::with_max_active_requests(Runtime::default(), 4)
            .unwrap()
            .with_policy(policy);
        executor
            .register_model(
                ModelDescriptor::language("synthetic", "synthetic"),
                crate::synthetic_artifact(KvGeometry {
                    capacity_bytes: 64 * 8 * 1024,
                    ..KvGeometry::default()
                }),
            )
            .unwrap();
        let ids = requests
            .iter()
            .enumerate()
            .map(|(index, &(prompt, max_new_tokens, eos))| {
                let mut spec = RequestSpec::text("synthetic", "");
                spec.inputs = vec![Input::Tokens(crate::synthetic_prompt(prompt, eos))];
                spec.generation.max_new_tokens = max_new_tokens;
                executor
                    .submit(spec, Timestamp::from_micros(index as u64 * 50))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut step = 0;
        while executor.state().queued_len() > 0 || executor.has_active_request() {
            step += 1;
            executor
                .step_iteration(Timestamp::from_micros(step * 1_000))
                .unwrap();
        }
        let outputs = ids
            .into_iter()
            .map(|id| executor.take_result(id).unwrap().token_ids)
            .collect();
        (outputs, executor.policy_descriptor().to_string())
    }

    #[test]
    fn the_serving_policy_re_plans_without_changing_outputs() {
        let requests = (0..40)
            .map(|index| {
                let long = index % 5 == 0;
                (
                    if long { 120 } else { 6 + index % 7 },
                    if long { 24 } else { 6 },
                    (index % 3 == 0).then_some(3),
                )
            })
            .collect::<Vec<_>>();
        // From the smallest budget, replays should move it.
        let adaptive = adaptive_token_budget_policy(
            smallest_token_budget(),
            CostModel::default(),
            4,
            OnlineTuning {
                interval: 5,
                window: 8,
                min_improvement_percent: 0,
                horizon: ReplayHorizon::Past,
            },
            None,
        )
        .unwrap();
        let (adapted, descriptor) = serve(adaptive, &requests);
        let (baseline, _) = serve(
            ComposedPolicy::default().with_admission(MemoryAwareAdmission),
            &requests,
        );
        // Budgets reorder work; greedy outputs stay the same.
        assert_eq!(adapted, baseline);
        assert!(!descriptor.contains("log2_max_tokens=4,"), "{descriptor}");

        // Logging every call changes nothing it decides.
        let records = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&records);
        let logged = adaptive_token_budget_policy(
            smallest_token_budget(),
            CostModel::default(),
            4,
            OnlineTuning {
                interval: 5,
                window: 8,
                min_improvement_percent: 0,
                horizon: ReplayHorizon::Past,
            },
            Some(Box::new(move |record: &ProposalRecord| {
                sink.lock().unwrap().push(record.clone());
            })),
        )
        .unwrap();
        let (logged_outputs, logged_descriptor) = serve(logged, &requests);
        assert_eq!(logged_outputs, adapted);
        assert_eq!(logged_descriptor, descriptor);
        let records = records.lock().unwrap();
        assert!(records.iter().any(|record| record.switched), "{records:?}");
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.call, index as u64 + 1);
            assert_eq!(record.switched, record.from != record.to);
            assert!(record.replayed || record.replays == 0);
        }
        // Each call starts from where the previous one left the vector.
        for pair in records.windows(2) {
            assert_eq!(pair[1].from, pair[0].to);
        }
    }

    #[test]
    fn keeps_the_current_vector_until_enough_work_has_completed() {
        let mut optimizer = ReplayOptimizer::new(settings(), budget).unwrap();
        let current = vec![("log2_max_tokens", "4".to_owned())];
        let proposed = optimizer
            .propose(&current, &[], AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, current);
        assert_eq!(optimizer.proposals(), 0);
    }

    #[test]
    fn replays_observed_work_and_leaves_a_poor_budget() {
        let (optimizer, run) = adapt(4);
        let optimizer = optimizer.lock().unwrap();
        // Live observations carried enough to rebuild the workload.
        assert!(optimizer.proposals() > 0);
        assert!(optimizer.switches() > 0);
        let fixed = run_scheduling_trial(
            &SchedulingTrace::overload(TraceParameters {
                seed: 3,
                requests: 80,
                mean_interarrival_micros: 2_000,
            }),
            budget(&[("log2_max_tokens", "4".into())]).unwrap(),
            &SimulationConfig::default(),
        )
        .unwrap();
        let score = |run: &crate::SchedulingRun| {
            AdaptiveObjectiveMetrics::from_observations(&run_observations(run))
                .score(settings().weights)
                .unwrap()
        };
        assert!(score(&run) < score(&fixed), "adapting did not help");
    }

    #[test]
    fn keeps_a_budget_that_replays_cannot_beat() {
        let (optimizer, _) = adapt(10);
        let optimizer = optimizer.lock().unwrap();
        assert!(optimizer.proposals() > 0);
        assert_eq!(optimizer.switches(), 0);
    }

    fn with_switching(cost: i64) -> ReplaySettings {
        ReplaySettings {
            min_improvement_percent: 0,
            switching: SwitchingCost::uniform(&["log2_max_tokens"], cost),
            ..settings()
        }
    }

    #[test]
    fn a_dominant_switching_cost_keeps_the_current_budget() {
        let (optimizer, _) = adapt_with(4, with_switching(1_000_000_000));
        let optimizer = optimizer.lock().unwrap();
        assert!(optimizer.proposals() > 0);
        assert_eq!(optimizer.switches(), 0);
        let last = optimizer.last_selection().unwrap();
        assert!(!last.switched);
        // Staying put costs nothing, so it is the argmin of J.
        assert_eq!(last.candidate, vec![("log2_max_tokens", "4".to_owned())]);
        assert_eq!(last.switching_cost, 0);
    }

    #[test]
    fn a_large_enough_improvement_pays_for_the_switch() {
        let (optimizer, _) = adapt_with(4, with_switching(1_000));
        let optimizer = optimizer.lock().unwrap();
        assert!(optimizer.switches() > 0);
        assert!(optimizer.replays() >= optimizer.proposals() * 8);
    }

    fn observation(
        id: u64,
        arrived: u64,
        size: Option<(usize, usize)>,
        generated: usize,
        completed: bool,
    ) -> AdaptiveRequestObservation {
        AdaptiveRequestObservation {
            request_id: RequestId::new(id),
            arrived_at: Timestamp::from_micros(arrived),
            first_token_at: None,
            completed_at: completed.then(|| Timestamp::from_micros(arrived + 1_000)),
            time_to_first_token_micros: None,
            completion_latency_micros: None,
            deadline_missed: false,
            capability_fallback: false,
            prompt_tokens: size.map(|(prompt, _)| prompt),
            max_new_tokens: size.map(|(_, limit)| limit),
            generated_tokens: (generated > 0).then_some(generated),
        }
    }

    #[test]
    fn the_state_horizon_replays_the_backlog_then_projected_arrivals() {
        let mut optimizer = ReplayOptimizer::new(
            ReplaySettings {
                horizon: ReplayHorizon::FromState,
                ..settings()
            },
            budget,
        )
        .unwrap();
        // Eight completed requests 100 µs apart; the last stopped early.
        let mut first = (1..=8)
            .map(|id| {
                observation(
                    id,
                    id * 100,
                    Some((10, 20)),
                    if id == 8 { 5 } else { 20 },
                    true,
                )
            })
            .collect::<Vec<_>>();
        // One decoding, one admitted but not yet prefilled, one queued.
        first.push(observation(9, 900, Some((30, 40)), 15, false));
        first.push(observation(10, 950, Some((50, 60)), 0, false));
        first.push(observation(11, 1_000, None, 0, false));
        optimizer.ingest(&first);
        let trace = optimizer.window_trace().unwrap();
        let shape = |request: &TraceRequest| {
            (
                request.arrival_micros,
                request.prompt_tokens,
                request.max_new_tokens,
                request.eos_after,
            )
        };
        let shapes = trace.requests.iter().map(shape).collect::<Vec<_>>();
        assert_eq!(
            shapes[0],
            (0, 1, 25, None),
            "decoding: remaining limit, stand-in prompt"
        );
        assert_eq!(shapes[1], (0, 50, 60, None), "prefilling: as admitted");
        assert_eq!(
            shapes[2],
            (0, 10, 20, None),
            "queued: borrows a completed size"
        );
        // Projected: the eight completed sizes at the recent gaps (100, 100,
        // ..., then 50 and 50 between the live arrivals), in order.
        assert_eq!(trace.requests.len(), 3 + 8);
        assert_eq!(shapes[3], (100, 10, 20, None));
        assert_eq!(shapes[10].3, Some(5));
        assert!(
            trace
                .requests
                .windows(2)
                .all(|w| w[0].arrival_micros <= w[1].arrival_micros)
        );

        // The queued request is no longer reported (for example, cancelled):
        // it leaves the backlog.
        optimizer.ingest(&[
            observation(9, 900, None, 16, false),
            observation(10, 950, None, 0, false),
        ]);
        let trace = optimizer.window_trace().unwrap();
        assert_eq!(trace.requests.len(), 2 + 8);
        assert_eq!(shape(&trace.requests[0]), (0, 1, 24, None));
    }

    #[test]
    fn a_regime_trigger_replays_only_when_the_regime_changes() {
        // 80 slow arrivals, then 80 fast ones, each completing at once.
        let mut arrivals = (0..80).map(|i| i * 400_000).collect::<Vec<u64>>();
        let switch = arrivals[79];
        arrivals.extend((1..=80).map(|i| switch + i * 20_000));
        let model = GaussianHmm::fit(&[ArrivalWindows::features(&arrivals, 8)], 2, 30).unwrap();
        let run = |refresh_calls| {
            let mut optimizer = ReplayOptimizer::new(
                ReplaySettings {
                    trigger: Some(RegimeTrigger {
                        model: model.clone(),
                        features: RegimeFeatures::Arrivals,
                        window_arrivals: 8,
                        confidence: 0.9,
                        refresh_calls,
                    }),
                    ..settings()
                },
                budget,
            )
            .unwrap();
            let current = vec![("log2_max_tokens", "10".to_owned())];
            for (index, &arrival) in arrivals.iter().enumerate() {
                let completed = observation(index as u64 + 1, arrival, Some((10, 8)), 8, true);
                optimizer
                    .propose(&current, &[completed], AdaptiveConstraints::default())
                    .unwrap();
            }
            (optimizer.proposals(), optimizer.regime().unwrap().0)
        };
        // The first confident regime, then the change: two replays in 160 calls.
        assert_eq!(run(None), (2, 0));
        // A refresh every 40 calls adds replays without a change.
        assert!(run(Some(40)).0 > 2);
    }

    #[test]
    fn a_size_trigger_replays_when_what_arrives_changes() {
        // A steady arrival rate throughout; requests turn from short to long
        // halfway, which only their sizes show.
        let size = |i: u64| {
            let jitter = (i % 3) as usize;
            if i < 80 {
                (10 + jitter, 8 + jitter)
            } else {
                (500 + jitter, 800 + jitter)
            }
        };
        let sizes = (0..160).map(size).collect::<Vec<_>>();
        let model = GaussianHmm::fit(&[SizeWindows::features(&sizes, 8)], 2, 30).unwrap();
        let mut optimizer = ReplayOptimizer::new(
            ReplaySettings {
                trigger: Some(RegimeTrigger {
                    model,
                    features: RegimeFeatures::Sizes,
                    window_arrivals: 8,
                    confidence: 0.9,
                    refresh_calls: None,
                }),
                ..settings()
            },
            budget,
        )
        .unwrap();
        let current = vec![("log2_max_tokens", "10".to_owned())];
        for (index, &(prompt, limit)) in sizes.iter().enumerate() {
            let at = index as u64 * 100_000;
            let completed = observation(index as u64 + 1, at, Some((prompt, limit)), limit, true);
            optimizer
                .propose(&current, &[completed], AdaptiveConstraints::default())
                .unwrap();
        }
        // The first confident regime, then the change to long requests.
        assert_eq!(optimizer.proposals(), 2);
        assert_eq!(optimizer.regime().unwrap().0, 1);
    }

    #[test]
    fn adaptation_cost_charges_replaying_calls_and_switches() {
        let record = |replayed, switched| ProposalRecord {
            call: 1,
            host_micros: 0,
            replayed,
            window_requests: 64,
            replays: 30,
            current_score: None,
            candidate_score: None,
            switched,
            from: BTreeMap::new(),
            to: BTreeMap::new(),
            error: None,
            regime: None,
        };
        let cost = AdaptationCost {
            per_switch_micros: 1_000,
            ..AdaptationCost::RTX_4090_E4
        };
        assert_eq!(cost.stall_micros(&record(false, false)), 0);
        assert_eq!(
            cost.stall_micros(&record(true, false)),
            6_476 + 59 * 30 * 64
        );
        assert_eq!(
            cost.stall_micros(&record(true, true)),
            6_476 + 59 * 30 * 64 + 1_000
        );
        assert_eq!(
            cost.latency_switching_cost(&["budget"]),
            SwitchingCost::uniform(&["budget"], 2_000)
        );
    }

    #[test]
    fn a_charged_policy_meters_its_calls_without_changing_its_choices() {
        let trace = SchedulingTrace::mixed_short_long(TraceParameters {
            seed: 3,
            requests: 80,
            mean_interarrival_micros: 1_000,
        });
        let config = SimulationConfig::default();
        let tuning = OnlineTuning {
            interval: 8,
            ..OnlineTuning::default()
        };
        let plain = serving_adaptive_policy(CostModel::default(), 8, tuning).unwrap();
        let (charged, meter) = charged_adaptive_policy(
            CostModel::default(),
            8,
            tuning,
            AdaptationCost::RTX_4090_E4,
            SwitchingCost::default(),
            None,
        )
        .unwrap();
        let plain = run_scheduling_trial(&trace, plain, &config).unwrap();
        // Metered but not charged: identical runs.
        let metered = run_scheduling_trial(&trace, charged, &config).unwrap();
        assert_eq!(metered, plain);
        let (calls, replaying, _) = meter.counts();
        assert!(calls > 0 && replaying > 0);
        assert!(meter.total_micros() >= replaying * 6_476);
        assert_eq!(meter.take_micros(), meter.total_micros());
        assert_eq!(meter.take_micros(), 0);
    }

    #[test]
    fn the_recorded_objective_is_the_one_compared() {
        let (optimizer, _) = adapt_with(4, with_switching(1_000));
        let optimizer = optimizer.lock().unwrap();
        let last = optimizer.last_selection().unwrap();
        assert_eq!(
            last.candidate_objective,
            last.candidate_score
                .map(|score| score + last.switching_cost)
        );
        if last.switched {
            assert!(last.candidate_objective.unwrap() < last.current_score.unwrap() - last.margin);
        }
    }

    #[test]
    fn unscorable_candidates_never_replace_the_current_vector() {
        fn reject(_: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
            Err(PolicyError::InvalidCandidate("rejected".into()))
        }
        let mut optimizer = ReplayOptimizer::new(with_switching(0), reject).unwrap();
        let observations = (0..8)
            .map(|index| AdaptiveRequestObservation {
                request_id: RequestId::new(index),
                arrived_at: Timestamp::from_micros(index * 100),
                first_token_at: Some(Timestamp::from_micros(index * 100 + 50)),
                completed_at: Some(Timestamp::from_micros(index * 100 + 90)),
                time_to_first_token_micros: Some(50),
                completion_latency_micros: Some(90),
                deadline_missed: false,
                capability_fallback: false,
                prompt_tokens: Some(8),
                max_new_tokens: Some(4),
                generated_tokens: Some(4),
            })
            .collect::<Vec<_>>();
        let current = vec![("log2_max_tokens", "6".to_owned())];
        let proposed = optimizer
            .propose(&current, &observations, AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, current);
        let last = optimizer.last_selection().unwrap();
        assert_eq!((last.current_score, last.candidate_objective), (None, None));
        assert!(!last.switched);
    }
}
