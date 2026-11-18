//! Beliefs about how many output tokens a request has left to generate.
//!
//! See `POLICY_STATE.md` in this crate. A request's *total* output length
//! is `O = min(L, m)`, where `L ≥ 1` is the length at which the model would
//! emit end-of-sequence and `m` is the caller's `max_new_tokens`. Neither the
//! policy nor the executor knows `L` in advance. After `g` tokens have been
//! generated, the quantity a memory policy needs is the *remaining* output
//! `R = O − g`, and this module only ever answers questions about `R`.
//!
//! # Estimator
//!
//! [`OutputLengthBelief`] keeps a bounded, sliding window of finished
//! requests. A request that emitted end-of-sequence is an exact observation
//! of `L`; a request that reached its limit only shows `L ≥ m` and is a
//! right-censored observation. The survival function `S(t) = P(L > t)` is
//! the Kaplan–Meier product-limit estimate over the window, which uses the
//! censored samples instead of mistaking them for natural stops. Conditioning
//! on the tokens already generated gives
//! `P(R > r) = S(g + r) / S(g)` for `g + r < m`, and zero at or beyond the
//! limit. Probability mass the window cannot place (survival left over past
//! the longest observed stop) is put at the request's own limit, which is
//! the conservative choice for memory.
//!
//! Samples are grouped into classes by model and by the power-of-two bucket
//! of the caller's output limit, the only output-related information known
//! at admission. A class with too few samples at risk falls back to the
//! model's pooled window; if that is also too thin, no belief is returned
//! and callers must use their deterministic worst case.
//!
//! # What this is not
//!
//! The estimate is empirical. Nothing here claims calibrated probabilities:
//! the window may come from a different workload than the next request, and
//! drift is only tracked by forgetting old samples. Check empirical
//! coverage of a quantile on held-out traces before relying on it. Beliefs
//! never change what physical KV capacity validation and the executor
//! enforce.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use ola_core::{ModelId, RequestId};
use ola_model::KvCacheProfile;

use crate::{
    AdmissionDecision, AdmissionOutcome, ExecutionPolicy, ExecutorEvent, ExecutorIteration,
    PlanningSnapshot, PolicyDecision, PolicyDescriptor, PolicyError,
};

/// Bounds and thresholds for an [`OutputLengthBelief`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputBeliefConfig {
    /// Finished requests kept per class and per model pool; older samples
    /// are forgotten first, which is the only drift handling.
    pub window: usize,
    /// Samples still at risk after the request's generated tokens that a
    /// query needs before it returns a distribution.
    pub min_samples: usize,
    /// Live requests tracked at once; the oldest admission is forgotten
    /// when the bound is reached, for example after an unobserved
    /// cancellation.
    pub max_tracked_requests: usize,
    /// Distinct `(model, limit bucket)` classes kept; finished requests in
    /// further classes still update their model's pooled window.
    pub max_classes: usize,
}

impl Default for OutputBeliefConfig {
    fn default() -> Self {
        Self {
            window: 256,
            min_samples: 16,
            max_tracked_requests: 4_096,
            max_classes: 64,
        }
    }
}

/// Why a remaining-output query returned no distribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeliefUnavailable {
    /// `max_new_tokens` is zero or the request already generated all of it.
    InvalidInput {
        /// Caller's output limit.
        max_new_tokens: usize,
        /// Tokens already generated.
        generated_tokens: usize,
    },
    /// Fewer finished samples outlive the generated tokens than required.
    InsufficientSamples {
        /// Samples still at risk after the generated tokens.
        at_risk: usize,
        /// Samples the configuration requires.
        required: usize,
    },
}

/// One finished request, as the estimator stores it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputSample {
    /// Output tokens generated before the request finished.
    generated_tokens: usize,
    /// Whether it stopped at end-of-sequence (`true`) or at its limit, in
    /// which case its natural length is only known to be at least
    /// `generated_tokens`.
    stopped_naturally: bool,
}

/// Sample class: model and `⌈log2(max_new_tokens)⌉`.
type ClassKey = (ModelId, u32);

#[derive(Clone, Debug)]
struct Tracked {
    class: ClassKey,
    max_new_tokens: usize,
    generated_tokens: usize,
    admitted: u64,
}

/// Empirical, censoring-aware belief over remaining output tokens.
///
/// Update it only from committed iterations ([`observe`](Self::observe)),
/// so planning reads a fixed belief. Memory is bounded by
/// [`OutputBeliefConfig`]: at most `max_classes + models` windows of
/// `window` samples, and `max_tracked_requests` live entries.
#[derive(Clone, Debug)]
pub struct OutputLengthBelief {
    config: OutputBeliefConfig,
    classes: BTreeMap<ClassKey, VecDeque<OutputSample>>,
    pooled: BTreeMap<ModelId, VecDeque<OutputSample>>,
    live: BTreeMap<RequestId, Tracked>,
    admissions: u64,
    recorded: u64,
}

impl Default for OutputLengthBelief {
    fn default() -> Self {
        Self::new(OutputBeliefConfig::default())
    }
}

/// Power-of-two bucket of an output limit; zero for limits of 0 or 1.
pub fn output_limit_bucket(max_new_tokens: usize) -> u32 {
    max_new_tokens.max(1).next_power_of_two().trailing_zeros()
}

impl OutputLengthBelief {
    /// Creates an empty belief. Zero bounds are raised to one.
    pub fn new(config: OutputBeliefConfig) -> Self {
        Self {
            config: OutputBeliefConfig {
                window: config.window.max(1),
                min_samples: config.min_samples.max(1),
                max_tracked_requests: config.max_tracked_requests.max(1),
                max_classes: config.max_classes.max(1),
            },
            classes: BTreeMap::new(),
            pooled: BTreeMap::new(),
            live: BTreeMap::new(),
            admissions: 0,
            recorded: 0,
        }
    }

    /// Returns the configuration after normalization.
    pub const fn config(&self) -> OutputBeliefConfig {
        self.config
    }

    /// Finished requests recorded since creation, including forgotten ones.
    pub const fn recorded_samples(&self) -> u64 {
        self.recorded
    }

    /// Live requests currently tracked.
    pub fn tracked_requests(&self) -> usize {
        self.live.len()
    }

    /// Samples currently held across all windows.
    pub fn stored_samples(&self) -> usize {
        self.classes
            .values()
            .chain(self.pooled.values())
            .map(VecDeque::len)
            .sum()
    }

    /// Output tokens a tracked request has generated, if it is tracked.
    pub fn generated_tokens(&self, request_id: RequestId) -> Option<usize> {
        self.live
            .get(&request_id)
            .map(|tracked| tracked.generated_tokens)
    }

    /// Starts tracking a request at its first admission. Re-admission after
    /// preemption keeps the original class and generated count.
    pub fn admit(&mut self, request_id: RequestId, model_id: &ModelId, max_new_tokens: usize) {
        if self.live.contains_key(&request_id) {
            return;
        }
        if self.live.len() >= self.config.max_tracked_requests
            && let Some(oldest) = self
                .live
                .iter()
                .min_by_key(|(_, tracked)| tracked.admitted)
                .map(|(id, _)| *id)
        {
            self.live.remove(&oldest);
        }
        self.admissions += 1;
        self.live.insert(
            request_id,
            Tracked {
                class: (model_id.clone(), output_limit_bucket(max_new_tokens)),
                max_new_tokens,
                generated_tokens: 0,
                admitted: self.admissions,
            },
        );
    }

    /// Records a generated token; `finished` ends tracking and, for a
    /// tracked request, stores it as a sample.
    pub fn token(
        &mut self,
        request_id: RequestId,
        generated_tokens: usize,
        finished: bool,
        end_of_sequence: bool,
    ) {
        if !finished {
            if let Some(tracked) = self.live.get_mut(&request_id) {
                tracked.generated_tokens = tracked.generated_tokens.max(generated_tokens);
            }
            return;
        }
        let Some(tracked) = self.live.remove(&request_id) else {
            return;
        };
        self.record(
            tracked.class,
            OutputSample {
                generated_tokens,
                stopped_naturally: end_of_sequence,
            },
        );
    }

    /// Stops tracking a request without learning from it: failed, rejected,
    /// or cancelled requests say nothing about natural output length.
    pub fn forget(&mut self, request_id: RequestId) {
        self.live.remove(&request_id);
    }

    /// Updates the belief from one committed iteration.
    pub fn observe(&mut self, iteration: &ExecutorIteration) {
        for record in &iteration.decision.admissions {
            if let AdmissionOutcome::Decided {
                context,
                decision: AdmissionDecision::Admit { .. },
                budget_selected: Some(true),
            } = &record.outcome
            {
                self.admit(record.request_id, &context.model_id, context.max_new_tokens);
            }
        }
        for event in &iteration.events {
            match event {
                ExecutorEvent::Token {
                    request_id,
                    token,
                    generated_tokens,
                    finished,
                    ..
                } => self.token(
                    *request_id,
                    *generated_tokens,
                    *finished,
                    token.is_end_of_sequence,
                ),
                ExecutorEvent::Failed { request_id, .. }
                | ExecutorEvent::Rejected { request_id, .. } => self.forget(*request_id),
                _ => {}
            }
        }
    }

    fn record(&mut self, class: ClassKey, sample: OutputSample) {
        self.recorded += 1;
        let window = self.config.window;
        let push = |samples: &mut VecDeque<OutputSample>| {
            if samples.len() >= window {
                samples.pop_front();
            }
            samples.push_back(sample);
        };
        push(self.pooled.entry(class.0.clone()).or_default());
        if self.classes.contains_key(&class) || self.classes.len() < self.config.max_classes {
            push(self.classes.entry(class).or_default());
        }
    }

    /// [`remaining`](Self::remaining) for a specific request. A tracked
    /// request (for example one resuming after preemption, whose queued
    /// limit is only what is left) is described by its original limit and
    /// generated count; an untracked one by `max_new_tokens` and nothing
    /// generated, which is exact for a request not yet admitted.
    ///
    /// # Errors
    ///
    /// As for [`remaining`](Self::remaining).
    pub fn remaining_for(
        &self,
        request_id: RequestId,
        model_id: &ModelId,
        max_new_tokens: usize,
    ) -> Result<RemainingOutputDistribution, BeliefUnavailable> {
        match self.live.get(&request_id) {
            Some(tracked) => self.remaining(
                &tracked.class.0,
                tracked.max_new_tokens,
                tracked.generated_tokens,
            ),
            None => self.remaining(model_id, max_new_tokens, 0),
        }
    }

    /// Distribution of the remaining output of a request on `model_id` with
    /// caller limit `max_new_tokens` that has generated `generated_tokens`.
    ///
    /// # Errors
    ///
    /// Returns [`BeliefUnavailable`] for impossible inputs or when neither
    /// the request's class nor its model pool has `min_samples` at risk;
    /// callers must then fall back to the worst case `max_new_tokens − g`.
    pub fn remaining(
        &self,
        model_id: &ModelId,
        max_new_tokens: usize,
        generated_tokens: usize,
    ) -> Result<RemainingOutputDistribution, BeliefUnavailable> {
        if max_new_tokens == 0 || generated_tokens >= max_new_tokens {
            return Err(BeliefUnavailable::InvalidInput {
                max_new_tokens,
                generated_tokens,
            });
        }
        let class = (model_id.clone(), output_limit_bucket(max_new_tokens));
        let mut best_at_risk = 0;
        for samples in [self.classes.get(&class), self.pooled.get(model_id)]
            .into_iter()
            .flatten()
        {
            let at_risk = samples
                .iter()
                .filter(|sample| sample.generated_tokens > generated_tokens)
                .count();
            if at_risk >= self.config.min_samples {
                return Ok(RemainingOutputDistribution::kaplan_meier(
                    samples,
                    max_new_tokens,
                    generated_tokens,
                    at_risk,
                ));
            }
            best_at_risk = best_at_risk.max(at_risk);
        }
        Err(BeliefUnavailable::InsufficientSamples {
            at_risk: best_at_risk,
            required: self.config.min_samples,
        })
    }
}

/// Discrete distribution of a request's remaining output tokens `R`.
///
/// Support points are strictly increasing, lie in `1..=max_remaining`, and
/// their probabilities sum to one (up to floating-point rounding).
#[derive(Clone, Debug, PartialEq)]
pub struct RemainingOutputDistribution {
    points: Vec<(usize, f64)>,
    max_remaining: usize,
    samples_at_risk: usize,
}

impl RemainingOutputDistribution {
    /// A distribution with all mass at the worst case: the request runs to
    /// its limit. Useful as an explicit conservative stand-in.
    pub fn worst_case(max_remaining: usize) -> Self {
        let max_remaining = max_remaining.max(1);
        Self {
            points: vec![(max_remaining, 1.0)],
            max_remaining,
            samples_at_risk: 0,
        }
    }

    fn kaplan_meier(
        samples: &VecDeque<OutputSample>,
        max_new_tokens: usize,
        generated_tokens: usize,
        samples_at_risk: usize,
    ) -> Self {
        // Natural stops strictly after `g` and strictly before the limit
        // are the only event times that move mass below the limit.
        let mut stops = BTreeMap::<usize, usize>::new();
        let mut exits = BTreeMap::<usize, usize>::new();
        for sample in samples {
            if sample.generated_tokens <= generated_tokens {
                continue;
            }
            *exits.entry(sample.generated_tokens).or_default() += 1;
            if sample.stopped_naturally {
                *stops.entry(sample.generated_tokens).or_default() += 1;
            }
        }
        let max_remaining = max_new_tokens - generated_tokens;
        let mut at_risk = samples_at_risk;
        let mut survival = 1.0_f64;
        let mut points = Vec::new();
        for (&time, &leaving) in &exits {
            if time >= max_new_tokens {
                break;
            }
            if let Some(&events) = stops.get(&time) {
                let mass = survival * events as f64 / at_risk as f64;
                if mass > 0.0 {
                    points.push((time - generated_tokens, mass));
                    survival -= mass;
                }
            }
            at_risk -= leaving;
        }
        // Whatever survives to the limit, including mass the window cannot
        // resolve, finishes at the limit.
        if survival > 0.0 || points.is_empty() {
            points.push((max_remaining, survival.max(0.0)));
        }
        Self {
            points,
            max_remaining,
            samples_at_risk,
        }
    }

    /// Support points and their probabilities, increasing in tokens.
    pub fn points(&self) -> &[(usize, f64)] {
        &self.points
    }

    /// Largest possible remaining output: the limit minus generated tokens.
    pub const fn max_remaining(&self) -> usize {
        self.max_remaining
    }

    /// Finished samples that informed the estimate, before conditioning.
    pub const fn samples_at_risk(&self) -> usize {
        self.samples_at_risk
    }

    /// `P(R > tokens)`.
    pub fn exceedance(&self, tokens: usize) -> f64 {
        self.points
            .iter()
            .filter(|(point, _)| *point > tokens)
            .map(|(_, probability)| probability)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Smallest `r` with `P(R ≤ r) ≥ probability`; the worst case for
    /// probabilities at or above one, or that are not numbers.
    pub fn quantile(&self, probability: f64) -> usize {
        if probability.is_nan() || probability >= 1.0 {
            return self.max_remaining;
        }
        let mut cumulative = 0.0;
        for &(point, mass) in &self.points {
            cumulative += mass;
            if cumulative >= probability {
                return point;
            }
        }
        self.max_remaining
    }

    /// Expected remaining output.
    pub fn mean(&self) -> f64 {
        self.points
            .iter()
            .map(|(point, probability)| *point as f64 * probability)
            .sum()
    }

    /// Inverse-CDF sample for a uniform `u` in `[0, 1)`.
    pub fn sample(&self, u: f64) -> usize {
        self.quantile(u.clamp(0.0, 1.0))
    }
}

/// Builds the policy that plans from a frozen belief.
type BeliefFactory = Box<dyn Fn(Arc<OutputLengthBelief>) -> Box<dyn ExecutionPolicy> + Send>;

/// An execution policy that plans from a remaining-output belief.
///
/// The belief `B_n` is updated only in [`observe`](ExecutionPolicy::observe),
/// after an iteration commits. When a finished request changes it, the
/// wrapper freezes a copy behind an [`Arc`] and rebuilds the inner policy
/// with `factory`; `plan` reads that frozen copy and never mutates.
/// Typical use: `factory` returns a `ComposedPolicy` with
/// [`QuantileAdmission`](crate::QuantileAdmission) and preemption.
pub struct BeliefPolicy {
    belief: OutputLengthBelief,
    factory: BeliefFactory,
    inner: Box<dyn ExecutionPolicy>,
    built_from: u64,
}

impl BeliefPolicy {
    /// Creates the wrapper and builds the first inner policy from `belief`.
    pub fn new(
        belief: OutputLengthBelief,
        factory: impl Fn(Arc<OutputLengthBelief>) -> Box<dyn ExecutionPolicy> + Send + 'static,
    ) -> Self {
        let inner = factory(Arc::new(belief.clone()));
        Self {
            built_from: belief.recorded_samples(),
            belief,
            factory: Box::new(factory),
            inner,
        }
    }

    /// The current belief.
    pub const fn belief(&self) -> &OutputLengthBelief {
        &self.belief
    }

    /// The policy currently planning.
    pub fn inner(&self) -> &dyn ExecutionPolicy {
        self.inner.as_ref()
    }
}

impl ExecutionPolicy for BeliefPolicy {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("belief")
            .with_config("window", self.belief.config().window)
            .with_config("min_samples", self.belief.config().min_samples)
            .with_component(self.inner.descriptor())
    }

    fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
        let mut decision = self.inner.plan(snapshot)?;
        decision.record.policy = PolicyDescriptor::new("belief")
            .with_config("window", self.belief.config().window)
            .with_config("min_samples", self.belief.config().min_samples)
            .with_component(decision.record.policy);
        Ok(decision)
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        self.inner.enforces_model_kv_capacity()
    }

    fn observe(&mut self, iteration: &ExecutorIteration) {
        self.inner.observe(iteration);
        self.belief.observe(iteration);
        if self.belief.recorded_samples() != self.built_from {
            self.built_from = self.belief.recorded_samples();
            self.inner = (self.factory)(Arc::new(self.belief.clone()));
        }
    }

    fn recovers_kv_exhaustion(&self) -> bool {
        self.inner.recovers_kv_exhaustion()
    }
}

/// Result of [`kv_overflow_risk`]: an empirical estimate, not a guarantee.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvRiskEstimate {
    /// Scenarios drawn.
    pub scenarios: usize,
    /// Scenarios in which the live set's demand exceeded capacity.
    pub exceeded: usize,
    /// `exceeded / scenarios`.
    pub estimate: f64,
    /// `estimate + sqrt(ln(1/δ) / 2n)`: with probability at least `1 − δ`
    /// over the scenario draws, the risk *under the belief model* is below
    /// this. It says nothing about error in the belief itself.
    pub sampling_upper_bound: f64,
    /// Live requests whose remaining output had no belief and were charged
    /// their worst case in every scenario.
    pub worst_case_requests: usize,
}

/// One live request as the risk estimator sees it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveKvDemand {
    /// Model whose belief describes the request.
    pub model_id: ModelId,
    /// KV positions the request already occupies.
    pub cached_tokens: usize,
    /// Caller's original output limit.
    pub max_new_tokens: usize,
    /// Output tokens generated so far.
    pub generated_tokens: usize,
}

/// Estimates `P(Σ final KV footprint of the live set > capacity)` for one
/// model pool by drawing `scenarios` remaining-output vectors from `belief`.
///
/// Each scenario charges every request `cached + R − 1` page-rounded
/// positions (the last generated token is never a decode input). Summing
/// final footprints overstates peak concurrent demand, because requests
/// finish at different times, so the estimate is conservative for the
/// question "will the current live set need preemption", and ignores any
/// future admissions. Draws are reproducible from `seed`. Validation of
/// actual plans is unaffected.
pub fn kv_overflow_risk(
    belief: &OutputLengthBelief,
    profile: KvCacheProfile,
    live: &[LiveKvDemand],
    scenarios: usize,
    delta: f64,
    seed: u64,
) -> KvRiskEstimate {
    let scenarios = scenarios.max(1);
    let distributions = live
        .iter()
        .map(|request| {
            belief
                .remaining(
                    &request.model_id,
                    request.max_new_tokens,
                    request.generated_tokens,
                )
                .ok()
        })
        .collect::<Vec<_>>();
    let worst_case_requests = distributions
        .iter()
        .filter(|belief| belief.is_none())
        .count();
    let mut state = seed;
    let mut exceeded = 0;
    for _ in 0..scenarios {
        let mut demand = 0_u64;
        for (request, distribution) in live.iter().zip(&distributions) {
            let worst = request
                .max_new_tokens
                .saturating_sub(request.generated_tokens);
            let remaining = match distribution {
                Some(distribution) => {
                    state = crate::adaptive::splitmix64(state);
                    distribution.sample((state >> 11) as f64 / (1_u64 << 53) as f64)
                }
                None => worst,
            };
            let tokens = request
                .cached_tokens
                .saturating_add(remaining.saturating_sub(1));
            demand = demand.saturating_add(profile.estimate_bytes(tokens).unwrap_or(u64::MAX));
        }
        if demand > profile.capacity_bytes() {
            exceeded += 1;
        }
    }
    let estimate = exceeded as f64 / scenarios as f64;
    let delta = if delta > 0.0 && delta < 1.0 {
        delta
    } else {
        0.05
    };
    KvRiskEstimate {
        scenarios,
        exceeded,
        estimate,
        sampling_upper_bound: (estimate + ((1.0 / delta).ln() / (2.0 * scenarios as f64)).sqrt())
            .min(1.0),
        worst_case_requests,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelId {
        ModelId::new("m")
    }

    fn id(value: u64) -> RequestId {
        RequestId::new(value)
    }

    fn finish(belief: &mut OutputLengthBelief, request: u64, limit: usize, generated: usize) {
        belief.admit(id(request), &model(), limit);
        belief.token(id(request), generated, true, generated < limit);
    }

    fn config(min_samples: usize) -> OutputBeliefConfig {
        OutputBeliefConfig {
            min_samples,
            ..OutputBeliefConfig::default()
        }
    }

    #[test]
    fn cold_start_reports_no_belief() {
        let belief = OutputLengthBelief::default();
        assert_eq!(
            belief.remaining(&model(), 64, 0),
            Err(BeliefUnavailable::InsufficientSamples {
                at_risk: 0,
                required: 16
            })
        );
    }

    #[test]
    fn rejects_impossible_queries() {
        let belief = OutputLengthBelief::default();
        assert!(matches!(
            belief.remaining(&model(), 0, 0),
            Err(BeliefUnavailable::InvalidInput { .. })
        ));
        assert!(matches!(
            belief.remaining(&model(), 8, 8),
            Err(BeliefUnavailable::InvalidInput { .. })
        ));
    }

    #[test]
    fn natural_stops_form_a_distribution_over_remaining_tokens() {
        let mut belief = OutputLengthBelief::new(config(4));
        for (request, length) in [10, 10, 20, 40].into_iter().enumerate() {
            finish(&mut belief, request as u64, 64, length);
        }
        let fresh = belief.remaining(&model(), 64, 0).unwrap();
        assert_eq!(fresh.points(), &[(10, 0.5), (20, 0.25), (40, 0.25)][..]);
        assert_eq!(fresh.quantile(0.5), 10);
        assert_eq!(fresh.quantile(0.9), 40);
        assert!((fresh.exceedance(15) - 0.5).abs() < 1e-12);
        assert!((fresh.mean() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn conditioning_on_generated_tokens_predicts_remaining_not_total() {
        let mut belief = OutputLengthBelief::new(config(2));
        for (request, length) in [10, 10, 20, 40].into_iter().enumerate() {
            finish(&mut belief, request as u64, 64, length);
        }
        // After 15 tokens only the 20- and 40-token samples remain at risk.
        let later = belief.remaining(&model(), 64, 15).unwrap();
        assert_eq!(later.points(), &[(5, 0.5), (25, 0.5)][..]);
        assert_eq!(later.max_remaining(), 49);
        // Too few samples outlive 30 tokens for this configuration.
        assert_eq!(
            belief.remaining(&model(), 64, 30),
            Err(BeliefUnavailable::InsufficientSamples {
                at_risk: 1,
                required: 2
            })
        );
    }

    #[test]
    fn requests_that_hit_their_limit_are_censored_not_natural_stops() {
        let mut belief = OutputLengthBelief::new(config(4));
        // Two natural stops at 4; two requests hit their 8-token limit.
        finish(&mut belief, 0, 8, 4);
        finish(&mut belief, 1, 8, 4);
        finish(&mut belief, 2, 8, 8);
        finish(&mut belief, 3, 8, 8);
        // For a 64-token request, the censored half must not be read as
        // stopping at 8: the unresolved mass goes to the limit.
        let wide = belief.remaining(&model(), 64, 0).unwrap();
        assert_eq!(wide.points(), &[(4, 0.5), (64, 0.5)][..]);
        assert_eq!(wide.quantile(0.75), 64);
    }

    #[test]
    fn mass_beyond_the_limit_finishes_at_the_limit() {
        let mut belief = OutputLengthBelief::new(config(2));
        finish(&mut belief, 0, 64, 10);
        finish(&mut belief, 1, 64, 50);
        let short = belief.remaining(&model(), 32, 0).unwrap();
        assert_eq!(short.points(), &[(10, 0.5), (32, 0.5)][..]);
        assert_eq!(short.quantile(1.0), 32);
    }

    #[test]
    fn preemption_keeps_the_class_and_count_and_failures_teach_nothing() {
        let mut belief = OutputLengthBelief::new(config(1));
        belief.admit(id(1), &model(), 64);
        belief.token(id(1), 5, false, false);
        // Re-admission after preemption sees the remaining limit only.
        belief.admit(id(1), &model(), 59);
        assert_eq!(belief.generated_tokens(id(1)), Some(5));
        belief.token(id(1), 9, true, true);
        assert_eq!(belief.recorded_samples(), 1);
        assert_eq!(belief.tracked_requests(), 0);

        belief.admit(id(2), &model(), 64);
        belief.forget(id(2));
        belief.token(id(2), 3, true, true);
        assert_eq!(belief.recorded_samples(), 1);
    }

    #[test]
    fn memory_is_bounded() {
        let mut belief = OutputLengthBelief::new(OutputBeliefConfig {
            window: 8,
            min_samples: 1,
            max_tracked_requests: 4,
            max_classes: 2,
        });
        for request in 0..100 {
            belief.admit(id(request), &model(), 1 + request as usize);
        }
        assert_eq!(belief.tracked_requests(), 4);
        for request in 0..100_u64 {
            finish(&mut belief, 1_000 + request, 1 << (request % 6), 1);
        }
        // Two class windows and one pooled window, eight samples each.
        assert_eq!(belief.stored_samples(), 24);
        assert_eq!(belief.recorded_samples(), 100);
    }

    #[test]
    fn sliding_window_forgets_an_old_regime() {
        let mut belief = OutputLengthBelief::new(OutputBeliefConfig {
            window: 4,
            min_samples: 4,
            ..OutputBeliefConfig::default()
        });
        for request in 0..4 {
            finish(&mut belief, request, 64, 8);
        }
        assert_eq!(belief.remaining(&model(), 64, 0).unwrap().quantile(0.5), 8);
        for request in 4..8 {
            finish(&mut belief, request, 64, 60);
        }
        assert_eq!(belief.remaining(&model(), 64, 0).unwrap().quantile(0.5), 60);
    }

    #[test]
    fn kv_risk_is_seeded_and_charges_unknown_requests_their_worst_case() {
        let mut belief = OutputLengthBelief::new(config(4));
        for (request, length) in [4, 4, 4, 60].into_iter().enumerate() {
            finish(&mut belief, request as u64, 64, length);
        }
        let profile = KvCacheProfile::new(1, 4, 140).unwrap();
        let live = vec![
            LiveKvDemand {
                model_id: model(),
                cached_tokens: 16,
                max_new_tokens: 64,
                generated_tokens: 0,
            },
            LiveKvDemand {
                model_id: model(),
                cached_tokens: 16,
                max_new_tokens: 64,
                generated_tokens: 0,
            },
        ];
        let first = kv_overflow_risk(&belief, profile, &live, 2_000, 0.05, 7);
        assert_eq!(
            first,
            kv_overflow_risk(&belief, profile, &live, 2_000, 0.05, 7)
        );
        // Overflow needs both requests long: about 1/16.
        assert!((first.estimate - 0.0625).abs() < 0.02, "{first:?}");
        assert!(first.sampling_upper_bound > first.estimate);
        assert_eq!(first.worst_case_requests, 0);

        let unknown = OutputLengthBelief::default();
        let worst = kv_overflow_risk(&unknown, profile, &live, 10, 0.05, 7);
        assert_eq!(worst.worst_case_requests, 2);
        assert_eq!(worst.exceeded, 10);
    }
}
