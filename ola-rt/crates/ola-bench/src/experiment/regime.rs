//! Does the preferred KV-reservation action change between workload
//! regimes, and does adapting to the change pay? In virtual time.
//!
//! See `REGIME_STUDY.md` in `ola-runtime`. The action is how much KV an
//! admitted request reserves beyond its prompt: a fixed headroom (watermark
//! admission), the worst case, or a quantile of the remaining-output belief.
//! Every action runs with newest-first preemption, so an over-committed
//! pool is recovered by recompute and a mid-run switch to the worst case is
//! safe. Every plan is validated by the executor as usual.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use ola_runtime::{
    AdaptiveConstraints, AdaptiveIntegerDomain, AdaptiveObjectiveWeights,
    AdaptiveOptimizationError, AdaptiveOptimizer, AdaptivePolicy, AdaptiveRequestObservation,
    BeliefPolicy, ComposedPolicy, ExecutionPolicy, MemoryAwareAdmission, NewestFirst,
    OutputBeliefConfig, OutputLengthBelief, PolicyError, QuantileAdmission, SwitchingCost,
    WatermarkAdmission,
};

use super::builtin::{as_f64, finish, run_error, values};
use super::statistics::bootstrap_mean_interval;
use super::{
    Direction, Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor,
    ExperimentError, ExperimentKind, ExperimentOutcome, MetricSpec, ParameterSpec, Parameters,
    TimeDomain, Unit, Variant,
};
use crate::{
    DEFAULT_STARVATION_FACTOR, KvGeometry, ReplayOptimizer, ReplaySettings, SchedulingTrace,
    SimulationConfig, TraceParameters, isolated_end_to_end, run_scheduling_trial, summarize_run,
};

/// `log2` of the watermark headroom; the top value means worst case.
const HEADROOM: AdaptiveIntegerDomain = AdaptiveIntegerDomain {
    name: "log2_headroom",
    min: 4,
    max: 9,
    step: 1,
};
const WORST_CASE_LOG2: i64 = 9;

const WEIGHTS: AdaptiveObjectiveWeights = AdaptiveObjectiveWeights {
    ttft: 1,
    completion: 1,
    throughput: 0,
    starvation: 0,
    deadline: 0,
    fallback: 0,
};

type Factory = fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError>;
type SharedReplay = Arc<Mutex<ReplayOptimizer<Factory>>>;

/// The policy one grid point names; all of them preempt newest-first.
fn headroom_policy(
    theta: &[(&'static str, String)],
) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let log2 = theta
        .iter()
        .find(|(key, _)| *key == HEADROOM.name)
        .and_then(|(_, value)| value.parse::<i64>().ok())
        .filter(|log2| (HEADROOM.min..=HEADROOM.max).contains(log2))
        .ok_or_else(|| PolicyError::InvalidCandidate("missing or invalid log2_headroom".into()))?;
    let policy = ComposedPolicy::default().with_preemption(NewestFirst);
    Ok(Box::new(if log2 == WORST_CASE_LOG2 {
        policy.with_admission(MemoryAwareAdmission)
    } else {
        policy.with_admission(WatermarkAdmission {
            headroom_tokens: 1 << log2,
        })
    }))
}

fn grid_point(log2: i64) -> Vec<(&'static str, String)> {
    vec![(HEADROOM.name, log2.to_string())]
}

fn grid_label(log2: i64) -> String {
    if log2 == WORST_CASE_LOG2 {
        "fixed-worst-case".into()
    } else {
        format!("fixed-h{}", 1_u64 << log2)
    }
}

/// One way of choosing the action, rebuilt fresh for every run.
#[derive(Clone, Copy)]
enum Controller {
    /// The repository default: worst-case reservation, no preemption.
    Baseline,
    /// One grid point for the whole run.
    Fixed(i64),
    /// Online replay over the grid, starting from the worst case.
    Replay { switching_cost: i64 },
    /// Quantile of the remaining-output belief, per request.
    Belief { quantile: f64, window: usize },
}

struct Tuning {
    config: SimulationConfig,
    interval: u64,
    window: usize,
    min_improvement_percent: u64,
}

fn build(
    controller: Controller,
    tuning: &Tuning,
) -> Result<(Box<dyn ExecutionPolicy>, Option<SharedReplay>), ExperimentError> {
    match controller {
        Controller::Baseline => Ok((
            Box::new(ComposedPolicy::default().with_admission(MemoryAwareAdmission)),
            None,
        )),
        Controller::Fixed(log2) => {
            Ok((headroom_policy(&grid_point(log2)).map_err(run_error)?, None))
        }
        Controller::Replay { switching_cost } => {
            let factory: Factory = headroom_policy;
            let optimizer = Arc::new(Mutex::new(
                ReplayOptimizer::new(
                    ReplaySettings {
                        domains: vec![HEADROOM],
                        weights: WEIGHTS,
                        config: tuning.config,
                        window: tuning.window,
                        min_requests: tuning.window.min(8),
                        min_improvement_percent: tuning.min_improvement_percent,
                        switching: SwitchingCost::uniform(&[HEADROOM.name], switching_cost),
                    },
                    factory,
                )
                .map_err(run_error)?,
            ));
            let shared = optimizer.clone();
            let start = grid_point(WORST_CASE_LOG2);
            let policy = AdaptivePolicy::new(headroom_policy(&start).map_err(run_error)?, start)
                .with_fallback(
                    ComposedPolicy::default()
                        .with_admission(MemoryAwareAdmission)
                        .with_preemption(NewestFirst),
                )
                .with_constraints(AdaptiveConstraints {
                    max_active_requests: Some(tuning.config.max_active_requests),
                    prefill_chunk_size: tuning.config.prefill_chunk_size,
                    available_memory_bytes: None,
                })
                .with_optimization_interval(tuning.interval)
                .with_factory(headroom_policy)
                .with_optimizer(
                    move |current: &[(&'static str, String)],
                          observations: &[AdaptiveRequestObservation],
                          constraints: AdaptiveConstraints|
                          -> Result<_, AdaptiveOptimizationError> {
                        shared
                            .lock()
                            .map_err(|_| AdaptiveOptimizationError::NoCandidate)?
                            .propose(current, observations, constraints)
                    },
                );
            Ok((Box::new(policy), Some(optimizer)))
        }
        Controller::Belief { quantile, window } => Ok((
            Box::new(BeliefPolicy::new(
                OutputLengthBelief::new(OutputBeliefConfig {
                    window,
                    ..OutputBeliefConfig::default()
                }),
                move |belief| {
                    Box::new(
                        ComposedPolicy::default()
                            .with_admission(QuantileAdmission::new(belief, quantile))
                            .with_preemption(NewestFirst),
                    )
                },
            )),
            None,
        )),
    }
}

/// Runs one controller on one trace and returns its summary and, for
/// replay, the optimizer's switch and replay counts.
fn evaluate(
    controller: Controller,
    tuning: &Tuning,
    trace: &SchedulingTrace,
    isolated: &[Option<u64>],
) -> Result<(crate::SchedulingSummary, u64, u64), ExperimentError> {
    let (policy, optimizer) = build(controller, tuning)?;
    let run = run_scheduling_trial(trace, policy, &tuning.config).map_err(run_error)?;
    let (switches, replays) = optimizer.map_or((0, 0), |optimizer| {
        optimizer.lock().map_or((0, 0), |optimizer| {
            (optimizer.switches(), optimizer.replays())
        })
    });
    Ok((
        summarize_run(&run, isolated, DEFAULT_STARVATION_FACTOR),
        switches,
        replays,
    ))
}

fn objective(summary: &crate::SchedulingSummary) -> f64 {
    as_f64(
        summary
            .completion
            .p95_micros
            .saturating_add(summary.time_to_first_token.p95_micros),
    )
}

/// Early-stopping requests followed by requests that run to their limits,
/// on a KV pool that binds, with the per-regime optimum measured first.
pub struct RegimeShiftStudy;

impl Experiment for RegimeShiftStudy {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "regime-shift-study",
            name: "Regime shift study",
            description: "KV reservation headroom under an early-stopping regime followed by a \
                          run-to-limit regime: per-regime best fixed action, hindsight static \
                          oracle, online replay with and without switching cost, and \
                          remaining-output belief quantiles",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::integer("seeds", "Trace seeds, one per repetition", 5),
                ParameterSpec::integer("requests", "Requests per regime", 60),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    1_000,
                ),
                ParameterSpec::integer("max_active_requests", "Executor slots", 24),
                ParameterSpec::integer("kv_capacity_tokens", "KV pool size in tokens", 2_048),
                ParameterSpec::integer(
                    "optimization_interval",
                    "Executor iterations between replay proposals",
                    40,
                ),
                ParameterSpec::integer("window", "Completed requests replayed", 32),
                ParameterSpec::integer(
                    "min_improvement_percent",
                    "Replay improvement required to switch (noise margin)",
                    5,
                ),
                ParameterSpec::integer(
                    "switching_cost_micros",
                    "Objective microseconds charged per headroom change",
                    5_000,
                ),
                ParameterSpec::integer(
                    "belief_quantile_percent",
                    "Remaining-output quantile reserved by the belief policy",
                    90,
                ),
                ParameterSpec::integer("belief_window", "Finished requests kept by the belief", 32),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let seeds = (1..=parameters.integer("seeds")?).collect::<Vec<_>>();
        let geometry = KvGeometry::default();
        let tuning = Tuning {
            config: SimulationConfig {
                max_active_requests: parameters.size("max_active_requests")?,
                kv: KvGeometry {
                    capacity_bytes: parameters
                        .integer("kv_capacity_tokens")?
                        .saturating_mul(geometry.bytes_per_token),
                    ..geometry
                },
                ..SimulationConfig::default()
            },
            interval: parameters.integer("optimization_interval")?.max(1),
            window: parameters.size("window")?.max(1),
            min_improvement_percent: parameters.integer("min_improvement_percent")?,
        };
        let switching_cost =
            i64::try_from(parameters.integer("switching_cost_micros")?).unwrap_or(i64::MAX);
        let quantile = parameters.integer("belief_quantile_percent")?.min(100) as f64 / 100.0;
        let belief_window = parameters.size("belief_window")?;
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };

        let mut controllers = vec![("baseline", Controller::Baseline)];
        let fixed_labels = (HEADROOM.min..=HEADROOM.max)
            .map(grid_label)
            .collect::<Vec<_>>();
        for (index, log2) in (HEADROOM.min..=HEADROOM.max).enumerate() {
            controllers.push((fixed_labels[index].as_str(), Controller::Fixed(log2)));
        }
        controllers.push(("replay", Controller::Replay { switching_cost: 0 }));
        controllers.push(("replay-switching", Controller::Replay { switching_cost }));
        controllers.push((
            "belief-quantile",
            Controller::Belief {
                quantile,
                window: belief_window,
            },
        ));

        let mut samples = controllers
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for &seed in &seeds {
            let early = SchedulingTrace::early_stopping(TraceParameters { seed, ..base });
            let limit = SchedulingTrace::run_to_limit(TraceParameters {
                seed: seed ^ 0x9e37_79b9_7f4a_7c15,
                ..base
            });
            let shifted = early.clone().followed_by(
                limit.clone(),
                "early_stopping+run_to_limit",
                base.mean_interarrival_micros,
            );
            for request in &shifted.requests {
                content.push_str(&format!("{request:?};"));
            }
            let isolated = [&early, &limit, &shifted]
                .map(|trace| isolated_end_to_end(trace, &tuning.config).map_err(run_error));
            let [early_isolated, limit_isolated, shifted_isolated] = isolated;
            let (early_isolated, limit_isolated, shifted_isolated) =
                (early_isolated?, limit_isolated?, shifted_isolated?);
            for ((_, controller), variant) in controllers.iter().zip(samples.iter_mut()) {
                let (alone_early, ..) = evaluate(*controller, &tuning, &early, &early_isolated)?;
                let (alone_limit, ..) = evaluate(*controller, &tuning, &limit, &limit_isolated)?;
                let (summary, switches, replays) =
                    evaluate(*controller, &tuning, &shifted, &shifted_isolated)?;
                variant.push(values(&[
                    ("objective", objective(&summary)),
                    ("completion_p95", as_f64(summary.completion.p95_micros)),
                    ("ttft_p95", as_f64(summary.time_to_first_token.p95_micros)),
                    (
                        "throughput",
                        summary.output_tokens_per_second().unwrap_or(0.0),
                    ),
                    ("peak_kv_cache_bytes", as_f64(summary.peak_kv_cache_bytes)),
                    ("preemptions", as_f64(summary.preemptions)),
                    ("failed", as_f64(summary.failed + summary.rejected)),
                    ("unfinished", as_f64(summary.unfinished)),
                    ("switches", as_f64(switches)),
                    ("replays", as_f64(replays)),
                    ("objective_early_alone", objective(&alone_early)),
                    ("objective_limit_alone", objective(&alone_limit)),
                ]));
            }
        }

        // Selections over the fixed grid, all with hindsight on the
        // evaluated traces themselves.
        let mean = |samples: &[BTreeMap<String, f64>], metric: &str| {
            samples.iter().map(|sample| sample[metric]).sum::<f64>() / samples.len().max(1) as f64
        };
        let fixed = 1..=fixed_labels.len();
        let best_fixed = |metric: &str| {
            fixed
                .clone()
                .min_by(|&left, &right| {
                    mean(&samples[left], metric).total_cmp(&mean(&samples[right], metric))
                })
                .unwrap_or(1)
        };
        let early_best = best_fixed("objective_early_alone");
        let limit_best = best_fixed("objective_limit_alone");
        let oracle = best_fixed("objective");
        // The preferred action changes only if, in each regime, its best
        // action beats the other regime's best on paired per-seed
        // differences whose 95% bootstrap interval excludes zero.
        let separated = |metric: &str, better: usize, worse: usize| {
            let differences = samples[worse]
                .iter()
                .zip(&samples[better])
                .map(|(worse, better)| worse[metric] - better[metric])
                .collect::<Vec<_>>();
            bootstrap_mean_interval(&differences).is_some_and(|interval| interval.low > 0.0)
        };
        let change = if early_best == limit_best {
            "does not change"
        } else if separated("objective_early_alone", early_best, limit_best)
            && separated("objective_limit_alone", limit_best, early_best)
        {
            "changes (paired intervals separate)"
        } else {
            "is not distinguishable (paired intervals overlap zero)"
        };
        let oracle_samples = samples[oracle].clone();
        let mut labels = controllers
            .iter()
            .map(|(label, controller)| {
                let policy = build(*controller, &tuning)
                    .map(|(policy, _)| policy.descriptor().to_string())
                    .unwrap_or_default();
                ((*label).to_owned(), policy)
            })
            .collect::<Vec<_>>();
        labels.push((
            "oracle-static".into(),
            format!(
                "hindsight best fixed action on the evaluated traces: {}",
                controllers[oracle].0
            ),
        ));
        samples.push(oracle_samples);

        let latency = |name| metric(name, Unit::Micros, Direction::Lower);
        let count = |name, better| metric(name, Unit::Count, better);
        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "early_stopping+run_to_limit",
            seeds.clone(),
            vec![
                latency("objective"),
                latency("completion_p95"),
                latency("ttft_p95"),
                metric("throughput", Unit::TokensPerSecond, Direction::Higher),
                metric("peak_kv_cache_bytes", Unit::Bytes, Direction::Neither),
                count("preemptions", Direction::Neither),
                count("failed", Direction::Lower),
                count("unfinished", Direction::Lower),
                count("switches", Direction::Neither),
                count("replays", Direction::Lower),
                latency("objective_early_alone"),
                latency("objective_limit_alone"),
            ],
            labels
                .into_iter()
                .zip(samples)
                .map(|((label, policy), samples)| {
                    (
                        Variant {
                            label,
                            policy: Some(policy),
                            kernel: None,
                        },
                        samples,
                    )
                })
                .collect(),
            String::new(),
        )?;
        let estimate = |variant: &str, metric: &str| {
            outcome
                .report
                .summary(variant)
                .map_or(f64::NAN, |summary| summary.estimates[metric].mean / 1_000.0)
        };
        let summary = format!(
            "best fixed alone: early_stopping {} ({:.1} ms), run_to_limit {} ({:.1} ms), \
             preferred action {} · shifted objective: baseline {:.1}, oracle-static ({}) {:.1}, \
             replay {:.1}, replay-switching {:.1}, belief-quantile {:.1} ms",
            controllers[early_best].0,
            estimate(controllers[early_best].0, "objective_early_alone"),
            controllers[limit_best].0,
            estimate(controllers[limit_best].0, "objective_limit_alone"),
            change,
            estimate("baseline", "objective"),
            controllers[oracle].0,
            estimate("oracle-static", "objective"),
            estimate("replay", "objective"),
            estimate("replay-switching", "objective"),
            estimate("belief-quantile", "objective"),
        );
        Ok(ExperimentOutcome { summary, ..outcome }.with_content_digest(&content))
    }
}

fn metric(name: &str, unit: Unit, better: Direction) -> MetricSpec {
    MetricSpec::new(name, unit, TimeDomain::Virtual, better)
}
