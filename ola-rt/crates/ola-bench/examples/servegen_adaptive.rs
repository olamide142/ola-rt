//! Does a real controller capture the gain `oracle-switch` finds on a
//! ServeGen production day?
//!
//! `servegen_regimes` shows that at 4 slots the busiest `language/m-large`
//! clients' night prefers FIFO and their day and evening prefer
//! shortest-total-job-first, and that installing each regime's best at its
//! first request beats `sjf-total` held all day by about 8% at night. This
//! replaces that oracle with the replay optimizer choosing between the two
//! policies online, its host stall charged (`AdaptationCost::RTX_4090_E4`):
//!
//! - `replay-past` and `replay-state` re-plan every 40 iterations against
//!   the recent past or the executor's live state;
//! - `hmm-past` and `hmm-state` check every iteration but replay only when
//!   an HMM over arrival windows (one state per regime by default) changes its most probable regime
//!   with 90% confidence; the HMM is fitted on held-out seeds (`--fit`);
//! - `hmm-state-free` is `hmm-state` with its stall uncharged, separating
//!   the decisions from their cost;
//! - `sizes-past`, `sizes-state` and `sizes-state-free` are the same with
//!   an HMM over windows of request sizes (`RegimeFeatures::Sizes`) instead
//!   of arrival times: the night differs from the day mainly in output
//!   length, which arrivals do not show.
//!
//! - `table-arrivals` and `table-sizes` replay nothing: they install a
//!   regime-to-policy table whenever the HMM's most probable regime reaches
//!   90%. The table is learned on the held-out seeds by decoding their
//!   regimes with the same HMM (not the true labels) and scoring each
//!   policy on each decoded regime's requests.
//!
//! The static policies run first. Their per-regime means pick the oracle's
//! policy for each regime, and their hourly means pick the best single
//! policy, which every controller starts at and is compared with on paired
//! per-seed differences. Scores are p95 completion plus p95 TTFT per
//! regime, over the whole day, and averaged over trace hours (the window an
//! SLO is typically judged over; a pooled p95 hides a gain confined to one
//! regime).
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example servegen_adaptive -- \
//!   sg-*.jsonl --fit fit-*.jsonl
//! ```
//!
//! `ADAPTIVE` (comma-separated) runs only the named replay and trigger
//! controllers. `MAX_ACTIVE` (default 4) overrides the slot count, `REGIME_STATES`
//! (default 3, one per regime) the HMMs' state count.

use std::collections::{BTreeMap, HashSet};

use ola_bench::{
    AdaptationCost, AdaptationMeter, ArrivalWindows, CostModel, DecodeCost, GaussianHmm,
    KvGeometry, OnlineTuning, RegimeFeatures, RegimeFilter, RegimeTrigger, ReplayHorizon,
    SchedulingRun, SchedulingTrace, SimulationConfig, SizeWindows, charged_adaptive_policy_over,
    run_charged_trial, run_scheduling_trial,
};
use ola_runtime::{
    AdaptiveIntegerDomain, AdaptivePolicy, ComposedPolicy, ExecutionPolicy, JobCost,
    MemoryAwareAdmission, PolicyError, ShortestJobFirst, SwitchingCost,
};

/// Regimes by ServeGen window start, in seconds into the day.
const REGIMES: [(&str, u64, u64); 3] = [
    ("night 00-08", 0, 8 * 3600),
    ("day 08-18", 8 * 3600, 18 * 3600),
    ("evening 18-24", 18 * 3600, 24 * 3600),
];

/// The controllers' grid: `policy` indexes this list.
const POLICIES: [&str; 2] = ["fifo", "sjf-total"];

const WINDOW_ARRIVALS: usize = 8;

/// The simulator cost model, overridden by `OLA_COST` when set; the same
/// format `reviewer_eval` reads.
fn cost_model() -> CostModel {
    let Ok(value) = std::env::var("OLA_COST") else {
        return CostModel::default();
    };
    let parts: Vec<u64> = value
        .split(',')
        .map(|x| x.trim().parse().expect("OLA_COST must be integers"))
        .collect();
    let (decode, per_padding) = match parts.len() {
        3 => (None, None),
        6 => (
            Some(DecodeCost {
                per_invocation_micros: parts[4],
                per_row_micros: parts[5],
            }),
            Some(parts[3]),
        ),
        _ => panic!(
            "OLA_COST=fixed,per_invocation,per_dense_token\
             [,per_padding_token,decode_per_invocation,decode_per_row]"
        ),
    };
    CostModel {
        fixed_micros: parts[0],
        per_invocation_micros: parts[1],
        per_dense_token_micros: parts[2],
        per_padding_token_micros: per_padding,
        decode,
    }
}

/// The policy `theta`'s `policy` entry indexes in [`POLICIES`].
fn indexed_policy(
    theta: &[(&'static str, String)],
) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let index = theta
        .iter()
        .find(|(key, _)| *key == "policy")
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .ok_or_else(|| PolicyError::InvalidCandidate("missing policy".into()))?;
    let base = ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    Ok(match POLICIES.get(index) {
        Some(&"fifo") => Box::new(base),
        Some(&"sjf-total") => {
            Box::new(base.with_scheduler(ShortestJobFirst::new(JobCost::PromptPlusOutputLimit)))
        }
        _ => {
            return Err(PolicyError::InvalidCandidate(format!(
                "unknown policy {index}"
            )));
        }
    })
}

fn theta(name: &str) -> Vec<(&'static str, String)> {
    let index = POLICIES.iter().position(|p| *p == name).expect("policy");
    vec![("policy", index.to_string())]
}

/// Installs `schedule[i].1` from virtual time `schedule[i].0` on.
fn oracle_switch(schedule: Vec<(u64, &'static str)>) -> AdaptivePolicy {
    let start = theta(schedule[0].1);
    AdaptivePolicy::new(indexed_policy(&start).expect("policy"), start)
        .with_factory(indexed_policy)
        .with_update(move |iteration, _, theta_now| {
            let now = iteration.at.as_micros();
            if let Some((_, name)) = schedule.iter().rev().find(|(at, _)| *at <= now) {
                *theta_now = theta(name);
            }
        })
}

fn p95(mut values: Vec<u64>) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[(values.len() * 95).div_ceil(100) - 1]
}

/// p95 completion plus p95 TTFT, in seconds, over the requests `keep`
/// selects by trace index.
fn objective(run: &SchedulingRun, keep: impl Fn(usize) -> bool) -> f64 {
    let kept = run.requests.iter().filter(|r| keep(r.trace_index));
    let (mut completion, mut ttft) = (Vec::new(), Vec::new());
    for request in kept {
        if let Some(done) = request.completed_micros {
            completion.push(done - request.arrival_micros);
        }
        if let Some(first) = request.first_token_micros {
            ttft.push(first - request.arrival_micros);
        }
    }
    (p95(completion) + p95(ttft)) as f64 / 1e6
}

/// Where a request falls in the ServeGen day.
#[derive(Clone, Copy)]
struct Label {
    regime: Option<usize>,
    /// Trace hour of the request's ServeGen window.
    hour: Option<u64>,
}

/// Objectives for each regime, the whole day, and the mean over trace
/// hours of each hour's objective: an hour is the window an SLO would
/// typically be judged over, and a pooled p95 hides a gain confined to
/// one regime.
fn scores(run: &SchedulingRun, labels: &[Label]) -> Vec<f64> {
    let mut out = (0..REGIMES.len())
        .map(|regime| objective(run, |index| labels[index].regime == Some(regime)))
        .collect::<Vec<_>>();
    out.push(objective(run, |_| true));
    let hours = labels
        .iter()
        .filter_map(|label| label.hour)
        .collect::<std::collections::BTreeSet<_>>();
    out.push(mean(
        &hours
            .into_iter()
            .map(|hour| objective(run, |index| labels[index].hour == Some(hour)))
            .collect::<Vec<_>>(),
    ));
    out
}

fn bootstrap(diffs: &[f64]) -> (f64, f64) {
    let mut state = 0x5EED_u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as usize
    };
    let n = diffs.len();
    let mut means = (0..4_000)
        .map(|_| (0..n).map(|_| diffs[next() % n]).sum::<f64>() / n as f64)
        .collect::<Vec<_>>();
    means.sort_by(f64::total_cmp);
    (means[100], means[3_899])
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

/// A trace and where each request falls in the day.
fn load(path: &str) -> (SchedulingTrace, Vec<Label>) {
    let text = std::fs::read_to_string(path).expect("trace file");
    let (trace, windows) = SchedulingTrace::from_json_lines("servegen", &text).expect("trace");
    let labels = windows
        .iter()
        .map(|window| Label {
            regime: window.and_then(|window| {
                REGIMES
                    .iter()
                    .position(|(_, start, end)| (*start..*end).contains(&window))
            }),
            hour: window.map(|window| window / 3600),
        })
        .collect();
    (trace, labels)
}

fn arrivals(trace: &SchedulingTrace) -> Vec<u64> {
    trace.requests.iter().map(|r| r.arrival_micros).collect()
}

/// A trigger's feature windows, fed per request.
struct Windows {
    features: RegimeFeatures,
    arrivals: ArrivalWindows,
    sizes: SizeWindows,
}

impl Windows {
    fn new(trigger: &RegimeTrigger) -> Self {
        Self {
            features: trigger.features,
            arrivals: ArrivalWindows::new(trigger.window_arrivals),
            sizes: SizeWindows::new(trigger.window_arrivals),
        }
    }

    fn arrival(&mut self, at: u64) -> Option<Vec<f64>> {
        (self.features == RegimeFeatures::Arrivals)
            .then(|| self.arrivals.push(at))
            .flatten()
    }

    fn size(&mut self, prompt: usize, limit: usize) -> Option<Vec<f64>> {
        (self.features == RegimeFeatures::Sizes)
            .then(|| self.sizes.push(prompt, limit))
            .flatten()
    }
}

/// The trigger model's most probable state for each request of `trace`:
/// that of the window the request completes or falls in, filtered online
/// in arrival order. Requests after the last window take its state.
fn decode(trace: &SchedulingTrace, trigger: &RegimeTrigger) -> Vec<usize> {
    let mut windows = Windows::new(trigger);
    let mut filter = RegimeFilter::new(trigger.model.clone());
    let mut states = Vec::new();
    let mut pending = 0;
    for request in &trace.requests {
        pending += 1;
        let features = windows
            .arrival(request.arrival_micros)
            .or_else(|| windows.size(request.prompt_tokens, request.max_new_tokens));
        if let Some(features) = features {
            filter.update(&features);
            let (state, _) = filter.most_probable().expect("posterior");
            states.extend(std::iter::repeat_n(state, pending));
            pending = 0;
        }
    }
    let last = states.last().copied().unwrap_or(0);
    states.extend(std::iter::repeat_n(last, pending));
    states
}

/// Each trigger state's best policy, learned offline on `fits`: requests
/// are grouped by their decoded state (not the true regime) and every
/// policy is scored per group.
fn learn_table(
    fits: &[SchedulingTrace],
    trigger: &RegimeTrigger,
    config: &SimulationConfig,
) -> Vec<&'static str> {
    let decoded = fits
        .iter()
        .map(|trace| decode(trace, trigger))
        .collect::<Vec<_>>();
    let runs = POLICIES.map(|name| {
        fits.iter()
            .map(|trace| {
                run_scheduling_trial(trace, indexed_policy(&theta(name)).expect("policy"), config)
                    .expect("trial")
            })
            .collect::<Vec<_>>()
    });
    (0..trigger.model.states())
        .map(|state| {
            let score = |policy: usize| {
                mean(
                    &runs[policy]
                        .iter()
                        .zip(&decoded)
                        .filter(|(_, states)| states.contains(&state))
                        .map(|(run, states)| objective(run, |index| states[index] == state))
                        .collect::<Vec<_>>(),
                )
            };
            let best = (0..POLICIES.len())
                .min_by(|a, b| score(*a).total_cmp(&score(*b)))
                .expect("policies");
            eprintln!(
                "{:?} state {state}: {}",
                trigger.features,
                POLICIES
                    .iter()
                    .enumerate()
                    .map(|(i, name)| format!("{name} {:.1}", score(i)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            POLICIES[best]
        })
        .collect()
}

/// Installs `table[state]` whenever the trigger model's most probable
/// state reaches its confidence: a learned regime-to-policy map applied
/// online, with no replay. Starts at `start`.
fn regime_table(
    trigger: RegimeTrigger,
    table: Vec<&'static str>,
    start: &'static str,
) -> AdaptivePolicy {
    let initial = theta(start);
    let mut windows = Windows::new(&trigger);
    let mut filter = RegimeFilter::new(trigger.model.clone());
    let (mut arrived, mut sized) = (HashSet::new(), HashSet::new());
    AdaptivePolicy::new(indexed_policy(&initial).expect("policy"), initial)
        .with_factory(indexed_policy)
        .with_update(move |_, observations, theta_now| {
            let mut arrivals = Vec::new();
            let mut sizes = Vec::new();
            for observation in observations {
                let at = observation.arrived_at.as_micros();
                if arrived.insert(observation.request_id) {
                    arrivals.push(at);
                }
                if let (Some(prompt), Some(limit)) =
                    (observation.prompt_tokens, observation.max_new_tokens)
                    && sized.insert(observation.request_id)
                {
                    sizes.push((at, prompt, limit));
                }
            }
            arrivals.sort_unstable();
            sizes.sort_unstable();
            let mut features = arrivals
                .into_iter()
                .filter_map(|at| windows.arrival(at))
                .collect::<Vec<_>>();
            features.extend(
                sizes
                    .into_iter()
                    .filter_map(|(_, prompt, limit)| windows.size(prompt, limit)),
            );
            for features in features {
                filter.update(&features);
            }
            if let Some((state, probability)) = filter.most_probable()
                && probability >= trigger.confidence
            {
                *theta_now = theta(table[state]);
            }
        })
}

/// A controller to run, with its meter if it has an optimizer.
type Controller = (
    &'static str,
    Box<dyn ExecutionPolicy>,
    Option<AdaptationMeter>,
);

/// One controller's run on one trace: its label, scores, replaying calls,
/// switches (when metered) and charged stall.
type RunRow = (&'static str, Vec<f64>, u64, Option<u64>, u64);

/// Runs every trace's controllers, the traces on parallel threads, and
/// returns each trace's rows in trace order.
fn run_all(
    traces: &[(SchedulingTrace, Vec<Label>)],
    config: &SimulationConfig,
    controllers: impl Fn(&SchedulingTrace, &[Label]) -> Vec<Controller> + Sync,
) -> Vec<Vec<RunRow>> {
    std::thread::scope(|scope| {
        let handles = traces
            .iter()
            .map(|(trace, labels)| {
                let controllers = &controllers;
                scope.spawn(move || {
                    controllers(trace, labels)
                        .into_iter()
                        .map(|(label, policy, meter)| {
                            let charge_meter = meter.clone();
                            let run = run_charged_trial(trace, policy, config, &[], &mut |_, _| {
                                charge_meter
                                    .as_ref()
                                    .map_or(0, AdaptationMeter::take_micros)
                            })
                            .expect("trial");
                            assert!(
                                run.requests.iter().all(|r| r.completed_micros.is_some()),
                                "{label} left work unfinished"
                            );
                            let replaying = meter.as_ref().map_or(0, |meter| meter.counts().1);
                            let switches = meter.as_ref().map(|meter| meter.counts().2);
                            let stall = meter.as_ref().map_or(0, AdaptationMeter::total_micros);
                            (label, scores(&run, labels), replaying, switches, stall)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("seed thread"))
            .collect()
    })
}

/// Per-controller run accounting, summed over seeds.
#[derive(Default)]
struct Tally {
    rows: Vec<Vec<f64>>,
    replaying_calls: u64,
    /// Switches the optimizer installed; `None` for controllers without
    /// one, whose switches are not counted.
    switches: Option<u64>,
    stall_micros: u64,
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let split = args.iter().position(|a| a == "--fit");
    let (eval, fit) = match split {
        Some(at) => (&args[..at], &args[at + 1..]),
        None => panic!("usage: servegen_adaptive TRACE.jsonl... --fit FIT.jsonl..."),
    };
    assert!(
        !eval.is_empty() && !fit.is_empty(),
        "need traces to score and to fit"
    );
    let slots = std::env::var("MAX_ACTIVE").map_or(4, |v| v.parse().expect("MAX_ACTIVE"));
    let cost = cost_model();
    let geometry = KvGeometry::default();
    let config = SimulationConfig {
        max_active_requests: slots,
        prefill_chunk_size: None,
        kv: KvGeometry {
            capacity_bytes: geometry.bytes_per_token.saturating_mul(1 << 20),
            ..geometry
        },
        cost,
        max_iterations: usize::MAX,
        ..SimulationConfig::default()
    };

    let states = std::env::var("REGIME_STATES").map_or(3, |v| v.parse().expect("REGIME_STATES"));
    let fits = fit.iter().map(|path| load(path).0).collect::<Vec<_>>();
    let fit_model = |sequences: Vec<Vec<Vec<f64>>>, features| {
        let model = GaussianHmm::fit(&sequences, states, 50).expect("training windows");
        eprintln!("{features:?} regime model: {model:?}");
        RegimeTrigger {
            model,
            features,
            window_arrivals: WINDOW_ARRIVALS,
            confidence: 0.9,
            refresh_calls: None,
        }
    };
    let arrival_trigger = fit_model(
        fits.iter()
            .map(|trace| ArrivalWindows::features(&arrivals(trace), WINDOW_ARRIVALS))
            .collect(),
        RegimeFeatures::Arrivals,
    );
    let size_trigger = fit_model(
        fits.iter()
            .map(|trace| {
                let sizes = trace
                    .requests
                    .iter()
                    .map(|r| (r.prompt_tokens, r.max_new_tokens))
                    .collect::<Vec<_>>();
                SizeWindows::features(&sizes, WINDOW_ARRIVALS)
            })
            .collect(),
        RegimeFeatures::Sizes,
    );

    let traces = eval.iter().map(|path| load(path)).collect::<Vec<_>>();
    eprintln!("{} traces, {slots} slots", traces.len());
    let grid = vec![AdaptiveIntegerDomain {
        name: "policy",
        min: 0,
        max: POLICIES.len() as i64 - 1,
        step: 1,
    }];
    let (past, state) = (ReplayHorizon::Past, ReplayHorizon::FromState);
    let charged = AdaptationCost::RTX_4090_E4;
    let (by_arrivals, by_sizes) = (Some(&arrival_trigger), Some(&size_trigger));
    let adaptive_controllers = [
        ("replay-past", charged, 40, past, None),
        ("replay-state", charged, 40, state, None),
        ("hmm-past", charged, 1, past, by_arrivals),
        ("hmm-state", charged, 1, state, by_arrivals),
        (
            "hmm-state-free",
            AdaptationCost::default(),
            1,
            state,
            by_arrivals,
        ),
        ("sizes-past", charged, 1, past, by_sizes),
        ("sizes-state", charged, 1, state, by_sizes),
        (
            "sizes-state-free",
            AdaptationCost::default(),
            1,
            state,
            by_sizes,
        ),
    ];
    let only = std::env::var("ADAPTIVE").ok();
    let adaptive_controllers = adaptive_controllers
        .into_iter()
        .filter(|c| {
            only.as_ref()
                .is_none_or(|only| only.split(',').any(|n| n == c.0))
        })
        .collect::<Vec<_>>();
    let tables = [
        ("table-arrivals", &arrival_trigger),
        ("table-sizes", &size_trigger),
    ]
    .map(|(label, trigger)| (label, trigger, learn_table(&fits, trigger, &config)));
    for (label, _, table) in &tables {
        eprintln!("{label}: {table:?}");
    }
    let names = POLICIES
        .into_iter()
        .chain(["oracle-switch"])
        .chain(adaptive_controllers.iter().map(|c| c.0))
        .chain(tables.iter().map(|t| t.0))
        .collect::<Vec<_>>();

    let mut results: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut record = |per_seed: Vec<Vec<RunRow>>| {
        for runs in per_seed {
            for (label, row, replaying, switches, stall) in runs {
                let tally = results.entry(label).or_default();
                tally.rows.push(row);
                tally.replaying_calls += replaying;
                if let Some(switches) = switches {
                    *tally.switches.get_or_insert(0) += switches;
                }
                tally.stall_micros += stall;
            }
        }
    };

    // The static policies first: they fix each regime's best policy, for
    // the oracle, and the best single policy, which every controller
    // starts at and is compared with.
    let statics = run_all(&traces, &config, |_, _| {
        POLICIES
            .into_iter()
            .map(|name| -> Controller {
                (name, indexed_policy(&theta(name)).expect("policy"), None)
            })
            .collect()
    });
    let mean_of = |name: &str, index: usize| {
        mean(
            &statics
                .iter()
                .flatten()
                .filter(|row| row.0 == name)
                .map(|row| row.1[index])
                .collect::<Vec<_>>(),
        )
    };
    let best_at = |index: usize| {
        POLICIES
            .into_iter()
            .min_by(|a, b| mean_of(a, index).total_cmp(&mean_of(b, index)))
            .expect("policies")
    };
    let regime_best = (0..REGIMES.len()).map(best_at).collect::<Vec<_>>();
    // The reference is the best single policy by the hourly score, the one
    // an SLO judged per hour would pick.
    let reference = best_at(REGIMES.len() + 1);
    eprintln!("per-regime best {regime_best:?}, best single policy {reference}");
    record(statics);

    let others = run_all(&traces, &config, |trace, labels| {
        let schedule = (0..REGIMES.len())
            .filter_map(|regime| {
                let first = labels
                    .iter()
                    .position(|label| label.regime == Some(regime))?;
                Some((trace.requests[first].arrival_micros, regime_best[regime]))
            })
            .collect::<Vec<_>>();
        let mut controllers: Vec<Controller> =
            vec![("oracle-switch", Box::new(oracle_switch(schedule)), None)];
        for (label, trigger, table) in &tables {
            let policy = regime_table((*trigger).clone(), table.clone(), reference);
            controllers.push((label, Box::new(policy), None));
        }
        for &(label, adaptation, interval, horizon, trigger) in &adaptive_controllers {
            let (policy, meter) = charged_adaptive_policy_over(
                theta(reference),
                grid.clone(),
                indexed_policy,
                cost,
                slots,
                OnlineTuning {
                    interval,
                    horizon,
                    ..OnlineTuning::default()
                },
                adaptation,
                SwitchingCost::default(),
                trigger.cloned(),
            )
            .expect("adaptive policy");
            controllers.push((label, Box::new(policy), Some(meter)));
        }
        controllers
    });
    record(others);

    let seeds = traces.len() as f64;
    let column = |name: &str, index: usize| -> Vec<f64> {
        results[name].rows.iter().map(|row| row[index]).collect()
    };
    let day = REGIMES.len();
    println!(
        "Mean objective (s), p95 completion + p95 TTFT; Δ vs {reference}, the best single \
         policy by hourly mean, on paired seeds [95% CI]. Per-regime best: {}. Replays, switches and stall \
         are per run.\n",
        regime_best.join(", ")
    );
    let header = REGIMES
        .iter()
        .map(|(name, ..)| *name)
        .chain(["whole day", "hourly mean"])
        .collect::<Vec<_>>()
        .join(" | ");
    println!("| controller | {header} | replays | switches | stall (s) |");
    println!("|---|{}---:|---:|---:|", "---|".repeat(day + 2));
    for name in names {
        let cells = (0..=day + 1)
            .map(|index| {
                let values = column(name, index);
                if name == reference {
                    return format!("{:.1}", mean(&values));
                }
                let diffs = values
                    .iter()
                    .zip(column(reference, index))
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let (lo, hi) = bootstrap(&diffs);
                format!(
                    "{:.1} ({:+.2} [{lo:+.2}, {hi:+.2}])",
                    mean(&values),
                    mean(&diffs)
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let tally = &results[name];
        println!(
            "| {name} | {cells} | {:.1} | {} | {:.2} |",
            tally.replaying_calls as f64 / seeds,
            tally.switches.map_or("–".into(), |switches| format!(
                "{:.1}",
                switches as f64 / seeds
            )),
            tally.stall_micros as f64 / seeds / 1e6,
        );
    }
}
