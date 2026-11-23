//! Does adapting the token budget across a regime change pay once the
//! optimizer's host stall is charged?
//!
//! `budget_regimes` finds one regime pair whose best fixed budgets differ
//! on paired intervals under the RTX 4090 cost model at 16 slots:
//! `prefill_decode_competition` prefers 512 tokens with half for prompts
//! under sustained overload, and an unbounded budget in a burst. This runs
//! the two phases back to back, in both orders, and compares:
//!
//! - `fixed-1024/100` and `fixed-512/50`, each phase's best for the whole run;
//! - `oracle-switch`, which installs each phase's best when that phase's
//!   first request arrives: the most any controller could gain, at no cost;
//! - `adaptive-free`, `ola serve --adaptive`'s policy with its optimizer
//!   stall uncharged, as every earlier virtual-time study ran it;
//! - `adaptive-charged`, the same with the stall measured on the RTX 4090
//!   host (`AdaptationCost::RTX_4090_E4`) added to the clock;
//! - `adaptive-charged-i10` and `-i160`, the charged policy re-planning every
//!   10 or 160 iterations instead of 40;
//! - `state-free`, `state-charged` and `state-charged-i160`, the same policy
//!   replaying from the executor's live state (`ReplayHorizon::FromState`)
//!   instead of the recent past;
//! - `past-hmm-charged` and `state-hmm-charged`, which check every iteration
//!   but replay only when a two-regime HMM over arrival windows, fitted on
//!   held-out seeds, changes its most probable regime with 90% confidence.
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example regime_switching -- out.csv
//! ```
//!
//! `SEEDS` (default 10) and `MAX_ACTIVE` (default 16) override the run.

use std::io::Write;

use ola_bench::{
    AdaptationCost, AdaptationMeter, ArrivalWindows, CostModel, DecodeCost, GaussianHmm,
    KvGeometry, OnlineTuning, RegimeTrigger, ReplayHorizon, SchedulingRun, SchedulingTrace,
    SimulationConfig, TraceParameters, charged_adaptive_policy, largest_token_budget,
    run_charged_trial, token_budget_policy,
};
use ola_runtime::{AdaptivePolicy, ExecutionPolicy, SwitchingCost};

const PHASE_REQUESTS: usize = 120;
const OVERLOAD_NOMINAL_MICROS: u64 = 1_000;
const BURST_NOMINAL_MICROS: u64 = 50;

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

/// As in `budget_regimes`: nominal gaps scale with the cost model.
fn load_scale(cost: CostModel) -> f64 {
    let reference = |c: CostModel| {
        let invocation = match c.decode {
            Some(decode) => decode.per_invocation_micros + 8 * decode.per_row_micros,
            None => c.per_invocation_micros + 8 * c.per_dense_token_micros,
        };
        (c.fixed_micros + invocation) as f64
    };
    reference(cost) / reference(CostModel::default())
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
}

fn budget(log2: u32, share: u32) -> Vec<(&'static str, String)> {
    vec![
        ("log2_max_tokens", log2.to_string()),
        ("prefill_share_percent", share.to_string()),
    ]
}

/// Installs `first` until `boundary_micros`, then `second`.
fn oracle_switch(
    first: Vec<(&'static str, String)>,
    second: Vec<(&'static str, String)>,
    boundary_micros: u64,
) -> AdaptivePolicy {
    AdaptivePolicy::new(token_budget_policy(&first).expect("budget"), first)
        .with_factory(token_budget_policy)
        .with_update(move |iteration, _, theta| {
            if iteration.at.as_micros() >= boundary_micros {
                *theta = second.clone();
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

/// p95 completion plus p95 TTFT over the requests `keep` selects.
fn objective(run: &SchedulingRun, keep: impl Fn(usize) -> bool) -> u64 {
    let requests = run.requests.iter().filter(|r| keep(r.trace_index));
    let (mut completion, mut ttft) = (Vec::new(), Vec::new());
    for request in requests {
        if let Some(done) = request.completed_micros {
            completion.push(done - request.arrival_micros);
        }
        if let Some(first) = request.first_token_micros {
            ttft.push(first - request.arrival_micros);
        }
    }
    p95(completion) + p95(ttft)
}

/// Arrival gaps per regime-filter window.
const WINDOW_ARRIVALS: usize = 8;

/// The two phases back to back: `order` is `overload>burst` or
/// `burst>overload`; phase 1 holds the first `PHASE_REQUESTS` requests.
fn shifted(order: &'static str, seed: u64, scale: f64) -> SchedulingTrace {
    let gap = |nominal: u64| ((nominal as f64) * scale).round().max(1.0) as u64;
    let phase = |nominal: u64, offset: u64| {
        SchedulingTrace::prefill_decode_competition(TraceParameters {
            seed: seed + offset,
            requests: PHASE_REQUESTS,
            mean_interarrival_micros: gap(nominal),
        })
    };
    let overload = phase(OVERLOAD_NOMINAL_MICROS, 0);
    let burst = phase(BURST_NOMINAL_MICROS, 1_000);
    if order == "overload>burst" {
        overload.followed_by(burst, order, gap(BURST_NOMINAL_MICROS))
    } else {
        burst.followed_by(overload, order, gap(OVERLOAD_NOMINAL_MICROS))
    }
}

/// A two-regime arrival model fitted on held-out seeds (101–130) of both
/// orders, so no evaluated trace is in its training data.
fn fit_regimes(scale: f64) -> GaussianHmm {
    let sequences = ["overload>burst", "burst>overload"]
        .into_iter()
        .flat_map(|order| (101..=130).map(move |seed| shifted(order, seed, scale)))
        .map(|trace| {
            let arrivals = trace
                .requests
                .iter()
                .map(|request| request.arrival_micros)
                .collect::<Vec<_>>();
            ArrivalWindows::features(&arrivals, WINDOW_ARRIVALS)
        })
        .collect::<Vec<_>>();
    GaussianHmm::fit(&sequences, 2, 50).expect("training windows")
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "regime_switching.csv".into());
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("output file"));
    let cost = cost_model();
    let scale = load_scale(cost);
    let seeds = env_number("SEEDS", 10);
    let max_active = env_number("MAX_ACTIVE", 16) as usize;
    eprintln!("load scale {scale:.3}, {seeds} seeds, {max_active} slots");
    let geometry = KvGeometry::default();
    let config = SimulationConfig {
        max_active_requests: max_active,
        prefill_chunk_size: None,
        kv: KvGeometry {
            capacity_bytes: geometry.bytes_per_token.saturating_mul(1 << 20),
            ..geometry
        },
        cost,
        ..SimulationConfig::default()
    };
    let regimes = fit_regimes(scale);
    eprintln!("regime model: {regimes:?}");
    let trigger = RegimeTrigger {
        model: regimes,
        window_arrivals: WINDOW_ARRIVALS,
        confidence: 0.9,
        refresh_calls: None,
    };
    let overload_budget = budget(9, 50);
    let burst_budget = largest_token_budget();
    writeln!(
        file,
        "order,seed,controller,completed,objective_us,phase1_objective_us,phase2_objective_us,\
         makespan_us,calls,replaying_calls,switches,stall_us"
    )
    .unwrap();
    for order in ["overload>burst", "burst>overload"] {
        for seed in 1..=seeds {
            let trace = shifted(order, seed, scale);
            let (first_best, second_best) = if order == "overload>burst" {
                (overload_budget.clone(), burst_budget.clone())
            } else {
                (burst_budget.clone(), overload_budget.clone())
            };
            let boundary = trace.requests[PHASE_REQUESTS].arrival_micros;
            let tuning = |interval: u64, horizon: ReplayHorizon| OnlineTuning {
                interval,
                horizon,
                ..OnlineTuning::default()
            };
            let adaptive = |adaptation: AdaptationCost,
                            interval: u64,
                            horizon: ReplayHorizon,
                            trigger: Option<RegimeTrigger>| {
                charged_adaptive_policy(
                    cost,
                    max_active,
                    tuning(interval, horizon),
                    adaptation,
                    SwitchingCost::default(),
                    trigger,
                )
                .expect("adaptive policy")
            };
            let mut controllers: Vec<(String, Box<dyn ExecutionPolicy>, Option<AdaptationMeter>)> = vec![
                (
                    "fixed-1024/100".into(),
                    token_budget_policy(&largest_token_budget()).expect("budget"),
                    None,
                ),
                (
                    "fixed-512/50".into(),
                    token_budget_policy(&overload_budget).expect("budget"),
                    None,
                ),
                (
                    "oracle-switch".into(),
                    Box::new(oracle_switch(first_best, second_best, boundary)),
                    None,
                ),
            ];
            let (past, state) = (ReplayHorizon::Past, ReplayHorizon::FromState);
            let charged = AdaptationCost::RTX_4090_E4;
            let hmm = Some(trigger.clone());
            for (label, adaptation, interval, horizon, trigger) in [
                ("adaptive-free", AdaptationCost::default(), 40, past, None),
                ("adaptive-charged", charged, 40, past, None),
                ("adaptive-charged-i10", charged, 10, past, None),
                ("adaptive-charged-i160", charged, 160, past, None),
                ("state-free", AdaptationCost::default(), 40, state, None),
                ("state-charged", charged, 40, state, None),
                ("state-charged-i160", charged, 160, state, None),
                ("past-hmm-charged", charged, 1, past, hmm.clone()),
                ("state-hmm-charged", charged, 1, state, hmm.clone()),
            ] {
                let (policy, meter) = adaptive(adaptation, interval, horizon, trigger);
                controllers.push((label.into(), Box::new(policy), Some(meter)));
            }
            for (label, policy, meter) in controllers {
                let charge_meter = meter.clone();
                let run = run_charged_trial(&trace, policy, &config, &[], &mut |_, _| {
                    charge_meter
                        .as_ref()
                        .map_or(0, AdaptationMeter::take_micros)
                })
                .expect("trial");
                let (calls, replaying, switches) =
                    meter.as_ref().map_or((0, 0, 0), AdaptationMeter::counts);
                writeln!(
                    file,
                    "{order},{seed},{label},{},{},{},{},{},{calls},{replaying},{switches},{}",
                    run.requests
                        .iter()
                        .filter(|r| r.completed_micros.is_some())
                        .count(),
                    objective(&run, |_| true),
                    objective(&run, |i| i < PHASE_REQUESTS),
                    objective(&run, |i| i >= PHASE_REQUESTS),
                    run.makespan_micros,
                    meter.as_ref().map_or(0, AdaptationMeter::total_micros),
                )
                .unwrap();
            }
        }
        eprintln!("{order} done");
    }
    eprintln!("wrote {out}");
}
