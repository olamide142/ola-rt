//! Does switching policy pay across the regimes of a production day?
//!
//! Reads ServeGen day traces written by `lab/replay/servegen.py` (one file
//! per seed) and replays each under every named policy in virtual time, at
//! `MAX_ACTIVE` slots (default 4) under the cost model `OLA_COST` names.
//! Requests are scored by the regime of the ServeGen window they came from:
//! for `language/m-large`'s busiest clients a day splits into a decode-heavy
//! night (00–08 h), a busy day of short requests (08–18 h), and a
//! prefill-heavy evening (18–24 h). The objective is p95 completion plus p95
//! TTFT, per regime and over the day.
//!
//! After the fixed policies, `oracle-switch` installs each regime's best
//! policy (by mean objective over the seeds) when that regime's first
//! request arrives: the most a regime-switching controller could gain, at
//! no cost. It is compared with the best single policy on paired per-seed
//! differences.
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example servegen_regimes -- sg-*.jsonl
//! ```

use std::collections::BTreeMap;

use ola_bench::{
    CostModel, DecodeCost, KvGeometry, SchedulingRun, SchedulingTrace, SimulationConfig,
    run_scheduling_trial,
};
use ola_runtime::{
    AdaptivePolicy, ComposedPolicy, ExecutionPolicy, HeadOfLine, JobCost, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, PolicyError, ShortestJobFirst,
};

/// Regimes by ServeGen window start, in seconds into the day.
const REGIMES: [(&str, u64, u64); 3] = [
    ("night 00-08", 0, 8 * 3600),
    ("day 08-18", 8 * 3600, 18 * 3600),
    ("evening 18-24", 18 * 3600, 24 * 3600),
];

const POLICIES: [&str; 8] = [
    "fifo",
    "sjf-prompt",
    "sjf-total",
    "fifo-backfill",
    "fifo-512/50",
    "fifo-128/100",
    "fifo-1024/25",
    "sjf-total-512/50",
];

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

/// The policy `theta`'s `policy` entry names; every one admits with
/// `MemoryAwareAdmission`.
fn named_policy(theta: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let name = theta
        .iter()
        .find(|(key, _)| *key == "policy")
        .map(|(_, value)| value.as_str())
        .ok_or_else(|| PolicyError::InvalidCandidate("missing policy".into()))?;
    let base = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    let budget = |tokens, prefill| MixedTokenBudgetPolicy::new(tokens, prefill).expect("budget");
    let sjf = ShortestJobFirst::new(JobCost::Prompt);
    let sjf_total = ShortestJobFirst::new(JobCost::PromptPlusOutputLimit);
    Ok(match name {
        "fifo" => Box::new(base()),
        "sjf-prompt" => Box::new(base().with_scheduler(sjf)),
        "sjf-total" => Box::new(base().with_scheduler(sjf_total)),
        "fifo-backfill" => {
            Box::new(base().with_head_of_line(HeadOfLine::Backfill { max_skipped: 4 }))
        }
        "fifo-512/50" => Box::new(base().with_batch(budget(512, 256))),
        "fifo-128/100" => Box::new(base().with_batch(budget(128, 128))),
        "fifo-1024/25" => Box::new(base().with_batch(budget(1024, 256))),
        "sjf-total-512/50" => Box::new(
            base()
                .with_scheduler(sjf_total)
                .with_batch(budget(512, 256)),
        ),
        other => {
            return Err(PolicyError::InvalidCandidate(format!(
                "unknown policy {other}"
            )));
        }
    })
}

fn theta(name: &str) -> Vec<(&'static str, String)> {
    vec![("policy", name.to_owned())]
}

/// Installs `schedule[i].1` from virtual time `schedule[i].0` on.
fn oracle_switch(schedule: Vec<(u64, &'static str)>) -> AdaptivePolicy {
    let start = theta(schedule[0].1);
    AdaptivePolicy::new(named_policy(&start).expect("policy"), start)
        .with_factory(named_policy)
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

/// Objectives for each regime, then the whole day.
fn scores(run: &SchedulingRun, regime_of: &[Option<usize>]) -> Vec<f64> {
    let mut out = (0..REGIMES.len())
        .map(|regime| objective(run, |index| regime_of[index] == Some(regime)))
        .collect::<Vec<_>>();
    out.push(objective(run, |_| true));
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

/// Per-seed objectives of `name` in column `index` (a regime, or the day).
fn column(results: &BTreeMap<String, Vec<Vec<f64>>>, name: &str, index: usize) -> Vec<f64> {
    results[name].iter().map(|row| row[index]).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn main() {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    assert!(!paths.is_empty(), "usage: servegen_regimes TRACE.jsonl...");
    let slots = std::env::var("MAX_ACTIVE").map_or(4, |v| v.parse().expect("MAX_ACTIVE"));
    let geometry = KvGeometry::default();
    let config = SimulationConfig {
        max_active_requests: slots,
        prefill_chunk_size: None,
        kv: KvGeometry {
            capacity_bytes: geometry.bytes_per_token.saturating_mul(1 << 20),
            ..geometry
        },
        cost: cost_model(),
        max_iterations: usize::MAX,
        ..SimulationConfig::default()
    };
    let traces = paths
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path).expect("trace file");
            let (trace, windows) =
                SchedulingTrace::from_json_lines("servegen", &text).expect("trace");
            let regime_of = windows
                .iter()
                .map(|window| {
                    let window = (*window)?;
                    REGIMES
                        .iter()
                        .position(|(_, start, end)| (*start..*end).contains(&window))
                })
                .collect::<Vec<_>>();
            (trace, regime_of)
        })
        .collect::<Vec<_>>();
    let counts = (0..REGIMES.len())
        .map(|regime| {
            mean(
                &traces
                    .iter()
                    .map(|(_, of)| of.iter().filter(|r| **r == Some(regime)).count() as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    eprintln!(
        "{} traces, {slots} slots, mean requests per regime {counts:?}",
        traces.len()
    );

    // name -> per-seed [regime objectives..., day]
    let mut results: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    for name in POLICIES {
        for (trace, regime_of) in &traces {
            let run =
                run_scheduling_trial(trace, named_policy(&theta(name)).expect("policy"), &config)
                    .expect("trial");
            assert!(
                run.requests.iter().all(|r| r.completed_micros.is_some()),
                "{name} left work unfinished"
            );
            results
                .entry(name.into())
                .or_default()
                .push(scores(&run, regime_of));
        }
        eprintln!("{name} done");
    }
    let best_for = |results: &BTreeMap<String, Vec<Vec<f64>>>, index: usize| {
        POLICIES
            .into_iter()
            .min_by(|a, b| {
                mean(&column(results, a, index)).total_cmp(&mean(&column(results, b, index)))
            })
            .expect("policies")
    };
    let per_regime_best = (0..REGIMES.len())
        .map(|index| best_for(&results, index))
        .collect::<Vec<_>>();

    let mut oracle_rows = Vec::new();
    for (trace, regime_of) in &traces {
        let schedule = (0..REGIMES.len())
            .filter_map(|regime| {
                let first = regime_of.iter().position(|r| *r == Some(regime))?;
                Some((
                    trace.requests[first].arrival_micros,
                    per_regime_best[regime],
                ))
            })
            .collect::<Vec<_>>();
        let run = run_scheduling_trial(trace, oracle_switch(schedule), &config).expect("trial");
        oracle_rows.push(scores(&run, regime_of));
    }
    results.insert("oracle-switch".into(), oracle_rows);

    println!("Mean objective (s), p95 completion + p95 TTFT; best per column in bold.\n");
    let header = REGIMES
        .iter()
        .map(|(name, ..)| *name)
        .collect::<Vec<_>>()
        .join(" | ");
    println!("| policy | {header} | whole day |");
    println!("|---|{}---:|", "---:|".repeat(REGIMES.len()));
    let day = REGIMES.len();
    let best_day = best_for(&results, day);
    for name in POLICIES.into_iter().chain(["oracle-switch"]) {
        let cells = (0..=day)
            .map(|index| {
                let value = format!("{:.1}", mean(&column(&results, name, index)));
                let best = if index == day {
                    best_day
                } else {
                    per_regime_best[index]
                };
                if name == best {
                    format!("**{value}**")
                } else {
                    value
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        println!("| {name} | {cells} |");
    }
    let diffs = column(&results, "oracle-switch", day)
        .iter()
        .zip(column(&results, best_day, day))
        .map(|(o, s)| o - s)
        .collect::<Vec<_>>();
    let (lo, hi) = bootstrap(&diffs);
    println!(
        "\nPer-regime best: {}. Best single policy for the day: {best_day}.",
        per_regime_best.join(", ")
    );
    println!(
        "oracle-switch − {best_day}, whole day: {:+.2} s [{lo:+.2}, {hi:+.2}] over {} seeds.",
        mean(&diffs),
        diffs.len()
    );
    for (regime, (name, ..)) in REGIMES.iter().enumerate() {
        let diffs = column(&results, "oracle-switch", regime)
            .iter()
            .zip(column(&results, best_day, regime))
            .map(|(o, s)| o - s)
            .collect::<Vec<_>>();
        let (lo, hi) = bootstrap(&diffs);
        println!("  {name}: {:+.2} s [{lo:+.2}, {hi:+.2}]", mean(&diffs));
    }
}
