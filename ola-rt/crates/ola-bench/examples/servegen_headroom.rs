//! How much could switching between fixed policies gain on a ServeGen
//! workload, at most?
//!
//! A screen for where adaptation has headroom, before building a controller
//! for it. Reads ServeGen traces written by `lab/replay/servegen.py` (one
//! file per seed), replays each under every named policy of
//! `servegen_regimes` in virtual time, and scores every trace hour (p95
//! completion plus p95 TTFT over the requests of that hour's ServeGen
//! windows). Then, per seed:
//!
//! - the best single policy is the one with the lowest hourly mean over the
//!   seeds;
//! - the hindsight bound takes, for each hour, the best policy's objective
//!   in that hour.
//!
//! The bound ignores the cost of switching and the backlog one hour leaves
//! the next, so it is an upper bound on what any controller over these
//! policies could gain; a small one rules a workload out. Prints one
//! markdown row, labelled by `LABEL`, so a sweep's rows form a table:
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 LABEL="m-large 4 clients" \
//!   cargo run --release -p ola-bench --example servegen_headroom -- sg-*.jsonl
//! ```
//!
//! `MAX_ACTIVE` (default 4) overrides the slot count; `HEADER=1` prints the
//! table header first.

use std::collections::BTreeSet;

use ola_bench::{
    CostModel, DecodeCost, KvGeometry, SchedulingRun, SchedulingTrace, SimulationConfig,
    run_scheduling_trial,
};
use ola_runtime::{
    ComposedPolicy, ExecutionPolicy, HeadOfLine, JobCost, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, PolicyError, ShortestJobFirst,
};

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

fn p95(mut values: Vec<u64>) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[(values.len() * 95).div_ceil(100) - 1]
}

/// p95 completion plus p95 TTFT, in seconds, of each trace hour's
/// requests, by hour.
fn hourly(run: &SchedulingRun, hour_of: &[Option<u64>]) -> Vec<f64> {
    let hours = hour_of.iter().flatten().copied().collect::<BTreeSet<_>>();
    hours
        .into_iter()
        .map(|hour| {
            let (mut completion, mut ttft) = (Vec::new(), Vec::new());
            for request in run
                .requests
                .iter()
                .filter(|r| hour_of[r.trace_index] == Some(hour))
            {
                if let Some(done) = request.completed_micros {
                    completion.push(done - request.arrival_micros);
                }
                if let Some(first) = request.first_token_micros {
                    ttft.push(first - request.arrival_micros);
                }
            }
            (p95(completion) + p95(ttft)) as f64 / 1e6
        })
        .collect()
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

fn main() {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    assert!(!paths.is_empty(), "usage: servegen_headroom TRACE.jsonl...");
    let label = std::env::var("LABEL").unwrap_or_else(|_| "servegen".into());
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
            let hour_of = windows
                .iter()
                .map(|window| window.map(|w| w / 3600))
                .collect::<Vec<_>>();
            (trace, hour_of)
        })
        .collect::<Vec<_>>();
    let requests = mean(
        &traces
            .iter()
            .map(|(trace, _)| trace.requests.len() as f64)
            .collect::<Vec<_>>(),
    );

    // [seed][policy][hour], seeds on parallel threads.
    let scores = std::thread::scope(|scope| {
        let handles = traces
            .iter()
            .map(|(trace, hour_of)| {
                let config = &config;
                scope.spawn(move || {
                    POLICIES
                        .iter()
                        .map(|name| {
                            let policy = named_policy(&theta(name)).expect("policy");
                            let run = run_scheduling_trial(trace, policy, config).expect("trial");
                            assert!(
                                run.requests.iter().all(|r| r.completed_micros.is_some()),
                                "{name} left work unfinished"
                            );
                            hourly(&run, hour_of)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("seed thread"))
            .collect::<Vec<_>>()
    });

    let hourly_mean = |seed: usize, policy: usize| mean(&scores[seed][policy]);
    let best = (0..POLICIES.len())
        .min_by(|&a, &b| {
            let total = |p| (0..scores.len()).map(|s| hourly_mean(s, p)).sum::<f64>();
            total(a).total_cmp(&total(b))
        })
        .expect("policies");
    let mut wins = vec![0_usize; POLICIES.len()];
    let bound = (0..scores.len())
        .map(|seed| {
            let hours = scores[seed][best].len();
            mean(
                &(0..hours)
                    .map(|hour| {
                        let (winner, value) = (0..POLICIES.len())
                            .map(|p| (p, scores[seed][p][hour]))
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .expect("policies");
                        // Count an hour as won only when it beats the best
                        // single policy by more than 1%.
                        if value < scores[seed][best][hour] * 0.99 {
                            wins[winner] += 1;
                        }
                        value
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let single = (0..scores.len())
        .map(|seed| hourly_mean(seed, best))
        .collect::<Vec<_>>();
    let gains = single
        .iter()
        .zip(&bound)
        .map(|(s, b)| s - b)
        .collect::<Vec<_>>();
    let (lo, hi) = bootstrap(&gains);
    let won = POLICIES
        .iter()
        .zip(&wins)
        .filter(|(_, n)| **n > 0)
        .map(|(name, n)| format!("{name} {n}"))
        .collect::<Vec<_>>()
        .join(", ");
    if std::env::var("HEADER").is_ok() {
        println!(
            "| workload | slots | requests | best single | hourly mean (s) | hindsight bound (s) | \
             headroom (s) [95% CI] | headroom % | hours won by others (>1%) |"
        );
        println!("|---|---:|---:|---|---:|---:|---|---:|---|");
    }
    println!(
        "| {label} | {slots} | {requests:.0} | {} | {:.1} | {:.1} | {:.2} [{lo:.2}, {hi:.2}] | {:.1} | {} |",
        POLICIES[best],
        mean(&single),
        mean(&bound),
        mean(&gains),
        100.0 * mean(&gains) / mean(&single),
        if won.is_empty() { "none".into() } else { won },
    );
}
