//! Does the best execution policy change between workload regimes, for any
//! knob a policy can switch at runtime?
//!
//! `budget_regimes` found the token budget almost inert. This sweeps named
//! policies that differ in scheduling order, head-of-line behaviour and
//! token budget, all with memory-aware admission, over the same trace
//! families and loads, in virtual time under the cost model `OLA_COST`
//! names. It writes one CSV row per run, with a `policy` column;
//! `lab/replay/regimes.py` tests which regime pairs separate (add `--across`
//! to compare workload mixes at the same load).
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example policy_regimes -- out.csv
//! ```
//!
//! `SEEDS` (default 10) and `MAX_ACTIVE` (default 4) override the sweep.

use std::io::Write;

use ola_bench::{
    CostModel, DEFAULT_STARVATION_FACTOR, DecodeCost, KvGeometry, SchedulingTrace,
    SimulationConfig, TraceParameters, run_scheduling_trial, summarize_run,
};
use ola_runtime::{
    ComposedPolicy, ExecutionPolicy, HeadOfLine, JobCost, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, ShortestJobFirst,
};

/// A trace generator taking only its parameters.
type TraceFamily = fn(TraceParameters) -> SchedulingTrace;

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

/// The named policies swept; every one admits with `MemoryAwareAdmission`.
fn policies() -> Vec<(&'static str, Box<dyn ExecutionPolicy>)> {
    let base = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    let budget = |tokens, prefill| MixedTokenBudgetPolicy::new(tokens, prefill).expect("budget");
    let sjf = ShortestJobFirst::new(JobCost::Prompt);
    let sjf_total = ShortestJobFirst::new(JobCost::PromptPlusOutputLimit);
    let backfill = HeadOfLine::Backfill { max_skipped: 4 };
    vec![
        ("fifo", Box::new(base())),
        ("sjf-prompt", Box::new(base().with_scheduler(sjf))),
        ("sjf-total", Box::new(base().with_scheduler(sjf_total))),
        (
            "fifo-backfill",
            Box::new(base().with_head_of_line(backfill)),
        ),
        ("fifo-512/50", Box::new(base().with_batch(budget(512, 256)))),
        (
            "fifo-128/100",
            Box::new(base().with_batch(budget(128, 128))),
        ),
        (
            "sjf-total-512/50",
            Box::new(
                base()
                    .with_scheduler(sjf_total)
                    .with_batch(budget(512, 256)),
            ),
        ),
    ]
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "policy_regimes.csv".into());
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("output file"));
    let cost = cost_model();
    let scale = load_scale(cost);
    let seeds = env_number("SEEDS", 10);
    let max_active = env_number("MAX_ACTIVE", 4) as usize;
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
    writeln!(
        file,
        "trace,load,interarrival_us,seed,policy,completed,completion_p50_us,completion_p95_us,\
         ttft_p50_us,ttft_p95_us,makespan_us,objective_us"
    )
    .unwrap();
    let traces: [(&str, TraceFamily); 4] = [
        ("mixed_short_long", SchedulingTrace::mixed_short_long),
        (
            "prefill_decode_competition",
            SchedulingTrace::prefill_decode_competition,
        ),
        ("priority_classes", SchedulingTrace::priority_classes),
        ("early_stopping", SchedulingTrace::early_stopping),
    ];
    for (trace_name, make) in traces {
        for (load, nominal) in [
            ("light", 4_000_u64),
            ("moderate", 2_000),
            ("overload", 1_000),
            ("burst", 50),
        ] {
            let interarrival = ((nominal as f64) * scale).round().max(1.0) as u64;
            for seed in 1..=seeds {
                let trace = make(TraceParameters {
                    seed,
                    requests: 120,
                    mean_interarrival_micros: interarrival,
                });
                for (policy_name, policy) in policies() {
                    let run = run_scheduling_trial(&trace, policy, &config).expect("trial");
                    let isolated = vec![None; trace.requests.len()];
                    let s = summarize_run(&run, &isolated, DEFAULT_STARVATION_FACTOR);
                    writeln!(
                        file,
                        "{trace_name},{load},{interarrival},{seed},{policy_name},{},{},{},{},{},{},{}",
                        s.completed,
                        s.completion.p50_micros,
                        s.completion.p95_micros,
                        s.time_to_first_token.p50_micros,
                        s.time_to_first_token.p95_micros,
                        s.makespan_micros,
                        s.completion.p95_micros + s.time_to_first_token.p95_micros,
                    )
                    .unwrap();
                }
            }
            eprintln!("{trace_name} {load} done");
        }
    }
    eprintln!("wrote {out}");
}
