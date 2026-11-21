//! Does the best fixed token budget change between workload regimes?
//!
//! Runs every point of the `ola serve --adaptive` grid
//! (`TOKEN_BUDGET_DOMAINS`) as a fixed policy on each trace family and load,
//! in virtual time under the cost model `OLA_COST` names (the default model
//! when unset), and writes one CSV row per run. Adapting the budget can only
//! pay between regimes whose best budgets differ; `lab/replay/regimes.py`
//! tests that on paired per-seed differences.
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example budget_regimes -- out.csv
//! ```
//!
//! `SEEDS` (default 20) and `MAX_ACTIVE` (default 4, the `ola serve`
//! default) override the sweep.

use std::io::Write;

use ola_bench::{
    CostModel, DEFAULT_STARVATION_FACTOR, DecodeCost, KvGeometry, SchedulingTrace,
    SimulationConfig, TOKEN_BUDGET_DOMAINS, TraceParameters, run_scheduling_trial, summarize_run,
    token_budget_policy,
};

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

/// Ratio of a one-invocation, eight-row decode iteration under `cost` to the
/// default model's, so nominal interarrival gaps give the same load under
/// any cost model (as in `reviewer_eval`).
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

/// A trace generator taking only its parameters.
type TraceFamily = fn(TraceParameters) -> SchedulingTrace;

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
}

/// Every grid point, in `TOKEN_BUDGET_DOMAINS` order.
fn grid() -> Vec<(i64, i64)> {
    let [tokens, share] = TOKEN_BUDGET_DOMAINS;
    let points = |d: ola_runtime::AdaptiveIntegerDomain| {
        (d.min..=d.max).step_by(d.step as usize).collect::<Vec<_>>()
    };
    points(tokens)
        .into_iter()
        .flat_map(|log2| points(share).into_iter().map(move |s| (log2, s)))
        .collect()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "budget_regimes.csv".into());
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("output file"));
    let cost = cost_model();
    let scale = load_scale(cost);
    let seeds = env_number("SEEDS", 20);
    let max_active = env_number("MAX_ACTIVE", 4) as usize;
    eprintln!("load scale {scale:.3}, {seeds} seeds, {max_active} slots");
    // The replay configuration `serving_adaptive_policy` uses: live slots,
    // no chunking, a KV pool that never binds.
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
        "trace,load,interarrival_us,seed,log2_max_tokens,prefill_share_percent,\
         completed,completion_p50_us,completion_p95_us,ttft_p50_us,ttft_p95_us,\
         makespan_us,objective_us"
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
                for (log2, share) in grid() {
                    let theta = [
                        ("log2_max_tokens", log2.to_string()),
                        ("prefill_share_percent", share.to_string()),
                    ];
                    let policy = token_budget_policy(&theta).expect("grid point");
                    let run = run_scheduling_trial(&trace, policy, &config).expect("trial");
                    let isolated = vec![None; trace.requests.len()];
                    let s = summarize_run(&run, &isolated, DEFAULT_STARVATION_FACTOR);
                    writeln!(
                        file,
                        "{trace_name},{load},{interarrival},{seed},{log2},{share},{},{},{},{},{},{},{}",
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
