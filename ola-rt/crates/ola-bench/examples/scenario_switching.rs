//! Do regime pairs whose best policies differ in isolation still reward a
//! switch when the regimes run back to back?
//!
//! `policy_regimes` finds pairs of regimes, each preferring a different
//! named policy on paired intervals. That is necessary for adaptation to
//! pay, not sufficient: when the first regime's backlog outlives its
//! arrivals, the executor stays in the first regime's condition after the
//! boundary. For each candidate pair this runs both orders back to back
//! (120 requests each) and compares each regime's best policy held for the
//! whole run with `oracle-switch`, which installs the second regime's best
//! when that regime's first request arrives: the most any controller could
//! gain, at no cost. Prints a markdown table.
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example scenario_switching
//! ```
//!
//! `SEEDS` (default 10) overrides the run.

use ola_bench::{
    CostModel, DecodeCost, KvGeometry, SchedulingRun, SchedulingTrace, SimulationConfig,
    TraceParameters, run_scheduling_trial,
};
use ola_runtime::{
    AdaptivePolicy, ComposedPolicy, ExecutionPolicy, JobCost, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, PolicyError, ShortestJobFirst,
};

const PHASE_REQUESTS: usize = 120;

/// A trace generator taking only its parameters.
type TraceFamily = fn(TraceParameters) -> SchedulingTrace;

/// One regime: a trace family at a nominal mean gap.
#[derive(Clone, Copy)]
struct Regime {
    name: &'static str,
    family: TraceFamily,
    nominal_micros: u64,
}

/// A candidate pair from `policy_regimes`, with each regime's best policy.
struct Scenario {
    slots: usize,
    first: Regime,
    first_best: &'static str,
    second: Regime,
    second_best: &'static str,
}

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

/// The named policies of `policy_regimes` that the scenarios use.
fn named_policy(theta: &[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
    let name = theta
        .iter()
        .find(|(key, _)| *key == "policy")
        .map(|(_, value)| value.as_str())
        .ok_or_else(|| PolicyError::InvalidCandidate("missing policy".into()))?;
    let base = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    let budget = || MixedTokenBudgetPolicy::new(512, 256).expect("budget");
    Ok(match name {
        "fifo" => Box::new(base()),
        "sjf-total" => {
            Box::new(base().with_scheduler(ShortestJobFirst::new(JobCost::PromptPlusOutputLimit)))
        }
        "fifo-512/50" => Box::new(base().with_batch(budget())),
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

/// Installs `first` until `boundary_micros`, then `second`.
fn oracle_switch(first: &str, second: &'static str, boundary_micros: u64) -> AdaptivePolicy {
    let start = theta(first);
    AdaptivePolicy::new(named_policy(&start).expect("policy"), start)
        .with_factory(named_policy)
        .with_update(move |iteration, _, theta_now| {
            if iteration.at.as_micros() >= boundary_micros {
                *theta_now = theta(second);
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

/// p95 completion plus p95 TTFT over every request.
fn objective(run: &SchedulingRun) -> f64 {
    let completion = run
        .requests
        .iter()
        .filter_map(|r| Some(r.completed_micros? - r.arrival_micros))
        .collect();
    let ttft = run
        .requests
        .iter()
        .filter_map(|r| Some(r.first_token_micros? - r.arrival_micros))
        .collect();
    (p95(completion) + p95(ttft)) as f64 / 1e6
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
    let cost = cost_model();
    let scale = load_scale(cost);
    let seeds = std::env::var("SEEDS").map_or(10, |v| v.parse().expect("SEEDS"));
    let pdc: TraceFamily = SchedulingTrace::prefill_decode_competition;
    let mixed: TraceFamily = SchedulingTrace::mixed_short_long;
    let early: TraceFamily = SchedulingTrace::early_stopping;
    let regime = |name, family, nominal_micros| Regime {
        name,
        family,
        nominal_micros,
    };
    let scenarios = [
        Scenario {
            slots: 4,
            first: regime("pdc light", pdc, 4_000),
            first_best: "sjf-total",
            second: regime("pdc moderate", pdc, 2_000),
            second_best: "fifo",
        },
        Scenario {
            slots: 4,
            first: regime("pdc light", pdc, 4_000),
            first_best: "sjf-total",
            second: regime("pdc overload", pdc, 1_000),
            second_best: "fifo",
        },
        Scenario {
            slots: 4,
            first: regime("mixed burst", mixed, 50),
            first_best: "sjf-total",
            second: regime("pdc burst", pdc, 50),
            second_best: "fifo",
        },
        Scenario {
            slots: 4,
            first: regime("early light", early, 4_000),
            first_best: "fifo",
            second: regime("pdc light", pdc, 4_000),
            second_best: "sjf-total",
        },
        Scenario {
            slots: 16,
            first: regime("pdc burst", pdc, 50),
            first_best: "sjf-total",
            second: regime("pdc overload", pdc, 1_000),
            second_best: "fifo-512/50",
        },
    ];
    eprintln!("load scale {scale:.3}, {seeds} seeds");
    println!(
        "| slots | order | best static | oracle-switch Δ vs best static (s) [95% CI] | \
         first's best (s) | second's best (s) | oracle-switch (s) |"
    );
    println!("|---:|---|---|---|---:|---:|---:|");
    for scenario in &scenarios {
        let geometry = KvGeometry::default();
        let config = SimulationConfig {
            max_active_requests: scenario.slots,
            prefill_chunk_size: None,
            kv: KvGeometry {
                capacity_bytes: geometry.bytes_per_token.saturating_mul(1 << 20),
                ..geometry
            },
            cost,
            ..SimulationConfig::default()
        };
        let orders = [
            (
                scenario.first,
                scenario.first_best,
                scenario.second,
                scenario.second_best,
            ),
            (
                scenario.second,
                scenario.second_best,
                scenario.first,
                scenario.first_best,
            ),
        ];
        for (a, a_best, b, b_best) in orders {
            let gap = |r: Regime| ((r.nominal_micros as f64) * scale).round().max(1.0) as u64;
            let (mut hold_a, mut hold_b, mut oracle) = (Vec::new(), Vec::new(), Vec::new());
            for seed in 1..=seeds {
                let phase = |r: Regime, offset: u64| {
                    (r.family)(TraceParameters {
                        seed: seed + offset,
                        requests: PHASE_REQUESTS,
                        mean_interarrival_micros: gap(r),
                    })
                };
                let trace = phase(a, 0).followed_by(phase(b, 1_000), "shifted", gap(b));
                let boundary = trace.requests[PHASE_REQUESTS].arrival_micros;
                let run = |policy: Box<dyn ExecutionPolicy>| {
                    objective(&run_scheduling_trial(&trace, policy, &config).expect("trial"))
                };
                hold_a.push(run(named_policy(&theta(a_best)).expect("policy")));
                hold_b.push(run(named_policy(&theta(b_best)).expect("policy")));
                oracle.push(run(Box::new(oracle_switch(a_best, b_best, boundary))));
            }
            let (static_name, best_static) = if mean(&hold_a) <= mean(&hold_b) {
                (a_best, &hold_a)
            } else {
                (b_best, &hold_b)
            };
            let diffs = oracle
                .iter()
                .zip(best_static)
                .map(|(o, s)| o - s)
                .collect::<Vec<_>>();
            let (lo, hi) = bootstrap(&diffs);
            println!(
                "| {} | {} ({a_best}) > {} ({b_best}) | {static_name} | {:+.2} [{lo:+.2}, {hi:+.2}] | \
                 {:.1} | {:.1} | {:.1} |",
                scenario.slots,
                a.name,
                b.name,
                mean(&diffs),
                mean(&hold_a),
                mean(&hold_b),
                mean(&oracle),
            );
        }
    }
}
