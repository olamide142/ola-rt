//! Evaluation runs requested in review.
//!
//! ```text
//! cargo run --release -p ola-bench --example reviewer_eval -- schedulers <out.csv>
//! cargo run --release -p ola-bench --example reviewer_eval -- bucketing  <out.csv>
//! cargo run --release -p ola-bench --example reviewer_eval -- timing     <out.csv>
//! cargo run --release -p ola-bench --example reviewer_eval -- prefill_sets <out.csv>
//! ```
//!
//! * `schedulers`: FIFO, priority, EDF, SJF, backfill on the
//!   `priority_classes` and `deadlines` traces at three loads, 30 seeds,
//!   one row per (trace, load, seed, policy) with per-class metrics.
//! * `bucketing`: unbucketed, power-of-two, and cost-aware grouping over
//!   many random prompt-length sets from several distributions.
//! * `timing`: host wall-clock duration of `plan`, `validate`, and
//!   `observe` per call, with the queue and live sizes at that call.
//! * `prefill_sets`: one seeded prompt-length set per distribution and set
//!   size (4, 8, 16) from the `bucketing` distributions, for running
//!   `llama-prefill-strategies` on a real model.
//!
//! Set `OLA_COST=fixed,per_invocation,per_dense_token` (microseconds) to
//! replace the simulator's default cost model with fitted coefficients,
//! e.g. from `ola experiment run cost-calibration --model <dir>`. Three more
//! values, `per_padding_token,decode_per_invocation,decode_per_row`, price
//! padding and decode separately, as in `replay_trace`. Arrival
//! rates and deadline estimates are then scaled by the ratio of a reference
//! decode iteration's cost (one invocation, eight rows) under the fitted and
//! default models, so "light", "moderate", and "overload" keep their meaning.

#![allow(missing_docs)]

use std::{
    io::Write,
    sync::{Arc, Mutex},
    time::Instant,
};

use ola_bench::{
    CostModel, DEFAULT_STARVATION_FACTOR, DecodeCost, KvGeometry, OnlineTuning, SchedulingTrace,
    SimulationConfig, TraceParameters, compare_schedulers, run_observed_trial,
    serving_adaptive_policy,
};
use ola_runtime::{
    ComposedPolicy, CostAwarePrefillBatcher, EarliestDeadlineFirst, ExecutionPlan, ExecutionPolicy,
    ExecutorIteration, HeadOfLine, JobCost, MemoryAwareAdmission, NewestFirst, PlanningSnapshot,
    PolicyDecision, PolicyDescriptor, PolicyError, PowerOfTwoPrefillBatcher, PrefillBatcher,
    PriorityScheduler, ShortestJobFirst, UnbucketedPrefillBatcher, WatermarkAdmission,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| format!("{mode}.csv"));
    let mut file = std::fs::File::create(&out).expect("output file");
    match mode {
        "schedulers" => schedulers(&mut file),
        "bucketing" => bucketing(&mut file),
        "timing" => timing(&mut file),
        "prefill_sets" => prefill_sets(&mut file),
        _ => panic!("mode must be schedulers, bucketing, timing, or prefill_sets"),
    }
    eprintln!("wrote {out}");
}

/// The simulator cost model, overridden by `OLA_COST` when set.
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

/// Ratio of a reference decode iteration's cost (one invocation, eight
/// one-token rows) under the active cost model to the default one. Exactly
/// 1.0 without `OLA_COST`.
fn load_scale() -> f64 {
    let reference = |c: CostModel| {
        let invocation = match c.decode {
            Some(decode) => decode.per_invocation_micros + 8 * decode.per_row_micros,
            None => c.per_invocation_micros + 8 * c.per_dense_token_micros,
        };
        (c.fixed_micros + invocation) as f64
    };
    reference(cost_model()) / reference(CostModel::default())
}

fn scaled(micros: u64) -> u64 {
    ((micros as f64) * load_scale()).round().max(1.0) as u64
}

// ---------------------------------------------------------------------------

fn scheduler_policies() -> Vec<(&'static str, Box<dyn ExecutionPolicy>)> {
    let fifo = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    vec![
        ("fifo", Box::new(fifo())),
        (
            "priority",
            Box::new(fifo().with_scheduler(PriorityScheduler::new())),
        ),
        (
            "edf",
            Box::new(fifo().with_scheduler(EarliestDeadlineFirst)),
        ),
        (
            "sjf",
            Box::new(fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt))),
        ),
        (
            "backfill",
            Box::new(fifo().with_head_of_line(HeadOfLine::Backfill { max_skipped: 4 })),
        ),
    ]
}

fn schedulers(out: &mut impl Write) {
    writeln!(
        out,
        "trace,interarrival_us,seed,policy,requests,completed,completion_p50_us,completion_p95_us,\
         ttft_p95_us,throughput_tps,with_deadline,deadline_missed,starved,\
         hi_requests,hi_completion_p50_us,hi_completion_p95_us,lo_requests,lo_completion_p50_us,lo_completion_p95_us,load"
    )
    .unwrap();
    eprintln!("load scale {:.3}", load_scale());
    let config = SimulationConfig {
        cost: cost_model(),
        ..SimulationConfig::default()
    };
    for trace_name in ["priority_classes", "deadlines"] {
        for (load, nominal) in [
            ("light", 4_000_u64),
            ("moderate", 2_000),
            ("overload", 1_000),
        ] {
            let interarrival = scaled(nominal);
            for seed in 1..=30_u64 {
                let p = TraceParameters {
                    seed,
                    requests: 120,
                    mean_interarrival_micros: interarrival,
                };
                let trace = match trace_name {
                    "priority_classes" => SchedulingTrace::priority_classes(p),
                    _ => SchedulingTrace::deadlines(p, scaled(504)),
                };
                let (labels, policies): (Vec<_>, Vec<_>) = scheduler_policies().into_iter().unzip();
                let comparison =
                    compare_schedulers(&trace, &config, policies, DEFAULT_STARVATION_FACTOR)
                        .expect("trial");
                for (label, s) in labels.iter().zip(&comparison.summaries) {
                    let class = |name: &str| s.classes.iter().find(|c| c.class == name);
                    let hi = class("priority=5");
                    let lo = class("priority=0");
                    writeln!(
                        out,
                        "{trace_name},{interarrival},{seed},{label},{},{},{},{},{},{:.3},{},{},{},{},{},{},{},{},{},{load}",
                        s.requests,
                        s.completed,
                        s.completion.p50_micros,
                        s.completion.p95_micros,
                        s.time_to_first_token.p95_micros,
                        s.output_tokens_per_second().unwrap_or(0.0),
                        s.with_deadline,
                        s.deadline_missed,
                        s.starved,
                        hi.map_or(0, |c| c.requests),
                        hi.map_or(0, |c| c.completion.p50_micros),
                        hi.map_or(0, |c| c.completion.p95_micros),
                        lo.map_or(0, |c| c.requests),
                        lo.map_or(0, |c| c.completion.p50_micros),
                        lo.map_or(0, |c| c.completion.p95_micros),
                    )
                    .unwrap();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64, lo: usize, hi: usize) -> usize {
    lo + (next(state) as usize) % (hi - lo + 1)
}

fn prefill_sets(out: &mut impl Write) {
    writeln!(out, "distribution,set_size,lengths").unwrap();
    // A different seed from `bucketing`, so these sets are fresh draws.
    let mut state = 20_261_119_u64;
    for dist in [
        "uniform_1_128",
        "bimodal_80_20",
        "heavy_tail",
        "clustered_48_64",
    ] {
        for set_size in [4_usize, 8, 16] {
            let lengths = (0..set_size)
                .map(|_| sample_length(dist, &mut state).to_string())
                .collect::<Vec<_>>();
            writeln!(out, "{dist},{set_size},\"{}\"", lengths.join(",")).unwrap();
        }
    }
}

fn sample_length(dist: &str, state: &mut u64) -> usize {
    match dist {
        "uniform_1_128" => uniform(state, 1, 128),
        "bimodal_80_20" => {
            if next(state) % 100 < 80 {
                uniform(state, 4, 16)
            } else {
                uniform(state, 96, 192)
            }
        }
        // Geometric-like heavy tail: lengths cluster small, a few long.
        "heavy_tail" => {
            let u = (next(state) >> 11) as f64 / (1u64 << 53) as f64;
            ((-(1.0 - u).ln() * 24.0).ceil() as usize).clamp(1, 512)
        }
        // Tight cluster: chat-like prompts of similar length.
        "clustered_48_64" => uniform(state, 48, 64),
        _ => unreachable!(),
    }
}

/// (padding positions, invocations) for one grouping of `lengths`.
fn shape(batcher: &dyn PrefillBatcher, lengths: &[usize]) -> (usize, usize) {
    let decision = batcher.group(lengths);
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (len, key) in lengths.iter().zip(&decision.buckets) {
        groups.entry(*key).or_default().push(*len);
    }
    let padding = groups
        .values()
        .map(|g| {
            let max = *g.iter().max().unwrap();
            g.iter().map(|l| max - l).sum::<usize>()
        })
        .sum();
    (padding, groups.len())
}

fn bucketing(out: &mut impl Write) {
    writeln!(
        out,
        "distribution,set_size,trial,strategy,useful_tokens,padding_tokens,invocations,modelled_us"
    )
    .unwrap();
    let cost = cost_model();
    // Launch cost in token-equivalents under the simulator's coefficients.
    let overhead = (cost.per_invocation_micros / cost.per_dense_token_micros.max(1)) as usize;
    let strategies: Vec<(&str, Box<dyn PrefillBatcher>)> = vec![
        ("unbucketed", Box::new(UnbucketedPrefillBatcher)),
        ("power_of_two", Box::new(PowerOfTwoPrefillBatcher)),
        (
            "cost_aware",
            Box::new(CostAwarePrefillBatcher::new(overhead)),
        ),
    ];
    let mut state = 42_u64;
    for dist in [
        "uniform_1_128",
        "bimodal_80_20",
        "heavy_tail",
        "clustered_48_64",
    ] {
        for set_size in [4_usize, 8, 16, 32] {
            for trial in 0..1_000 {
                let lengths: Vec<usize> = (0..set_size)
                    .map(|_| sample_length(dist, &mut state))
                    .collect();
                let useful: usize = lengths.iter().sum();
                for (name, batcher) in &strategies {
                    let (padding, invocations) = shape(batcher.as_ref(), &lengths);
                    let per_padding = cost
                        .per_padding_token_micros
                        .unwrap_or(cost.per_dense_token_micros);
                    let modelled = invocations as u64 * cost.per_invocation_micros
                        + useful as u64 * cost.per_dense_token_micros
                        + padding as u64 * per_padding;
                    writeln!(
                        out,
                        "{dist},{set_size},{trial},{name},{useful},{padding},{invocations},{modelled}"
                    )
                    .unwrap();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[derive(Default)]
struct Timings {
    rows: Vec<(String, &'static str, usize, usize, usize, u128)>,
    samples: Vec<(PlanningSnapshot, ExecutionPlan, bool)>,
}

struct Timed {
    label: String,
    inner: Box<dyn ExecutionPolicy>,
    store: Arc<Mutex<Timings>>,
}

impl ExecutionPolicy for Timed {
    fn descriptor(&self) -> PolicyDescriptor {
        self.inner.descriptor()
    }
    fn plan(&self, s: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
        let start = Instant::now();
        let decision = self.inner.plan(s);
        let ns = start.elapsed().as_nanos();
        let live = s.prefilling.len() + s.decoding.len();
        let mut store = self.store.lock().unwrap();
        if let Ok(d) = &decision {
            let rows = d.plan.batches.iter().map(|b| b.rows.len()).sum();
            store
                .rows
                .push((self.label.clone(), "plan", s.queued.len(), live, rows, ns));
            store.samples.push((
                s.clone(),
                d.plan.clone(),
                self.inner.enforces_model_kv_capacity(),
            ));
        }
        decision
    }
    fn enforces_model_kv_capacity(&self) -> bool {
        self.inner.enforces_model_kv_capacity()
    }
    fn observe(&mut self, iteration: &ExecutorIteration) {
        let start = Instant::now();
        self.inner.observe(iteration);
        let ns = start.elapsed().as_nanos();
        let rows = iteration
            .execution_plan
            .batches
            .iter()
            .map(|b| b.rows.len())
            .sum();
        self.store.lock().unwrap().rows.push((
            self.label.clone(),
            "observe",
            iteration.queued_requests,
            iteration.active_requests,
            rows,
            ns,
        ));
    }
    fn recovers_kv_exhaustion(&self) -> bool {
        self.inner.recovers_kv_exhaustion()
    }
}

fn timing(out: &mut impl Write) {
    writeln!(out, "policy,max_active,call,queued,live,rows,nanos").unwrap();
    let geometry = KvGeometry::default();
    for max_active in [8_usize, 32, 128] {
        let config = SimulationConfig {
            max_active_requests: max_active,
            kv: KvGeometry {
                capacity_bytes: 16_384 * geometry.bytes_per_token,
                ..geometry
            },
            cost: cost_model(),
            ..SimulationConfig::default()
        };
        let labels = ["fifo", "sjf", "watermark+preemption", "adaptive-replay"];
        for label in labels {
            for seed in 1..=3_u64 {
                // Heavy overload so the queue grows into the hundreds.
                let trace = SchedulingTrace::overload(TraceParameters {
                    seed,
                    requests: 600,
                    mean_interarrival_micros: scaled(150),
                });
                let store = Arc::new(Mutex::new(Timings::default()));
                let timed = Timed {
                    label: label.to_owned(),
                    inner: policy_clone(label, max_active),
                    store: store.clone(),
                };
                run_observed_trial(&trace, timed, &config, &[], &mut |_, _| {}).expect("trial");
                let mut store = store.lock().unwrap();
                for (l, call, q, live, rows, ns) in store.rows.drain(..) {
                    writeln!(out, "{l},{max_active},{call},{q},{live},{rows},{ns}").unwrap();
                }
                // Validation timed apart from planning, repeated for a
                // stable per-call estimate.
                for (s, plan, enforce) in store.samples.drain(..) {
                    let reps = 20;
                    let start = Instant::now();
                    for _ in 0..reps {
                        std::hint::black_box(plan.validate(&s, enforce)).expect("valid");
                    }
                    let ns = start.elapsed().as_nanos() / reps;
                    let live = s.prefilling.len() + s.decoding.len();
                    let rows: usize = plan.batches.iter().map(|b| b.rows.len()).sum();
                    writeln!(
                        out,
                        "{label},{max_active},validate,{},{live},{rows},{ns}",
                        s.queued.len()
                    )
                    .unwrap();
                }
            }
        }
    }
}

fn policy_clone(label: &str, max_active: usize) -> Box<dyn ExecutionPolicy> {
    match label {
        "fifo" => Box::new(ComposedPolicy::default().with_admission(MemoryAwareAdmission)),
        "sjf" => Box::new(
            ComposedPolicy::default()
                .with_admission(MemoryAwareAdmission)
                .with_scheduler(ShortestJobFirst::new(JobCost::Prompt)),
        ),
        "watermark+preemption" => Box::new(
            ComposedPolicy::default()
                .with_admission(WatermarkAdmission {
                    headroom_tokens: 32,
                })
                .with_preemption(NewestFirst),
        ),
        _ => Box::new(
            serving_adaptive_policy(CostModel::default(), max_active, OnlineTuning::default())
                .expect("adaptive"),
        ),
    }
}
