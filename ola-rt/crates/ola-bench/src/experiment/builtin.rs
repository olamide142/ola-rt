//! Built-in experiments that need no loaded model.

use std::{collections::BTreeMap, time::Duration, time::Instant};

use ola_core::{ModelDescriptor, Timestamp};
use ola_runtime::{
    ComposedPolicy, EarliestDeadlineFirst, ExecutionPolicy, HeadOfLine, JobCost, KvBlockManager,
    MemoryAwareAdmission, PriorityScheduler, Runtime, ShortestJobFirst,
};

use super::{
    Direction, Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor,
    ExperimentError, ExperimentIdentity, ExperimentKind, ExperimentOutcome, ExperimentRegistry,
    ExperimentReport, MetricSpec, ParameterSpec, Parameters, Protocol, ReportParts, RunMetadata,
    Sample, TimeDomain, Unit, Variant, WorkloadIdentity, content_digest,
};
use crate::{
    DEFAULT_STARVATION_FACTOR, FixedWorkload, KvGeometry, SchedulingTrace, SimulationConfig,
    TraceParameters, analyze_cost_aware_prefill, compare_prefill_batching, compare_schedulers,
};

impl ExperimentRegistry {
    /// Returns every built-in experiment, in display order.
    pub fn builtin() -> Self {
        Self::new(vec![
            Box::new(KvBlockPressure),
            Box::new(FifoWorkload),
            Box::new(PrefillBucketing),
            Box::new(AdaptivePrefillCost),
            Box::new(SchedulerComparison),
        ])
    }
}

/// Fixed-size allocate/append/free pressure on the KV block manager.
pub struct KvBlockPressure;

impl Experiment for KvBlockPressure {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "kv-block-pressure",
            name: "KV block pressure",
            description: "Allocate, grow, and free sequences in the KV block manager",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Probe,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![ParameterSpec::integer(
                "sequences",
                "Sequences to allocate; every third is freed",
                128,
            )],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        const CAPACITY: u64 = 64 * 1024 * 1024;
        const TOKENS_PER_BLOCK: usize = 16;
        const BYTES_PER_TOKEN: u64 = 4 * 1024;
        let sequences = parameters.size("sequences")?;

        let started = Instant::now();
        let mut manager =
            KvBlockManager::new(CAPACITY, TOKENS_PER_BLOCK, BYTES_PER_TOKEN).map_err(run_error)?;
        let mut allocated = Vec::with_capacity(sequences);
        for index in 0..sequences {
            let sequence = manager.allocate().map_err(run_error)?;
            manager
                .append(sequence, 1 + index % 96)
                .map_err(run_error)?;
            allocated.push(sequence);
        }
        for (index, sequence) in allocated.into_iter().enumerate() {
            if index.is_multiple_of(3) {
                manager.free(sequence).map_err(run_error)?;
            }
        }
        let elapsed = started.elapsed();
        let metrics = manager.metrics();
        let fragmentation = if metrics.allocated_bytes == 0 {
            0.0
        } else {
            as_f64(metrics.internal_fragmentation_bytes) / as_f64(metrics.allocated_bytes)
        };

        let summary = format!(
            "{} live · {} blocks / {} · {} · {:.1}% fragment · {}",
            metrics.live_sequences,
            metrics.allocated_blocks,
            metrics.total_blocks,
            format_bytes(metrics.allocated_bytes),
            fragmentation * 100.0,
            format_duration(elapsed),
        );
        let count = |name| metric(name, Unit::Count, TimeDomain::Untimed, Direction::Neither);
        finish(
            &self.descriptor(),
            context,
            parameters,
            "kv-block-pressure",
            Vec::new(),
            vec![
                count("live_sequences"),
                count("allocated_blocks"),
                count("total_blocks"),
                metric(
                    "allocated_bytes",
                    Unit::Bytes,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
                metric(
                    "fragmentation",
                    Unit::Ratio,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric("elapsed", Unit::Micros, TimeDomain::Host, Direction::Lower),
            ],
            vec![(
                plain_variant("block-manager"),
                vec![values(&[
                    ("live_sequences", as_f64(metrics.live_sequences)),
                    ("allocated_blocks", as_f64(metrics.allocated_blocks)),
                    ("total_blocks", as_f64(metrics.total_blocks)),
                    ("allocated_bytes", as_f64(metrics.allocated_bytes)),
                    ("fragmentation", fragmentation),
                    ("elapsed", micros(elapsed)),
                ])],
            )],
            summary,
        )
    }
}

/// Replays a fixed workload through the runtime and checks FIFO order.
pub struct FifoWorkload;

impl Experiment for FifoWorkload {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "fifo-workload",
            name: "FIFO workload",
            description: "Replay and complete a deterministic runtime trace",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Probe,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![ParameterSpec::integer(
                "requests",
                "Requests in the fixed workload",
                100,
            )],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let requests = parameters.size("requests")?;
        let workload = FixedWorkload {
            model: "fifo-probe".into(),
            requests,
            prompt_tokens: 32,
            output_tokens: 16,
            interarrival: Duration::from_micros(100),
        }
        .build()
        .map_err(run_error)?;
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("fifo-probe", "probe"))
            .map_err(run_error)?;

        let started = Instant::now();
        let submitted = workload.replay(&mut runtime).map_err(run_error)?;
        let mut observed = Vec::with_capacity(requests);
        for index in 0..requests {
            let base = 20_000 + u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(3);
            let session = runtime
                .start_queue_head(Timestamp::from_micros(base))
                .map_err(run_error)?
                .ok_or_else(|| {
                    ExperimentError::Run("FIFO returned no request while work remained".into())
                })?;
            observed.push(session.request_id);
            runtime
                .mark_first_token(session.request_id, Timestamp::from_micros(base + 1))
                .map_err(run_error)?;
            runtime
                .complete(session.request_id, Timestamp::from_micros(base + 2))
                .map_err(run_error)?;
        }
        if observed != submitted {
            return Err(ExperimentError::Run(
                "FIFO execution order differed from arrival order".into(),
            ));
        }
        let elapsed = started.elapsed();

        finish(
            &self.descriptor(),
            context,
            parameters,
            "fixed-workload",
            Vec::new(),
            vec![
                metric(
                    "requests",
                    Unit::Count,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
                metric("elapsed", Unit::Micros, TimeDomain::Host, Direction::Lower),
            ],
            vec![(
                plain_variant("fifo"),
                vec![values(&[
                    ("requests", as_f64(requests)),
                    ("elapsed", micros(elapsed)),
                ])],
            )],
            format!(
                "{requests} requests · FIFO order verified · {}",
                format_duration(elapsed)
            ),
        )
    }
}

/// Padding and invocations of unbucketed versus power-of-two prefill.
pub struct PrefillBucketing;

impl Experiment for PrefillBucketing {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "prefill-bucketing",
            name: "Prefill bucketing",
            description: "Compare unbucketed and power-of-two padding work",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Probe,
            domain: TimeDomain::Untimed,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![ParameterSpec::text(
                "prompt_lengths",
                "Comma-separated prompt lengths",
                "5,7,9,15,17,31,33,63",
            )],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let lengths = lengths(parameters, "prompt_lengths")?;
        let comparison = compare_prefill_batching(&lengths).map_err(run_error)?;
        let sample = |summary: crate::PrefillBatchSummary| {
            values(&[
                ("padding_tokens", as_f64(summary.padding_tokens)),
                ("invocations", as_f64(summary.invocations)),
                (
                    "input_efficiency",
                    summary.input_token_efficiency().unwrap_or(0.0),
                ),
            ])
        };
        let summary = format!(
            "{} prompts · padding {} → {} ({:.1}% less) · invocations {} → {} · efficiency {:.1}% → {:.1}%",
            lengths.len(),
            comparison.unbucketed.padding_tokens,
            comparison.power_of_two.padding_tokens,
            comparison.padding_reduction().unwrap_or(0.0) * 100.0,
            comparison.unbucketed.invocations,
            comparison.power_of_two.invocations,
            comparison
                .unbucketed
                .input_token_efficiency()
                .unwrap_or(0.0)
                * 100.0,
            comparison
                .power_of_two
                .input_token_efficiency()
                .unwrap_or(0.0)
                * 100.0,
        );
        finish(
            &self.descriptor(),
            context,
            parameters,
            "prompt-lengths",
            Vec::new(),
            vec![
                metric(
                    "padding_tokens",
                    Unit::Tokens,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric(
                    "invocations",
                    Unit::Count,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric(
                    "input_efficiency",
                    Unit::Ratio,
                    TimeDomain::Untimed,
                    Direction::Higher,
                ),
            ],
            vec![
                (
                    plain_variant("unbucketed"),
                    vec![sample(comparison.unbucketed)],
                ),
                (
                    plain_variant("power-of-two"),
                    vec![sample(comparison.power_of_two)],
                ),
            ],
            summary,
        )
    }
}

/// Which grouping the cost-aware rule chooses for diverse and clustered
/// prompt lengths.
pub struct AdaptivePrefillCost;

impl Experiment for AdaptivePrefillCost {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "adaptive-prefill-cost",
            name: "Adaptive prefill cost",
            description: "Balance dense padding against invocation overhead",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Probe,
            domain: TimeDomain::Untimed,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::integer(
                    "invocation_overhead_tokens",
                    "Token-equivalent cost of one invocation",
                    32,
                ),
                ParameterSpec::text("diverse", "Diverse prompt lengths", "5,7,9,63"),
                ParameterSpec::text("clustered", "Clustered prompt lengths", "5,6,7,8"),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let overhead = parameters.size("invocation_overhead_tokens")?;
        let analyze = |name| {
            let lengths = lengths(parameters, name)?;
            analyze_cost_aware_prefill(&lengths, overhead)
                .map(|analysis| (lengths, analysis))
                .map_err(run_error)
        };
        let (diverse_lengths, diverse) = analyze("diverse")?;
        let (clustered_lengths, clustered) = analyze("clustered")?;
        let sample = |analysis: &crate::CostAwarePrefillAnalysis| {
            values(&[
                (
                    "unbucketed_cost",
                    as_f64(analysis.decision.unbucketed.estimated_cost),
                ),
                (
                    "power_of_two_cost",
                    as_f64(analysis.decision.power_of_two.estimated_cost),
                ),
                ("selected_padding", as_f64(analysis.selected.padding_tokens)),
                (
                    "selected_invocations",
                    as_f64(analysis.selected.invocations),
                ),
                (
                    "chose_power_of_two",
                    if analysis.decision.selected
                        == ola_runtime::PrefillGroupingStrategy::PowerOfTwo
                    {
                        1.0
                    } else {
                        0.0
                    },
                ),
            ])
        };
        let summary = format!(
            "{overhead} token-equivalent call cost · diverse {:?}: cost {} → {}, {} padding / {} calls · clustered {:?}: cost {} → {}, {} padding / {} calls",
            diverse.decision.selected,
            diverse.decision.unbucketed.estimated_cost,
            diverse.decision.power_of_two.estimated_cost,
            diverse.selected.padding_tokens,
            diverse.selected.invocations,
            clustered.decision.selected,
            clustered.decision.unbucketed.estimated_cost,
            clustered.decision.power_of_two.estimated_cost,
            clustered.selected.padding_tokens,
            clustered.selected.invocations,
        );
        let cost = |name| metric(name, Unit::Tokens, TimeDomain::Untimed, Direction::Lower);
        finish(
            &self.descriptor(),
            context,
            parameters,
            "prompt-lengths",
            Vec::new(),
            vec![
                cost("unbucketed_cost"),
                cost("power_of_two_cost"),
                cost("selected_padding"),
                metric(
                    "selected_invocations",
                    Unit::Count,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric(
                    "chose_power_of_two",
                    Unit::Ratio,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
            ],
            vec![
                (
                    Variant {
                        label: "diverse".into(),
                        policy: Some(format!("{diverse_lengths:?}")),
                        kernel: None,
                    },
                    vec![sample(&diverse)],
                ),
                (
                    Variant {
                        label: "clustered".into(),
                        policy: Some(format!("{clustered_lengths:?}")),
                        kernel: None,
                    },
                    vec![sample(&clustered)],
                ),
            ],
            summary,
        )
    }
}

/// FIFO, priority, EDF, SJF, and backfill on one trace generator across
/// several seeds, in virtual time.
pub struct SchedulerComparison;

/// Single-request iteration estimate used to scale `deadlines` traces under
/// the default cost model (200 + 300 + 4 µs).
const DEADLINE_ITERATION_MICROS: u64 = 504;

impl SchedulerComparison {
    fn policies() -> Vec<(&'static str, Box<dyn ExecutionPolicy>)> {
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
}

impl Experiment for SchedulerComparison {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "scheduler-comparison",
            name: "Scheduler comparison",
            description: "Scheduling baselines on seeded traces in virtual time",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::text(
                    "trace",
                    "mixed_short_long, prefill_decode_competition, priority_classes, \
                     deadlines, overload, or impossible",
                    "overload",
                ),
                ParameterSpec::integer("seeds", "Trace seeds, one per repetition", 5),
                ParameterSpec::integer("requests", "Requests per trace", 120),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    2_000,
                ),
                ParameterSpec::integer("max_active_requests", "Executor slots", 8),
                ParameterSpec::integer("kv_capacity_tokens", "KV pool size in tokens", 1_024),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let trace_name = parameters.text("trace")?.to_owned();
        let seeds = (1..=parameters.integer("seeds")?).collect::<Vec<_>>();
        let geometry = KvGeometry::default();
        let config = SimulationConfig {
            max_active_requests: parameters.size("max_active_requests")?,
            kv: KvGeometry {
                capacity_bytes: parameters
                    .integer("kv_capacity_tokens")?
                    .saturating_mul(geometry.bytes_per_token),
                ..geometry
            },
            ..SimulationConfig::default()
        };
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };
        let generate = |seed| {
            let parameters = TraceParameters { seed, ..base };
            Ok(match trace_name.as_str() {
                "mixed_short_long" => SchedulingTrace::mixed_short_long(parameters),
                "prefill_decode_competition" => {
                    SchedulingTrace::prefill_decode_competition(parameters)
                }
                "priority_classes" => SchedulingTrace::priority_classes(parameters),
                "deadlines" => SchedulingTrace::deadlines(parameters, DEADLINE_ITERATION_MICROS),
                "overload" => SchedulingTrace::overload(parameters),
                "impossible" => {
                    SchedulingTrace::impossible(parameters, config.kv.capacity_tokens())
                }
                _ => {
                    return Err(ExperimentError::InvalidParameter {
                        name: "trace".into(),
                        reason: format!("unknown trace `{trace_name}`"),
                    });
                }
            })
        };

        let labels = Self::policies()
            .into_iter()
            .map(|(label, policy)| (label, policy.descriptor().to_string()))
            .collect::<Vec<_>>();
        let mut samples = labels
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for &seed in &seeds {
            let trace = generate(seed)?;
            for request in &trace.requests {
                content.push_str(&format!("{request:?};"));
            }
            let comparison = compare_schedulers(
                &trace,
                &config,
                Self::policies()
                    .into_iter()
                    .map(|(_, policy)| policy)
                    .collect(),
                DEFAULT_STARVATION_FACTOR,
            )
            .map_err(run_error)?;
            for (variant, summary) in samples.iter_mut().zip(&comparison.summaries) {
                variant.push(values(&[
                    ("completion_p50", as_f64(summary.completion.p50_micros)),
                    ("completion_p95", as_f64(summary.completion.p95_micros)),
                    ("ttft_p95", as_f64(summary.time_to_first_token.p95_micros)),
                    ("queue_p95", as_f64(summary.queue.p95_micros)),
                    (
                        "throughput",
                        summary.output_tokens_per_second().unwrap_or(0.0),
                    ),
                    ("jain_fairness", summary.slowdown.jain_fairness),
                    ("starved", as_f64(summary.starved)),
                    ("deadline_missed", as_f64(summary.deadline_missed)),
                    ("admission_deferrals", as_f64(summary.admission_deferrals)),
                    ("backfill_skips", as_f64(summary.backfill_skips)),
                ]));
            }
        }

        let latency = |name| metric(name, Unit::Micros, TimeDomain::Virtual, Direction::Lower);
        let count = |name, better| metric(name, Unit::Count, TimeDomain::Virtual, better);
        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            &trace_name,
            seeds.clone(),
            vec![
                latency("completion_p50"),
                latency("completion_p95"),
                latency("ttft_p95"),
                latency("queue_p95"),
                metric(
                    "throughput",
                    Unit::TokensPerSecond,
                    TimeDomain::Virtual,
                    Direction::Higher,
                ),
                metric(
                    "jain_fairness",
                    Unit::Ratio,
                    TimeDomain::Virtual,
                    Direction::Higher,
                ),
                count("starved", Direction::Lower),
                count("deadline_missed", Direction::Lower),
                count("admission_deferrals", Direction::Neither),
                count("backfill_skips", Direction::Neither),
            ],
            labels
                .iter()
                .zip(samples)
                .map(|((label, policy), samples)| {
                    (Variant::policy(*label, policy.clone()), samples)
                })
                .collect(),
            String::new(),
        )?
        .with_content_digest(&content);
        let summary = format!(
            "{trace_name}, {} seeds × {} requests, virtual time · completion p95 (95% CI): {}",
            seeds.len(),
            base.requests,
            outcome
                .report
                .summaries
                .iter()
                .map(|summary| {
                    let estimate = &summary.estimates["completion_p95"];
                    let interval = estimate.ci95.map_or_else(String::new, |interval| {
                        format!(
                            " [{:.1}–{:.1}]",
                            interval.low / 1_000.0,
                            interval.high / 1_000.0
                        )
                    });
                    format!(
                        "{} {:.1} ms{interval}",
                        summary.variant,
                        estimate.mean / 1_000.0
                    )
                })
                .collect::<Vec<_>>()
                .join(" · ")
        );
        Ok(ExperimentOutcome { summary, ..outcome })
    }
}

impl ExperimentOutcome {
    /// Replaces the workload digest with one over the generated content,
    /// keeping the report valid.
    fn with_content_digest(mut self, content: &str) -> Self {
        self.report.workload.digest = content_digest(content.as_bytes());
        self
    }
}

#[allow(clippy::too_many_arguments, reason = "one call site per experiment")]
fn finish(
    descriptor: &ExperimentDescriptor,
    context: &ExperimentContext,
    parameters: &Parameters,
    workload: &str,
    seeds: Vec<u64>,
    metrics: Vec<MetricSpec>,
    variants: Vec<(Variant, Vec<BTreeMap<String, f64>>)>,
    summary: String,
) -> Result<ExperimentOutcome, ExperimentError> {
    let repetitions = variants.first().map_or(0, |(_, samples)| samples.len());
    let parameter_strings = parameters.as_strings();
    let digest = content_digest(format!("{workload}{parameter_strings:?}").as_bytes());
    let mut samples = Vec::new();
    let mut declared = Vec::new();
    for (variant, values) in variants {
        for (repetition, values) in values.into_iter().enumerate() {
            samples.push(Sample {
                repetition,
                variant: variant.label.clone(),
                values,
            });
        }
        declared.push(variant);
    }
    let report = ExperimentReport::new(ReportParts {
        experiment: ExperimentIdentity {
            id: descriptor.id.to_owned(),
            payload_version: descriptor.payload_version,
            kind: descriptor.kind,
            domain: descriptor.domain,
        },
        metadata: RunMetadata::now(context.model_label.clone(), context.device.clone()),
        workload: WorkloadIdentity {
            name: workload.to_owned(),
            seeds,
            parameters: parameter_strings,
            digest,
        },
        protocol: Protocol {
            warmups: 0,
            repetitions,
            ordering: if repetitions > 1 {
                "one seed per repetition".into()
            } else {
                "single run".into()
            },
        },
        metrics,
        variants: declared,
        samples,
        details: None,
    })?;
    Ok(ExperimentOutcome { report, summary })
}

fn metric(name: &str, unit: Unit, domain: TimeDomain, better: Direction) -> MetricSpec {
    MetricSpec::new(name, unit, domain, better)
}

fn plain_variant(label: &str) -> Variant {
    Variant {
        label: label.to_owned(),
        policy: None,
        kernel: None,
    }
}

fn values(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

fn lengths(parameters: &Parameters, name: &str) -> Result<Vec<usize>, ExperimentError> {
    parameters
        .text(name)?
        .split(',')
        .map(|length| {
            length
                .trim()
                .parse()
                .map_err(|_| ExperimentError::InvalidParameter {
                    name: name.to_owned(),
                    reason: "expected comma-separated positive integers".into(),
                })
        })
        .collect()
}

fn run_error(error: impl std::fmt::Display) -> ExperimentError {
    ExperimentError::Run(error.to_string())
}

fn micros(duration: Duration) -> f64 {
    as_f64(duration.as_micros())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "measurements are far below 2^52"
)]
fn as_f64(value: impl TryInto<u64>) -> f64 {
    value.try_into().map_or(f64::MAX, |value: u64| value as f64)
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2} s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.1} ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{} µs", duration.as_micros())
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", as_f64(bytes) / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", as_f64(bytes) / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
