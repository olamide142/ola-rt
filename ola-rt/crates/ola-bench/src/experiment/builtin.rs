//! Built-in experiments that need no loaded model, and the built-in registry.

use std::{collections::BTreeMap, time::Duration, time::Instant};

#[cfg(any(feature = "cubecl-cpu", feature = "wgpu"))]
use burn::tensor::Device;
#[cfg(feature = "wgpu")]
use burn::tensor::DeviceKind;
#[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
use burn::tensor::{Tensor, TensorData};

use ola_core::{ModelDescriptor, Timestamp};
use ola_runtime::{
    AdaptiveIntegerDomain, AdaptiveObjectiveWeights, ComposedPolicy, EarliestDeadlineFirst,
    ExecutionPolicy, HeadOfLine, JobCost, KvBlockManager, MemoryAwareAdmission,
    MixedTokenBudgetPolicy, NewestFirst, PolicyError, PriorityScheduler, Runtime, ShortestJobFirst,
    WatermarkAdmission,
};

use super::cluster::AdaptiveClusterRoutingStudy;
use super::{
    Direction, Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor,
    ExperimentError, ExperimentIdentity, ExperimentKind, ExperimentOutcome, ExperimentRegistry,
    ExperimentReport, MetricSpec, ParameterSpec, Parameters, Protocol, ReportParts, RunMetadata,
    Sample, TimeDomain, Unit, Variant, WorkloadIdentity, content_digest,
};
use crate::{
    DEFAULT_STARVATION_FACTOR, FixedWorkload, KvGeometry, SchedulingTrace, SimulationConfig,
    TraceParameters, analyze_cost_aware_prefill, compare_prefill_batching, compare_schedulers,
    compare_schedulers_observed, evaluate_virtual_time_held_out,
};

#[cfg(any(feature = "cubecl-cpu", feature = "wgpu"))]
use ola_kernel::{CubePagedAttention, KernelTarget, PortablePagedAttention};
#[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
use ola_kernel::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvBlockTable,
    PagedKvBlocks, PagedKvPage,
};

impl ExperimentRegistry {
    /// Returns every built-in experiment, in display order.
    pub fn builtin() -> Self {
        #[allow(unused_mut)]
        let mut experiments: Vec<Box<dyn Experiment>> = vec![
            Box::new(super::LlamaGeneration),
            Box::new(super::PrefillStrategies),
            Box::new(super::ChunkedPrefill),
            Box::new(super::CostCalibration),
            Box::new(KvBlockPressure),
            Box::new(FifoWorkload),
            Box::new(PrefillBucketing),
            Box::new(AdaptivePrefillCost),
            Box::new(SchedulerComparison),
            Box::new(AdaptivePolicyStudy),
            Box::new(KvPressure),
            Box::new(PrefixCache),
            Box::new(KvPageSize),
            Box::new(super::ClusterRouting),
            Box::new(AdaptiveClusterRoutingStudy),
        ];
        #[cfg(any(feature = "cubecl-cpu", feature = "wgpu"))]
        experiments.push(Box::new(PagedAttentionKernelStudy));
        #[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
        experiments.push(Box::new(super::PagedAttentionOccupancy));
        Self::new(experiments)
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

    fn traceable(&self) -> bool {
        true
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
            // One trace run per policy and seed; the first write error is
            // kept and reported after the trial.
            let mut runs = labels
                .iter()
                .map(|(label, _)| {
                    context
                        .trace
                        .as_ref()
                        .map(|trace| trace.run(format!("{label}/seed={seed}")))
                })
                .collect::<Vec<_>>();
            let mut trace_error = None;
            let comparison = compare_schedulers_observed(
                &trace,
                &config,
                Self::policies()
                    .into_iter()
                    .map(|(_, policy)| policy)
                    .collect(),
                DEFAULT_STARVATION_FACTOR,
                &mut |index, iteration, state| {
                    if let Some(Some(run)) = runs.get_mut(index)
                        && trace_error.is_none()
                        && let Err(error) = run.observe(iteration, state)
                    {
                        trace_error = Some(error);
                    }
                },
            )
            .map_err(run_error)?;
            if let Some(error) = trace_error {
                return Err(error);
            }
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

/// Selects a token budget on a training trace and evaluates it on held-out
/// seeded traces against a fixed FIFO baseline.
pub struct AdaptivePolicyStudy;

impl AdaptivePolicyStudy {
    fn adaptive_policy(
        theta: &[(&'static str, String)],
    ) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
        let max_tokens = theta
            .iter()
            .find(|(name, _)| *name == "max_tokens")
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .ok_or_else(|| PolicyError::InvalidCandidate("missing max_tokens".into()))?;
        let max_prefill_tokens = theta
            .iter()
            .find(|(name, _)| *name == "max_prefill_tokens")
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .ok_or_else(|| PolicyError::InvalidCandidate("missing max_prefill_tokens".into()))?;
        let batch =
            MixedTokenBudgetPolicy::new(max_tokens, max_prefill_tokens).ok_or_else(|| {
                PolicyError::InvalidCandidate(
                    "max_prefill_tokens must be positive and no greater than max_tokens".into(),
                )
            })?;
        Ok(Box::new(
            ComposedPolicy::default()
                .with_admission(MemoryAwareAdmission)
                .with_batch(batch),
        ))
    }

    fn baseline_policy(
        _: &[(&'static str, String)],
    ) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
        Ok(Box::new(
            ComposedPolicy::default().with_admission(MemoryAwareAdmission),
        ))
    }
}

impl Experiment for AdaptivePolicyStudy {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "adaptive-policy-study",
            name: "Adaptive policy study",
            description: "Solver-selected token budgets evaluated on held-out virtual-time traces",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::integer("seeds", "Held-out trace seeds", 5),
                ParameterSpec::integer("requests", "Requests per trace", 40),
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

    fn traceable(&self) -> bool {
        true
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
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
        let training = SchedulingTrace::overload(base);
        let held_out = seeds
            .iter()
            .map(|&seed| SchedulingTrace::overload(TraceParameters { seed, ..base }))
            .collect::<Vec<_>>();
        let study = evaluate_virtual_time_held_out(
            &training,
            &held_out,
            &config,
            vec![
                AdaptiveIntegerDomain {
                    name: "max_tokens",
                    min: 16,
                    max: 32,
                    step: 8,
                },
                AdaptiveIntegerDomain {
                    name: "max_prefill_tokens",
                    min: 2,
                    max: 16,
                    step: 2,
                },
            ],
            vec![
                ("max_tokens", "16".into()),
                ("max_prefill_tokens", "8".into()),
            ],
            Self::adaptive_policy,
            Self::baseline_policy,
            AdaptiveObjectiveWeights {
                ttft: 1,
                completion: 1,
                ..AdaptiveObjectiveWeights::default()
            },
        )
        .map_err(run_error)?;

        let adaptive_descriptor = Self::adaptive_policy(&study.theta)
            .map_err(run_error)?
            .descriptor()
            .to_string();
        let baseline_descriptor = Self::baseline_policy(&[])
            .map_err(run_error)?
            .descriptor()
            .to_string();
        let labels = vec![
            ("adaptive", adaptive_descriptor),
            ("fifo", baseline_descriptor),
        ];
        let mut samples = labels
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for trace in &held_out {
            for request in &trace.requests {
                content.push_str(&format!("{request:?};"));
            }
            let comparison = compare_schedulers_observed(
                trace,
                &config,
                vec![
                    Self::adaptive_policy(&study.theta).map_err(run_error)?,
                    Self::baseline_policy(&[]).map_err(run_error)?,
                ],
                DEFAULT_STARVATION_FACTOR,
                &mut |_, _, _| {},
            )
            .map_err(run_error)?;
            for (variant, summary) in samples.iter_mut().zip(&comparison.summaries) {
                variant.push(values(&[
                    ("completion_p95", as_f64(summary.completion.p95_micros)),
                    ("ttft_p95", as_f64(summary.time_to_first_token.p95_micros)),
                    (
                        "throughput",
                        summary.output_tokens_per_second().unwrap_or(0.0),
                    ),
                    ("starved", as_f64(summary.starved)),
                ]));
            }
        }

        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "adaptive-policy-study",
            seeds.clone(),
            vec![
                metric(
                    "completion_p95",
                    Unit::Micros,
                    TimeDomain::Virtual,
                    Direction::Lower,
                ),
                metric(
                    "ttft_p95",
                    Unit::Micros,
                    TimeDomain::Virtual,
                    Direction::Lower,
                ),
                metric(
                    "throughput",
                    Unit::TokensPerSecond,
                    TimeDomain::Virtual,
                    Direction::Higher,
                ),
                metric(
                    "starved",
                    Unit::Count,
                    TimeDomain::Virtual,
                    Direction::Lower,
                ),
            ],
            labels
                .into_iter()
                .zip(samples)
                .map(|((label, policy), samples)| {
                    (
                        Variant {
                            label: label.into(),
                            policy: Some(policy),
                            kernel: None,
                        },
                        samples,
                    )
                })
                .collect(),
            format!(
                "train seed 0, {} held-out seeds, selected max_tokens={}, max_prefill_tokens={}",
                seeds.len(),
                study
                    .theta
                    .iter()
                    .find(|(name, _)| *name == "max_tokens")
                    .map_or("?", |(_, value)| value.as_str()),
                study
                    .theta
                    .iter()
                    .find(|(name, _)| *name == "max_prefill_tokens")
                    .map_or("?", |(_, value)| value.as_str())
            ),
        )?;
        Ok(outcome.with_content_digest(&content))
    }
}

/// Worst-case KV reservation against watermark admission with preemption,
/// on a pool that binds, in virtual time.
pub struct KvPressure;

impl KvPressure {
    fn policies(headroom_tokens: usize) -> Vec<(&'static str, Box<dyn ExecutionPolicy>)> {
        vec![
            (
                "worst-case",
                Box::new(ComposedPolicy::default().with_admission(MemoryAwareAdmission)),
            ),
            (
                "watermark",
                Box::new(
                    ComposedPolicy::default()
                        .with_admission(WatermarkAdmission { headroom_tokens })
                        .with_preemption(NewestFirst),
                ),
            ),
        ]
    }
}

impl Experiment for KvPressure {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "kv-pressure",
            name: "KV pressure",
            description: "Worst-case KV reservation versus watermark admission with preemption",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            // 2: input tokens, to measure recompute overhead.
            payload_version: 2,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::text(
                    "trace",
                    "early_stopping, mixed_short_long, or overload",
                    "early_stopping",
                ),
                ParameterSpec::integer("seeds", "Trace seeds, one per repetition", 5),
                ParameterSpec::integer("requests", "Requests per trace", 120),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    1_000,
                ),
                ParameterSpec::integer("max_active_requests", "Executor slots", 24),
                ParameterSpec::integer("kv_capacity_tokens", "KV pool size in tokens", 2_048),
                ParameterSpec::integer(
                    "headroom_tokens",
                    "Watermark positions reserved beyond the prompt",
                    32,
                ),
            ],
        }
    }

    fn traceable(&self) -> bool {
        true
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
        let headroom_tokens = parameters.size("headroom_tokens")?;
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };
        let generate = |seed| {
            let parameters = TraceParameters { seed, ..base };
            Ok(match trace_name.as_str() {
                "early_stopping" => SchedulingTrace::early_stopping(parameters),
                "mixed_short_long" => SchedulingTrace::mixed_short_long(parameters),
                "overload" => SchedulingTrace::overload(parameters),
                _ => {
                    return Err(ExperimentError::InvalidParameter {
                        name: "trace".into(),
                        reason: format!("unknown trace `{trace_name}`"),
                    });
                }
            })
        };

        let labels = Self::policies(headroom_tokens)
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
            let mut runs = labels
                .iter()
                .map(|(label, _)| {
                    context
                        .trace
                        .as_ref()
                        .map(|trace| trace.run(format!("{label}/seed={seed}")))
                })
                .collect::<Vec<_>>();
            let mut trace_error = None;
            let comparison = compare_schedulers_observed(
                &trace,
                &config,
                Self::policies(headroom_tokens)
                    .into_iter()
                    .map(|(_, policy)| policy)
                    .collect(),
                DEFAULT_STARVATION_FACTOR,
                &mut |index, iteration, state| {
                    if let Some(Some(run)) = runs.get_mut(index)
                        && trace_error.is_none()
                        && let Err(error) = run.observe(iteration, state)
                    {
                        trace_error = Some(error);
                    }
                },
            )
            .map_err(run_error)?;
            if let Some(error) = trace_error {
                return Err(error);
            }
            for (variant, summary) in samples.iter_mut().zip(&comparison.summaries) {
                variant.push(values(&[
                    ("completion_p50", as_f64(summary.completion.p50_micros)),
                    ("completion_p95", as_f64(summary.completion.p95_micros)),
                    ("ttft_p95", as_f64(summary.time_to_first_token.p95_micros)),
                    (
                        "throughput",
                        summary.output_tokens_per_second().unwrap_or(0.0),
                    ),
                    ("peak_kv_cache_bytes", as_f64(summary.peak_kv_cache_bytes)),
                    ("preemptions", as_f64(summary.preemptions)),
                    ("input_tokens", as_f64(summary.input_tokens)),
                    ("failed", as_f64(summary.failed + summary.rejected)),
                    ("unfinished", as_f64(summary.unfinished)),
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
                metric(
                    "throughput",
                    Unit::TokensPerSecond,
                    TimeDomain::Virtual,
                    Direction::Higher,
                ),
                metric(
                    "peak_kv_cache_bytes",
                    Unit::Bytes,
                    TimeDomain::Virtual,
                    Direction::Neither,
                ),
                count("preemptions", Direction::Neither),
                metric(
                    "input_tokens",
                    Unit::Tokens,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
                count("failed", Direction::Lower),
                count("unfinished", Direction::Lower),
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
        let estimate = |variant: &str, metric: &str| {
            outcome
                .report
                .summaries
                .iter()
                .find(|summary| summary.variant == variant)
                .map_or(0.0, |summary| summary.estimates[metric].mean)
        };
        let summary = format!(
            "{trace_name}, {} seeds × {} requests, virtual time · completion p95 \
             worst-case {:.1} ms vs watermark {:.1} ms · throughput {:.0} vs {:.0} tok/s · \
             {:.1} preemptions per run",
            seeds.len(),
            base.requests,
            estimate("worst-case", "completion_p95") / 1_000.0,
            estimate("watermark", "completion_p95") / 1_000.0,
            estimate("worst-case", "throughput"),
            estimate("watermark", "throughput"),
            estimate("watermark", "preemptions"),
        );
        Ok(ExperimentOutcome { summary, ..outcome })
    }
}

/// The synthetic pool with and without its prefix cache, on prompts that
/// share a system prompt, in virtual time.
pub struct PrefixCache;

impl Experiment for PrefixCache {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "prefix-cache",
            name: "Prefix cache",
            description: "Prefill reuse for prompts that share a system prompt",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::integer("seeds", "Trace seeds, one per repetition", 5),
                ParameterSpec::integer("requests", "Requests per trace", 120),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    2_000,
                ),
                ParameterSpec::integer("prefix_tokens", "Shared prompt prefix length", 256),
                ParameterSpec::integer("max_active_requests", "Executor slots", 8),
                ParameterSpec::integer("kv_capacity_tokens", "KV pool size in tokens", 4_096),
            ],
        }
    }

    fn traceable(&self) -> bool {
        true
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
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
        let prefix_tokens = parameters.size("prefix_tokens")?;
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };
        let policy = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
        let descriptor = policy().descriptor().to_string();
        let variants = [("no-cache", false), ("prefix-cache", true)];
        let mut samples = variants
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for &seed in &seeds {
            let trace =
                SchedulingTrace::shared_prefix(TraceParameters { seed, ..base }, prefix_tokens);
            for request in &trace.requests {
                content.push_str(&format!("{request:?};"));
            }
            for ((label, cached), samples) in variants.iter().zip(&mut samples) {
                let config = SimulationConfig {
                    prefix_cache: *cached,
                    ..config
                };
                let mut run = context
                    .trace
                    .as_ref()
                    .map(|trace| trace.run(format!("{label}/seed={seed}")));
                let mut trace_error = None;
                let comparison = compare_schedulers_observed(
                    &trace,
                    &config,
                    vec![Box::new(policy())],
                    DEFAULT_STARVATION_FACTOR,
                    &mut |_, iteration, state| {
                        if let Some(run) = &mut run
                            && trace_error.is_none()
                            && let Err(error) = run.observe(iteration, state)
                        {
                            trace_error = Some(error);
                        }
                    },
                )
                .map_err(run_error)?;
                if let Some(error) = trace_error {
                    return Err(error);
                }
                let summary = &comparison.summaries[0];
                samples.push(values(&[
                    ("ttft_p50", as_f64(summary.time_to_first_token.p50_micros)),
                    ("ttft_p95", as_f64(summary.time_to_first_token.p95_micros)),
                    ("completion_p95", as_f64(summary.completion.p95_micros)),
                    (
                        "throughput",
                        summary.output_tokens_per_second().unwrap_or(0.0),
                    ),
                    ("input_tokens", as_f64(summary.input_tokens)),
                    ("completed", as_f64(summary.completed)),
                ]));
            }
        }

        let latency = |name| metric(name, Unit::Micros, TimeDomain::Virtual, Direction::Lower);
        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "shared_prefix",
            seeds.clone(),
            vec![
                latency("ttft_p50"),
                latency("ttft_p95"),
                latency("completion_p95"),
                metric(
                    "throughput",
                    Unit::TokensPerSecond,
                    TimeDomain::Virtual,
                    Direction::Higher,
                ),
                metric(
                    "input_tokens",
                    Unit::Tokens,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric(
                    "completed",
                    Unit::Count,
                    TimeDomain::Untimed,
                    Direction::Higher,
                ),
            ],
            variants
                .iter()
                .zip(samples)
                .map(|((label, cached), samples)| {
                    (
                        Variant::policy(*label, format!("{descriptor} prefix_cache={cached}")),
                        samples,
                    )
                })
                .collect(),
            String::new(),
        )?
        .with_content_digest(&content);
        let estimate = |variant: &str, metric: &str| {
            outcome
                .report
                .summaries
                .iter()
                .find(|summary| summary.variant == variant)
                .map_or(0.0, |summary| summary.estimates[metric].mean)
        };
        let summary = format!(
            "shared_prefix ({prefix_tokens} tokens), {} seeds × {} requests, virtual time · \
             TTFT p95 {:.1} → {:.1} ms · prefill tokens {:.0} → {:.0} · throughput {:.0} → {:.0} tok/s",
            seeds.len(),
            base.requests,
            estimate("no-cache", "ttft_p95") / 1_000.0,
            estimate("prefix-cache", "ttft_p95") / 1_000.0,
            estimate("no-cache", "input_tokens"),
            estimate("prefix-cache", "input_tokens"),
            estimate("no-cache", "throughput"),
            estimate("prefix-cache", "throughput"),
        );
        Ok(ExperimentOutcome { summary, ..outcome })
    }
}

/// Portable versus CubeCL paged-attention latency and numerical conformance.
pub struct PagedAttentionKernelStudy;

impl Experiment for PagedAttentionKernelStudy {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "paged-attention-kernel",
            name: "Paged-attention kernel",
            description: "Warm-up and repeated portable-versus-CubeCL attention measurements",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::text("page_sizes", "Tokens per logical page", "4,8,16"),
                ParameterSpec::integer("sequence_tokens", "Populated tokens per request", 23),
                ParameterSpec::integer("batch", "Independent requests per dispatch", 2),
                ParameterSpec::integer(
                    "pool_slack_blocks",
                    "Unused physical KV blocks per request",
                    0,
                ),
                ParameterSpec::integer("warmups", "Discarded dispatches per variant", 1),
                ParameterSpec::integer("repetitions", "Measured dispatches per variant", 5),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        #[cfg(not(any(feature = "cubecl-cpu", feature = "wgpu")))]
        {
            let _ = (context, parameters);
            return Err(ExperimentError::Run(
                "paged-attention-kernel requires the ola-bench cubecl-cpu feature".into(),
            ));
        }

        #[cfg(any(feature = "cubecl-cpu", feature = "wgpu"))]
        {
            let page_sizes = lengths(parameters, "page_sizes")?;
            let batch = parameters.size("batch")?;
            let sequence_tokens = parameters.size("sequence_tokens")?;
            let pool_slack_blocks = parameters.size("pool_slack_blocks")?;
            let warmups = parameters.size("warmups")?;
            let repetitions = parameters.size("repetitions")?;
            if batch == 0 || repetitions == 0 {
                return Err(ExperimentError::Run(
                    "batch and repetitions must be greater than zero".into(),
                ));
            }
            #[cfg(feature = "wgpu")]
            let (device, target) = (
                Device::wgpu(DeviceKind::IntegratedGpu(0)),
                KernelTarget::Wgpu,
            );
            #[cfg(all(not(feature = "wgpu"), feature = "cubecl-cpu"))]
            let (device, target) = (Device::cpu(), KernelTarget::Cpu);
            let portable = PortablePagedAttention::new();
            let cube = CubePagedAttention::new(target).map_err(run_error)?;
            let mut variants = Vec::new();
            let mut content = String::new();
            for page_size in page_sizes {
                let config = PagedAttentionConfig::new(4, 2, 4).map_err(run_error)?;
                let input = kernel_input(
                    &device,
                    config,
                    page_size,
                    sequence_tokens,
                    pool_slack_blocks,
                )?;
                let inputs = vec![input; batch];
                let logical_pages = sequence_tokens.div_ceil(page_size);
                let padding_tokens = logical_pages * page_size - sequence_tokens;
                let physical_blocks = logical_pages
                    .checked_add(1)
                    .and_then(|blocks| blocks.checked_add(pool_slack_blocks))
                    .ok_or_else(|| {
                        ExperimentError::Run("physical KV block count overflowed".into())
                    })?;
                let unused_blocks = physical_blocks - logical_pages;
                let pool_fragmentation = unused_blocks as f64 / physical_blocks as f64;
                content.push_str(&format!(
                    "page={page_size};tokens={sequence_tokens};batch={batch};pool_slack={pool_slack_blocks};physical_blocks={physical_blocks};"
                ));
                let expected = run_kernel(&portable, &inputs)?.1;
                let mut portable_samples = Vec::with_capacity(repetitions);
                let mut cube_samples = Vec::with_capacity(repetitions);
                for _ in 0..warmups {
                    let _ = run_kernel(&portable, &inputs)?;
                    let _ = run_kernel(&cube, &inputs)?;
                }
                for _ in 0..repetitions {
                    let (portable_time, _) = run_kernel(&portable, &inputs)?;
                    portable_samples.push(values(&[
                        ("latency_micros", micros(portable_time)),
                        ("max_abs_error", 0.0),
                        ("batch_size", batch as f64),
                        ("page_tokens", page_size as f64),
                        ("sequence_tokens", sequence_tokens as f64),
                        ("logical_pages", logical_pages as f64),
                        ("padding_tokens", padding_tokens as f64),
                        ("physical_blocks", physical_blocks as f64),
                        ("unused_blocks", unused_blocks as f64),
                        ("pool_fragmentation", pool_fragmentation),
                    ]));
                    let (cube_time, actual) = run_kernel(&cube, &inputs)?;
                    let error = max_error(&expected, &actual);
                    cube_samples.push(values(&[
                        ("latency_micros", micros(cube_time)),
                        ("max_abs_error", error),
                        ("batch_size", batch as f64),
                        ("page_tokens", page_size as f64),
                        ("sequence_tokens", sequence_tokens as f64),
                        ("logical_pages", logical_pages as f64),
                        ("padding_tokens", padding_tokens as f64),
                        ("physical_blocks", physical_blocks as f64),
                        ("unused_blocks", unused_blocks as f64),
                        ("pool_fragmentation", pool_fragmentation),
                    ]));
                }
                variants.push((
                    Variant {
                        label: format!("portable-page={page_size}"),
                        policy: None,
                        kernel: Some(portable.descriptor().name().to_owned()),
                    },
                    portable_samples,
                ));
                variants.push((
                    Variant {
                        label: format!("cubecl-page={page_size}"),
                        policy: None,
                        kernel: Some(cube.descriptor().name().to_owned()),
                    },
                    cube_samples,
                ));
            }
            let mut outcome = finish(
                &self.descriptor(),
                context,
                parameters,
                if target == KernelTarget::Wgpu {
                    "paged-attention-wgpu"
                } else {
                    "paged-attention-cpu"
                },
                (0..repetitions as u64).collect(),
                vec![
                    metric(
                        "latency_micros",
                        Unit::Micros,
                        TimeDomain::Host,
                        Direction::Lower,
                    ),
                    metric(
                        "max_abs_error",
                        Unit::Ratio,
                        TimeDomain::Untimed,
                        Direction::Lower,
                    ),
                    metric(
                        "batch_size",
                        Unit::Count,
                        TimeDomain::Untimed,
                        Direction::Neither,
                    ),
                    metric(
                        "page_tokens",
                        Unit::Tokens,
                        TimeDomain::Untimed,
                        Direction::Neither,
                    ),
                    metric(
                        "sequence_tokens",
                        Unit::Tokens,
                        TimeDomain::Untimed,
                        Direction::Neither,
                    ),
                    metric(
                        "logical_pages",
                        Unit::Count,
                        TimeDomain::Untimed,
                        Direction::Neither,
                    ),
                    metric(
                        "padding_tokens",
                        Unit::Tokens,
                        TimeDomain::Untimed,
                        Direction::Lower,
                    ),
                    metric(
                        "physical_blocks",
                        Unit::Count,
                        TimeDomain::Untimed,
                        Direction::Neither,
                    ),
                    metric(
                        "unused_blocks",
                        Unit::Count,
                        TimeDomain::Untimed,
                        Direction::Lower,
                    ),
                    metric(
                        "pool_fragmentation",
                        Unit::Ratio,
                        TimeDomain::Untimed,
                        Direction::Lower,
                    ),
                ],
                variants,
                format!(
                    "{} page sizes × {} requests; pool_slack_blocks={pool_slack_blocks}; warmups={warmups}, repetitions={repetitions}",
                    parameters.text("page_sizes")?,
                    batch
                ),
            )?;
            outcome.report.protocol.warmups = warmups;
            outcome.report.protocol.repetitions = repetitions;
            outcome.report.protocol.ordering = "portable then CubeCL per page size".into();
            outcome.report.workload.digest = content_digest(content.as_bytes());
            Ok(outcome)
        }
    }
}

#[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
pub(super) fn kernel_input(
    device: &burn::tensor::Device,
    config: PagedAttentionConfig,
    page_size: usize,
    sequence_tokens: usize,
    pool_slack_blocks: usize,
) -> Result<PagedAttentionInput, ExperimentError> {
    if page_size == 0 || sequence_tokens == 0 {
        return Err(ExperimentError::Run(
            "page sizes and sequence tokens must be greater than zero".into(),
        ));
    }
    let heads = config.query_heads();
    let kv_heads = config.key_value_heads();
    let dim = config.head_dim();
    let query = Tensor::from_data(
        TensorData::new(
            (0..heads * dim)
                .map(|index| index as f32 * 0.01)
                .collect::<Vec<_>>(),
            [1, heads, 1, dim],
        ),
        device,
    );
    let logical_pages = sequence_tokens.div_ceil(page_size);
    let physical_blocks = logical_pages
        .checked_add(1)
        .and_then(|blocks| blocks.checked_add(pool_slack_blocks))
        .ok_or_else(|| ExperimentError::Run("physical KV block count overflowed".into()))?;
    let mut physical_key = Tensor::zeros([physical_blocks, kv_heads, page_size, dim], device);
    let mut physical_value = Tensor::zeros([physical_blocks, kv_heads, page_size, dim], device);
    let mut pages = Vec::with_capacity(logical_pages);
    let mut block_ids = Vec::with_capacity(logical_pages);
    for logical in 0..logical_pages {
        let used = (sequence_tokens - logical * page_size).min(page_size);
        let values = (0..kv_heads * used * dim)
            .map(|index| ((index + logical * 7) % 17) as f32 * 0.02 - 0.1)
            .collect::<Vec<_>>();
        let key = Tensor::from_data(
            TensorData::new(values.clone(), [1, kv_heads, used, dim]),
            device,
        );
        let value = Tensor::from_data(TensorData::new(values, [1, kv_heads, used, dim]), device);
        let physical = logical + 1;
        physical_key = physical_key.slice_assign(
            [physical..physical + 1, 0..kv_heads, 0..used, 0..dim],
            key.clone(),
        );
        physical_value = physical_value.slice_assign(
            [physical..physical + 1, 0..kv_heads, 0..used, 0..dim],
            value.clone(),
        );
        pages.push(PagedKvPage::new(key, value));
        block_ids.push(physical);
    }
    let table = PagedKvBlockTable::new(block_ids, physical_blocks, device).map_err(run_error)?;
    let blocks = PagedKvBlocks::new(
        physical_key,
        physical_value,
        table,
        page_size,
        sequence_tokens,
    )
    .map_err(run_error)?;
    PagedAttentionInput::new(query, pages, config)
        .map_err(run_error)?
        .with_blocks(blocks)
        .map_err(run_error)
}

#[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
pub(super) fn run_kernel(
    kernel: &dyn PagedAttentionKernel,
    inputs: &[PagedAttentionInput],
) -> Result<(Duration, Vec<Vec<f32>>), ExperimentError> {
    let started = Instant::now();
    let outputs = kernel.execute_batch(inputs.to_vec()).map_err(run_error)?;
    let values = outputs
        .into_iter()
        .map(|output| output.into_data().try_to_vec::<f32>().map_err(run_error))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((started.elapsed(), values))
}

#[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
pub(super) fn max_error(expected: &[Vec<f32>], actual: &[Vec<f32>]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .flat_map(|(expected, actual)| expected.iter().zip(actual))
        .map(|(expected, actual)| f64::from((actual - expected).abs()))
        .fold(0.0, f64::max)
}

/// KV page size: internal fragmentation against prefix-cache reuse.
pub struct KvPageSize;

impl Experiment for KvPageSize {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "kv-page-size",
            name: "KV page size",
            description: "Internal fragmentation and prefix reuse across KV page sizes",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::text("page_sizes", "Tokens per page to compare", "4,8,16,32,64"),
                ParameterSpec::integer("seeds", "Trace seeds, one per repetition", 5),
                ParameterSpec::integer("requests", "Requests per trace", 120),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    2_000,
                ),
                ParameterSpec::integer(
                    "prefix_tokens",
                    "Shared prompt prefix length; not a multiple of most page sizes",
                    100,
                ),
                ParameterSpec::integer("max_active_requests", "Executor slots", 8),
                ParameterSpec::integer("kv_capacity_tokens", "KV pool size in tokens", 4_096),
            ],
        }
    }

    fn traceable(&self) -> bool {
        false
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let page_sizes = lengths(parameters, "page_sizes")?;
        let seeds = (1..=parameters.integer("seeds")?).collect::<Vec<_>>();
        let capacity_tokens = parameters.integer("kv_capacity_tokens")?;
        let prefix_tokens = parameters.size("prefix_tokens")?;
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };
        let max_active_requests = parameters.size("max_active_requests")?;
        let policy = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
        let descriptor = policy().descriptor().to_string();
        let mut samples = page_sizes
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for &seed in &seeds {
            let trace =
                SchedulingTrace::shared_prefix(TraceParameters { seed, ..base }, prefix_tokens);
            for request in &trace.requests {
                content.push_str(&format!("{request:?};"));
            }
            for (&page, samples) in page_sizes.iter().zip(&mut samples) {
                let geometry = KvGeometry::default();
                let config = SimulationConfig {
                    max_active_requests,
                    kv: KvGeometry {
                        tokens_per_block: page,
                        capacity_bytes: capacity_tokens.saturating_mul(geometry.bytes_per_token),
                        ..geometry
                    },
                    ..SimulationConfig::default()
                };
                let trial = |prefix_cache| {
                    compare_schedulers(
                        &trace,
                        &SimulationConfig {
                            prefix_cache,
                            ..config
                        },
                        vec![Box::new(policy())],
                        DEFAULT_STARVATION_FACTOR,
                    )
                    .map(|comparison| comparison.summaries[0].clone())
                    .map_err(run_error)
                };
                let plain = trial(false)?;
                let cached = trial(true)?;
                samples.push(values(&[
                    ("kv_fragmentation", plain.kv_internal_fragmentation()),
                    ("prefix_hit_tokens", as_f64(cached.prefix_hit_tokens)),
                    ("input_tokens", as_f64(cached.input_tokens)),
                    ("completion_p95", as_f64(cached.completion.p95_micros)),
                    ("ttft_p95", as_f64(cached.time_to_first_token.p95_micros)),
                ]));
            }
        }

        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "shared_prefix",
            seeds.clone(),
            vec![
                metric(
                    "kv_fragmentation",
                    Unit::Ratio,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric(
                    "prefix_hit_tokens",
                    Unit::Tokens,
                    TimeDomain::Untimed,
                    Direction::Higher,
                ),
                metric(
                    "input_tokens",
                    Unit::Tokens,
                    TimeDomain::Untimed,
                    Direction::Lower,
                ),
                metric(
                    "completion_p95",
                    Unit::Micros,
                    TimeDomain::Virtual,
                    Direction::Lower,
                ),
                metric(
                    "ttft_p95",
                    Unit::Micros,
                    TimeDomain::Virtual,
                    Direction::Lower,
                ),
            ],
            page_sizes
                .iter()
                .zip(samples)
                .map(|(page, samples)| {
                    (
                        Variant::policy(
                            format!("page={page}"),
                            format!("{descriptor} tokens_per_block={page}"),
                        ),
                        samples,
                    )
                })
                .collect(),
            String::new(),
        )?
        .with_content_digest(&content);
        let summary = format!(
            "shared_prefix ({prefix_tokens} tokens), {} seeds × {} requests · {}",
            seeds.len(),
            base.requests,
            outcome
                .report
                .summaries
                .iter()
                .map(|summary| {
                    format!(
                        "{}: waste {:.1}%, reused {:.0} tokens",
                        summary.variant,
                        summary.estimates["kv_fragmentation"].mean * 100.0,
                        summary.estimates["prefix_hit_tokens"].mean
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
    pub(super) fn with_content_digest(mut self, content: &str) -> Self {
        self.report.workload.digest = content_digest(content.as_bytes());
        self
    }
}

#[allow(clippy::too_many_arguments, reason = "one call site per experiment")]
pub(super) fn finish(
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

pub(super) fn metric(name: &str, unit: Unit, domain: TimeDomain, better: Direction) -> MetricSpec {
    MetricSpec::new(name, unit, domain, better)
}

pub(super) fn plain_variant(label: &str) -> Variant {
    Variant {
        label: label.to_owned(),
        policy: None,
        kernel: None,
    }
}

pub(super) fn values(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

pub(super) fn lengths(parameters: &Parameters, name: &str) -> Result<Vec<usize>, ExperimentError> {
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

pub(super) fn run_error(error: impl std::fmt::Display) -> ExperimentError {
    ExperimentError::Run(error.to_string())
}

fn micros(duration: Duration) -> f64 {
    as_f64(duration.as_micros())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "measurements are far below 2^52"
)]
pub(super) fn as_f64(value: impl TryInto<u64>) -> f64 {
    value.try_into().map_or(f64::MAX, |value: u64| value as f64)
}

pub(super) fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2} s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.1} ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{} µs", duration.as_micros())
    }
}

pub(super) fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", as_f64(bytes) / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", as_f64(bytes) / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
