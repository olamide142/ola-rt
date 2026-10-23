//! Built-in experiments that need a loaded model, and import of legacy
//! prefill reports.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use ola_core::{Clock, GenerationConfig, ModelDescriptor, MonotonicClock, RequestSpec};
use ola_model::CausalLanguageModel;
use ola_observe::TokenGaps;
use ola_runtime::{ComposedPolicy, Executor, ExecutorEvent, MemoryAwareAdmission};

use super::builtin::{
    as_f64, finish, format_bytes, format_duration, lengths, plain_variant, run_error,
};
use super::{
    BuildInfo, Direction, Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor,
    ExperimentError, ExperimentIdentity, ExperimentKind, ExperimentOutcome, ExperimentReport,
    HOST_STUDY_MIN_REPETITIONS, HOST_STUDY_MIN_WARMUPS, MetricSpec, ParameterSpec, Parameters,
    Protocol, ReportError, ReportParts, Requirement, RunMetadata, Sample, TimeDomain, Unit,
    Variant, WorkloadIdentity, content_digest,
};
use crate::{
    PrefillBenchmarkReport, PrefillExecutionConfig, PrefillFairnessLatency, PrefillFairnessRun,
    PrefillReportSample, PrefillThreeWayTrialConfig, PrefillTrialConfig,
    benchmark_chunked_prefill_execution, benchmark_prefill_fairness_trials,
    benchmark_prefill_trials, benchmark_three_way_prefill_trials,
    calibrate_prefill_invocation_overhead,
};

const MODEL_ID: &str = "experiment-model";
const PREFILL_STRATEGIES: &str = "llama-prefill-strategies";

fn loaded(context: &ExperimentContext) -> Result<&CausalLanguageModel, ExperimentError> {
    context
        .model
        .as_ref()
        .ok_or(ExperimentError::MissingRequirement(
            Requirement::LoadedModel,
        ))
}

/// Greedy generation latency and throughput on the loaded model.
pub struct LlamaGeneration;

impl Experiment for LlamaGeneration {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "llama-generation",
            name: "Llama generation",
            description: "Cached greedy generation: TTFT, inter-token latency, and throughput",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Probe,
            domain: TimeDomain::Host,
            // 2: real per-token times; TTFT and inter-token gaps.
            payload_version: 2,
            requirements: vec![Requirement::LoadedModel],
            parameters: vec![
                ParameterSpec::text("prompt", "Prompt text", "Hello"),
                ParameterSpec::integer("tokens", "Tokens to generate", 8),
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
        let model = loaded(context)?;
        let clock = Arc::new(MonotonicClock::start());
        let mut executor = Executor::default()
            .with_policy(ComposedPolicy::default().with_admission(MemoryAwareAdmission))
            .with_clock(clock.clone());
        executor
            .register_model(ModelDescriptor::language(MODEL_ID, "llama"), model.clone())
            .map_err(run_error)?;
        let mut request = RequestSpec::text(MODEL_ID, parameters.text("prompt")?);
        request.generation = GenerationConfig {
            max_new_tokens: parameters.size("tokens")?,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        };
        let request_id = executor.submit(request, clock.now()).map_err(run_error)?;
        let mut trace = context.trace.as_ref().map(|trace| trace.run("greedy"));
        let mut kernels = BTreeSet::new();
        let started = Instant::now();
        'generation: for _ in 0..10_000 {
            let iteration = executor.step_iteration(clock.now()).map_err(run_error)?;
            record_kernel_names(&iteration.execution_batches, &mut kernels);
            if let Some(trace) = &mut trace {
                trace.observe(&iteration, executor.state())?;
            }
            for event in iteration.events {
                match event {
                    ExecutorEvent::Token { finished: true, .. } => break 'generation,
                    ExecutorEvent::Failed { message, .. } => {
                        return Err(ExperimentError::Run(message));
                    }
                    ExecutorEvent::Rejected { reason, .. } => {
                        return Err(ExperimentError::Run(format!(
                            "generation admission rejected: {reason}"
                        )));
                    }
                    _ => {}
                }
            }
        }
        let elapsed = started.elapsed();
        let result = executor
            .result(request_id)
            .ok_or_else(|| ExperimentError::Run("generation did not finish".into()))?;
        let generated = result.metrics.generated_tokens;
        let timing = executor
            .state()
            .request(request_id)
            .map(|record| record.timing)
            .unwrap_or_default();
        let ttft = timing.first_token_at.map_or(0, |first| {
            first
                .as_micros()
                .saturating_sub(timing.queued_at.as_micros())
        });
        let gaps = TokenGaps::of(&result.token_times).unwrap_or_default();
        let throughput = if elapsed.is_zero() {
            0.0
        } else {
            as_f64(generated) / elapsed.as_secs_f64()
        };
        finish(
            &self.descriptor(),
            context,
            parameters,
            "single-prompt",
            Vec::new(),
            vec![
                MetricSpec::new(
                    "generated_tokens",
                    Unit::Tokens,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
                MetricSpec::new("elapsed", Unit::Micros, TimeDomain::Host, Direction::Lower),
                MetricSpec::new("ttft", Unit::Micros, TimeDomain::Host, Direction::Lower),
                MetricSpec::new("itl_p50", Unit::Micros, TimeDomain::Host, Direction::Lower),
                MetricSpec::new("itl_p90", Unit::Micros, TimeDomain::Host, Direction::Lower),
                MetricSpec::new("itl_max", Unit::Micros, TimeDomain::Host, Direction::Lower),
                MetricSpec::new(
                    "throughput",
                    Unit::TokensPerSecond,
                    TimeDomain::Host,
                    Direction::Higher,
                ),
                MetricSpec::new(
                    "kv_cache_bytes",
                    Unit::Bytes,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
            ],
            vec![(
                Variant {
                    label: "greedy".into(),
                    policy: None,
                    kernel: joined_kernel_names(kernels),
                },
                vec![values(&[
                    ("generated_tokens", as_f64(generated)),
                    ("elapsed", as_f64(elapsed.as_micros())),
                    ("ttft", as_f64(ttft)),
                    ("itl_p50", as_f64(gaps.p50_micros)),
                    ("itl_p90", as_f64(gaps.p90_micros)),
                    ("itl_max", as_f64(gaps.max_micros)),
                    ("throughput", throughput),
                    ("kv_cache_bytes", as_f64(result.metrics.kv_cache_bytes)),
                ])],
            )],
            format!(
                "{generated} tokens · TTFT {} · inter-token p50 {} max {} · {throughput:.2} tok/s · {} KV",
                format_duration(Duration::from_micros(ttft)),
                format_duration(Duration::from_micros(gaps.p50_micros)),
                format_duration(Duration::from_micros(gaps.max_micros)),
                format_bytes(result.metrics.kv_cache_bytes)
            ),
        )
    }
}

/// Calibrates invocation overhead, then compares unbucketed, power-of-two,
/// and cost-aware prefill grouping in rotated host-time trials.
pub struct PrefillStrategies;

impl Experiment for PrefillStrategies {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: PREFILL_STRATEGIES,
            name: "Llama prefill strategies",
            description: "Calibrate and compare three prefill groupers",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: vec![Requirement::LoadedModel],
            parameters: vec![
                ParameterSpec::text(
                    "prompt_lengths",
                    "Comma-separated prompt lengths",
                    "2,3,5,7",
                ),
                ParameterSpec::integer("max_new_tokens", "Tokens generated per prompt", 2),
                ParameterSpec::integer("calibration_warmups", "Discarded calibration trials", 1),
                ParameterSpec::integer("calibration_trials", "Measured calibration trials", 4),
                ParameterSpec::integer("warmups", "Discarded comparison trials", 1),
                ParameterSpec::integer(
                    "trials",
                    "Measured comparison trials, rotating strategy order",
                    5,
                ),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let model = loaded(context)?;
        let execution = PrefillExecutionConfig {
            prompts: lengths(parameters, "prompt_lengths")?
                .into_iter()
                .map(|length| vec![1; length])
                .collect(),
            max_new_tokens: parameters.size("max_new_tokens")?,
        };
        let calibration_comparison = benchmark_prefill_trials(
            model,
            &PrefillTrialConfig {
                execution: execution.clone(),
                warmup_trials: parameters.size("calibration_warmups")?,
                measured_trials: parameters.size("calibration_trials")?,
            },
        )
        .map_err(run_error)?;
        let calibration =
            calibrate_prefill_invocation_overhead(&calibration_comparison).map_err(run_error)?;
        let config = PrefillThreeWayTrialConfig {
            execution: execution.clone(),
            warmup_trials: parameters.size("warmups")?,
            measured_trials: parameters.size("trials")?,
            adaptive_invocation_overhead_tokens: calibration.invocation_overhead_tokens,
        };
        let comparison = benchmark_three_way_prefill_trials(model, &config).map_err(run_error)?;

        let strategies = [
            (&comparison.unbucketed, "unbucketed"),
            (&comparison.power_of_two, "power-of-two"),
            (&comparison.cost_aware, "cost-aware"),
        ];
        let (workload_name, digest) =
            prefill_workload(&execution.prompts, execution.max_new_tokens);
        let mut parts = report_parts(
            &self.descriptor(),
            context,
            parameters,
            workload_name,
            digest,
            Protocol {
                warmups: config.warmup_trials,
                repetitions: config.measured_trials,
                ordering: "rotated".into(),
            },
            calibration.invocation_overhead_tokens,
            strategies.iter().map(|(trials, label)| {
                (
                    *label,
                    trials
                        .samples
                        .iter()
                        .map(PrefillReportSample::from)
                        .collect::<Vec<_>>(),
                )
            }),
        );
        for ((trials, _), variant) in strategies.iter().zip(&mut parts.variants) {
            variant.kernel = summaries_kernel_identity(&trials.samples);
        }
        parts.details = Some(serde_json::json!({
            "calibration": {
                "invocation_overhead_tokens": calibration.invocation_overhead_tokens,
                "unbucketed_p50_micros": calibration.unbucketed_p50_micros,
                "power_of_two_p50_micros": calibration.power_of_two_p50_micros,
            },
            "strategy_order_by_trial": comparison.strategy_order_by_trial,
        }));
        let report = ExperimentReport::new(parts)?;

        let shape = |index: usize| &strategies[index].0.samples[0];
        let p50 = |index: usize| {
            format_duration(Duration::from_micros(
                strategies[index].0.prefill_iteration_latency.p50_micros,
            ))
        };
        let summary = format!(
            "{} prompts · calibrated {} token-equivalent call cost from {} → {} p50 · {} rotating trials + {} warmup · prefill p50 U/P/A {} / {} / {} · padding U/P/A {} / {} / {} · calls U/P/A {} / {} / {} · throughput U/P/A {:.2} / {:.2} / {:.2} tok/s · peak KV U/P/A {} / {} / {}",
            execution.prompts.len(),
            calibration.invocation_overhead_tokens,
            format_duration(Duration::from_micros(calibration.unbucketed_p50_micros)),
            format_duration(Duration::from_micros(calibration.power_of_two_p50_micros)),
            config.measured_trials,
            config.warmup_trials,
            p50(0),
            p50(1),
            p50(2),
            shape(0).prefill_padding_tokens,
            shape(1).prefill_padding_tokens,
            shape(2).prefill_padding_tokens,
            shape(0).prefill_invocations,
            shape(1).prefill_invocations,
            shape(2).prefill_invocations,
            strategies[0].0.output_tokens_per_second().unwrap_or(0.0),
            strategies[1].0.output_tokens_per_second().unwrap_or(0.0),
            strategies[2].0.output_tokens_per_second().unwrap_or(0.0),
            format_bytes(strategies[0].0.peak_kv_cache_bytes()),
            format_bytes(strategies[1].0.peak_kv_cache_bytes()),
            format_bytes(strategies[2].0.peak_kv_cache_bytes()),
        );
        Ok(ExperimentOutcome { report, summary })
    }
}

/// Complete versus resumable prompts, plus a chunk-size fairness sweep.
pub struct ChunkedPrefill;

impl Experiment for ChunkedPrefill {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "llama-chunked-prefill",
            name: "Llama chunked prefill",
            description: "Compare complete prompts with resumable prompt segments",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Probe,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: vec![Requirement::LoadedModel],
            parameters: vec![
                ParameterSpec::integer("chunk_size", "Resumable segment length", 3),
                ParameterSpec::text(
                    "fairness_chunk_sizes",
                    "Comma-separated chunk sizes for the fairness sweep",
                    "2,4,8",
                ),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let model = loaded(context)?;
        let chunk_size = parameters.size("chunk_size")?;
        let fairness_chunk_sizes = lengths(parameters, "fairness_chunk_sizes")?;
        let comparison = benchmark_chunked_prefill_execution(
            model,
            &PrefillExecutionConfig {
                prompts: vec![vec![1; 7], vec![1; 5]],
                max_new_tokens: 2,
            },
            chunk_size,
        )
        .map_err(run_error)?;
        let fairness = benchmark_prefill_fairness_trials(
            model,
            &PrefillExecutionConfig {
                prompts: vec![vec![1; 2], vec![1; 15], vec![1; 4], vec![1; 7]],
                max_new_tokens: 3,
            },
            2,
            &fairness_chunk_sizes,
            1,
            3,
        )
        .map_err(run_error)?;

        let request_iterations = |run: &PrefillFairnessRun| {
            run.requests
                .iter()
                .map(|request| {
                    format!(
                        "{}/{}",
                        request.first_token_iteration, request.completion_iteration
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let request_latency = |latencies: &[PrefillFairnessLatency]| {
            latencies
                .iter()
                .map(|latency| {
                    format!(
                        "{}:{}/{}:{}/{}",
                        latency.request_index,
                        latency.ttft.p50_micros / 1_000,
                        latency.ttft.p95_micros / 1_000,
                        latency.completion.p50_micros / 1_000,
                        latency.completion.p95_micros / 1_000
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let chunked_latencies = fairness.chunked_latency();
        let fairness_summary = fairness
            .chunked
            .iter()
            .enumerate()
            .map(|(index, runs)| {
                let run = &runs[0];
                format!(
                    "{}tok [{}] / {} iterations p50/p95 TTFT/completion ms [{}]",
                    run.chunk_size.unwrap_or_default(),
                    request_iterations(run),
                    run.iterations,
                    request_latency(&chunked_latencies[index])
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let summary = format!(
            "2 prompts (7/5 tokens) · chunk size {chunk_size} · prefill calls {} → {} · tensor calls {} → {} · iterations {} → {} · prefill {} → {} · output preserved · peak KV {} → {} · fairness (1 warm-up, 3 measured, two active) request TTFT/completion iterations baseline [{}], p50/p95 ms [{}], chunks {fairness_summary}",
            comparison.unchunked.prefill_invocations,
            comparison.chunked.prefill_invocations,
            comparison.unchunked.tensor_prefill_invocations,
            comparison.chunked.tensor_prefill_invocations,
            comparison.unchunked.iterations,
            comparison.chunked.iterations,
            format_duration(comparison.unchunked.prefill_iteration_elapsed),
            format_duration(comparison.chunked.prefill_iteration_elapsed),
            format_bytes(comparison.unchunked.peak_kv_cache_bytes),
            format_bytes(comparison.chunked.peak_kv_cache_bytes),
            request_iterations(&fairness.unchunked[0]),
            request_latency(&fairness.unchunked_latency()),
        );
        let sample = |summary: &crate::PrefillExecutionSummary| {
            values(&[
                ("prefill_invocations", as_f64(summary.prefill_invocations)),
                (
                    "tensor_prefill_invocations",
                    as_f64(summary.tensor_prefill_invocations),
                ),
                ("iterations", as_f64(summary.iterations)),
                (
                    "prefill_iteration",
                    as_f64(summary.prefill_iteration_elapsed.as_micros()),
                ),
                ("peak_kv_cache_bytes", as_f64(summary.peak_kv_cache_bytes)),
            ])
        };
        let count =
            |name| MetricSpec::new(name, Unit::Count, TimeDomain::Untimed, Direction::Neither);
        let complete_variant = Variant {
            label: "complete".into(),
            policy: None,
            kernel: comparison.unchunked.kernel.clone(),
        };
        let mut outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "chunked-prompts",
            Vec::new(),
            vec![
                count("prefill_invocations"),
                count("tensor_prefill_invocations"),
                count("iterations"),
                MetricSpec::new(
                    "prefill_iteration",
                    Unit::Micros,
                    TimeDomain::Host,
                    Direction::Lower,
                ),
                MetricSpec::new(
                    "peak_kv_cache_bytes",
                    Unit::Bytes,
                    TimeDomain::Untimed,
                    Direction::Neither,
                ),
            ],
            vec![
                (complete_variant, vec![sample(&comparison.unchunked)]),
                (
                    Variant {
                        label: "chunked".into(),
                        policy: Some(format!("prefill_chunk_size={chunk_size}")),
                        kernel: comparison.chunked.kernel.clone(),
                    },
                    vec![sample(&comparison.chunked)],
                ),
            ],
            summary,
        )?;
        outcome.report.details = Some(serde_json::json!({
            "fairness": fairness
                .chunked
                .iter()
                .zip(&chunked_latencies)
                .map(|(runs, latencies)| serde_json::json!({
                    "chunk_size": runs[0].chunk_size,
                    "iterations": runs[0].iterations,
                    "requests": runs[0].requests.iter().map(|request| serde_json::json!({
                        "request_index": request.request_index,
                        "first_token_iteration": request.first_token_iteration,
                        "completion_iteration": request.completion_iteration,
                    })).collect::<Vec<_>>(),
                    "latency_micros": latencies.iter().map(|latency| serde_json::json!({
                        "request_index": latency.request_index,
                        "ttft_p50": latency.ttft.p50_micros,
                        "ttft_p95": latency.ttft.p95_micros,
                        "completion_p50": latency.completion.p50_micros,
                        "completion_p95": latency.completion.p95_micros,
                    })).collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>(),
        }));
        Ok(outcome)
    }
}

impl ExperimentReport {
    /// Converts a legacy prefill report (schema v2) into an
    /// `llama-prefill-strategies` report with the same metrics as a live run,
    /// so old and new results can be compared.
    ///
    /// # Errors
    ///
    /// Returns [`ReportError`] if the converted report does not validate.
    pub fn from_prefill_v2(legacy: &PrefillBenchmarkReport) -> Result<Self, ReportError> {
        let protocol = &legacy.protocol;
        let study = protocol.measured_trials >= HOST_STUDY_MIN_REPETITIONS
            && protocol.warmup_trials >= HOST_STUDY_MIN_WARMUPS;
        let (workload_name, digest) = prefill_workload(&protocol.prompts, protocol.max_new_tokens);
        let mut strategies = vec![
            (
                "unbucketed",
                legacy
                    .trials
                    .iter()
                    .map(|trial| trial.unbucketed.clone())
                    .collect::<Vec<_>>(),
            ),
            (
                "power-of-two",
                legacy
                    .trials
                    .iter()
                    .map(|trial| trial.power_of_two.clone())
                    .collect(),
            ),
        ];
        if legacy.trials.iter().all(|trial| trial.cost_aware.is_some()) && !legacy.trials.is_empty()
        {
            strategies.push((
                "cost-aware",
                legacy
                    .trials
                    .iter()
                    .filter_map(|trial| trial.cost_aware.clone())
                    .collect(),
            ));
        }
        let metadata = &legacy.metadata;
        let mut parts = report_parts_from(
            ExperimentIdentity {
                id: PREFILL_STRATEGIES.to_owned(),
                payload_version: 1,
                kind: if study {
                    ExperimentKind::Study
                } else {
                    ExperimentKind::Probe
                },
                domain: TimeDomain::Host,
            },
            RunMetadata {
                recorded_at_unix_micros: metadata.recorded_at_unix_micros,
                model: Some(metadata.model.clone()),
                device: Some(metadata.device.clone()),
                build: BuildInfo {
                    crate_version: "unknown".into(),
                    revision: metadata.software_revision.clone(),
                    profile: "unknown".into(),
                    target: "unknown".into(),
                    features: Vec::new(),
                },
            },
            WorkloadIdentity {
                name: workload_name.to_owned(),
                seeds: Vec::new(),
                parameters: BTreeMap::from([
                    (
                        "prompt_lengths".to_owned(),
                        protocol
                            .prompts
                            .iter()
                            .map(|prompt| prompt.len().to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    ),
                    (
                        "max_new_tokens".to_owned(),
                        protocol.max_new_tokens.to_string(),
                    ),
                ]),
                digest,
            },
            Protocol {
                warmups: protocol.warmup_trials,
                repetitions: protocol.measured_trials,
                ordering: "rotated".into(),
            },
            protocol.adaptive_invocation_overhead_tokens.unwrap_or(0),
            strategies.into_iter(),
        );
        parts.details = Some(serde_json::json!({
            "imported_from": "prefill benchmark report schema v2",
            "strategy_order_by_trial": legacy
                .trials
                .iter()
                .map(|trial| trial.strategy_order.clone())
                .collect::<Vec<_>>(),
        }));
        Self::new(parts)
    }

    /// Reads an experiment report, or a legacy prefill report converted with
    /// [`from_prefill_v2`](Self::from_prefill_v2).
    ///
    /// # Errors
    ///
    /// Returns [`ReportError`] if the file is neither, or does not validate.
    pub fn read_compatible(path: impl AsRef<Path>) -> Result<Self, ReportError> {
        let text = fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if value.get("schema").is_some() {
            return Self::from_json(&text);
        }
        let legacy = PrefillBenchmarkReport::from_json(&text).map_err(|error| {
            ReportError::Inconsistent(format!("legacy prefill report: {error}"))
        })?;
        Self::from_prefill_v2(&legacy)
    }
}

/// Workload name and digest shared by live runs and imported reports.
fn prefill_workload(prompts: &[Vec<u32>], max_new_tokens: usize) -> (&'static str, String) {
    (
        "prefill-prompts",
        content_digest(format!("{prompts:?}/{max_new_tokens}").as_bytes()),
    )
}

fn record_kernel_names(batches: &[ola_runtime::ExecutionBatch], names: &mut BTreeSet<String>) {
    names.extend(batches.iter().filter_map(|batch| {
        batch
            .kernel
            .filter(|kernel| kernel.calls > 0)
            .map(|kernel| kernel.descriptor.name().to_owned())
    }));
}

fn joined_kernel_names(names: BTreeSet<String>) -> Option<String> {
    (!names.is_empty()).then(|| names.into_iter().collect::<Vec<_>>().join(","))
}

fn summaries_kernel_identity(samples: &[crate::PrefillExecutionSummary]) -> Option<String> {
    let names = samples
        .iter()
        .filter_map(|sample| sample.kernel.as_deref())
        .flat_map(|names| names.split(','))
        .map(str::to_owned)
        .collect();
    joined_kernel_names(names)
}

/// The metrics a prefill-strategies report declares.
fn prefill_metrics() -> Vec<MetricSpec> {
    let host = |name, unit, better| MetricSpec::new(name, unit, TimeDomain::Host, better);
    let untimed = |name, unit, better| MetricSpec::new(name, unit, TimeDomain::Untimed, better);
    vec![
        host("prefill_iteration", Unit::Micros, Direction::Lower),
        host("total", Unit::Micros, Direction::Lower),
        host("throughput", Unit::TokensPerSecond, Direction::Higher),
        untimed("padding_tokens", Unit::Tokens, Direction::Lower),
        untimed("invocations", Unit::Count, Direction::Lower),
        untimed("tensor_invocations", Unit::Count, Direction::Neither),
        untimed("iterations", Unit::Count, Direction::Neither),
        untimed("peak_kv_cache_bytes", Unit::Bytes, Direction::Neither),
        untimed("peak_reserved_kv_bytes", Unit::Bytes, Direction::Neither),
    ]
}

fn prefill_values(sample: &PrefillReportSample) -> BTreeMap<String, f64> {
    let throughput = if sample.total_micros == 0 {
        0.0
    } else {
        as_f64(sample.output_tokens) * 1_000_000.0 / as_f64(sample.total_micros)
    };
    values(&[
        ("prefill_iteration", as_f64(sample.prefill_iteration_micros)),
        ("total", as_f64(sample.total_micros)),
        ("throughput", throughput),
        ("padding_tokens", as_f64(sample.prefill_padding_tokens)),
        ("invocations", as_f64(sample.prefill_invocations)),
        (
            "tensor_invocations",
            as_f64(sample.tensor_prefill_invocations),
        ),
        ("iterations", as_f64(sample.iterations)),
        ("peak_kv_cache_bytes", as_f64(sample.peak_kv_cache_bytes)),
        (
            "peak_reserved_kv_bytes",
            as_f64(sample.peak_reserved_kv_bytes),
        ),
    ])
}

#[allow(clippy::too_many_arguments, reason = "mirrors the report envelope")]
fn report_parts(
    descriptor: &ExperimentDescriptor,
    context: &ExperimentContext,
    parameters: &Parameters,
    workload_name: &str,
    digest: String,
    protocol: Protocol,
    overhead: usize,
    strategies: impl Iterator<Item = (&'static str, Vec<PrefillReportSample>)>,
) -> ReportParts {
    report_parts_from(
        ExperimentIdentity {
            id: descriptor.id.to_owned(),
            payload_version: descriptor.payload_version,
            kind: descriptor.kind,
            domain: descriptor.domain,
        },
        RunMetadata::now(context.model_label.clone(), context.device.clone()),
        WorkloadIdentity {
            name: workload_name.to_owned(),
            seeds: Vec::new(),
            parameters: parameters.as_strings(),
            digest,
        },
        protocol,
        overhead,
        strategies,
    )
}

fn report_parts_from(
    experiment: ExperimentIdentity,
    metadata: RunMetadata,
    workload: WorkloadIdentity,
    protocol: Protocol,
    overhead: usize,
    strategies: impl Iterator<Item = (&'static str, Vec<PrefillReportSample>)>,
) -> ReportParts {
    let mut variants = Vec::new();
    let mut samples = Vec::new();
    for (label, trials) in strategies {
        let policy = match label {
            "unbucketed" => "unbucketed-prefill".to_owned(),
            "power-of-two" => "power-of-two-prefill".to_owned(),
            _ => format!("cost-aware-prefill{{invocation_overhead_tokens={overhead}}}"),
        };
        variants.push(Variant::policy(label, policy));
        samples.extend(
            trials
                .iter()
                .enumerate()
                .map(|(repetition, sample)| Sample {
                    repetition,
                    variant: label.to_owned(),
                    values: prefill_values(sample),
                }),
        );
    }
    ReportParts {
        experiment,
        metadata,
        workload,
        protocol,
        metrics: prefill_metrics(),
        variants,
        samples,
        details: None,
    }
}

fn values(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

/// Fits iteration cost on the loaded model over a grid of batch widths and
/// prompt lengths, each measured batched and split.
pub struct CostCalibration;

impl Experiment for CostCalibration {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: crate::COST_CALIBRATION,
            name: "Cost calibration",
            description: "Fit fixed, per-token, and per-call iteration cost by least squares",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: vec![Requirement::LoadedModel],
            parameters: vec![
                ParameterSpec::text("widths", "Comma-separated batch widths", "1,2,4"),
                ParameterSpec::text("lengths", "Comma-separated prompt lengths", "4,8,16"),
                ParameterSpec::integer("warmups", "Discarded grid sweeps", 1),
                ParameterSpec::integer("repetitions", "Measured grid sweeps, one fit each", 5),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let model = loaded(context)?;
        let widths = lengths(parameters, "widths")?;
        let prompt_lengths = lengths(parameters, "lengths")?;
        let sweep =
            || crate::measure_iteration_costs(model, &widths, &prompt_lengths).map_err(run_error);
        let warmups = parameters.size("warmups")?;
        for _ in 0..warmups {
            sweep()?;
        }
        let mut pooled = Vec::new();
        let mut fits = Vec::new();
        for _ in 0..parameters.integer("repetitions")? {
            let observations = sweep()?;
            fits.push(crate::fit_cost_model(&observations).map_err(run_error)?);
            pooled.extend(observations);
        }
        let pooled_fit = crate::fit_cost_model(&pooled).map_err(run_error)?;

        let micros = |name| MetricSpec::new(name, Unit::Micros, TimeDomain::Host, Direction::Lower);
        let mut outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "shape-grid",
            Vec::new(),
            vec![
                micros("fixed_micros"),
                micros("per_dense_token_micros"),
                micros("per_invocation_micros"),
                MetricSpec::new(
                    "invocation_overhead_tokens",
                    Unit::Tokens,
                    TimeDomain::Host,
                    Direction::Neither,
                ),
                micros("residual_rms_micros"),
            ],
            vec![(
                plain_variant("fit"),
                fits.iter()
                    .map(|fit| {
                        values(&[
                            ("fixed_micros", fit.fixed_micros),
                            ("per_dense_token_micros", fit.per_dense_token_micros),
                            ("per_invocation_micros", fit.per_invocation_micros),
                            (
                                "invocation_overhead_tokens",
                                fit.invocation_overhead_tokens().map_or(0.0, as_f64),
                            ),
                            ("residual_rms_micros", fit.residual_rms_micros),
                        ])
                    })
                    .collect(),
            )],
            String::new(),
        )?;
        // `finish` records single-pass probes; this study ran warm-up sweeps.
        outcome.report.protocol.warmups = warmups;
        outcome.report.protocol.ordering = "full grid sweep per repetition".into();
        let interval_json = |interval: Option<super::Interval>| {
            interval.map(|interval| serde_json::json!([interval.low, interval.high]))
        };
        let intervals = pooled_fit.intervals;
        let cost = pooled_fit.cost_model();
        outcome.report.details = Some(serde_json::json!({
            "pooled_fit": {
                "observations": pooled_fit.observations,
                "fixed_micros": pooled_fit.fixed_micros,
                "per_dense_token_micros": pooled_fit.per_dense_token_micros,
                "per_invocation_micros": pooled_fit.per_invocation_micros,
                "residual_rms_micros": pooled_fit.residual_rms_micros,
                "ci95": {
                    "fixed_micros": interval_json(intervals.map(|i| i.fixed_micros)),
                    "per_dense_token_micros": interval_json(intervals.map(|i| i.per_dense_token_micros)),
                    "per_invocation_micros": interval_json(intervals.map(|i| i.per_invocation_micros)),
                },
                "cost_model": {
                    "fixed_micros": cost.fixed_micros,
                    "per_invocation_micros": cost.per_invocation_micros,
                    "per_dense_token_micros": cost.per_dense_token_micros,
                },
                "invocation_overhead_tokens": pooled_fit.invocation_overhead_tokens(),
            },
            "observations": pooled
                .iter()
                .map(|observation| serde_json::json!([
                    observation.dense_tokens,
                    observation.invocations,
                    observation.micros,
                ]))
                .collect::<Vec<_>>(),
        }));
        let estimate = |name: &str| outcome.report.summaries[0].estimates[name];
        let describe = |name: &str| {
            let estimate = estimate(name);
            estimate.ci95.map_or_else(
                || format!("{:.1} µs", estimate.mean),
                |interval| {
                    format!(
                        "{:.1} µs [{:.1}–{:.1}]",
                        estimate.mean, interval.low, interval.high
                    )
                },
            )
        };
        outcome.summary = format!(
            "{} sweeps × {} shapes · fixed {} · per token {} · per call {} · overhead ≈ {} tokens · residual {:.1} µs",
            fits.len(),
            widths.len() * prompt_lengths.len() * 2,
            describe("fixed_micros"),
            describe("per_dense_token_micros"),
            describe("per_invocation_micros"),
            pooled_fit
                .invocation_overhead_tokens()
                .map_or_else(|| "n/a".to_owned(), |tokens| tokens.to_string()),
            pooled_fit.residual_rms_micros,
        );
        Ok(outcome)
    }
}

#[cfg(test)]
mod kernel_identity_tests {
    use super::*;

    #[test]
    fn observed_kernel_names_are_stable_and_deduplicated() {
        let names = ["z-kernel", "a-kernel", "z-kernel"]
            .into_iter()
            .map(str::to_owned)
            .collect();

        assert_eq!(
            joined_kernel_names(names).as_deref(),
            Some("a-kernel,z-kernel")
        );
        assert_eq!(joined_kernel_names(BTreeSet::new()), None);
    }
}
