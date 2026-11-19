use std::{
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    io::{self, BufReader, BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    LatencySummary, OutputCheck, PrefillExecutionConfig, PrefillExecutionSummary, PrefillStrategy,
    PrefillStrategyTrials, PrefillThreeWayComparison, PrefillThreeWayTrialConfig,
    PrefillTrialComparison, PrefillTrialConfig,
};

/// Current on-disk schema version for prefill benchmark reports.
pub const PREFILL_REPORT_SCHEMA_VERSION: u32 = 2;

/// Reproducibility metadata supplied by the benchmark caller.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillBenchmarkMetadata {
    /// Unix time in microseconds when the report was created.
    pub recorded_at_unix_micros: u64,
    /// Human-readable loaded model or artifact identity.
    pub model: String,
    /// Human-readable execution backend and device identity.
    pub device: String,
    /// Source revision, build identifier, or package version.
    pub software_revision: String,
}

/// Exact workload and repetition protocol used to produce a report.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillReportProtocol {
    /// Complete token sequences submitted as prompts.
    pub prompts: Vec<Vec<u32>>,
    /// Maximum generated tokens per request.
    pub max_new_tokens: usize,
    /// Paired trials executed and discarded before measurement.
    pub warmup_trials: usize,
    /// Paired trials retained in the report.
    pub measured_trials: usize,
    /// Token-equivalent invocation cost used by an adaptive third strategy.
    pub adaptive_invocation_overhead_tokens: Option<usize>,
}

impl From<&PrefillTrialConfig> for PrefillReportProtocol {
    fn from(config: &PrefillTrialConfig) -> Self {
        Self {
            prompts: config.execution.prompts.clone(),
            max_new_tokens: config.execution.max_new_tokens,
            warmup_trials: config.warmup_trials,
            measured_trials: config.measured_trials,
            adaptive_invocation_overhead_tokens: None,
        }
    }
}

impl From<&PrefillReportProtocol> for PrefillTrialConfig {
    fn from(protocol: &PrefillReportProtocol) -> Self {
        Self {
            execution: PrefillExecutionConfig {
                prompts: protocol.prompts.clone(),
                max_new_tokens: protocol.max_new_tokens,
                output_check: OutputCheck::Exact,
            },
            warmup_trials: protocol.warmup_trials,
            measured_trials: protocol.measured_trials,
        }
    }
}

/// Stable, JSON-friendly representation of one raw strategy execution.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillReportSample {
    /// Requests executed by the sample.
    pub requests: usize,
    /// Executor iteration boundaries required for completion.
    pub iterations: usize,
    /// Output tokens emitted across requests.
    pub output_tokens: usize,
    /// Model invocations that performed prefill.
    pub prefill_invocations: usize,
    /// Prefill invocations using a shared tensor call.
    pub tensor_prefill_invocations: usize,
    /// Prefill invocations executed individually.
    pub individual_prefill_invocations: usize,
    /// Real prompt positions processed during prefill.
    pub prefill_input_tokens: usize,
    /// Artificial right-padding positions processed during prefill.
    pub prefill_padding_tokens: usize,
    /// Complete prefill-containing iteration wall time in microseconds.
    pub prefill_iteration_micros: u64,
    /// Complete run wall time in microseconds.
    pub total_micros: u64,
    /// Highest physical KV-cache byte observation.
    pub peak_kv_cache_bytes: u64,
    /// Highest conservative KV reservation observation.
    pub peak_reserved_kv_bytes: u64,
}

impl From<&PrefillExecutionSummary> for PrefillReportSample {
    fn from(summary: &PrefillExecutionSummary) -> Self {
        Self {
            requests: summary.requests,
            iterations: summary.iterations,
            output_tokens: summary.output_tokens,
            prefill_invocations: summary.prefill_invocations,
            tensor_prefill_invocations: summary.tensor_prefill_invocations,
            individual_prefill_invocations: summary.individual_prefill_invocations,
            prefill_input_tokens: summary.prefill_input_tokens,
            prefill_padding_tokens: summary.prefill_padding_tokens,
            prefill_iteration_micros: duration_micros(summary.prefill_iteration_elapsed),
            total_micros: duration_micros(summary.total_elapsed),
            peak_kv_cache_bytes: summary.peak_kv_cache_bytes,
            peak_reserved_kv_bytes: summary.peak_reserved_kv_bytes,
        }
    }
}

/// Both strategy samples belonging to one measured paired trial.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillReportTrial {
    /// Zero-based position among retained trials.
    pub index: usize,
    /// Complete strategy order used for this trial.
    pub strategy_order: Vec<PrefillStrategy>,
    /// Raw unbucketed measurement.
    pub unbucketed: PrefillReportSample,
    /// Raw power-of-two measurement.
    pub power_of_two: PrefillReportSample,
    /// Raw adaptive measurement when this was a three-strategy report.
    pub cost_aware: Option<PrefillReportSample>,
}

/// Derived measurements for one strategy across all retained trials.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillReportStrategySummary {
    /// Prefill-containing iteration latency distribution in microseconds.
    pub prefill_iteration_latency: LatencySummary,
    /// Complete run latency distribution in microseconds.
    pub total_latency: LatencySummary,
    /// Output tokens emitted across all retained trials.
    pub aggregate_output_tokens: usize,
    /// Complete wall time across all retained trials in microseconds.
    ///
    /// Together with `aggregate_output_tokens`, this preserves the exact
    /// inputs needed to derive throughput without persisting a lossy float.
    pub aggregate_elapsed_micros: u64,
    /// Highest physical KV-cache observation across retained trials.
    pub peak_kv_cache_bytes: u64,
    /// Highest conservative KV reservation across retained trials.
    pub peak_reserved_kv_bytes: u64,
}

/// Versioned, self-contained JSON record of one repeated comparison.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillBenchmarkReport {
    /// Schema version used to decode this record.
    pub schema_version: u32,
    /// Model, device, build, and creation-time identity.
    pub metadata: PrefillBenchmarkMetadata,
    /// Exact workload and repetition protocol.
    pub protocol: PrefillReportProtocol,
    /// Raw paired measurements in execution order.
    pub trials: Vec<PrefillReportTrial>,
    /// Derived unbucketed measurements.
    pub unbucketed: PrefillReportStrategySummary,
    /// Derived power-of-two measurements.
    pub power_of_two: PrefillReportStrategySummary,
    /// Derived adaptive measurements when included by the protocol.
    pub cost_aware: Option<PrefillReportStrategySummary>,
}

impl PrefillBenchmarkReport {
    /// Builds a report after validating that configuration and samples align.
    pub fn new(
        metadata: PrefillBenchmarkMetadata,
        config: &PrefillTrialConfig,
        comparison: &PrefillTrialComparison,
    ) -> Result<Self, PrefillReportError> {
        let measured = config.measured_trials;
        if comparison.warmup_trials != config.warmup_trials
            || comparison.first_strategy_by_trial.len() != measured
            || comparison.unbucketed.samples.len() != measured
            || comparison.power_of_two.samples.len() != measured
        {
            return Err(PrefillReportError::InconsistentComparison);
        }
        let trials = (0..measured)
            .map(|index| PrefillReportTrial {
                index,
                strategy_order: vec![
                    comparison.first_strategy_by_trial[index],
                    match comparison.first_strategy_by_trial[index] {
                        PrefillStrategy::Unbucketed => PrefillStrategy::PowerOfTwo,
                        PrefillStrategy::PowerOfTwo => PrefillStrategy::Unbucketed,
                        PrefillStrategy::CostAware => {
                            unreachable!("paired reports exclude cost-aware grouping")
                        }
                    },
                ],
                unbucketed: (&comparison.unbucketed.samples[index]).into(),
                power_of_two: (&comparison.power_of_two.samples[index]).into(),
                cost_aware: None,
            })
            .collect();
        Ok(Self {
            schema_version: PREFILL_REPORT_SCHEMA_VERSION,
            metadata,
            protocol: config.into(),
            trials,
            unbucketed: strategy_summary(&comparison.unbucketed),
            power_of_two: strategy_summary(&comparison.power_of_two),
            cost_aware: None,
        })
    }

    /// Builds a three-strategy report with calibrated adaptive measurements.
    pub fn new_three_way(
        metadata: PrefillBenchmarkMetadata,
        config: &PrefillThreeWayTrialConfig,
        comparison: &PrefillThreeWayComparison,
    ) -> Result<Self, PrefillReportError> {
        let measured = config.measured_trials;
        if comparison.warmup_trials != config.warmup_trials
            || comparison.strategy_order_by_trial.len() != measured
            || comparison.unbucketed.samples.len() != measured
            || comparison.power_of_two.samples.len() != measured
            || comparison.cost_aware.samples.len() != measured
        {
            return Err(PrefillReportError::InconsistentComparison);
        }
        let trials = (0..measured)
            .map(|index| PrefillReportTrial {
                index,
                strategy_order: comparison.strategy_order_by_trial[index].to_vec(),
                unbucketed: (&comparison.unbucketed.samples[index]).into(),
                power_of_two: (&comparison.power_of_two.samples[index]).into(),
                cost_aware: Some((&comparison.cost_aware.samples[index]).into()),
            })
            .collect();
        Ok(Self {
            schema_version: PREFILL_REPORT_SCHEMA_VERSION,
            metadata,
            protocol: PrefillReportProtocol {
                prompts: config.execution.prompts.clone(),
                max_new_tokens: config.execution.max_new_tokens,
                warmup_trials: config.warmup_trials,
                measured_trials: config.measured_trials,
                adaptive_invocation_overhead_tokens: Some(
                    config.adaptive_invocation_overhead_tokens,
                ),
            },
            trials,
            unbucketed: strategy_summary(&comparison.unbucketed),
            power_of_two: strategy_summary(&comparison.power_of_two),
            cost_aware: Some(strategy_summary(&comparison.cost_aware)),
        })
    }

    /// Encodes this report as stable pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String, PrefillReportError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Decodes and validates a report from JSON text.
    pub fn from_json(json: &str) -> Result<Self, PrefillReportError> {
        let report: Self = serde_json::from_str(json)?;
        report.validate_schema()?;
        Ok(report)
    }

    /// Creates a new JSON file without replacing an existing report.
    ///
    /// The parent directory must already exist. Callers choose directory
    /// creation policy and file naming; this method guarantees create-new
    /// semantics for the final path.
    pub fn write_json_new(&self, path: impl AsRef<Path>) -> Result<(), PrefillReportError> {
        self.validate_schema()?;
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    /// Reads and validates a report from a JSON file.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, PrefillReportError> {
        let file = fs::File::open(path)?;
        let report: Self = serde_json::from_reader(BufReader::new(file))?;
        report.validate_schema()?;
        Ok(report)
    }

    fn validate_schema(&self) -> Result<(), PrefillReportError> {
        if self.schema_version != PREFILL_REPORT_SCHEMA_VERSION {
            return Err(PrefillReportError::UnsupportedSchema(self.schema_version));
        }
        if self.trials.len() != self.protocol.measured_trials
            || self
                .trials
                .iter()
                .enumerate()
                .any(|(index, trial)| trial.index != index)
        {
            return Err(PrefillReportError::InconsistentComparison);
        }
        let adaptive = self.cost_aware.is_some();
        if self.protocol.adaptive_invocation_overhead_tokens.is_some() != adaptive
            || self.trials.iter().any(|trial| {
                trial.cost_aware.is_some() != adaptive
                    || trial.strategy_order.len() != if adaptive { 3 } else { 2 }
            })
        {
            return Err(PrefillReportError::InconsistentComparison);
        }
        Ok(())
    }
}

fn strategy_summary(trials: &PrefillStrategyTrials) -> PrefillReportStrategySummary {
    PrefillReportStrategySummary {
        prefill_iteration_latency: trials.prefill_iteration_latency,
        total_latency: trials.total_latency,
        aggregate_output_tokens: aggregate_output_tokens(&trials.samples),
        aggregate_elapsed_micros: aggregate_elapsed_micros(&trials.samples),
        peak_kv_cache_bytes: trials.peak_kv_cache_bytes(),
        peak_reserved_kv_bytes: trials.peak_reserved_kv_bytes(),
    }
}

fn duration_micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn aggregate_output_tokens(samples: &[PrefillExecutionSummary]) -> usize {
    samples.iter().fold(0, |total, sample| {
        total.saturating_add(sample.output_tokens)
    })
}

fn aggregate_elapsed_micros(samples: &[PrefillExecutionSummary]) -> u64 {
    samples.iter().fold(0, |total, sample| {
        total.saturating_add(duration_micros(sample.total_elapsed))
    })
}

/// Failure to construct, encode, decode, or persist a benchmark report.
#[derive(Debug)]
pub enum PrefillReportError {
    /// Configuration and retained trial vectors do not describe the same run.
    InconsistentComparison,
    /// The record uses a schema newer or otherwise different from this build.
    UnsupportedSchema(u32),
    /// Filesystem access failed.
    Io(io::Error),
    /// JSON encoding or decoding failed.
    Json(serde_json::Error),
}

impl fmt::Display for PrefillReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InconsistentComparison => {
                formatter.write_str("prefill report configuration and samples are inconsistent")
            }
            Self::UnsupportedSchema(version) => {
                write!(
                    formatter,
                    "unsupported prefill report schema version {version}"
                )
            }
            Self::Io(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl Error for PrefillReportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for PrefillReportError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for PrefillReportError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{PrefillExecutionSummary, PrefillStrategyTrials};

    use super::*;

    #[test]
    fn report_round_trips_every_protocol_and_raw_sample_field() {
        let (config, comparison) = fixture();
        let report = PrefillBenchmarkReport::new(metadata(), &config, &comparison).unwrap();

        let json = report.to_json_pretty().unwrap();
        let decoded = PrefillBenchmarkReport::from_json(&json).unwrap();

        assert_eq!(decoded, report);
        assert_eq!(decoded.schema_version, PREFILL_REPORT_SCHEMA_VERSION);
        assert_eq!(decoded.protocol.prompts, config.execution.prompts);
        assert_eq!(
            decoded.trials[0].strategy_order,
            vec![PrefillStrategy::Unbucketed, PrefillStrategy::PowerOfTwo]
        );
        assert_eq!(decoded.trials[0].unbucketed.total_micros, 20);
        assert!(json.contains("\"power_of_two\""));
    }

    #[test]
    fn report_file_creation_never_overwrites_an_existing_result() {
        let (config, comparison) = fixture();
        let report = PrefillBenchmarkReport::new(metadata(), &config, &comparison).unwrap();
        let directory = std::env::temp_dir().join(format!(
            "ola-bench-report-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("report.json");

        report.write_json_new(&path).unwrap();
        let decoded = PrefillBenchmarkReport::read_json(&path).unwrap();
        let second = report.write_json_new(&path).unwrap_err();

        assert_eq!(decoded, report);
        assert!(
            matches!(second, PrefillReportError::Io(ref error) if error.kind() == io::ErrorKind::AlreadyExists)
        );
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn report_rejects_an_unknown_schema_version() {
        let (config, comparison) = fixture();
        let mut report = PrefillBenchmarkReport::new(metadata(), &config, &comparison).unwrap();
        report.schema_version += 1;
        let json = serde_json::to_string(&report).unwrap();

        assert!(matches!(
            PrefillBenchmarkReport::from_json(&json),
            Err(PrefillReportError::UnsupportedSchema(3))
        ));
    }

    #[test]
    fn three_way_report_retains_adaptive_protocol_and_samples() {
        let config = PrefillThreeWayTrialConfig {
            execution: PrefillExecutionConfig {
                prompts: vec![vec![1, 2], vec![3, 4, 5]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
            warmup_trials: 1,
            measured_trials: 1,
            adaptive_invocation_overhead_tokens: 10,
        };
        let comparison = PrefillThreeWayComparison {
            warmup_trials: 1,
            strategy_order_by_trial: vec![[
                PrefillStrategy::Unbucketed,
                PrefillStrategy::PowerOfTwo,
                PrefillStrategy::CostAware,
            ]],
            unbucketed: strategy(vec![sample(11, 20)]),
            power_of_two: strategy(vec![sample(7, 16)]),
            cost_aware: strategy(vec![sample(7, 17)]),
        };

        let report =
            PrefillBenchmarkReport::new_three_way(metadata(), &config, &comparison).unwrap();
        let decoded = PrefillBenchmarkReport::from_json(&report.to_json_pretty().unwrap()).unwrap();

        assert_eq!(decoded, report);
        assert_eq!(
            decoded.protocol.adaptive_invocation_overhead_tokens,
            Some(10)
        );
        assert!(decoded.trials[0].cost_aware.is_some());
        assert!(decoded.cost_aware.is_some());
    }

    #[test]
    fn legacy_reports_convert_into_comparable_experiment_reports() {
        let config = PrefillThreeWayTrialConfig {
            execution: PrefillExecutionConfig {
                prompts: vec![vec![1, 2], vec![3, 4, 5]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
            warmup_trials: 1,
            measured_trials: 2,
            adaptive_invocation_overhead_tokens: 10,
        };
        let order = [
            PrefillStrategy::Unbucketed,
            PrefillStrategy::PowerOfTwo,
            PrefillStrategy::CostAware,
        ];
        let comparison = PrefillThreeWayComparison {
            warmup_trials: 1,
            strategy_order_by_trial: vec![order, order],
            unbucketed: strategy(vec![sample(11, 20), sample(11, 22)]),
            power_of_two: strategy(vec![sample(7, 16), sample(7, 18)]),
            cost_aware: strategy(vec![sample(7, 17), sample(7, 19)]),
        };
        let legacy =
            PrefillBenchmarkReport::new_three_way(metadata(), &config, &comparison).unwrap();

        let converted = crate::ExperimentReport::from_prefill_v2(&legacy).unwrap();

        assert_eq!(converted.experiment.id, "llama-prefill-strategies");
        // Two measured trials are too few for a host-time study.
        assert_eq!(converted.experiment.kind, crate::ExperimentKind::Probe);
        assert_eq!(converted.protocol.repetitions, 2);
        assert_eq!(
            converted
                .variants
                .iter()
                .map(|variant| variant.label.as_str())
                .collect::<Vec<_>>(),
            ["unbucketed", "power-of-two", "cost-aware"]
        );
        assert_eq!(
            converted.variants[2].policy.as_deref(),
            Some("cost-aware-prefill{invocation_overhead_tokens=10}")
        );
        for (repetition, trial) in legacy.trials.iter().enumerate() {
            let sample = converted
                .samples
                .iter()
                .find(|sample| sample.variant == "unbucketed" && sample.repetition == repetition)
                .unwrap();
            assert_eq!(
                sample.values["padding_tokens"],
                trial.unbucketed.prefill_padding_tokens as f64
            );
            assert_eq!(
                sample.values["prefill_iteration"],
                trial.unbucketed.prefill_iteration_micros as f64
            );
        }
        assert_eq!(
            converted.metadata.build.revision,
            legacy.metadata.software_revision
        );

        let directory = std::env::temp_dir().join(format!(
            "ola-legacy-import-{}-{}",
            std::process::id(),
            converted.metadata.recorded_at_unix_micros
        ));
        fs::create_dir_all(&directory).unwrap();
        legacy
            .write_json_new(directory.join("legacy.json"))
            .unwrap();
        let read = crate::ExperimentReport::read_compatible(directory.join("legacy.json")).unwrap();
        assert_eq!(read, converted);
        fs::remove_dir_all(directory).unwrap();
    }

    fn metadata() -> PrefillBenchmarkMetadata {
        PrefillBenchmarkMetadata {
            recorded_at_unix_micros: 123,
            model: "fixture".into(),
            device: "cpu".into(),
            software_revision: "test".into(),
        }
    }

    fn fixture() -> (PrefillTrialConfig, PrefillTrialComparison) {
        let config = PrefillTrialConfig {
            execution: PrefillExecutionConfig {
                prompts: vec![vec![1, 2], vec![3, 4, 5]],
                max_new_tokens: 2,
                output_check: OutputCheck::Exact,
            },
            warmup_trials: 1,
            measured_trials: 2,
        };
        let unbucketed = vec![sample(11, 20), sample(13, 22)];
        let power_of_two = vec![sample(7, 16), sample(9, 18)];
        let comparison = PrefillTrialComparison {
            warmup_trials: 1,
            first_strategy_by_trial: vec![PrefillStrategy::Unbucketed, PrefillStrategy::PowerOfTwo],
            unbucketed: strategy(unbucketed),
            power_of_two: strategy(power_of_two),
        };
        (config, comparison)
    }

    fn strategy(samples: Vec<PrefillExecutionSummary>) -> PrefillStrategyTrials {
        PrefillStrategyTrials {
            prefill_iteration_latency: crate::summarize_latencies(
                &samples
                    .iter()
                    .map(|sample| sample.prefill_iteration_elapsed)
                    .collect::<Vec<_>>(),
            ),
            total_latency: crate::summarize_latencies(
                &samples
                    .iter()
                    .map(|sample| sample.total_elapsed)
                    .collect::<Vec<_>>(),
            ),
            samples,
        }
    }

    fn sample(prefill_micros: u64, total_micros: u64) -> PrefillExecutionSummary {
        PrefillExecutionSummary {
            requests: 2,
            iterations: 2,
            output_tokens: 4,
            prefill_invocations: 1,
            tensor_prefill_invocations: 1,
            individual_prefill_invocations: 0,
            prefill_input_tokens: 5,
            prefill_padding_tokens: 1,
            prefill_iteration_elapsed: Duration::from_micros(prefill_micros),
            total_elapsed: Duration::from_micros(total_micros),
            peak_kv_cache_bytes: 128,
            peak_reserved_kv_bytes: 256,
            kernel: None,
        }
    }
}
