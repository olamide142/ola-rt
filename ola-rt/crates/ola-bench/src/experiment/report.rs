//! The versioned experiment report envelope.
//!
//! See `EXPERIMENTS.md` in this crate. A report keeps raw samples beside the
//! summaries derived from them. Validation recomputes the summaries, so a
//! report whose derived numbers disagree with its samples is rejected.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    io::{self, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use super::{MetricSpec, TimeDomain};

/// Schema identifier written into every report.
pub const EXPERIMENT_SCHEMA: &str = "ola-experiment/1";

/// Whether a result may support a claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentKind {
    /// An operational check. Not for publication.
    Probe,
    /// A protocol-conforming measurement with repetitions and intervals.
    Study,
}

/// Which experiment produced a report.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ExperimentIdentity {
    /// Stable experiment ID, such as `scheduler-comparison`.
    pub id: String,
    /// Version of this experiment's metrics and parameters.
    pub payload_version: u32,
    /// Probe or study.
    pub kind: ExperimentKind,
    /// Clock its timed metrics come from.
    pub domain: TimeDomain,
}

/// Compile-time identity of the binary that ran the experiment.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct BuildInfo {
    /// `ola-bench` crate version.
    pub crate_version: String,
    /// Git revision, `OLA_REVISION`, or `unknown`.
    pub revision: String,
    /// Cargo profile.
    pub profile: String,
    /// Target triple.
    pub target: String,
    /// Enabled `ola-bench` features.
    pub features: Vec<String>,
}

impl BuildInfo {
    /// Returns the identity captured when this crate was built.
    pub fn current() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            revision: env!("OLA_BUILD_REVISION").to_owned(),
            profile: env!("OLA_BUILD_PROFILE").to_owned(),
            target: env!("OLA_BUILD_TARGET").to_owned(),
            features: env!("OLA_BUILD_FEATURES")
                .split(',')
                .filter(|feature| !feature.is_empty())
                .map(str::to_owned)
                .collect(),
        }
    }
}

/// When, where, and with what an experiment ran.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct RunMetadata {
    /// Unix time in microseconds when the report was created.
    pub recorded_at_unix_micros: u64,
    /// Loaded model identity, if the experiment used one.
    pub model: Option<String>,
    /// Backend and device identity, if relevant.
    pub device: Option<String>,
    /// Build identity.
    pub build: BuildInfo,
}

impl RunMetadata {
    /// Stamps the current time and build.
    pub fn now(model: Option<String>, device: Option<String>) -> Self {
        Self {
            recorded_at_unix_micros: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| {
                    u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
                }),
            model,
            device,
            build: BuildInfo::current(),
        }
    }
}

/// What workload the experiment measured.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct WorkloadIdentity {
    /// Workload or trace generator name.
    pub name: String,
    /// Seeds used, one per repetition for virtual-time studies.
    pub seeds: Vec<u64>,
    /// Generator parameters as stable strings.
    pub parameters: BTreeMap<String, String>,
    /// Digest of the workload content; see [`content_digest`].
    pub digest: String,
}

/// How samples were collected.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Protocol {
    /// Discarded runs before measurement.
    pub warmups: usize,
    /// Measured repetitions per variant.
    pub repetitions: usize,
    /// How variant order was arranged, such as `rotated`.
    pub ordering: String,
}

/// One configuration under test.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Variant {
    /// Unique label within the report.
    pub label: String,
    /// Policy descriptor, when a policy defines the variant.
    pub policy: Option<String>,
    /// Kernel identity, when a kernel defines the variant.
    pub kernel: Option<String>,
}

impl Variant {
    /// Declares a variant defined by a policy descriptor.
    pub fn policy(label: impl Into<String>, policy: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            policy: Some(policy.into()),
            kernel: None,
        }
    }
}

/// Raw metric values from one repetition of one variant.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Sample {
    /// Zero-based repetition.
    pub repetition: usize,
    /// Variant label.
    pub variant: String,
    /// Value of every declared metric.
    pub values: BTreeMap<String, f64>,
}

/// A 95% interval for a mean.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct Interval {
    /// Lower bound.
    pub low: f64,
    /// Upper bound.
    pub high: f64,
}

/// Distribution of one metric across repetitions.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct Estimate {
    /// Repetitions.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Nearest-rank median.
    pub p50: f64,
    /// Nearest-rank 95th percentile.
    pub p95: f64,
    /// Minimum.
    pub min: f64,
    /// Maximum.
    pub max: f64,
    /// 95% interval for the mean, when computed.
    pub ci95: Option<Interval>,
}

impl Estimate {
    /// Summarizes values; `None` when there are none.
    pub fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = |percent: usize| {
            let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
            sorted[index.min(sorted.len() - 1)]
        };
        Some(Self {
            n: sorted.len(),
            mean: sorted.iter().sum::<f64>() / count(sorted.len()),
            p50: rank(50),
            p95: rank(95),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            ci95: None,
        })
    }
}

/// Derived estimates for one variant.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct VariantSummary {
    /// Variant label.
    pub variant: String,
    /// Estimate per metric name.
    pub estimates: BTreeMap<String, Estimate>,
}

/// One experiment run: identity, raw samples, and derived summaries.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ExperimentReport {
    /// Always [`EXPERIMENT_SCHEMA`].
    pub schema: String,
    /// Which experiment ran.
    pub experiment: ExperimentIdentity,
    /// When, where, and with what.
    pub metadata: RunMetadata,
    /// What workload.
    pub workload: WorkloadIdentity,
    /// How samples were collected.
    pub protocol: Protocol,
    /// Declared metrics.
    pub metrics: Vec<MetricSpec>,
    /// Configurations under test, in display order.
    pub variants: Vec<Variant>,
    /// Raw samples.
    pub samples: Vec<Sample>,
    /// Summaries derived from `samples`, in variant order.
    pub summaries: Vec<VariantSummary>,
    /// Experiment-specific extras; never needed to compare reports.
    pub details: Option<serde_json::Value>,
}

/// The parts of a report a caller provides; summaries are derived.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportParts {
    /// Which experiment ran.
    pub experiment: ExperimentIdentity,
    /// When, where, and with what.
    pub metadata: RunMetadata,
    /// What workload.
    pub workload: WorkloadIdentity,
    /// How samples were collected.
    pub protocol: Protocol,
    /// Declared metrics.
    pub metrics: Vec<MetricSpec>,
    /// Configurations under test.
    pub variants: Vec<Variant>,
    /// Raw samples.
    pub samples: Vec<Sample>,
    /// Experiment-specific extras.
    pub details: Option<serde_json::Value>,
}

impl ExperimentReport {
    /// Derives summaries from the samples and validates the whole report.
    ///
    /// # Errors
    ///
    /// Returns [`ReportError::Inconsistent`] if the parts do not describe a
    /// complete, well-formed run.
    pub fn new(parts: ReportParts) -> Result<Self, ReportError> {
        let summaries = summarize(&parts.variants, &parts.metrics, &parts.samples);
        let report = Self {
            schema: EXPERIMENT_SCHEMA.to_owned(),
            experiment: parts.experiment,
            metadata: parts.metadata,
            workload: parts.workload,
            protocol: parts.protocol,
            metrics: parts.metrics,
            variants: parts.variants,
            samples: parts.samples,
            summaries,
            details: parts.details,
        };
        report.validate()?;
        Ok(report)
    }

    /// Checks the schema, structural consistency, and that every summary is
    /// exactly what the samples imply.
    ///
    /// # Errors
    ///
    /// Returns the first problem found.
    pub fn validate(&self) -> Result<(), ReportError> {
        if self.schema != EXPERIMENT_SCHEMA {
            return Err(ReportError::UnsupportedSchema(self.schema.clone()));
        }
        let inconsistent = |message: String| Err(ReportError::Inconsistent(message));
        let metric_names = unique(self.metrics.iter().map(|metric| metric.name.as_str()))
            .map_err(|name| ReportError::Inconsistent(format!("metric `{name}` declared twice")))?;
        let labels = unique(self.variants.iter().map(|variant| variant.label.as_str())).map_err(
            |label| ReportError::Inconsistent(format!("variant `{label}` declared twice")),
        )?;
        if metric_names.is_empty() || labels.is_empty() {
            return inconsistent("a report needs at least one metric and one variant".into());
        }
        if self.protocol.repetitions == 0 {
            return inconsistent("a report needs at least one repetition".into());
        }
        let mut seen = BTreeSet::new();
        for sample in &self.samples {
            if !labels.contains(sample.variant.as_str()) {
                return inconsistent(format!("sample for unknown variant `{}`", sample.variant));
            }
            if sample.repetition >= self.protocol.repetitions {
                return inconsistent(format!(
                    "repetition {} exceeds the protocol's {}",
                    sample.repetition, self.protocol.repetitions
                ));
            }
            if !seen.insert((sample.variant.as_str(), sample.repetition)) {
                return inconsistent(format!(
                    "variant `{}` has repetition {} twice",
                    sample.variant, sample.repetition
                ));
            }
            let keys = sample
                .values
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if keys != metric_names {
                return inconsistent(format!(
                    "variant `{}` repetition {} does not report exactly the declared metrics",
                    sample.variant, sample.repetition
                ));
            }
            if let Some((name, _)) = sample.values.iter().find(|(_, value)| !value.is_finite()) {
                return inconsistent(format!("metric `{name}` has a non-finite value"));
            }
        }
        let expected = labels.len() * self.protocol.repetitions;
        if seen.len() != expected {
            return inconsistent(format!(
                "expected {expected} samples (every variant × repetition), found {}",
                seen.len()
            ));
        }
        if self.summaries != summarize(&self.variants, &self.metrics, &self.samples) {
            return inconsistent("summaries do not match the samples they derive from".into());
        }
        Ok(())
    }

    /// Encodes the report as human-readable JSON.
    ///
    /// # Errors
    ///
    /// Returns [`ReportError::Json`] if encoding fails.
    pub fn to_json_pretty(&self) -> Result<String, ReportError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Decodes and validates a report.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed JSON, an unknown schema, or an
    /// inconsistent report.
    pub fn from_json(json: &str) -> Result<Self, ReportError> {
        let report: Self = serde_json::from_str(json)?;
        report.validate()?;
        Ok(report)
    }

    /// Writes the report into `directory` under a new, unique file name and
    /// returns its path. Never overwrites an existing file.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or written.
    pub fn write_new(&self, directory: impl AsRef<Path>) -> Result<PathBuf, ReportError> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let stem = format!(
            "{}-{}",
            file_safe(&self.experiment.id),
            self.metadata.recorded_at_unix_micros
        );
        for attempt in 0..1_000 {
            let path = directory.join(format!("{stem}-{attempt}.json"));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    let mut writer = BufWriter::new(file);
                    serde_json::to_writer_pretty(&mut writer, self)?;
                    writer.write_all(b"\n")?;
                    writer.flush()?;
                    return Ok(path);
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(ReportError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a unique report file name",
        )))
    }

    /// Reads and validates a report file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or does not validate.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, ReportError> {
        let report: Self = serde_json::from_reader(BufReader::new(fs::File::open(path)?))?;
        report.validate()?;
        Ok(report)
    }

    /// Returns the summary for a variant label.
    pub fn summary(&self, variant: &str) -> Option<&VariantSummary> {
        self.summaries
            .iter()
            .find(|summary| summary.variant == variant)
    }
}

/// Stable 64-bit FNV-1a digest of workload content, as 16 hex digits.
pub fn content_digest(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("{hash:016x}")
}

fn summarize(
    variants: &[Variant],
    metrics: &[MetricSpec],
    samples: &[Sample],
) -> Vec<VariantSummary> {
    variants
        .iter()
        .map(|variant| {
            let mut ordered = samples
                .iter()
                .filter(|sample| sample.variant == variant.label)
                .collect::<Vec<_>>();
            ordered.sort_by_key(|sample| sample.repetition);
            let estimates = metrics
                .iter()
                .filter_map(|metric| {
                    let values = ordered
                        .iter()
                        .filter_map(|sample| sample.values.get(&metric.name).copied())
                        .collect::<Vec<_>>();
                    Estimate::of(&values).map(|estimate| (metric.name.clone(), estimate))
                })
                .collect();
            VariantSummary {
                variant: variant.label.clone(),
                estimates,
            }
        })
        .collect()
}

fn unique<'a>(names: impl Iterator<Item = &'a str>) -> Result<BTreeSet<&'a str>, String> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(name.to_owned());
        }
    }
    Ok(seen)
}

fn file_safe(id: &str) -> String {
    id.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[allow(clippy::cast_precision_loss, reason = "repetition counts are tiny")]
fn count(value: usize) -> f64 {
    value as f64
}

/// Failure to build, encode, decode, or persist a report.
#[derive(Debug)]
pub enum ReportError {
    /// The schema identifier is not one this build reads.
    UnsupportedSchema(String),
    /// The report does not describe a complete, consistent run.
    Inconsistent(String),
    /// Filesystem access failed.
    Io(io::Error),
    /// JSON encoding or decoding failed.
    Json(serde_json::Error),
}

impl fmt::Display for ReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(schema) => {
                write!(formatter, "unsupported experiment report schema `{schema}`")
            }
            Self::Inconsistent(message) => write!(formatter, "inconsistent report: {message}"),
            Self::Io(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl Error for ReportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ReportError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ReportError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
