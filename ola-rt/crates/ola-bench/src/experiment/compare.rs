//! Comparing two reports of the same experiment.

use std::{collections::BTreeMap, error::Error, fmt};

use super::{Direction, Estimate, ExperimentReport, MetricSpec};

/// Options controlling when a difference counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompareOptions {
    /// Minimum relative change of the mean to classify a metric as improved
    /// or regressed.
    pub relative_threshold: f64,
    /// Compare reports whose workload digests differ.
    pub allow_workload_mismatch: bool,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            relative_threshold: 0.05,
            allow_workload_mismatch: false,
        }
    }
}

/// Verdict for one metric of one variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Change {
    /// Better beyond the threshold, with non-overlapping 95% intervals.
    Improved,
    /// Worse beyond the threshold, with non-overlapping 95% intervals.
    Regressed,
    /// Intervals overlap, are missing (one repetition), or the change is
    /// below the threshold.
    Inconclusive,
    /// The metric has no direction of improvement.
    Informational,
}

/// One metric of one variant in both reports.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricComparison {
    /// Variant label.
    pub variant: String,
    /// Metric declaration.
    pub metric: MetricSpec,
    /// Baseline estimate.
    pub baseline: Estimate,
    /// Candidate estimate.
    pub candidate: Estimate,
    /// Candidate mean minus baseline mean.
    pub delta: f64,
    /// `delta` relative to the baseline mean, when that is non-zero.
    pub relative: Option<f64>,
    /// Verdict.
    pub change: Change,
}

/// An identity field that differs between the two reports.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityDifference {
    /// Field, such as `model` or `variant fifo policy`.
    pub field: String,
    /// Baseline value.
    pub baseline: String,
    /// Candidate value.
    pub candidate: String,
}

/// Everything that differs between two reports of one experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportComparison {
    /// Experiment ID.
    pub experiment: String,
    /// What changed between the runs.
    pub identity_differences: Vec<IdentityDifference>,
    /// Per-variant, per-metric verdicts, in baseline order.
    pub metrics: Vec<MetricComparison>,
    /// Variant or metric names present in only one report, or declared
    /// differently.
    pub unmatched: Vec<String>,
}

impl ReportComparison {
    /// Returns the metrics that regressed.
    pub fn regressions(&self) -> impl Iterator<Item = &MetricComparison> {
        self.metrics
            .iter()
            .filter(|metric| metric.change == Change::Regressed)
    }
}

/// Reports that cannot be meaningfully compared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompareError {
    /// The reports come from different experiments.
    DifferentExperiment {
        /// Baseline experiment ID.
        baseline: String,
        /// Candidate experiment ID.
        candidate: String,
    },
    /// The same experiment at different payload versions.
    DifferentPayloadVersion {
        /// Baseline version.
        baseline: u32,
        /// Candidate version.
        candidate: u32,
    },
    /// The reports measure different clocks.
    DifferentDomain,
    /// The workloads differ and the options do not allow it.
    DifferentWorkload {
        /// Baseline digest.
        baseline: String,
        /// Candidate digest.
        candidate: String,
    },
}

impl fmt::Display for CompareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentExperiment {
                baseline,
                candidate,
            } => write!(
                formatter,
                "cannot compare experiment `{baseline}` with `{candidate}`"
            ),
            Self::DifferentPayloadVersion {
                baseline,
                candidate,
            } => write!(
                formatter,
                "payload versions differ ({baseline} vs {candidate})"
            ),
            Self::DifferentDomain => formatter.write_str("reports measure different time domains"),
            Self::DifferentWorkload {
                baseline,
                candidate,
            } => write!(
                formatter,
                "workloads differ ({baseline} vs {candidate}); allow the mismatch to compare anyway"
            ),
        }
    }
}

impl Error for CompareError {}

/// Compares `candidate` against `baseline`.
///
/// # Errors
///
/// Returns [`CompareError`] when the reports are not comparable.
pub fn compare_reports(
    baseline: &ExperimentReport,
    candidate: &ExperimentReport,
    options: CompareOptions,
) -> Result<ReportComparison, CompareError> {
    let (base, cand) = (&baseline.experiment, &candidate.experiment);
    if base.id != cand.id {
        return Err(CompareError::DifferentExperiment {
            baseline: base.id.clone(),
            candidate: cand.id.clone(),
        });
    }
    if base.payload_version != cand.payload_version {
        return Err(CompareError::DifferentPayloadVersion {
            baseline: base.payload_version,
            candidate: cand.payload_version,
        });
    }
    if base.domain != cand.domain {
        return Err(CompareError::DifferentDomain);
    }
    let workload_differs = baseline.workload.digest != candidate.workload.digest;
    if workload_differs && !options.allow_workload_mismatch {
        return Err(CompareError::DifferentWorkload {
            baseline: baseline.workload.digest.clone(),
            candidate: candidate.workload.digest.clone(),
        });
    }

    let mut identity_differences = Vec::new();
    let mut differ = |field: &str, left: String, right: String| {
        if left != right {
            identity_differences.push(IdentityDifference {
                field: field.to_owned(),
                baseline: left,
                candidate: right,
            });
        }
    };
    let optional = |value: &Option<String>| value.clone().unwrap_or_else(|| "none".into());
    let (left, right) = (&baseline.metadata, &candidate.metadata);
    differ("model", optional(&left.model), optional(&right.model));
    differ("device", optional(&left.device), optional(&right.device));
    differ(
        "revision",
        left.build.revision.clone(),
        right.build.revision.clone(),
    );
    differ(
        "profile",
        left.build.profile.clone(),
        right.build.profile.clone(),
    );
    differ(
        "target",
        left.build.target.clone(),
        right.build.target.clone(),
    );
    differ(
        "features",
        left.build.features.join(","),
        right.build.features.join(","),
    );
    differ(
        "protocol",
        format!("{:?}", baseline.protocol),
        format!("{:?}", candidate.protocol),
    );
    if workload_differs {
        differ(
            "workload",
            baseline.workload.digest.clone(),
            candidate.workload.digest.clone(),
        );
    }
    for variant in &baseline.variants {
        if let Some(other) = candidate
            .variants
            .iter()
            .find(|other| other.label == variant.label)
        {
            differ(
                &format!("variant {} policy", variant.label),
                optional(&variant.policy),
                optional(&other.policy),
            );
            differ(
                &format!("variant {} kernel", variant.label),
                optional(&variant.kernel),
                optional(&other.kernel),
            );
        }
    }

    let candidate_metrics = candidate
        .metrics
        .iter()
        .map(|metric| (metric.name.as_str(), metric))
        .collect::<BTreeMap<_, _>>();
    let mut unmatched = Vec::new();
    for metric in &baseline.metrics {
        match candidate_metrics.get(metric.name.as_str()) {
            Some(other) if *other == metric => {}
            Some(_) => unmatched.push(format!("metric {} (declared differently)", metric.name)),
            None => unmatched.push(format!("metric {} (baseline only)", metric.name)),
        }
    }
    for metric in &candidate.metrics {
        if !baseline.metrics.iter().any(|base| base.name == metric.name) {
            unmatched.push(format!("metric {} (candidate only)", metric.name));
        }
    }
    for variant in &candidate.variants {
        if baseline.summary(&variant.label).is_none() {
            unmatched.push(format!("variant {} (candidate only)", variant.label));
        }
    }

    let mut metrics = Vec::new();
    for summary in &baseline.summaries {
        let Some(other) = candidate.summary(&summary.variant) else {
            unmatched.push(format!("variant {} (baseline only)", summary.variant));
            continue;
        };
        for metric in &baseline.metrics {
            if candidate_metrics.get(metric.name.as_str()) != Some(&metric) {
                continue;
            }
            let (Some(before), Some(after)) = (
                summary.estimates.get(&metric.name),
                other.estimates.get(&metric.name),
            ) else {
                continue;
            };
            metrics.push(classify(&summary.variant, metric, *before, *after, options));
        }
    }

    Ok(ReportComparison {
        experiment: base.id.clone(),
        identity_differences,
        metrics,
        unmatched,
    })
}

fn classify(
    variant: &str,
    metric: &MetricSpec,
    baseline: Estimate,
    candidate: Estimate,
    options: CompareOptions,
) -> MetricComparison {
    let delta = candidate.mean - baseline.mean;
    let relative = (baseline.mean != 0.0).then(|| delta / baseline.mean.abs());
    let change = match metric.better {
        Direction::Neither => Change::Informational,
        better => {
            let separated = match (baseline.ci95, candidate.ci95) {
                (Some(before), Some(after)) => after.low > before.high || after.high < before.low,
                _ => false,
            };
            let large =
                relative.is_some_and(|relative| relative.abs() >= options.relative_threshold);
            if separated && large {
                let increased = delta > 0.0;
                if increased == (better == Direction::Higher) {
                    Change::Improved
                } else {
                    Change::Regressed
                }
            } else {
                Change::Inconclusive
            }
        }
    };
    MetricComparison {
        variant: variant.to_owned(),
        metric: metric.clone(),
        baseline,
        candidate,
        delta,
        relative,
        change,
    }
}
