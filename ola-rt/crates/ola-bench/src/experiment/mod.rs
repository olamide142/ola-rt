//! Generic experiment identity, metrics, and versioned reports.
//!
//! See `EXPERIMENTS.md` in this crate.

mod compare;
mod metric;
mod report;
mod statistics;

pub use compare::{
    Change, CompareError, CompareOptions, IdentityDifference, MetricComparison, ReportComparison,
    compare_reports,
};
pub use metric::{Direction, MetricSpec, TimeDomain, Unit};
pub use report::{
    BuildInfo, EXPERIMENT_SCHEMA, Estimate, ExperimentIdentity, ExperimentKind, ExperimentReport,
    Interval, Protocol, ReportError, ReportParts, RunMetadata, Sample, Variant, VariantSummary,
    WorkloadIdentity, content_digest,
};

pub use statistics::BOOTSTRAP_RESAMPLES;

#[cfg(test)]
mod tests;
