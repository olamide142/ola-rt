//! Generic experiment identity, metrics, and versioned reports.
//!
//! See `EXPERIMENTS.md` in this crate.

mod metric;
mod report;

pub use metric::{Direction, MetricSpec, TimeDomain, Unit};
pub use report::{
    BuildInfo, EXPERIMENT_SCHEMA, Estimate, ExperimentIdentity, ExperimentKind, ExperimentReport,
    Interval, Protocol, ReportError, ReportParts, RunMetadata, Sample, Variant, VariantSummary,
    WorkloadIdentity, content_digest,
};

#[cfg(test)]
mod tests;
