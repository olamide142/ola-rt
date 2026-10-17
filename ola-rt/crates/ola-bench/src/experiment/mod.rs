//! Generic experiment identity, metrics, and versioned reports.
//!
//! See `EXPERIMENTS.md` in this crate.

mod builtin;
mod compare;
mod metric;
mod model;
mod registry;
mod report;
mod statistics;

pub use builtin::{
    AdaptivePrefillCost, FifoWorkload, KvBlockPressure, PrefillBucketing, SchedulerComparison,
};
pub use compare::{
    Change, CompareError, CompareOptions, IdentityDifference, MetricComparison, ReportComparison,
    compare_reports,
};
pub use metric::{Direction, MetricSpec, TimeDomain, Unit};
pub use model::{ChunkedPrefill, CostCalibration, LlamaGeneration, PrefillStrategies};
pub use registry::{
    Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor, ExperimentError,
    ExperimentOutcome, ExperimentRegistry, ExperimentTrace, HOST_STUDY_MIN_REPETITIONS,
    HOST_STUDY_MIN_WARMUPS, ParameterSpec, ParameterValue, Parameters, Requirement,
    STUDY_MIN_REPETITIONS,
};
pub use report::{
    BuildInfo, EXPERIMENT_SCHEMA, Estimate, ExperimentIdentity, ExperimentKind, ExperimentReport,
    Interval, Protocol, ReportError, ReportParts, RunMetadata, Sample, Variant, VariantSummary,
    WorkloadIdentity, content_digest,
};

pub use statistics::BOOTSTRAP_RESAMPLES;

#[cfg(test)]
mod tests;
