//! What a measured number means.

use serde::{Deserialize, Serialize};

/// Unit of a metric value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// Microseconds.
    Micros,
    /// Tokens.
    Tokens,
    /// Bytes.
    Bytes,
    /// A count of events or items.
    Count,
    /// A dimensionless ratio.
    Ratio,
    /// Tokens per second.
    TokensPerSecond,
}

/// Which clock a metric's time comes from. Comparisons never mix domains.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeDomain {
    /// Deterministic simulated time; exactly reproducible on any machine.
    Virtual,
    /// Host wall-clock time; noisy and machine specific.
    Host,
    /// No time involved, such as counts or padding.
    Untimed,
}

/// Which direction of change is an improvement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Smaller is better, such as latency.
    Lower,
    /// Larger is better, such as throughput.
    Higher,
    /// Informational only; never a regression.
    Neither,
}

/// Declared meaning of one named metric in a report.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct MetricSpec {
    /// Stable snake_case name, such as `completion_p95`.
    pub name: String,
    /// Unit of every value.
    pub unit: Unit,
    /// Clock the value comes from.
    pub domain: TimeDomain,
    /// Direction of improvement.
    pub better: Direction,
}

impl MetricSpec {
    /// Declares a metric.
    pub fn new(name: impl Into<String>, unit: Unit, domain: TimeDomain, better: Direction) -> Self {
        Self {
            name: name.into(),
            unit,
            domain,
            better,
        }
    }
}
