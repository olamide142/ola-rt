#![doc = include_str!("../README.md")]

use std::{error::Error, fmt, time::Duration};

use ola_core::{Input, ModelId, RequestId, RequestSpec, Timestamp};
use ola_runtime::{Runtime, RuntimeError, Scheduler};

#[derive(Clone, Debug, PartialEq)]
/// One request arrival in a workload trace.
pub struct WorkloadEvent {
    /// Arrival time relative to the trace epoch.
    pub at: Timestamp,
    /// Request submitted at that time.
    pub request: RequestSpec,
}

#[derive(Clone, Debug, Default, PartialEq)]
/// Arrival-ordered, replayable request trace.
pub struct Workload {
    events: Vec<WorkloadEvent>,
}

impl Workload {
    /// Creates a trace after validating nondecreasing arrival order.
    pub fn new(events: Vec<WorkloadEvent>) -> Result<Self, WorkloadError> {
        if events.windows(2).any(|window| window[0].at > window[1].at) {
            return Err(WorkloadError::EventsOutOfOrder);
        }
        Ok(Self { events })
    }

    /// Returns trace events in arrival order.
    pub fn events(&self) -> &[WorkloadEvent] {
        &self.events
    }

    /// Submits every event using its recorded arrival timestamp.
    ///
    /// Replay performs admission only; the experiment remains responsible for
    /// advancing scheduling and execution.
    pub fn replay<S: Scheduler>(
        &self,
        runtime: &mut Runtime<S>,
    ) -> Result<Vec<RequestId>, RuntimeError> {
        self.events
            .iter()
            .map(|event| runtime.submit(event.request.clone(), event.at))
            .collect()
    }
}

#[derive(Clone, Debug)]
/// Deterministic workload with identical request shapes and constant arrivals.
pub struct FixedWorkload {
    /// Registered model used by every request.
    pub model: ModelId,
    /// Number of arrivals in the trace.
    pub requests: usize,
    /// Token count supplied to every request.
    pub prompt_tokens: usize,
    /// Generation limit supplied to every request.
    pub output_tokens: usize,
    /// Constant time between adjacent arrivals.
    pub interarrival: Duration,
}

impl FixedWorkload {
    /// Materializes the fixed trace, rejecting empty or overflowing workloads.
    pub fn build(&self) -> Result<Workload, WorkloadError> {
        if self.requests == 0 {
            return Err(WorkloadError::NoRequests);
        }
        if self.output_tokens == 0 {
            return Err(WorkloadError::NoOutputTokens);
        }

        let interval = u64::try_from(self.interarrival.as_micros())
            .map_err(|_| WorkloadError::TimestampOverflow)?;
        let events = (0..self.requests)
            .map(|index| {
                let index = u64::try_from(index).map_err(|_| WorkloadError::TimestampOverflow)?;
                let at = interval
                    .checked_mul(index)
                    .ok_or(WorkloadError::TimestampOverflow)?;
                let mut request = RequestSpec::text(self.model.clone(), "");
                request.inputs = vec![Input::Tokens(vec![0; self.prompt_tokens])];
                request.generation.max_new_tokens = self.output_tokens;
                Ok(WorkloadEvent {
                    at: Timestamp::from_micros(at),
                    request,
                })
            })
            .collect::<Result<Vec<_>, WorkloadError>>()?;
        Workload::new(events)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Nearest-rank latency distribution summary in microseconds.
pub struct LatencySummary {
    /// Number of observed durations.
    pub samples: usize,
    /// Minimum duration.
    pub min_micros: u64,
    /// 50th percentile duration.
    pub p50_micros: u64,
    /// 95th percentile duration.
    pub p95_micros: u64,
    /// 99th percentile duration.
    pub p99_micros: u64,
    /// Maximum duration.
    pub max_micros: u64,
}

/// Summarizes durations using sorted nearest-rank percentiles.
///
/// An empty slice returns an all-zero summary with zero samples. Durations that
/// exceed `u64` microseconds saturate to `u64::MAX`.
pub fn summarize_latencies(samples: &[Duration]) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary::default();
    }
    let mut micros = samples
        .iter()
        .map(|sample| u64::try_from(sample.as_micros()).unwrap_or(u64::MAX))
        .collect::<Vec<_>>();
    micros.sort_unstable();

    LatencySummary {
        samples: micros.len(),
        min_micros: micros[0],
        p50_micros: percentile(&micros, 50),
        p95_micros: percentile(&micros, 95),
        p99_micros: percentile(&micros, 99),
        max_micros: micros[micros.len() - 1],
    }
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Invalid workload definition.
pub enum WorkloadError {
    /// Generator was configured with zero arrivals.
    NoRequests,
    /// Generator was configured with a zero output-token limit.
    NoOutputTokens,
    /// Explicit trace arrival times decrease.
    EventsOutOfOrder,
    /// Arrival time cannot be represented by `Timestamp`.
    TimestampOverflow,
}

impl fmt::Display for WorkloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NoRequests => "workload must contain at least one request",
            Self::NoOutputTokens => "workload requests must generate at least one token",
            Self::EventsOutOfOrder => "workload events must be ordered by arrival time",
            Self::TimestampOverflow => "workload timestamps exceed the supported range",
        };
        formatter.write_str(message)
    }
}

impl Error for WorkloadError {}

#[cfg(test)]
mod tests {
    use ola_core::ModelDescriptor;

    use super::*;

    #[test]
    fn fixed_workload_replays_deterministically() {
        let workload = FixedWorkload {
            model: "tiny".into(),
            requests: 3,
            prompt_tokens: 4,
            output_tokens: 2,
            interarrival: Duration::from_micros(10),
        }
        .build()
        .unwrap();
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "tiny-decoder"))
            .unwrap();

        let ids = workload.replay(&mut runtime).unwrap();

        assert_eq!(
            ids.iter().map(|id| id.get()).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(workload.events()[2].at, Timestamp::from_micros(20));
    }

    #[test]
    fn summarizes_tail_latency_with_nearest_rank_percentiles() {
        let samples = [1, 2, 3, 4, 100].map(Duration::from_micros);

        let summary = summarize_latencies(&samples);

        assert_eq!(summary.p50_micros, 3);
        assert_eq!(summary.p95_micros, 100);
        assert_eq!(summary.p99_micros, 100);
    }
}
