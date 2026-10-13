#![doc = include_str!("../README.md")]

use std::{error::Error, fmt, time::Duration};

use ola_core::{Input, ModelId, RequestId, RequestSpec, Timestamp};
use ola_runtime::{
    PowerOfTwoPrefillBatcher, PrefillBatcher, Runtime, RuntimeError, Scheduler,
    UnbucketedPrefillBatcher,
};

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

/// Deterministic compute-shape summary for one prefill grouping strategy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefillBatchSummary {
    /// Prompt rows represented by the trace.
    pub requests: usize,
    /// Model invocations produced after grouping.
    pub invocations: usize,
    /// Invocations containing at least two prompt rows.
    pub tensor_invocations: usize,
    /// Single-row invocations using the individual execution path.
    pub individual_invocations: usize,
    /// Real prompt positions across every invocation.
    pub input_tokens: usize,
    /// Artificial right-padding positions across every invocation.
    pub padding_tokens: usize,
}

impl PrefillBatchSummary {
    /// Returns real plus artificial dense input positions.
    pub fn dense_input_tokens(self) -> usize {
        self.input_tokens.saturating_add(self.padding_tokens)
    }

    /// Returns the fraction of dense positions carrying real prompt tokens.
    pub fn input_token_efficiency(self) -> Option<f64> {
        let dense = self.dense_input_tokens();
        if dense == 0 {
            None
        } else {
            Some(self.input_tokens as f64 / dense as f64)
        }
    }
}

/// Side-by-side deterministic shape comparison for the built-in strategies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillBatchComparison {
    /// All prompts grouped together after model compatibility.
    pub unbucketed: PrefillBatchSummary,
    /// Prompts grouped by next-power-of-two length.
    pub power_of_two: PrefillBatchSummary,
}

impl PrefillBatchComparison {
    /// Returns padding positions avoided by power-of-two grouping.
    pub fn padding_tokens_saved(self) -> usize {
        self.unbucketed
            .padding_tokens
            .saturating_sub(self.power_of_two.padding_tokens)
    }

    /// Returns the fraction of baseline padding avoided by bucketing.
    pub fn padding_reduction(self) -> Option<f64> {
        if self.unbucketed.padding_tokens == 0 {
            None
        } else {
            Some(self.padding_tokens_saved() as f64 / self.unbucketed.padding_tokens as f64)
        }
    }
}

/// Analyzes one same-model prompt-length trace with a caller-supplied grouper.
///
/// This measures deterministic tensor shape, not elapsed performance. Groups
/// retain first-seen bucket order, matching the executor's stable grouping
/// semantics. Every prompt length must be positive.
pub fn summarize_prefill_batches(
    prompt_lengths: &[usize],
    batcher: &mut dyn PrefillBatcher,
) -> Result<PrefillBatchSummary, WorkloadError> {
    if prompt_lengths.is_empty() {
        return Err(WorkloadError::NoPromptLengths);
    }
    if prompt_lengths.contains(&0) {
        return Err(WorkloadError::EmptyPrompt);
    }

    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for &prompt_tokens in prompt_lengths {
        let bucket = batcher.bucket(prompt_tokens);
        if let Some((_, lengths)) = groups.iter_mut().find(|(key, _)| *key == bucket) {
            lengths.push(prompt_tokens);
        } else {
            groups.push((bucket, vec![prompt_tokens]));
        }
    }

    let mut summary = PrefillBatchSummary {
        requests: prompt_lengths.len(),
        invocations: groups.len(),
        ..PrefillBatchSummary::default()
    };
    for (_, lengths) in groups {
        if lengths.len() > 1 {
            summary.tensor_invocations += 1;
        } else {
            summary.individual_invocations += 1;
        }
        let input_tokens = lengths.iter().try_fold(0_usize, |total, length| {
            total
                .checked_add(*length)
                .ok_or(WorkloadError::TokenCountOverflow)
        })?;
        let padded_length = lengths.iter().copied().max().unwrap_or(0);
        let dense_tokens = padded_length
            .checked_mul(lengths.len())
            .ok_or(WorkloadError::TokenCountOverflow)?;
        summary.input_tokens = summary
            .input_tokens
            .checked_add(input_tokens)
            .ok_or(WorkloadError::TokenCountOverflow)?;
        summary.padding_tokens = summary
            .padding_tokens
            .checked_add(dense_tokens - input_tokens)
            .ok_or(WorkloadError::TokenCountOverflow)?;
    }
    Ok(summary)
}

/// Compares the unbucketed baseline with power-of-two prompt grouping.
pub fn compare_prefill_batching(
    prompt_lengths: &[usize],
) -> Result<PrefillBatchComparison, WorkloadError> {
    let unbucketed = summarize_prefill_batches(prompt_lengths, &mut UnbucketedPrefillBatcher)?;
    let power_of_two = summarize_prefill_batches(prompt_lengths, &mut PowerOfTwoPrefillBatcher)?;
    Ok(PrefillBatchComparison {
        unbucketed,
        power_of_two,
    })
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
    /// A prefill-shape trace must include at least one prompt.
    NoPromptLengths,
    /// Model prefill does not accept a zero-token prompt.
    EmptyPrompt,
    /// Prompt shape arithmetic exceeded the platform token-count range.
    TokenCountOverflow,
}

impl fmt::Display for WorkloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NoRequests => "workload must contain at least one request",
            Self::NoOutputTokens => "workload requests must generate at least one token",
            Self::EventsOutOfOrder => "workload events must be ordered by arrival time",
            Self::TimestampOverflow => "workload timestamps exceed the supported range",
            Self::NoPromptLengths => "prefill workload must contain at least one prompt",
            Self::EmptyPrompt => "prefill workload prompts must contain at least one token",
            Self::TokenCountOverflow => "prefill workload token counts overflowed",
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

    #[test]
    fn compares_prefill_padding_without_claiming_elapsed_performance() {
        let comparison = compare_prefill_batching(&[5, 9, 7]).unwrap();

        assert_eq!(
            comparison.unbucketed,
            PrefillBatchSummary {
                requests: 3,
                invocations: 1,
                tensor_invocations: 1,
                individual_invocations: 0,
                input_tokens: 21,
                padding_tokens: 6,
            }
        );
        assert_eq!(
            comparison.power_of_two,
            PrefillBatchSummary {
                requests: 3,
                invocations: 2,
                tensor_invocations: 1,
                individual_invocations: 1,
                input_tokens: 21,
                padding_tokens: 2,
            }
        );
        assert_eq!(comparison.padding_tokens_saved(), 4);
        assert_eq!(comparison.padding_reduction(), Some(2.0 / 3.0));
        assert_eq!(
            comparison.power_of_two.input_token_efficiency(),
            Some(21.0 / 23.0)
        );
    }

    #[test]
    fn rejects_empty_or_zero_length_prefill_traces() {
        assert_eq!(
            compare_prefill_batching(&[]),
            Err(WorkloadError::NoPromptLengths)
        );
        assert_eq!(
            compare_prefill_batching(&[4, 0]),
            Err(WorkloadError::EmptyPrompt)
        );
    }
}
