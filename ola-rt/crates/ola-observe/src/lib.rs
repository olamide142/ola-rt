#![doc = include_str!("../README.md")]

use std::{collections::BTreeMap, error::Error, fmt};

use ola_core::{RequestId, Timestamp};
use ola_runtime::{
    BatchExecutionMode, BatchPhase, ExecutorEvent, ExecutorIteration, RequestRecord, RequestStatus,
    RuntimeState,
};

/// Terminal lifecycle outcome represented by one request measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestOutcome {
    /// Generation completed successfully.
    Completed,
    /// The caller cancelled queued or running work.
    Cancelled,
    /// Admission or execution failed.
    Failed,
}

/// Latency and output measurements for one terminal request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestMeasurement {
    /// Runtime-local request identifier.
    pub request_id: RequestId,
    /// Terminal lifecycle outcome.
    pub outcome: RequestOutcome,
    /// Generated tokens observed by this observer.
    pub output_tokens: usize,
    /// Time from queue insertion to execution start.
    pub queue_micros: Option<u64>,
    /// Time from queue insertion to first generated token.
    pub time_to_first_token_micros: Option<u64>,
    /// Time from execution start to first generated token.
    pub prefill_micros: Option<u64>,
    /// Time from execution start to terminal completion.
    pub service_micros: Option<u64>,
    /// Time from queue insertion to terminal completion.
    pub end_to_end_micros: u64,
    /// Time from first generated token to terminal completion.
    pub decode_span_micros: Option<u64>,
    /// Decode span divided by the number of inter-token gaps.
    pub mean_inter_token_micros: Option<u64>,
}

impl RequestMeasurement {
    /// Derives a measurement when the supplied record is terminal.
    ///
    /// Returns `None` for queued or running records. Durations are saturating
    /// because lifecycle code already enforces monotonic timestamps and an
    /// observer should not introduce arithmetic panics in telemetry paths.
    pub fn from_record(
        request_id: RequestId,
        record: &RequestRecord,
        output_tokens: usize,
    ) -> Option<Self> {
        let outcome = match &record.status {
            RequestStatus::Completed => RequestOutcome::Completed,
            RequestStatus::Cancelled => RequestOutcome::Cancelled,
            RequestStatus::Failed(_) => RequestOutcome::Failed,
            RequestStatus::Queued | RequestStatus::Running => return None,
        };
        let completed_at = record.timing.completed_at?;
        let queued_at = record.timing.queued_at;
        let started_at = record.timing.started_at;
        let first_token_at = record.timing.first_token_at;
        let decode_span_micros = first_token_at.map(|first| elapsed(completed_at, first));
        let mean_inter_token_micros = decode_span_micros.and_then(|span| {
            let gaps = output_tokens.checked_sub(1)?;
            let gaps = u64::try_from(gaps).ok()?;
            span.checked_div(gaps)
        });

        Some(Self {
            request_id,
            outcome,
            output_tokens,
            queue_micros: started_at.map(|started| elapsed(started, queued_at)),
            time_to_first_token_micros: first_token_at.map(|first| elapsed(first, queued_at)),
            prefill_micros: started_at
                .zip(first_token_at)
                .map(|(started, first)| elapsed(first, started)),
            service_micros: started_at.map(|started| elapsed(completed_at, started)),
            end_to_end_micros: elapsed(completed_at, queued_at),
            decode_span_micros,
            mean_inter_token_micros,
        })
    }
}

/// Measurements for one observed executor iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IterationMeasurement {
    /// Timestamp supplied to the executor iteration.
    pub at: Timestamp,
    /// Time since the preceding observed iteration, absent for the first.
    pub elapsed_micros: Option<u64>,
    /// Input token positions selected by batch policy.
    pub scheduled_input_tokens: usize,
    /// Selected prompt operations.
    pub scheduled_prefills: usize,
    /// Selected cached decode operations.
    pub scheduled_decodes: usize,
    /// Operations left for a later iteration by batch policy.
    pub batch_deferred_operations: usize,
    /// Model invocations that packed more than one request into tensors.
    pub tensor_batches: usize,
    /// Request operations executed through tensor batches.
    pub tensor_batched_operations: usize,
    /// Request operations executed through individual model invocations.
    pub individual_operations: usize,
    /// Meaningful input positions submitted to model invocations.
    pub executed_input_tokens: usize,
    /// Artificial input positions computed because prefill rows were padded.
    pub padding_input_tokens: usize,
    /// Generated tokens emitted during the iteration.
    pub emitted_output_tokens: usize,
    /// Successfully completed requests.
    pub completed_requests: usize,
    /// Requests terminated by execution failure.
    pub failed_requests: usize,
    /// Requests permanently rejected by admission.
    pub rejected_requests: usize,
    /// Temporary admission-policy deferrals.
    pub admission_deferrals: usize,
    /// Live sessions after the iteration.
    pub active_requests: usize,
    /// Queued requests after the iteration.
    pub queued_requests: usize,
    /// Physical KV bytes attributed to live sessions.
    pub kv_cache_bytes: u64,
    /// Worst-case KV bytes committed to live requests.
    pub reserved_kv_bytes: u64,
}

impl IterationMeasurement {
    /// Returns selected input-token positions per second over the observation
    /// interval, or `None` for the first/zero-duration interval.
    pub fn input_tokens_per_second(self) -> Option<f64> {
        rate(self.scheduled_input_tokens, self.elapsed_micros?)
    }

    /// Returns emitted output tokens per second over the observation interval,
    /// or `None` for the first/zero-duration interval.
    pub fn output_tokens_per_second(self) -> Option<f64> {
        rate(self.emitted_output_tokens, self.elapsed_micros?)
    }

    /// Returns dense model-input positions per second, including padding.
    pub fn dense_input_tokens_per_second(self) -> Option<f64> {
        rate(
            self.executed_input_tokens
                .saturating_add(self.padding_input_tokens),
            self.elapsed_micros?,
        )
    }

    /// Returns the fraction of dense input positions carrying real tokens.
    pub fn input_token_efficiency(self) -> Option<f64> {
        let dense = self
            .executed_input_tokens
            .saturating_add(self.padding_input_tokens);
        if dense == 0 {
            None
        } else {
            Some(self.executed_input_tokens as f64 / dense as f64)
        }
    }
}

/// Cumulative counters across every iteration accepted by one observer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ObservationTotals {
    /// Number of observed iterations.
    pub iterations: usize,
    /// Selected input token positions.
    pub scheduled_input_tokens: usize,
    /// Emitted output tokens.
    pub emitted_output_tokens: usize,
    /// Successfully completed requests.
    pub completed_requests: usize,
    /// Execution failures.
    pub failed_requests: usize,
    /// Permanent admission rejections.
    pub rejected_requests: usize,
    /// Temporary admission deferrals.
    pub admission_deferrals: usize,
    /// Batch-policy deferrals.
    pub batch_deferrals: usize,
    /// Model invocations reported as tensorized.
    pub tensor_batches: usize,
    /// Request operations executed through tensor batches.
    pub tensor_batched_operations: usize,
    /// Request operations executed individually.
    pub individual_operations: usize,
    /// Meaningful input positions submitted to model invocations.
    pub executed_input_tokens: usize,
    /// Artificial prefill positions computed due to right padding.
    pub padding_input_tokens: usize,
}

/// Stateful collector for deterministic executor observations.
#[derive(Clone, Debug, Default)]
pub struct ExecutionObserver {
    previous_iteration_at: Option<Timestamp>,
    observed_output_tokens: BTreeMap<RequestId, usize>,
    terminal_requests: BTreeMap<RequestId, RequestMeasurement>,
    totals: ObservationTotals,
}

impl ExecutionObserver {
    /// Observes one completed iteration and its corresponding runtime state.
    ///
    /// Calls must follow nondecreasing iteration timestamps. A generated-token
    /// count is taken from each event's cumulative request count, so attaching
    /// an observer after generation has started does not undercount that
    /// request's terminal output.
    pub fn observe_iteration(
        &mut self,
        iteration: &ExecutorIteration,
        state: &RuntimeState,
    ) -> Result<IterationMeasurement, ObservationError> {
        if let Some(previous) = self.previous_iteration_at
            && iteration.at < previous
        {
            return Err(ObservationError::TimestampRegression {
                previous,
                observed: iteration.at,
            });
        }
        let elapsed_micros = self
            .previous_iteration_at
            .map(|previous| elapsed(iteration.at, previous));
        let mut measurement = IterationMeasurement {
            at: iteration.at,
            elapsed_micros,
            scheduled_input_tokens: iteration.plan.scheduled_tokens(),
            scheduled_prefills: iteration
                .plan
                .scheduled
                .iter()
                .filter(|candidate| candidate.phase == BatchPhase::Prefill)
                .count(),
            scheduled_decodes: iteration
                .plan
                .scheduled
                .iter()
                .filter(|candidate| candidate.phase == BatchPhase::Decode)
                .count(),
            batch_deferred_operations: iteration.plan.deferred.len(),
            tensor_batches: iteration
                .execution_batches
                .iter()
                .filter(|batch| batch.mode == BatchExecutionMode::Tensor)
                .count(),
            tensor_batched_operations: iteration
                .execution_batches
                .iter()
                .filter(|batch| batch.mode == BatchExecutionMode::Tensor)
                .map(|batch| batch.request_ids.len())
                .sum(),
            individual_operations: iteration
                .execution_batches
                .iter()
                .filter(|batch| batch.mode == BatchExecutionMode::Individual)
                .map(|batch| batch.request_ids.len())
                .sum(),
            executed_input_tokens: iteration
                .execution_batches
                .iter()
                .map(|batch| batch.input_tokens)
                .sum(),
            padding_input_tokens: iteration
                .execution_batches
                .iter()
                .map(|batch| batch.padding_tokens)
                .sum(),
            emitted_output_tokens: 0,
            completed_requests: 0,
            failed_requests: 0,
            rejected_requests: 0,
            admission_deferrals: 0,
            active_requests: iteration.active_requests,
            queued_requests: iteration.queued_requests,
            kv_cache_bytes: iteration.kv_cache_bytes,
            reserved_kv_bytes: iteration.reserved_kv_bytes,
        };
        let mut output_updates: BTreeMap<RequestId, usize> = BTreeMap::new();
        let mut terminal = Vec::new();
        for event in &iteration.events {
            match event {
                ExecutorEvent::Token {
                    request_id,
                    generated_tokens,
                    finished,
                    ..
                } => {
                    measurement.emitted_output_tokens += 1;
                    output_updates
                        .entry(*request_id)
                        .and_modify(|tokens| *tokens = (*tokens).max(*generated_tokens))
                        .or_insert(*generated_tokens);
                    if *finished {
                        measurement.completed_requests += 1;
                        terminal.push(*request_id);
                    }
                }
                ExecutorEvent::Failed { request_id, .. } => {
                    measurement.failed_requests += 1;
                    terminal.push(*request_id);
                }
                ExecutorEvent::Rejected { request_id, .. } => {
                    measurement.rejected_requests += 1;
                    terminal.push(*request_id);
                }
                ExecutorEvent::Deferred { .. } => measurement.admission_deferrals += 1,
                ExecutorEvent::Idle
                | ExecutorEvent::PrefillProgress { .. }
                | ExecutorEvent::BatchDeferred { .. } => {}
            }
        }
        let mut terminal_measurements = Vec::with_capacity(terminal.len());
        for request_id in terminal {
            let record = state
                .request(request_id)
                .ok_or(ObservationError::UnknownRequest(request_id))?;
            let previous_tokens = self
                .observed_output_tokens
                .get(&request_id)
                .copied()
                .unwrap_or(0);
            let output_tokens = output_updates
                .get(&request_id)
                .copied()
                .unwrap_or(previous_tokens)
                .max(previous_tokens);
            let request_measurement =
                RequestMeasurement::from_record(request_id, record, output_tokens)
                    .ok_or(ObservationError::RequestNotTerminal(request_id))?;
            terminal_measurements.push(request_measurement);
        }
        for (request_id, generated_tokens) in output_updates {
            self.observed_output_tokens
                .entry(request_id)
                .and_modify(|tokens| *tokens = (*tokens).max(generated_tokens))
                .or_insert(generated_tokens);
        }
        for request_measurement in terminal_measurements {
            self.terminal_requests
                .insert(request_measurement.request_id, request_measurement);
        }

        self.previous_iteration_at = Some(iteration.at);
        self.totals.iterations += 1;
        self.totals.scheduled_input_tokens += measurement.scheduled_input_tokens;
        self.totals.emitted_output_tokens += measurement.emitted_output_tokens;
        self.totals.completed_requests += measurement.completed_requests;
        self.totals.failed_requests += measurement.failed_requests;
        self.totals.rejected_requests += measurement.rejected_requests;
        self.totals.admission_deferrals += measurement.admission_deferrals;
        self.totals.batch_deferrals += measurement.batch_deferred_operations;
        self.totals.tensor_batches += measurement.tensor_batches;
        self.totals.tensor_batched_operations += measurement.tensor_batched_operations;
        self.totals.individual_operations += measurement.individual_operations;
        self.totals.executed_input_tokens += measurement.executed_input_tokens;
        self.totals.padding_input_tokens += measurement.padding_input_tokens;
        Ok(measurement)
    }

    /// Captures a terminal request not surfaced by an iteration event, such as
    /// explicit cancellation.
    pub fn observe_terminal_request(
        &mut self,
        request_id: RequestId,
        state: &RuntimeState,
    ) -> Result<RequestMeasurement, ObservationError> {
        let record = state
            .request(request_id)
            .ok_or(ObservationError::UnknownRequest(request_id))?;
        let output_tokens = self
            .observed_output_tokens
            .get(&request_id)
            .copied()
            .unwrap_or(0);
        let measurement = RequestMeasurement::from_record(request_id, record, output_tokens)
            .ok_or(ObservationError::RequestNotTerminal(request_id))?;
        self.terminal_requests.insert(request_id, measurement);
        Ok(measurement)
    }

    /// Returns the retained terminal measurement for one request.
    pub fn request(&self, request_id: RequestId) -> Option<&RequestMeasurement> {
        self.terminal_requests.get(&request_id)
    }

    /// Iterates retained terminal request measurements in request-ID order.
    pub fn requests(&self) -> impl ExactSizeIterator<Item = &RequestMeasurement> {
        self.terminal_requests.values()
    }

    /// Returns cumulative counters for accepted iterations.
    pub const fn totals(&self) -> ObservationTotals {
        self.totals
    }
}

/// Invalid input supplied to an execution observer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationError {
    /// An iteration preceded the observer's previous iteration timestamp.
    TimestampRegression {
        /// Previously observed iteration time.
        previous: Timestamp,
        /// Regressive timestamp supplied by the caller.
        observed: Timestamp,
    },
    /// Runtime state did not retain an event's request.
    UnknownRequest(RequestId),
    /// A caller attempted to capture a queued or running request as terminal.
    RequestNotTerminal(RequestId),
}

impl fmt::Display for ObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimestampRegression { previous, observed } => write!(
                formatter,
                "observation timestamp regressed from {previous} to {observed}"
            ),
            Self::UnknownRequest(request) => {
                write!(
                    formatter,
                    "observation references unknown request `{request}`"
                )
            }
            Self::RequestNotTerminal(request) => {
                write!(formatter, "request `{request}` is not terminal")
            }
        }
    }
}

impl Error for ObservationError {}

fn elapsed(later: Timestamp, earlier: Timestamp) -> u64 {
    later.as_micros().saturating_sub(earlier.as_micros())
}

fn rate(units: usize, elapsed_micros: u64) -> Option<f64> {
    if elapsed_micros == 0 {
        return None;
    }
    Some(units as f64 * 1_000_000.0 / elapsed_micros as f64)
}

#[cfg(test)]
mod tests {
    use ola_core::{ModelDescriptor, RequestSpec};
    use ola_model::GeneratedToken;
    use ola_runtime::{
        BatchCandidate, ExecutionPhase, ExecutionPlan, FifoScheduler, IterationPlan, Runtime,
    };

    use super::*;

    #[test]
    fn derives_documented_request_latencies_from_lifecycle_boundaries() {
        let (runtime, request_id) = completed_runtime();
        let measurement = RequestMeasurement::from_record(
            request_id,
            runtime.state().request(request_id).unwrap(),
            3,
        )
        .unwrap();

        assert_eq!(measurement.outcome, RequestOutcome::Completed);
        assert_eq!(measurement.queue_micros, Some(10));
        assert_eq!(measurement.time_to_first_token_micros, Some(20));
        assert_eq!(measurement.prefill_micros, Some(10));
        assert_eq!(measurement.service_micros, Some(50));
        assert_eq!(measurement.end_to_end_micros, 60);
        assert_eq!(measurement.decode_span_micros, Some(40));
        assert_eq!(measurement.mean_inter_token_micros, Some(20));
        assert_eq!(
            RequestMeasurement::from_record(
                request_id,
                runtime.state().request(request_id).unwrap(),
                1,
            )
            .unwrap()
            .mean_inter_token_micros,
            None
        );
    }

    #[test]
    fn observes_iteration_work_rates_totals_and_terminal_output() {
        let (runtime, request_id) = completed_runtime();
        let iteration = ExecutorIteration {
            at: Timestamp::from_micros(70),
            events: vec![ExecutorEvent::Token {
                request_id,
                token: GeneratedToken {
                    id: 3,
                    is_end_of_sequence: false,
                },
                generated_tokens: 3,
                kv_cache_bytes: 64,
                phase: ExecutionPhase::Decode,
                finished: true,
            }],
            execution_plan: ExecutionPlan::default(),
            plan: IterationPlan {
                scheduled: vec![BatchCandidate {
                    request_id,
                    phase: BatchPhase::Decode,
                    token_cost: 1,
                }],
                deferred: Vec::new(),
            },
            execution_batches: vec![ola_runtime::ExecutionBatch {
                phase: BatchPhase::Decode,
                request_ids: vec![request_id],
                mode: BatchExecutionMode::Individual,
                input_tokens: 1,
                padding_tokens: 0,
            }],
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        let mut observer = ExecutionObserver::default();

        let first = observer
            .observe_iteration(&iteration, runtime.state())
            .unwrap();
        assert_eq!(first.elapsed_micros, None);
        assert_eq!(first.output_tokens_per_second(), None);
        assert_eq!(observer.request(request_id).unwrap().output_tokens, 3);

        let later = ExecutorIteration {
            at: Timestamp::from_micros(170),
            events: vec![ExecutorEvent::Idle],
            plan: IterationPlan::default(),
            execution_plan: ExecutionPlan::default(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        let second = observer.observe_iteration(&later, runtime.state()).unwrap();
        assert_eq!(second.elapsed_micros, Some(100));
        assert_eq!(second.input_tokens_per_second(), Some(0.0));
        assert_eq!(
            observer.totals(),
            ObservationTotals {
                iterations: 2,
                scheduled_input_tokens: 1,
                emitted_output_tokens: 1,
                completed_requests: 1,
                individual_operations: 1,
                executed_input_tokens: 1,
                ..ObservationTotals::default()
            }
        );
    }

    #[test]
    fn rejects_regressive_observation_time_without_updating_totals() {
        let runtime = Runtime::<FifoScheduler>::default();
        let mut observer = ExecutionObserver::default();
        let iteration = |at| ExecutorIteration {
            at: Timestamp::from_micros(at),
            events: vec![ExecutorEvent::Idle],
            plan: IterationPlan::default(),
            execution_plan: ExecutionPlan::default(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        observer
            .observe_iteration(&iteration(20), runtime.state())
            .unwrap();

        let error = observer
            .observe_iteration(&iteration(19), runtime.state())
            .unwrap_err();

        assert_eq!(
            error,
            ObservationError::TimestampRegression {
                previous: Timestamp::from_micros(20),
                observed: Timestamp::from_micros(19),
            }
        );
        assert_eq!(observer.totals().iterations, 1);
    }

    #[test]
    fn observes_dense_prefill_padding_and_efficiency() {
        let runtime = Runtime::<FifoScheduler>::default();
        let iteration = ExecutorIteration {
            at: Timestamp::from_micros(10),
            events: Vec::new(),
            plan: IterationPlan::default(),
            execution_plan: ExecutionPlan::default(),
            execution_batches: vec![ola_runtime::ExecutionBatch {
                phase: BatchPhase::Prefill,
                request_ids: vec![RequestId::new(1), RequestId::new(2)],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 13,
                padding_tokens: 3,
            }],
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        let mut observer = ExecutionObserver::default();

        let measurement = observer
            .observe_iteration(&iteration, runtime.state())
            .unwrap();

        assert_eq!(measurement.executed_input_tokens, 13);
        assert_eq!(measurement.padding_input_tokens, 3);
        assert_eq!(measurement.input_token_efficiency(), Some(13.0 / 16.0));
        assert_eq!(measurement.dense_input_tokens_per_second(), None);
        assert_eq!(observer.totals().executed_input_tokens, 13);
        assert_eq!(observer.totals().padding_input_tokens, 3);
    }

    #[test]
    fn invalid_terminal_event_does_not_partially_update_observer_state() {
        let runtime = Runtime::<FifoScheduler>::default();
        let request_id = RequestId::new(99);
        let iteration = ExecutorIteration {
            at: Timestamp::from_micros(10),
            events: vec![ExecutorEvent::Token {
                request_id,
                token: GeneratedToken {
                    id: 1,
                    is_end_of_sequence: false,
                },
                generated_tokens: 4,
                kv_cache_bytes: 0,
                phase: ExecutionPhase::Decode,
                finished: true,
            }],
            plan: IterationPlan::default(),
            execution_plan: ExecutionPlan::default(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        let mut observer = ExecutionObserver::default();

        let error = observer
            .observe_iteration(&iteration, runtime.state())
            .unwrap_err();

        assert_eq!(error, ObservationError::UnknownRequest(request_id));
        assert_eq!(observer.totals(), ObservationTotals::default());
        assert!(observer.observed_output_tokens.is_empty());
        assert!(observer.terminal_requests.is_empty());
    }

    fn completed_runtime() -> (Runtime<FifoScheduler>, RequestId) {
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "test"))
            .unwrap();
        let request_id = runtime
            .submit(
                RequestSpec::text("tiny", "hello"),
                Timestamp::from_micros(10),
            )
            .unwrap();
        runtime.schedule_next(Timestamp::from_micros(20)).unwrap();
        runtime
            .mark_first_token(request_id, Timestamp::from_micros(30))
            .unwrap();
        runtime
            .complete(request_id, Timestamp::from_micros(70))
            .unwrap();
        (runtime, request_id)
    }
}
