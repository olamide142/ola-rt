//! Per-request traces assembled from executor iterations.
//!
//! A [`Tracer`] reads each [`ExecutorIteration`] and the runtime state after
//! it, never mutating either, and returns [`TraceRecord`]s: one stream per
//! request covering arrival, every scheduling decision about it, every model
//! invocation it took part in, every token, and its terminal outcome. Records
//! carry the request and iteration IDs, so they join with plan reports and
//! with each other. They serialize as JSON Lines, one record per line.

use std::{
    collections::BTreeSet,
    io::{self, BufRead, Write},
};

use ola_core::{RequestId, Timestamp};
use ola_runtime::{
    AdmissionDecision, AdmissionOutcome, BatchExecutionMode, BatchPhase, ExecutionPhase,
    ExecutorEvent, ExecutorIteration, IterationId, RuntimeState, SelectionStage,
};
use serde::{Deserialize, Serialize};

/// One fact about one request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TraceRecord {
    /// Independent executor run the record belongs to, when one output
    /// holds several (for example one per policy and seed in an
    /// experiment). Request IDs are unique only within a run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    /// Request the record is about.
    pub request: u64,
    /// Iteration that produced the record; absent for facts outside any
    /// iteration, such as a cancellation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration: Option<u64>,
    /// When the fact became true, in microseconds of the runtime's epoch:
    /// arrival time, planning time for decisions, invocation start, token
    /// time, or terminal time.
    pub at_us: u64,
    /// What happened.
    #[serde(flatten)]
    pub event: TraceEvent,
}

/// What a trace record says.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TraceEvent {
    /// The request was queued.
    Arrived {
        /// Model it targets.
        model: String,
        /// Output-token limit.
        max_new_tokens: usize,
        /// Relative priority.
        priority: i32,
        /// Absolute deadline, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline_us: Option<u64>,
    },
    /// The scheduler ranked the request for admission.
    Ranked {
        /// Zero-based position in the ranking.
        rank: usize,
        /// Number of requests ranked.
        of: usize,
    },
    /// What admission decided for the ranked request.
    Admission {
        /// `admitted`, `deferred`, `rejected`, `no_slot`, or
        /// `preparation_failed`.
        outcome: String,
        /// Why, for deferrals, rejections, and preparation failures.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// KV bytes reserved by an admission.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reserved_kv_bytes: Option<u64>,
        /// Whether an admitted request's first prefill fit the token budget.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        budget_selected: Option<bool>,
    },
    /// Backfill admitted later-ranked work past this deferred request.
    BackfillSkipped,
    /// The batch policy considered the request's next operation.
    Selection {
        /// `decode`, `continued_prefill`, or `admission`.
        stage: String,
        /// Whether it was selected for this iteration.
        selected: bool,
        /// Input positions the operation would process.
        token_cost: usize,
    },
    /// The request took part in a model invocation.
    Invocation {
        /// Index of the invocation within its iteration.
        batch: usize,
        /// `prefill` or `decode`.
        phase: String,
        /// `tensor` (shared with `rows - 1` others) or `individual`.
        mode: String,
        /// Requests in the invocation.
        rows: usize,
        /// Meaningful input positions across all rows.
        input_tokens: usize,
        /// Padding positions across all rows.
        padding_tokens: usize,
        /// When the invocation finished; see `ExecutionBatch::finished_at`.
        finished_us: u64,
    },
    /// A prompt segment was processed without selecting a token.
    PrefillProgress {
        /// Prompt tokens processed so far.
        processed_tokens: usize,
        /// Prompt length.
        total_tokens: usize,
    },
    /// A token was produced.
    Token {
        /// One-based position in the output.
        index: usize,
        /// Selected token ID.
        token_id: u32,
        /// `prefill` or `decode`.
        phase: String,
    },
    /// Generation finished successfully.
    Completed,
    /// Execution failed.
    Failed {
        /// Diagnostic.
        message: String,
    },
    /// Admission determined the request can never fit.
    Rejected {
        /// Why.
        reason: String,
    },
    /// The request was cancelled.
    Cancelled,
    /// The request was returned to the queue to free KV blocks. It resumes
    /// later, prefilling its prompt and output together; its next `token`
    /// continues the output.
    Preempted {
        /// Why, such as `kv_shortfall`.
        reason: String,
        /// Blocks the pool was short before preemption.
        shortfall_blocks: usize,
        /// Output tokens kept for the resume.
        generated_tokens: usize,
    },
}

impl TraceEvent {
    /// Returns whether no record follows this one for its request.
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Rejected { .. } | Self::Cancelled
        )
    }
}

/// Builds request traces from executor iterations.
///
/// Feed every iteration, in order, together with the runtime state right
/// after it and before any terminal record is forgotten; report
/// cancellations, which happen outside iterations, with
/// [`observe_cancellation`](Self::observe_cancellation). The tracer only
/// remembers which unfinished requests it has announced.
#[derive(Clone, Debug, Default)]
pub struct Tracer {
    announced: BTreeSet<RequestId>,
}

impl Tracer {
    /// Returns the records produced by one iteration, in order: arrivals,
    /// ranking, admission, backfill, and selection decisions; then each
    /// invocation in execution order followed by what it produced for each
    /// member; then terminal outcomes decided at planning time.
    pub fn observe_iteration(
        &mut self,
        iteration: &ExecutorIteration,
        state: &RuntimeState,
    ) -> Vec<TraceRecord> {
        let mut records = Vec::new();
        let id = iteration.id;
        let planned = iteration.at;
        let push = |records: &mut Vec<TraceRecord>, request, at: Timestamp, event| {
            records.push(record(request, Some(id), at, event));
        };

        let decision = &iteration.decision;
        let ranked = &decision.ranking.ranked;
        for (rank, request) in ranked.iter().enumerate() {
            self.announce(*request, state, &mut records);
            push(
                &mut records,
                *request,
                planned,
                TraceEvent::Ranked {
                    rank,
                    of: ranked.len(),
                },
            );
        }
        for admission in &decision.admissions {
            self.announce(admission.request_id, state, &mut records);
            push(
                &mut records,
                admission.request_id,
                planned,
                admission_event(&admission.outcome),
            );
        }
        for request in &decision.backfill_skips {
            push(&mut records, *request, planned, TraceEvent::BackfillSkipped);
        }
        for selection in &decision.selections {
            for candidate in &selection.context.candidates {
                self.announce(candidate.request_id, state, &mut records);
                push(
                    &mut records,
                    candidate.request_id,
                    planned,
                    TraceEvent::Selection {
                        stage: stage(selection.stage).into(),
                        selected: selection.selected.contains(&candidate.request_id),
                        token_cost: candidate.token_cost,
                    },
                );
            }
        }

        // Each request runs in at most one invocation per iteration, so an
        // event belongs to the invocation containing its request, if any.
        let batch_of = |request: RequestId| {
            iteration
                .execution_batches
                .iter()
                .position(|batch| batch.request_ids.contains(&request))
        };
        let mut terminal = Vec::new();
        for event in &iteration.events {
            if let Some(request) = event_request(event)
                && batch_of(request).is_none()
            {
                // Decided before any invocation: rejections and failures to
                // start.
                self.announce(request, state, &mut records);
                self.produced(event, planned, id, &mut records, &mut terminal);
            }
        }
        for (index, batch) in iteration.execution_batches.iter().enumerate() {
            for request in &batch.request_ids {
                self.announce(*request, state, &mut records);
                push(
                    &mut records,
                    *request,
                    batch.started_at,
                    TraceEvent::Invocation {
                        batch: index,
                        phase: match batch.phase {
                            BatchPhase::Prefill => "prefill",
                            BatchPhase::Decode => "decode",
                        }
                        .into(),
                        mode: match batch.mode {
                            BatchExecutionMode::Tensor => "tensor",
                            BatchExecutionMode::Individual => "individual",
                        }
                        .into(),
                        rows: batch.request_ids.len(),
                        input_tokens: batch.input_tokens,
                        padding_tokens: batch.padding_tokens,
                        finished_us: batch.finished_at.as_micros(),
                    },
                );
                for event in &iteration.events {
                    if event_request(event) == Some(*request) {
                        self.produced(event, batch.finished_at, id, &mut records, &mut terminal);
                    }
                }
            }
        }
        for request in terminal {
            self.announced.remove(&request);
        }
        records
    }

    /// Returns the records for a cancellation, which happens between
    /// iterations, including the arrival if the request was never traced.
    pub fn observe_cancellation(
        &mut self,
        request: RequestId,
        state: &RuntimeState,
    ) -> Vec<TraceRecord> {
        let mut records = Vec::new();
        self.announce(request, state, &mut records);
        let at = state
            .request(request)
            .and_then(|record| record.timing.completed_at)
            .unwrap_or_default();
        records.push(record(request, None, at, TraceEvent::Cancelled));
        self.announced.remove(&request);
        records
    }

    fn announce(
        &mut self,
        request: RequestId,
        state: &RuntimeState,
        records: &mut Vec<TraceRecord>,
    ) {
        if !self.announced.insert(request) {
            return;
        }
        let Some(record_) = state.request(request) else {
            return;
        };
        let spec = &record_.request.spec;
        records.push(record(
            request,
            None,
            record_.timing.queued_at,
            TraceEvent::Arrived {
                model: spec.model.to_string(),
                max_new_tokens: spec.generation.max_new_tokens,
                priority: spec.priority.get(),
                deadline_us: spec.deadline.map(Timestamp::as_micros),
            },
        ));
    }

    /// Appends what one executor event says, stamped at `at` unless the
    /// event carries its own time.
    fn produced(
        &self,
        event: &ExecutorEvent,
        at: Timestamp,
        iteration: IterationId,
        records: &mut Vec<TraceRecord>,
        terminal: &mut Vec<RequestId>,
    ) {
        let mut push =
            |request, at, event| records.push(record(request, Some(iteration), at, event));
        match event {
            ExecutorEvent::Token {
                request_id,
                token,
                generated_tokens,
                phase,
                finished,
                at,
                ..
            } => {
                push(
                    *request_id,
                    *at,
                    TraceEvent::Token {
                        index: *generated_tokens,
                        token_id: token.id,
                        phase: match phase {
                            ExecutionPhase::Prefill => "prefill",
                            ExecutionPhase::Decode => "decode",
                        }
                        .into(),
                    },
                );
                if *finished {
                    push(*request_id, *at, TraceEvent::Completed);
                    terminal.push(*request_id);
                }
            }
            ExecutorEvent::PrefillProgress {
                request_id,
                processed_tokens,
                total_tokens,
                ..
            } => push(
                *request_id,
                at,
                TraceEvent::PrefillProgress {
                    processed_tokens: *processed_tokens,
                    total_tokens: *total_tokens,
                },
            ),
            ExecutorEvent::Failed {
                request_id,
                message,
            } => {
                push(
                    *request_id,
                    at,
                    TraceEvent::Failed {
                        message: message.clone(),
                    },
                );
                terminal.push(*request_id);
            }
            ExecutorEvent::Rejected { request_id, reason } => {
                push(
                    *request_id,
                    at,
                    TraceEvent::Rejected {
                        reason: reason.to_string(),
                    },
                );
                terminal.push(*request_id);
            }
            ExecutorEvent::Preempted {
                request_id,
                reason,
                generated_tokens,
            } => {
                let ola_runtime::PreemptionReason::KvShortfall { shortfall_blocks } = reason;
                push(
                    *request_id,
                    at,
                    TraceEvent::Preempted {
                        reason: "kv_shortfall".into(),
                        shortfall_blocks: *shortfall_blocks,
                        generated_tokens: *generated_tokens,
                    },
                );
            }
            // Deferrals are already recorded as decisions.
            ExecutorEvent::Idle
            | ExecutorEvent::Deferred { .. }
            | ExecutorEvent::BatchDeferred { .. } => {}
        }
    }
}

fn record(
    request: RequestId,
    iteration: Option<IterationId>,
    at: Timestamp,
    event: TraceEvent,
) -> TraceRecord {
    TraceRecord {
        run: None,
        request: request.get(),
        iteration: iteration.map(IterationId::get),
        at_us: at.as_micros(),
        event,
    }
}

const fn event_request(event: &ExecutorEvent) -> Option<RequestId> {
    match event {
        ExecutorEvent::Idle => None,
        ExecutorEvent::PrefillProgress { request_id, .. }
        | ExecutorEvent::Token { request_id, .. }
        | ExecutorEvent::Failed { request_id, .. }
        | ExecutorEvent::Deferred { request_id, .. }
        | ExecutorEvent::BatchDeferred { request_id, .. }
        | ExecutorEvent::Rejected { request_id, .. }
        | ExecutorEvent::Preempted { request_id, .. } => Some(*request_id),
    }
}

const fn stage(stage: SelectionStage) -> &'static str {
    match stage {
        SelectionStage::Decode => "decode",
        SelectionStage::ContinuedPrefill => "continued_prefill",
        SelectionStage::Admission => "admission",
    }
}

fn admission_event(outcome: &AdmissionOutcome) -> TraceEvent {
    let event = |outcome: &str, reason: Option<String>| TraceEvent::Admission {
        outcome: outcome.into(),
        reason,
        reserved_kv_bytes: None,
        budget_selected: None,
    };
    match outcome {
        AdmissionOutcome::NoSlot => event("no_slot", None),
        AdmissionOutcome::PreparationFailed { message } => {
            event("preparation_failed", Some(message.clone()))
        }
        AdmissionOutcome::Decided {
            decision,
            budget_selected,
            ..
        } => match decision {
            AdmissionDecision::Admit { reserved_kv_bytes } => TraceEvent::Admission {
                outcome: "admitted".into(),
                reason: None,
                reserved_kv_bytes: Some(*reserved_kv_bytes),
                budget_selected: *budget_selected,
            },
            AdmissionDecision::Defer(reason) => event("deferred", Some(reason.to_string())),
            AdmissionDecision::Reject(reason) => event("rejected", Some(reason.to_string())),
        },
    }
}

/// Writes records as JSON Lines.
///
/// # Errors
///
/// Returns any error from `writer`.
pub fn write_jsonl<'a>(
    records: impl IntoIterator<Item = &'a TraceRecord>,
    mut writer: impl Write,
) -> io::Result<()> {
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

/// Reads records written by [`write_jsonl`], skipping blank lines.
///
/// # Errors
///
/// Returns an I/O error, or `InvalidData` for a line that is not a record.
pub fn read_jsonl(reader: impl BufRead) -> io::Result<Vec<TraceRecord>> {
    reader
        .lines()
        .filter(|line| line.as_ref().map_or(true, |line| !line.trim().is_empty()))
        .map(|line| {
            serde_json::from_str(&line?)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        })
        .collect()
}
