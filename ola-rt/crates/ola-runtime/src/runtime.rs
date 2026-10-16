use ola_core::{ModelDescriptor, Request, RequestId, RequestSpec, Timestamp};

use crate::{
    FifoScheduler, InferenceSession, RequestRecord, RequestStatus, RequestTiming, RuntimeError,
    RuntimeState, Scheduler,
};

#[derive(Debug)]
/// Owns model registration, request identity, lifecycle state, and a scheduler.
///
/// `Runtime` is intentionally synchronous in the foundation release. Async
/// transport and device execution can wrap it without changing state semantics.
pub struct Runtime<S = FifoScheduler> {
    state: RuntimeState,
    scheduler: S,
    next_request_id: u64,
}

impl Default for Runtime<FifoScheduler> {
    fn default() -> Self {
        Self::new(FifoScheduler)
    }
}

impl<S: Scheduler> Runtime<S> {
    /// Creates an empty runtime using the supplied scheduler implementation.
    pub fn new(scheduler: S) -> Self {
        Self {
            state: RuntimeState::default(),
            scheduler,
            next_request_id: 1,
        }
    }

    /// Returns the read-only state observed by scheduling and telemetry code.
    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    /// Returns controlled mutable access for resource-accounting updates.
    ///
    /// Lifecycle collections remain private inside `RuntimeState`.
    pub fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.state
    }

    /// Adds model metadata to the runtime registry.
    pub fn register_model(&mut self, model: ModelDescriptor) -> Result<(), RuntimeError> {
        self.state.models.register(model)
    }

    /// Validates, identifies, and enqueues a request at a caller-supplied time.
    pub fn submit(
        &mut self,
        spec: RequestSpec,
        arrival_at: Timestamp,
    ) -> Result<RequestId, RuntimeError> {
        spec.validate()?;
        let model = self
            .state
            .models
            .get(&spec.model)
            .ok_or_else(|| RuntimeError::UnknownModel(spec.model.clone()))?;
        if let Some(modality) = spec
            .inputs
            .iter()
            .map(ola_core::Input::modality)
            .find(|modality| !model.supported_modalities.contains(modality))
        {
            return Err(RuntimeError::UnsupportedModality {
                model: spec.model,
                modality,
            });
        }

        let id = RequestId::new(self.next_request_id);
        let next = self
            .next_request_id
            .checked_add(1)
            .ok_or(RuntimeError::RequestIdExhausted)?;
        let request = Request::new(id, arrival_at, spec)?;
        let record = RequestRecord {
            request,
            status: RequestStatus::Queued,
            timing: RequestTiming {
                queued_at: arrival_at,
                ..RequestTiming::default()
            },
        };

        if self.state.requests.insert(id, record).is_some() {
            return Err(RuntimeError::DuplicateRequest(id));
        }
        self.state.queue.push_back(id);
        self.next_request_id = next;
        Ok(id)
    }

    /// Asks the scheduler for one request and atomically moves it to running.
    ///
    /// `Ok(None)` means no work is currently queued.
    pub fn schedule_next(
        &mut self,
        started_at: Timestamp,
    ) -> Result<Option<InferenceSession>, RuntimeError> {
        let Some(id) = self.select_next()? else {
            return Ok(None);
        };

        self.start_selected(id, started_at).map(Some)
    }

    /// Asks the scheduler for the next queued request without transitioning it.
    ///
    /// This two-stage form lets an executor apply admission policy to the
    /// scheduler's candidate while leaving deferred work queued.
    pub fn select_next(&mut self) -> Result<Option<RequestId>, RuntimeError> {
        let Some(id) = self.scheduler.select(&self.state) else {
            return Ok(None);
        };

        if !self.state.queue.contains(&id) {
            return Err(RuntimeError::InvalidSchedule(id));
        }
        let record = self
            .state
            .requests
            .get(&id)
            .ok_or(RuntimeError::InvalidSchedule(id))?;
        if record.status != RequestStatus::Queued {
            return Err(RuntimeError::InvalidSchedule(id));
        }
        Ok(Some(id))
    }

    /// Ranks up to `limit` queued requests without changing lifecycle state.
    pub fn rank_next(&mut self, limit: usize) -> Result<Vec<RequestId>, RuntimeError> {
        let ranked = self.scheduler.rank(&self.state, limit);
        let mut seen = std::collections::BTreeSet::new();
        for id in &ranked {
            let queued = self
                .state
                .requests
                .get(id)
                .is_some_and(|record| record.status == RequestStatus::Queued);
            if ranked.len() > limit || !queued || !seen.insert(*id) {
                return Err(RuntimeError::InvalidSchedule(*id));
            }
        }
        Ok(ranked)
    }

    /// Starts a scheduler-selected queued request after external admission.
    pub fn start_selected(
        &mut self,
        id: RequestId,
        started_at: Timestamp,
    ) -> Result<InferenceSession, RuntimeError> {
        let Some(position) = self.state.queue.iter().position(|queued| *queued == id) else {
            return Err(RuntimeError::InvalidSchedule(id));
        };
        let record = self
            .state
            .requests
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidSchedule(id))?;
        if record.status != RequestStatus::Queued {
            return Err(RuntimeError::InvalidSchedule(id));
        }
        ensure_not_before(id, "start", record.timing.queued_at, started_at)?;

        let model_id = record.request.spec.model.clone();
        record.status = RequestStatus::Running;
        record.timing.started_at = Some(started_at);
        self.state.queue.remove(position);
        self.state.active.insert(id);

        Ok(InferenceSession {
            request_id: id,
            model_id,
            started_at,
        })
    }

    /// Terminates a queued request that cannot satisfy admission constraints.
    pub fn reject(
        &mut self,
        request_id: RequestId,
        message: impl Into<String>,
        rejected_at: Timestamp,
    ) -> Result<(), RuntimeError> {
        let record = self
            .state
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        if record.status != RequestStatus::Queued {
            return Err(RuntimeError::InvalidTransition {
                request: request_id,
                operation: "reject",
            });
        }
        ensure_not_before(
            request_id,
            "admission rejection",
            record.timing.queued_at,
            rejected_at,
        )?;
        self.state.queue.retain(|queued| *queued != request_id);
        record.status = RequestStatus::Failed(message.into());
        record.timing.completed_at = Some(rejected_at);
        Ok(())
    }

    /// Records the first externally observable generated token.
    pub fn mark_first_token(
        &mut self,
        request_id: RequestId,
        at: Timestamp,
    ) -> Result<(), RuntimeError> {
        let record = self.record_for_running(request_id, "record first token for")?;
        if record.timing.first_token_at.is_some() {
            return Err(RuntimeError::InvalidTransition {
                request: request_id,
                operation: "record another first token for",
            });
        }
        let started_at = record
            .timing
            .started_at
            .expect("running requests always have a start timestamp");
        ensure_not_before(request_id, "first token", started_at, at)?;
        record.timing.first_token_at = Some(at);
        Ok(())
    }

    /// Moves a running request to the successful terminal state.
    pub fn complete(
        &mut self,
        request_id: RequestId,
        completed_at: Timestamp,
    ) -> Result<(), RuntimeError> {
        let record = self.record_for_running(request_id, "complete")?;
        let previous = record
            .timing
            .first_token_at
            .or(record.timing.started_at)
            .expect("running requests always have a start timestamp");
        ensure_not_before(request_id, "completion", previous, completed_at)?;
        record.status = RequestStatus::Completed;
        record.timing.completed_at = Some(completed_at);
        self.state.active.remove(&request_id);
        Ok(())
    }

    /// Moves a running request to the failed terminal state.
    pub fn fail(
        &mut self,
        request_id: RequestId,
        message: impl Into<String>,
        failed_at: Timestamp,
    ) -> Result<(), RuntimeError> {
        let record = self.record_for_running(request_id, "fail")?;
        let previous = record
            .timing
            .first_token_at
            .or(record.timing.started_at)
            .expect("running requests always have a start timestamp");
        ensure_not_before(request_id, "failure", previous, failed_at)?;
        record.status = RequestStatus::Failed(message.into());
        record.timing.completed_at = Some(failed_at);
        self.state.active.remove(&request_id);
        Ok(())
    }

    /// Cancels a queued or running request and removes it from the corresponding
    /// lifecycle collection.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
        cancelled_at: Timestamp,
    ) -> Result<(), RuntimeError> {
        let record = self
            .state
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        let previous = record.timing.started_at.unwrap_or(record.timing.queued_at);
        ensure_not_before(request_id, "cancellation", previous, cancelled_at)?;
        match record.status {
            RequestStatus::Queued => {
                self.state.queue.retain(|queued| *queued != request_id);
            }
            RequestStatus::Running => {
                self.state.active.remove(&request_id);
            }
            _ => {
                return Err(RuntimeError::InvalidTransition {
                    request: request_id,
                    operation: "cancel",
                });
            }
        }
        record.status = RequestStatus::Cancelled;
        record.timing.completed_at = Some(cancelled_at);
        Ok(())
    }

    fn record_for_running(
        &mut self,
        request_id: RequestId,
        operation: &'static str,
    ) -> Result<&mut RequestRecord, RuntimeError> {
        let record = self
            .state
            .requests
            .get_mut(&request_id)
            .ok_or(RuntimeError::UnknownRequest(request_id))?;
        if record.status != RequestStatus::Running {
            return Err(RuntimeError::InvalidTransition {
                request: request_id,
                operation,
            });
        }
        Ok(record)
    }
}

fn ensure_not_before(
    request: RequestId,
    event: &'static str,
    previous: Timestamp,
    observed: Timestamp,
) -> Result<(), RuntimeError> {
    if observed < previous {
        return Err(RuntimeError::TimestampRegression {
            request,
            event,
            previous,
            observed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ola_core::{ModelDescriptor, RequestSpec, Timestamp};

    use super::*;

    #[test]
    fn fifo_runtime_tracks_the_full_request_lifecycle() {
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "tiny-decoder"))
            .unwrap();

        let first = runtime
            .submit(
                RequestSpec::text("tiny", "first"),
                Timestamp::from_micros(10),
            )
            .unwrap();
        let second = runtime
            .submit(
                RequestSpec::text("tiny", "second"),
                Timestamp::from_micros(11),
            )
            .unwrap();

        let session = runtime
            .schedule_next(Timestamp::from_micros(20))
            .unwrap()
            .unwrap();
        assert_eq!(session.request_id, first);
        assert_eq!(
            runtime.state().queued_ids().collect::<Vec<_>>(),
            vec![second]
        );

        runtime
            .mark_first_token(first, Timestamp::from_micros(30))
            .unwrap();
        runtime.complete(first, Timestamp::from_micros(40)).unwrap();

        let record = runtime.state().request(first).unwrap();
        assert_eq!(record.status, RequestStatus::Completed);
        assert_eq!(
            record.timing.first_token_at,
            Some(Timestamp::from_micros(30))
        );
        assert_eq!(runtime.state().active_len(), 0);
    }

    #[test]
    fn rejects_requests_for_unregistered_models() {
        let mut runtime = Runtime::default();
        let result = runtime.submit(RequestSpec::text("missing", "hello"), Timestamp::ZERO);

        assert_eq!(result, Err(RuntimeError::UnknownModel("missing".into())));
    }

    #[test]
    fn rejects_modalities_not_supported_by_the_model() {
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "tiny-decoder"))
            .unwrap();
        let mut request = RequestSpec::text("tiny", "hello");
        request.inputs = vec![ola_core::Input::Image(ola_core::ImageInput {
            bytes: vec![0],
            width: 1,
            height: 1,
            channels: 1,
        })];

        let result = runtime.submit(request, Timestamp::ZERO);

        assert_eq!(
            result,
            Err(RuntimeError::UnsupportedModality {
                model: "tiny".into(),
                modality: ola_core::Modality::Image,
            })
        );
    }

    #[test]
    fn rejects_non_monotonic_lifecycle_measurements() {
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "tiny-decoder"))
            .unwrap();
        let id = runtime
            .submit(
                RequestSpec::text("tiny", "hello"),
                Timestamp::from_micros(10),
            )
            .unwrap();

        let result = runtime.schedule_next(Timestamp::from_micros(9));

        assert_eq!(
            result,
            Err(RuntimeError::TimestampRegression {
                request: id,
                event: "start",
                previous: Timestamp::from_micros(10),
                observed: Timestamp::from_micros(9),
            })
        );
        assert_eq!(runtime.state().queued_ids().collect::<Vec<_>>(), vec![id]);
    }
}
