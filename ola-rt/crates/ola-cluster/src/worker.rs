//! An executor that speaks the coordinator's message vocabulary.

use std::collections::BTreeMap;

use ola_core::{ClusterRequestId, RequestId, Timestamp, WorkerId};
use ola_runtime::{Executor, ExecutorError, ExecutorEvent, ExecutorIteration, RequestStatus};

use crate::{WorkerCommand, WorkerHealth, WorkerMessage, WorkerSummary};

/// Wraps one [`Executor`] as a cluster worker.
///
/// The executor stays authoritative for request lifecycle: the host only
/// translates coordinator commands into executor calls and executor events
/// into worker messages, keyed by cluster request ID.
pub struct WorkerHost {
    id: WorkerId,
    executor: Executor,
    health: WorkerHealth,
    admitted: u64,
    by_local: BTreeMap<RequestId, ClusterRequestId>,
    by_cluster: BTreeMap<ClusterRequestId, RequestId>,
}

impl WorkerHost {
    /// Wraps a configured executor with its models attached.
    pub fn new(id: impl Into<WorkerId>, executor: Executor) -> Self {
        Self {
            id: id.into(),
            executor,
            health: WorkerHealth::Ready,
            admitted: 0,
            by_local: BTreeMap::new(),
            by_cluster: BTreeMap::new(),
        }
    }

    /// The worker's identity.
    pub const fn id(&self) -> &WorkerId {
        &self.id
    }

    /// The wrapped executor.
    pub const fn executor(&self) -> &Executor {
        &self.executor
    }

    /// Whether the worker accepts new requests.
    pub const fn health(&self) -> WorkerHealth {
        self.health
    }

    /// Changes whether the worker accepts new requests. Owned requests run
    /// to completion either way.
    pub fn set_health(&mut self, health: WorkerHealth) {
        self.health = health;
    }

    /// The worker's runtime-local ID for a cluster request it owns.
    pub fn local_id(&self, request: ClusterRequestId) -> Option<RequestId> {
        self.by_cluster.get(&request).copied()
    }

    /// Cluster requests the worker owns, with their local IDs, in cluster-ID
    /// order.
    pub fn owned(&self) -> impl ExactSizeIterator<Item = (ClusterRequestId, RequestId)> + '_ {
        self.by_cluster
            .iter()
            .map(|(cluster, local)| (*cluster, *local))
    }

    /// Summarizes the worker's current load.
    pub fn summary(&self, at: Timestamp) -> WorkerSummary {
        WorkerSummary::from_executor(
            self.id.clone(),
            &self.executor,
            self.health,
            self.admitted,
            at,
        )
    }

    /// Whether the executor has queued or live work.
    pub fn has_work(&self) -> bool {
        self.executor.state().queued_len() > 0 || self.executor.has_active_request()
    }

    /// Applies one coordinator command.
    ///
    /// Admission failures become [`WorkerMessage::Refused`] rather than
    /// errors, since the coordinator must learn about them.
    ///
    /// # Errors
    ///
    /// Fails if the executor cannot cancel an owned, live request.
    pub fn handle(
        &mut self,
        command: WorkerCommand,
        at: Timestamp,
    ) -> Result<Vec<WorkerMessage>, ExecutorError> {
        match command {
            WorkerCommand::Admit {
                request,
                spec,
                arrival_at,
            } => {
                self.admitted += 1;
                if self.health != WorkerHealth::Ready {
                    return Ok(vec![WorkerMessage::Refused {
                        request,
                        message: format!("worker `{}` is draining", self.id),
                    }]);
                }
                Ok(vec![match self.executor.submit(spec, arrival_at) {
                    Ok(local) => {
                        self.by_local.insert(local, request);
                        self.by_cluster.insert(request, local);
                        WorkerMessage::Accepted { request, local }
                    }
                    Err(error) => WorkerMessage::Refused {
                        request,
                        message: error.to_string(),
                    },
                }])
            }
            WorkerCommand::Cancel { request } => {
                let Some(&local) = self.by_cluster.get(&request) else {
                    // Already terminal here; the terminal message is on its way.
                    return Ok(Vec::new());
                };
                let live = self.executor.state().request(local).is_some_and(|record| {
                    matches!(
                        record.status,
                        RequestStatus::Queued | RequestStatus::Running
                    )
                });
                if !live {
                    return Ok(Vec::new());
                }
                self.executor.cancel(local, at)?;
                self.unmap(local);
                Ok(vec![WorkerMessage::Cancelled { request }])
            }
            WorkerCommand::MigrateOut { request } => {
                let Some(local) = self.by_cluster.get(&request).copied() else {
                    return Ok(Vec::new());
                };
                let state = self.executor.export_request(local, at)?;
                self.unmap(local);
                Ok(vec![WorkerMessage::Migrated { request, state }])
            }
            WorkerCommand::AdmitMigrated {
                request,
                state,
                arrival_at,
            } => {
                if self.health != WorkerHealth::Ready {
                    return Ok(vec![WorkerMessage::Refused {
                        request,
                        message: format!("worker `{}` is draining", self.id),
                    }]);
                }
                let local = self.executor.import_request(state, arrival_at)?;
                self.admitted += 1;
                self.by_local.insert(local, request);
                self.by_cluster.insert(request, local);
                Ok(vec![WorkerMessage::Accepted { request, local }])
            }
        }
    }

    /// Runs one executor iteration if there is work.
    ///
    /// # Errors
    ///
    /// Propagates executor failures; the executor then changed nothing.
    pub fn step(
        &mut self,
        at: Timestamp,
    ) -> Result<Option<(ExecutorIteration, Vec<WorkerMessage>)>, ExecutorError> {
        if !self.has_work() {
            return Ok(None);
        }
        let iteration = self.executor.step_iteration(at)?;
        let mut messages = Vec::new();
        for event in &iteration.events {
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
                    let Some(&request) = self.by_local.get(request_id) else {
                        continue;
                    };
                    messages.push(WorkerMessage::Token {
                        request,
                        token_id: token.id,
                        generated_tokens: *generated_tokens,
                        phase: *phase,
                        at: *at,
                    });
                    if *finished {
                        let result = self.executor.take_result(*request_id);
                        self.unmap(*request_id);
                        messages.push(match result {
                            Some(result) => WorkerMessage::Finished {
                                request,
                                token_ids: result.token_ids,
                                text: result.text,
                            },
                            None => WorkerMessage::Failed {
                                request,
                                message: "completed without a retained result".into(),
                            },
                        });
                    }
                }
                ExecutorEvent::Failed {
                    request_id,
                    message,
                } => {
                    if let Some(request) = self.unmap(*request_id) {
                        messages.push(WorkerMessage::Failed {
                            request,
                            message: message.clone(),
                        });
                    }
                }
                ExecutorEvent::Rejected { request_id, reason } => {
                    if let Some(request) = self.unmap(*request_id) {
                        messages.push(WorkerMessage::Failed {
                            request,
                            message: reason.to_string(),
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(Some((iteration, messages)))
    }

    fn unmap(&mut self, local: RequestId) -> Option<ClusterRequestId> {
        let request = self.by_local.remove(&local)?;
        self.by_cluster.remove(&request);
        Some(request)
    }
}
