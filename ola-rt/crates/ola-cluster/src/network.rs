//! Coordinator orchestration over connected worker links.
//!
//! Every link forwards its frames into one channel (see [`crate::client`]),
//! so the coordinator sees all workers' traffic in arrival order. Callers
//! drive it in one of two ways:
//!
//! - an event loop calls [`TcpCoordinator::next_frame`], which is cancel-safe
//!   and can sit in a `select!`, then [`TcpCoordinator::apply_frame`];
//! - tests and scripts call [`TcpCoordinator::poll`] or
//!   [`TcpCoordinator::receive`], which do both.
//!
//! Every helper keeps the output of every frame it applies, whichever worker
//! sent it, so no client event is dropped while waiting for a particular one.

use std::{collections::BTreeMap, error::Error, fmt, time::Duration};

use ola_core::{ClusterRequestId, RequestSpec, Timestamp, WorkerId};
use tokio::sync::mpsc;

use crate::{
    ClusterFeedbackSnapshot, ClusterMigrationObservation, ClusterPolicy,
    ClusterPolicyUpdateObservation, ClusterRequestObservation, ClusterRequestStatus,
    ClusterWorkerObservation, Coordinator, CoordinatorError, CoordinatorLinkError,
    CoordinatorLinkEvent, CoordinatorOutput, CoordinatorWorkerLink, LinkFrame, WorkerCommand,
    WorkerMessage,
};

/// Errors raised while coordinating connected TCP workers.
#[derive(Debug)]
pub enum TcpCoordinatorError {
    /// The sans-I/O coordinator rejected a state transition.
    Coordinator(CoordinatorError),
    /// A worker link failed.
    Link(CoordinatorLinkError),
    /// No connected link exists for a worker selected by the coordinator.
    MissingLink(WorkerId),
    /// A worker the caller was waiting on disconnected. Its requests were
    /// recovered onto other workers.
    Disconnected(WorkerId),
    /// A target rejected a migration and the source was restored.
    MigrationRefused {
        /// Worker that refused the target admission.
        worker: WorkerId,
        /// Worker-provided diagnostic.
        message: String,
    },
}

/// Coordinator-side link timing estimate from worker heartbeats.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinkCalibration {
    /// Number of heartbeat samples incorporated.
    pub samples: u64,
    /// Mean of coordinator receive time minus worker send time, in
    /// microseconds: the clock offset plus the one-way delay.
    pub offset_us: i64,
}

impl fmt::Display for TcpCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordinator(error) => error.fmt(formatter),
            Self::Link(error) => error.fmt(formatter),
            Self::MissingLink(worker) => write!(formatter, "no TCP link for worker `{worker}`"),
            Self::Disconnected(worker) => write!(formatter, "worker `{worker}` disconnected"),
            Self::MigrationRefused { worker, message } => {
                write!(formatter, "worker `{worker}` refused migration: {message}")
            }
        }
    }
}

impl Error for TcpCoordinatorError {}

impl From<CoordinatorError> for TcpCoordinatorError {
    fn from(error: CoordinatorError) -> Self {
        Self::Coordinator(error)
    }
}

impl From<CoordinatorLinkError> for TcpCoordinatorError {
    fn from(error: CoordinatorLinkError) -> Self {
        Self::Link(error)
    }
}

/// Owns a sans-I/O coordinator and its connected worker links.
pub struct TcpCoordinator {
    coordinator: Coordinator,
    links: BTreeMap<WorkerId, CoordinatorWorkerLink>,
    calibration: BTreeMap<WorkerId, LinkCalibration>,
    frames_tx: mpsc::UnboundedSender<LinkFrame>,
    frames_rx: mpsc::UnboundedReceiver<LinkFrame>,
}

impl TcpCoordinator {
    /// Creates an empty networked coordinator.
    pub fn new<P: ClusterPolicy + 'static>(policy: P) -> Self {
        let (frames_tx, frames_rx) = mpsc::unbounded_channel();
        Self {
            coordinator: Coordinator::new(policy),
            links: BTreeMap::new(),
            calibration: BTreeMap::new(),
            frames_tx,
            frames_rx,
        }
    }

    /// Connects one worker and registers its initial summary.
    ///
    /// # Errors
    ///
    /// Fails if the handshake fails or the worker already joined.
    pub async fn connect_worker(
        &mut self,
        address: impl tokio::net::ToSocketAddrs,
        expected_worker: Option<WorkerId>,
    ) -> Result<WorkerId, TcpCoordinatorError> {
        let (link, summary) =
            CoordinatorWorkerLink::connect(address, expected_worker, self.frames_tx.clone())
                .await?;
        let worker = link.worker().clone();
        self.coordinator.join(summary)?;
        self.calibration
            .insert(worker.clone(), LinkCalibration::default());
        self.links.insert(worker.clone(), link);
        Ok(worker)
    }

    /// The underlying sans-I/O coordinator.
    pub const fn coordinator(&self) -> &Coordinator {
        &self.coordinator
    }

    /// Recent terminal request outcomes, oldest first.
    pub fn recent_request_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterRequestObservation> {
        self.coordinator.recent_request_observations()
    }

    /// Recent worker-state observations, oldest first.
    pub fn recent_worker_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterWorkerObservation> {
        self.coordinator.recent_worker_observations()
    }

    /// Recent committed migrations, oldest first.
    pub fn recent_migration_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterMigrationObservation> {
        self.coordinator.recent_migration_observations()
    }

    /// Recent acknowledged worker policy updates, oldest first.
    pub fn recent_policy_update_observations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ClusterPolicyUpdateObservation> {
        self.coordinator.recent_policy_update_observations()
    }

    /// Returns a bounded snapshot of distributed adaptation feedback.
    pub fn feedback_snapshot(&self) -> ClusterFeedbackSnapshot {
        self.coordinator.feedback_snapshot()
    }

    /// Workers with an open link, in ID order.
    pub fn connected(&self) -> impl ExactSizeIterator<Item = &WorkerId> {
        self.links.keys()
    }

    /// Returns the current status of a cluster request.
    pub fn status(&self, request: ClusterRequestId) -> Option<&ClusterRequestStatus> {
        self.coordinator.status(request)
    }

    /// Returns the latest heartbeat-derived timing estimate for a worker.
    pub fn link_calibration(&self, worker: &WorkerId) -> Option<LinkCalibration> {
        self.calibration.get(worker).copied()
    }

    /// Installs new adaptive routing weights on the coordinator policy.
    ///
    /// # Errors
    ///
    /// Returns an error when the active policy does not support weight
    /// updates or rejects the proposal.
    pub fn update_policy_weights(
        &mut self,
        weights: crate::ClusterWeights,
    ) -> Result<(), TcpCoordinatorError> {
        self.coordinator
            .update_policy_weights(weights)
            .map_err(CoordinatorError::Policy)?;
        Ok(())
    }

    /// Expires silent worker leases without performing network I/O.
    pub fn expire_workers(&mut self, now: Timestamp, timeout: Duration) -> Vec<WorkerId> {
        self.coordinator.expire_workers(now, timeout)
    }

    /// Expires silent workers, requeues their requests, and dispatches a new
    /// planning pass. Recovered requests restart by recompute; see
    /// [`Coordinator::recover_lost_requests`].
    ///
    /// # Errors
    ///
    /// Fails if planning or dispatch fails.
    pub async fn recover_lost(
        &mut self,
        now: Timestamp,
        timeout: Duration,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        self.coordinator.recover_lost_requests(now, timeout);
        self.plan_pending(now).await
    }

    /// Delivers commands recovery queued, then plans and dispatches any
    /// pending requests, such as those deferred while no worker was ready.
    ///
    /// # Errors
    ///
    /// Fails if planning or dispatch fails.
    pub async fn plan_pending(
        &mut self,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        let mut output = CoordinatorOutput {
            commands: self.coordinator.take_commands(),
            ..CoordinatorOutput::default()
        };
        if let Some((_, planned)) = self.coordinator.plan(at)? {
            output.extend(planned);
        }
        self.dispatch(&output.commands).await?;
        Ok(output)
    }

    /// Submits a request, plans it, and dispatches any resulting commands.
    ///
    /// # Errors
    ///
    /// Fails if the request is invalid, or planning or dispatch fails.
    pub async fn submit_and_plan(
        &mut self,
        spec: RequestSpec,
        at: Timestamp,
    ) -> Result<(ClusterRequestId, CoordinatorOutput), TcpCoordinatorError> {
        let (request, mut output) = self.coordinator.submit(spec, at)?;
        output.extend(self.plan_pending(at).await?);
        Ok((request, output))
    }

    /// Cancels a request and dispatches the resulting worker command.
    ///
    /// # Errors
    ///
    /// Fails for an unknown request or a failed dispatch.
    pub async fn cancel(
        &mut self,
        request: ClusterRequestId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        let output = self.coordinator.cancel(request, at)?;
        self.dispatch(&output.commands).await?;
        Ok(output)
    }

    /// Waits for the next frame from any worker.
    ///
    /// Cancel-safe: dropping the future loses nothing, so it can sit in a
    /// `select!`. Pass the frame to [`apply_frame`](Self::apply_frame).
    pub async fn next_frame(&mut self) -> LinkFrame {
        self.frames_rx
            .recv()
            .await
            .expect("the coordinator holds a sender, so the channel stays open")
    }

    /// Applies one frame from [`next_frame`](Self::next_frame).
    ///
    /// Heartbeats are acknowledged and renew the lease; messages update
    /// coordinator state, and any commands they produce (such as cancelling
    /// a stale copy) are dispatched. A failed or closed link loses its
    /// worker: its requests are recovered and replanned at once.
    ///
    /// # Errors
    ///
    /// Fails if the coordinator rejects a message or a dispatch fails.
    pub async fn apply_frame(
        &mut self,
        frame: LinkFrame,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        let mut output = CoordinatorOutput::default();
        let (worker, event) = self.accept(frame, at, &mut output).await?;
        if let Some(event) = event {
            output.extend(self.apply_event(&worker, event, at).await?);
        }
        Ok(output)
    }

    /// Waits for and applies the next frame from any worker.
    ///
    /// # Errors
    ///
    /// As [`apply_frame`](Self::apply_frame).
    pub async fn poll(
        &mut self,
        at: Timestamp,
    ) -> Result<(WorkerId, CoordinatorOutput), TcpCoordinatorError> {
        let frame = self.next_frame().await;
        let worker = frame.0.clone();
        Ok((worker, self.apply_frame(frame, at).await?))
    }

    /// Applies frames from every worker until `worker` sends a domain
    /// message, and returns everything those frames produced.
    ///
    /// # Errors
    ///
    /// [`TcpCoordinatorError::Disconnected`] if `worker` disconnects first,
    /// and as [`apply_frame`](Self::apply_frame).
    pub async fn receive(
        &mut self,
        worker: &WorkerId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        if !self.links.contains_key(worker) {
            return Err(TcpCoordinatorError::MissingLink(worker.clone()));
        }
        let mut output = CoordinatorOutput::default();
        loop {
            let frame = self.next_frame().await;
            let (from, event) = self.accept(frame, at, &mut output).await?;
            match event {
                None if &from == worker => {
                    return Err(TcpCoordinatorError::Disconnected(from));
                }
                None => {}
                Some(event) => {
                    let message = matches!(event, CoordinatorLinkEvent::Message(_));
                    output.extend(self.apply_event(&from, event, at).await?);
                    if message && &from == worker {
                        return Ok(output);
                    }
                }
            }
        }
    }

    /// Moves a running request from `source` to `target` with a two-phase
    /// handoff.
    ///
    /// The source exports first. Only after the payload arrives does
    /// ownership move to the target and an `AdmitMigrated` command go out.
    /// The target's acceptance commits the handoff; its refusal, or its
    /// loss, rolls the request back to the source. If the request finishes,
    /// fails, or is cancelled on the source before the export, there is
    /// nothing to move and this returns what happened.
    ///
    /// # Errors
    ///
    /// [`TcpCoordinatorError::MigrationRefused`] if the target refused or
    /// was lost, [`TcpCoordinatorError::Disconnected`] if the source was lost
    /// before exporting, and as [`apply_frame`](Self::apply_frame).
    pub async fn migrate(
        &mut self,
        request: ClusterRequestId,
        source: &WorkerId,
        target: WorkerId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        self.link(source)?
            .send_command(&WorkerCommand::MigrateOut { request })
            .await?;
        let mut output = CoordinatorOutput::default();
        let state = loop {
            let frame = self.next_frame().await;
            let (from, event) = self.accept(frame, at, &mut output).await?;
            match event {
                None if &from == source => {
                    return Err(TcpCoordinatorError::Disconnected(from));
                }
                Some(CoordinatorLinkEvent::Message(WorkerMessage::Migrated {
                    request: migrated,
                    state,
                })) if &from == source && migrated == request => break state,
                None => {}
                Some(event) => output.extend(self.apply_event(&from, event, at).await?),
            }
            if !matches!(
                self.coordinator.status(request),
                Some(ClusterRequestStatus::Running { worker, .. }) if worker == source
            ) {
                // Finished or moved before the export: nothing to hand off.
                return Ok(output);
            }
        };
        let staged =
            self.coordinator
                .stage_migration(request, source, target.clone(), state, at)?;
        self.dispatch(&staged.commands).await?;
        output.extend(staged);
        loop {
            let frame = self.next_frame().await;
            let (from, event) = self.accept(frame, at, &mut output).await?;
            match event {
                Some(CoordinatorLinkEvent::Message(WorkerMessage::Accepted {
                    request: accepted,
                    local,
                })) if from == target && accepted == request => {
                    output.extend(self.coordinator.handle(
                        &target,
                        WorkerMessage::Accepted { request, local },
                        at,
                    )?);
                    output.extend(self.coordinator.finish_migration(request, at));
                    // The source may have been lost meanwhile; its exported
                    // state then has nowhere to be freed.
                    if let Some(link) = self.links.get_mut(source) {
                        link.send_command(&WorkerCommand::CommitMigration { request })
                            .await?;
                    }
                    return Ok(output);
                }
                Some(CoordinatorLinkEvent::Message(WorkerMessage::Refused {
                    request: refused,
                    message,
                })) if from == target && refused == request => {
                    self.coordinator
                        .rollback_migration_at(request, source, Some(at))?;
                    self.link(source)?
                        .send_command(&WorkerCommand::AbortMigration { request })
                        .await?;
                    // The source's `Accepted` for the restored copy arrives
                    // later and is applied like any other message.
                    return Err(TcpCoordinatorError::MigrationRefused {
                        worker: target,
                        message,
                    });
                }
                None => {}
                Some(event) => output.extend(self.apply_event(&from, event, at).await?),
            }
            if self.coordinator.staged_migration(request).is_none() {
                // Recovery rolled the handoff back after losing the target,
                // and has already told the source to restore the request.
                return Err(TcpCoordinatorError::MigrationRefused {
                    worker: target,
                    message: "target worker was lost during the handoff".into(),
                });
            }
        }
    }

    /// Places a worker into draining mode. Existing requests continue, while
    /// subsequent plans exclude it from admission.
    ///
    /// # Errors
    ///
    /// Fails if the worker has no link or the command cannot be sent.
    pub async fn drain_worker(&mut self, worker: &WorkerId) -> Result<(), TcpCoordinatorError> {
        self.link(worker)?
            .send_command(&WorkerCommand::Drain)
            .await?;
        Ok(())
    }

    /// Pushes adaptive routing weights to a worker over its control link.
    ///
    /// The worker validates and records them; it does not yet use them for
    /// its own scheduling.
    ///
    /// # Errors
    ///
    /// Fails if the worker has no link or the command cannot be sent.
    pub async fn update_worker_policy(
        &mut self,
        worker: &WorkerId,
        weights: crate::ClusterWeights,
    ) -> Result<(), TcpCoordinatorError> {
        self.link(worker)?
            .send_command(&WorkerCommand::PolicyUpdate {
                load_weight: weights.load,
                kv_headroom_weight: weights.kv_headroom,
                network_weight: weights.network,
            })
            .await?;
        Ok(())
    }

    /// Changes a worker's prefill/decode disaggregation role.
    ///
    /// # Errors
    ///
    /// Fails if the worker has no link or the command cannot be sent.
    pub async fn set_worker_role(
        &mut self,
        worker: &WorkerId,
        role: crate::WorkerRole,
    ) -> Result<(), TcpCoordinatorError> {
        self.link(worker)?
            .send_command(&WorkerCommand::SetRole { role })
            .await?;
        Ok(())
    }

    /// Supplies a calibrated network cost for a worker to routing policy.
    pub fn set_worker_network_cost(&mut self, worker: WorkerId, cost_us: u64) {
        self.coordinator.set_network_cost(worker, cost_us);
    }

    /// Closes one worker connection gracefully.
    ///
    /// # Errors
    ///
    /// Fails if the worker has no link or the goodbye cannot be sent.
    pub async fn close_worker(&mut self, worker: &WorkerId) -> Result<(), TcpCoordinatorError> {
        self.link(worker)?.close().await?;
        Ok(())
    }

    fn link(
        &mut self,
        worker: &WorkerId,
    ) -> Result<&mut CoordinatorWorkerLink, TcpCoordinatorError> {
        self.links
            .get_mut(worker)
            .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))
    }

    /// Decodes a frame. A failed link loses its worker: the link is dropped,
    /// its requests are recovered and replanned into `output`, and the event
    /// is `None`.
    async fn accept(
        &mut self,
        (worker, frame): LinkFrame,
        at: Timestamp,
        output: &mut CoordinatorOutput,
    ) -> Result<(WorkerId, Option<CoordinatorLinkEvent>), TcpCoordinatorError> {
        let Some(link) = self.links.get_mut(&worker) else {
            // A frame still queued from a link that was already dropped.
            return Ok((worker, None));
        };
        match link.accept_frame(frame).await {
            Ok(event) => Ok((worker, Some(event))),
            Err(_) => {
                self.links.remove(&worker);
                self.coordinator.recover_worker(&worker, at)?;
                output.extend(self.plan_pending(at).await?);
                Ok((worker, None))
            }
        }
    }

    async fn apply_event(
        &mut self,
        worker: &WorkerId,
        event: CoordinatorLinkEvent,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        match event {
            CoordinatorLinkEvent::Message(message) => {
                let output = self.coordinator.handle(worker, message, at)?;
                self.dispatch(&output.commands).await?;
                Ok(output)
            }
            CoordinatorLinkEvent::Heartbeat { epoch, sent_at_us } => {
                match self.coordinator.heartbeat(worker, epoch, at) {
                    // A delayed heartbeat from a replaced process is dropped.
                    Ok(()) | Err(CoordinatorError::StaleWorkerEpoch { .. }) => {}
                    Err(error) => return Err(error.into()),
                }
                self.record_calibration(worker, at, sent_at_us);
                Ok(CoordinatorOutput::default())
            }
        }
    }

    async fn dispatch(
        &mut self,
        commands: &[(WorkerId, WorkerCommand)],
    ) -> Result<(), TcpCoordinatorError> {
        for (worker, command) in commands {
            self.link(worker)?.send_command(command).await?;
        }
        Ok(())
    }

    fn record_calibration(&mut self, worker: &WorkerId, at: Timestamp, sent_at_us: u64) {
        let offset = i128::from(at.as_micros()) - i128::from(sent_at_us);
        let offset = offset.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        if let Some(calibration) = self.calibration.get_mut(worker) {
            calibration.samples = calibration.samples.saturating_add(1);
            let samples = i128::from(calibration.samples);
            let previous = i128::from(calibration.offset_us) * samples.saturating_sub(1);
            calibration.offset_us = ((previous + i128::from(offset)) / samples)
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                as i64;
        }
    }
}
