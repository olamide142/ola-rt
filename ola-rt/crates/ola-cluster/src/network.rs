//! Coordinator orchestration over connected worker links.

use std::{collections::BTreeMap, error::Error, fmt, time::Duration};

use ola_core::{ClusterRequestId, RequestSpec, Timestamp, WorkerId};

use crate::{
    ClusterPolicy, Coordinator, CoordinatorError, CoordinatorLinkError, CoordinatorLinkEvent,
    CoordinatorOutput, CoordinatorWorkerLink, WorkerCommand, WorkerMessage,
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
}

/// Coordinator-side link timing estimate from worker heartbeats.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinkCalibration {
    /// Number of heartbeat samples incorporated.
    pub samples: u64,
    /// Estimated coordinator-minus-worker clock offset in microseconds.
    pub offset_us: i64,
}

impl fmt::Display for TcpCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordinator(error) => error.fmt(formatter),
            Self::Link(error) => error.fmt(formatter),
            Self::MissingLink(worker) => write!(formatter, "no TCP link for worker `{worker}`"),
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
}

impl TcpCoordinator {
    /// Creates an empty networked coordinator.
    pub fn new<P: ClusterPolicy + 'static>(policy: P) -> Self {
        Self {
            coordinator: Coordinator::new(policy),
            links: BTreeMap::new(),
            calibration: BTreeMap::new(),
        }
    }

    /// Connects one worker and registers its initial summary.
    pub async fn connect_worker(
        &mut self,
        address: impl tokio::net::ToSocketAddrs,
        expected_worker: Option<WorkerId>,
    ) -> Result<WorkerId, TcpCoordinatorError> {
        let (link, summary) = CoordinatorWorkerLink::connect(address, expected_worker).await?;
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

    /// Returns the current status of a cluster request.
    pub fn status(&self, request: ClusterRequestId) -> Option<&crate::ClusterRequestStatus> {
        self.coordinator.status(request)
    }

    /// Returns the latest heartbeat-derived timing estimate for a worker.
    pub fn link_calibration(&self, worker: &WorkerId) -> Option<LinkCalibration> {
        self.calibration.get(worker).copied()
    }

    /// Expires silent worker leases without performing network I/O.
    pub fn expire_workers(&mut self, now: Timestamp, timeout: Duration) -> Vec<WorkerId> {
        self.coordinator.expire_workers(now, timeout)
    }

    /// Expires silent workers, requeues their requests, and dispatches a new
    /// planning pass. Existing generated output is not reconstructed by this
    /// path; use the migration commands when continuity is required.
    pub async fn recover_lost(
        &mut self,
        now: Timestamp,
        timeout: Duration,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        let recovered = self.coordinator.recover_lost_requests(now, timeout);
        if recovered.is_empty() {
            return Ok(CoordinatorOutput::default());
        }
        let mut output = CoordinatorOutput::default();
        if let Some((_, planned)) = self.coordinator.plan(now)? {
            self.dispatch(&planned.commands).await?;
            output.extend(planned);
        }
        Ok(output)
    }

    /// Submits a request, plans it, and dispatches any resulting commands.
    pub async fn submit_and_plan(
        &mut self,
        spec: RequestSpec,
        at: Timestamp,
    ) -> Result<(ClusterRequestId, CoordinatorOutput), TcpCoordinatorError> {
        let (request, mut output) = self.coordinator.submit(spec, at)?;
        if let Some((_, planned)) = self.coordinator.plan(at)? {
            self.dispatch(&planned.commands).await?;
            output.extend(planned);
        }
        Ok((request, output))
    }

    /// Cancels a request and dispatches the resulting worker command.
    pub async fn cancel(
        &mut self,
        request: ClusterRequestId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        let output = self.coordinator.cancel(request, at)?;
        self.dispatch(&output.commands).await?;
        Ok(output)
    }

    /// Performs the coordinator-managed half of a migration handoff. The
    /// source exports first; only after the payload arrives does ownership
    /// move to the target and an `AdmitMigrated` command get dispatched.
    pub async fn migrate(
        &mut self,
        request: ClusterRequestId,
        source: &WorkerId,
        target: WorkerId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        self.links
            .get_mut(source)
            .ok_or_else(|| TcpCoordinatorError::MissingLink(source.clone()))?
            .send_command(&WorkerCommand::MigrateOut { request })
            .await?;
        let state = loop {
            let event = self
                .links
                .get_mut(source)
                .ok_or_else(|| TcpCoordinatorError::MissingLink(source.clone()))?
                .receive_event()
                .await?;
            match event {
                CoordinatorLinkEvent::Message(WorkerMessage::Migrated {
                    request: migrated_request,
                    state,
                }) if migrated_request == request => break state,
                CoordinatorLinkEvent::Message(message) => {
                    let _ = self.coordinator.handle(source, message, at)?;
                }
                CoordinatorLinkEvent::Heartbeat { epoch, sent_at_us } => {
                    self.coordinator.heartbeat(source, epoch, at)?;
                    self.record_calibration(source, at, sent_at_us);
                }
            }
        };
        let output = self
            .coordinator
            .stage_migration(request, source, target, state, at)?;
        self.dispatch(&output.commands).await?;
        Ok(output)
    }

    /// Places a worker into draining mode. Existing requests continue, while
    /// subsequent plans exclude it from admission.
    pub async fn drain_worker(&mut self, worker: &WorkerId) -> Result<(), TcpCoordinatorError> {
        let link = self
            .links
            .get_mut(worker)
            .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?;
        link.send_command(&WorkerCommand::Drain).await?;
        Ok(())
    }

    /// Pushes adaptive routing weights to a worker over its control link.
    pub async fn update_worker_policy(
        &mut self,
        worker: &WorkerId,
        weights: crate::ClusterWeights,
    ) -> Result<(), TcpCoordinatorError> {
        let link = self
            .links
            .get_mut(worker)
            .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?;
        link.send_command(&WorkerCommand::PolicyUpdate {
            load_weight: weights.load,
            kv_headroom_weight: weights.kv_headroom,
        })
        .await?;
        Ok(())
    }

    /// Receives one message from a worker and applies it to coordinator state.
    pub async fn receive(
        &mut self,
        worker: &WorkerId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        loop {
            let event = self
                .links
                .get_mut(worker)
                .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?
                .receive_event()
                .await?;
            match event {
                CoordinatorLinkEvent::Message(message) => {
                    return Ok(self.coordinator.handle(worker, message, at)?);
                }
                CoordinatorLinkEvent::Heartbeat { epoch, sent_at_us } => {
                    self.coordinator.heartbeat(worker, epoch, at)?;
                    self.record_calibration(worker, at, sent_at_us);
                }
            }
        }
    }

    /// Closes one worker connection gracefully.
    pub async fn close_worker(&mut self, worker: &WorkerId) -> Result<(), TcpCoordinatorError> {
        let link = self
            .links
            .get_mut(worker)
            .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?;
        link.close().await?;
        Ok(())
    }

    async fn dispatch(
        &mut self,
        commands: &[(WorkerId, WorkerCommand)],
    ) -> Result<(), TcpCoordinatorError> {
        for (worker, command) in commands {
            let link = self
                .links
                .get_mut(worker)
                .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?;
            link.send_command(command).await?;
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
