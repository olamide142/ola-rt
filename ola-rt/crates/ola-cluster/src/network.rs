//! Coordinator orchestration over connected worker links.

use std::{collections::BTreeMap, error::Error, fmt, time::Duration};

use ola_core::{ClusterRequestId, RequestSpec, Timestamp, WorkerId};

use crate::{
    Coordinator, CoordinatorError, CoordinatorLinkError, CoordinatorOutput, ClusterPolicy,
    CoordinatorLinkEvent, CoordinatorWorkerLink, WorkerCommand,
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
}

impl TcpCoordinator {
    /// Creates an empty networked coordinator.
    pub fn new<P: ClusterPolicy + 'static>(policy: P) -> Self {
        Self {
            coordinator: Coordinator::new(policy),
            links: BTreeMap::new(),
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

    /// Receives one message from a worker and applies it to coordinator state.
    pub async fn receive(
        &mut self,
        worker: &WorkerId,
        at: Timestamp,
    ) -> Result<CoordinatorOutput, TcpCoordinatorError> {
        let link = self
            .links
            .get_mut(worker)
            .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?;
        loop {
            match link.receive_event().await? {
                CoordinatorLinkEvent::Message(message) => {
                    return Ok(self.coordinator.handle(worker, message, at)?);
                }
                CoordinatorLinkEvent::Heartbeat { epoch, .. } => {
                    self.coordinator.heartbeat(worker, epoch, at)?;
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

    async fn dispatch(&mut self, commands: &[(WorkerId, WorkerCommand)]) -> Result<(), TcpCoordinatorError> {
        for (worker, command) in commands {
            let link = self
                .links
                .get_mut(worker)
                .ok_or_else(|| TcpCoordinatorError::MissingLink(worker.clone()))?;
            link.send_command(command).await?;
        }
        Ok(())
    }
}
