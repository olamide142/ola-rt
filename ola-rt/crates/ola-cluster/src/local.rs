//! A coordinator and its workers in one process.

use std::{collections::BTreeMap, error::Error, fmt};

use ola_core::{ClusterRequestId, RequestSpec, Timestamp, WorkerId};
use ola_runtime::{ExecutorError, ExecutorIteration};

use crate::{
    ClientEvent, ClusterIteration, ClusterPolicy, ClusterTraceRecord, Coordinator,
    CoordinatorError, CoordinatorOutput, WorkerHost,
};

/// Everything that happened in one [`LocalCluster::step`], plus anything
/// produced by submissions and cancellations since the previous step.
#[derive(Clone, Debug, Default)]
pub struct LocalClusterStep {
    /// The cluster iteration, if any request was pending.
    pub cluster: Option<ClusterIteration>,
    /// Executor iterations, in worker-ID order, for workers that had work.
    pub workers: Vec<(WorkerId, ExecutorIteration)>,
    /// Client events in delivery order.
    pub client: Vec<(ClusterRequestId, ClientEvent)>,
    /// Cluster trace records in order.
    pub trace: Vec<ClusterTraceRecord>,
}

/// A failure inside a local cluster.
#[derive(Debug)]
pub enum LocalClusterError {
    /// The coordinator rejected a call.
    Coordinator(CoordinatorError),
    /// A worker's executor failed.
    Executor {
        /// Worker whose executor failed.
        worker: WorkerId,
        /// Executor failure.
        error: ExecutorError,
    },
}

impl fmt::Display for LocalClusterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordinator(error) => error.fmt(formatter),
            Self::Executor { worker, error } => write!(formatter, "worker `{worker}`: {error}"),
        }
    }
}

impl Error for LocalClusterError {}

impl From<CoordinatorError> for LocalClusterError {
    fn from(error: CoordinatorError) -> Self {
        Self::Coordinator(error)
    }
}

/// Connects a [`Coordinator`] to [`WorkerHost`]s in one process.
///
/// Messages are delivered synchronously and workers step in ID order, so a
/// run is fully determined by its inputs and timestamps.
pub struct LocalCluster {
    coordinator: Coordinator,
    workers: BTreeMap<WorkerId, WorkerHost>,
    buffered: LocalClusterStep,
}

impl LocalCluster {
    /// Creates a cluster with no workers.
    pub fn new<P: ClusterPolicy + 'static>(policy: P) -> Self {
        Self {
            coordinator: Coordinator::new(policy),
            workers: BTreeMap::new(),
            buffered: LocalClusterStep::default(),
        }
    }

    /// Joins a worker.
    ///
    /// # Errors
    ///
    /// Fails for an empty or duplicate worker ID.
    pub fn add_worker(
        &mut self,
        worker: WorkerHost,
        at: Timestamp,
    ) -> Result<(), LocalClusterError> {
        self.coordinator.join(worker.summary(at))?;
        self.workers.insert(worker.id().clone(), worker);
        Ok(())
    }

    /// The coordinator.
    pub const fn coordinator(&self) -> &Coordinator {
        &self.coordinator
    }

    /// A worker by ID.
    pub fn worker(&self, id: &WorkerId) -> Option<&WorkerHost> {
        self.workers.get(id)
    }

    /// Mutable access to a worker, for changing its health.
    pub fn worker_mut(&mut self, id: &WorkerId) -> Option<&mut WorkerHost> {
        self.workers.get_mut(id)
    }

    /// Submits a request; it is routed at the next step.
    ///
    /// # Errors
    ///
    /// Fails if the request is invalid.
    pub fn submit(
        &mut self,
        spec: RequestSpec,
        at: Timestamp,
    ) -> Result<ClusterRequestId, LocalClusterError> {
        let (id, output) = self.coordinator.submit(spec, at)?;
        self.deliver(output, at)?;
        Ok(id)
    }

    /// Cancels a request.
    ///
    /// # Errors
    ///
    /// Fails for an unknown request or if the owner cannot cancel it.
    pub fn cancel(&mut self, id: ClusterRequestId, at: Timestamp) -> Result<(), LocalClusterError> {
        let output = self.coordinator.cancel(id, at)?;
        self.deliver(output, at)
    }

    /// Whether nothing is pending and no worker has work.
    pub fn is_idle(&self) -> bool {
        self.coordinator.pending_len() == 0 && self.workers.values().all(|w| !w.has_work())
    }

    /// Routes pending requests, steps every worker with work once, and
    /// refreshes each worker's summary.
    ///
    /// # Errors
    ///
    /// Fails on a coordinator or executor error. Work committed before the
    /// failure stays committed.
    pub fn step(&mut self, at: Timestamp) -> Result<LocalClusterStep, LocalClusterError> {
        if let Some((iteration, output)) = self.coordinator.plan(at)? {
            self.buffered.cluster = Some(iteration);
            self.deliver(output, at)?;
        }
        let ids = self.workers.keys().cloned().collect::<Vec<_>>();
        for id in ids {
            let worker = self.workers.get_mut(&id).expect("ids come from the map");
            let stepped = worker
                .step(at)
                .map_err(|error| LocalClusterError::Executor {
                    worker: id.clone(),
                    error,
                })?;
            if let Some((iteration, messages)) = stepped {
                self.buffered.workers.push((id.clone(), iteration));
                for message in messages {
                    let output = self.coordinator.handle(&id, message, at)?;
                    self.deliver(output, at)?;
                }
            }
            let summary = self.workers[&id].summary(at);
            self.coordinator
                .handle(&id, crate::WorkerMessage::Summary(summary), at)?;
        }
        Ok(std::mem::take(&mut self.buffered))
    }

    /// Delivers commands to workers and their replies to the coordinator
    /// until nothing is left to send.
    fn deliver(
        &mut self,
        output: CoordinatorOutput,
        at: Timestamp,
    ) -> Result<(), LocalClusterError> {
        let mut queue = std::collections::VecDeque::from([output]);
        while let Some(output) = queue.pop_front() {
            self.buffered.client.extend(output.client);
            self.buffered.trace.extend(output.trace);
            for (id, command) in output.commands {
                let worker = self
                    .workers
                    .get_mut(&id)
                    .ok_or_else(|| CoordinatorError::UnknownWorker(id.clone()))?;
                let messages =
                    worker
                        .handle(command, at)
                        .map_err(|error| LocalClusterError::Executor {
                            worker: id.clone(),
                            error,
                        })?;
                for message in messages {
                    queue.push_back(self.coordinator.handle(&id, message, at)?);
                }
            }
        }
        Ok(())
    }
}
