use std::{error::Error, fmt};

use crate::{ConnectionState, RunState, RuntimeSummary, WorkItem, WorkspaceSnapshot};

/// Integration boundary between the terminal shell and OlaRT capabilities.
///
/// Implementations may call in-process crates, communicate with a server, or
/// read from channels owned by background workers. Methods are synchronous at
/// this boundary; long-running implementations should enqueue work and expose
/// progress through subsequent snapshots.
pub trait UiBackend {
    /// Returns the newest coherent data snapshot.
    fn snapshot(&mut self) -> Result<WorkspaceSnapshot, UiBackendError>;

    /// Submits one prompt to the selected language model.
    fn submit_prompt(&mut self, prompt: &str) -> Result<(), UiBackendError>;

    /// Starts the benchmark identified by `id`.
    fn start_benchmark(&mut self, id: &str) -> Result<(), UiBackendError>;

    /// Starts the experiment identified by `id`.
    fn start_experiment(&mut self, id: &str) -> Result<(), UiBackendError>;
}

/// Human-readable adapter failure suitable for the UI status line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiBackendError {
    message: String,
}

impl UiBackendError {
    /// Creates an integration error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for UiBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for UiBackendError {}

/// Honest preview backend used until concrete OlaRT adapters are attached.
#[derive(Clone, Debug)]
pub struct DisconnectedBackend {
    snapshot: WorkspaceSnapshot,
}

impl Default for DisconnectedBackend {
    fn default() -> Self {
        Self {
            snapshot: WorkspaceSnapshot {
                runtime: RuntimeSummary {
                    connection: ConnectionState::Disconnected,
                    backend: "not connected".into(),
                    ..RuntimeSummary::default()
                },
                benchmarks: vec![
                    placeholder(
                        "llama-generation",
                        "Llama generation",
                        "Latency and token-throughput benchmark",
                    ),
                    placeholder(
                        "kv-block-pressure",
                        "KV block pressure",
                        "Allocation, fragmentation, and capacity workload",
                    ),
                ],
                experiments: vec![
                    placeholder(
                        "scheduler-policy",
                        "Scheduler policy",
                        "Compare FIFO with future cache-aware policies",
                    ),
                    placeholder(
                        "llama-conformance",
                        "Llama conformance",
                        "Compare model logits with the pinned scalar oracle",
                    ),
                ],
                activity: vec!["UI shell ready; integrations are not connected".into()],
                notice: Some(
                    "Attach a UiBackend implementation to connect models, workloads, and experiments"
                        .into(),
                ),
                ..WorkspaceSnapshot::default()
            },
        }
    }
}

impl UiBackend for DisconnectedBackend {
    fn snapshot(&mut self) -> Result<WorkspaceSnapshot, UiBackendError> {
        Ok(self.snapshot.clone())
    }

    fn submit_prompt(&mut self, _prompt: &str) -> Result<(), UiBackendError> {
        Err(not_connected("LLM"))
    }

    fn start_benchmark(&mut self, _id: &str) -> Result<(), UiBackendError> {
        Err(not_connected("benchmark runner"))
    }

    fn start_experiment(&mut self, _id: &str) -> Result<(), UiBackendError> {
        Err(not_connected("experiment runner"))
    }
}

fn placeholder(id: &str, name: &str, description: &str) -> WorkItem {
    WorkItem {
        id: id.into(),
        name: name.into(),
        description: description.into(),
        state: RunState::Unavailable,
        last_result: None,
    }
}

fn not_connected(capability: &str) -> UiBackendError {
    UiBackendError::new(format!("{capability} is not connected"))
}
