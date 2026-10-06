/// Connectivity state for one UI integration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ConnectionState {
    /// The integration can accept work.
    Connected,
    /// The integration is performing work or reconnecting.
    Busy,
    /// No implementation has been attached.
    #[default]
    Disconnected,
    /// The most recent integration operation failed.
    Failed(String),
}

impl ConnectionState {
    pub(crate) fn label(&self) -> &str {
        match self {
            Self::Connected => "connected",
            Self::Busy => "busy",
            Self::Disconnected => "disconnected",
            Self::Failed(_) => "failed",
        }
    }
}

/// Runtime measurements displayed by the overview screen.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeSummary {
    /// Runtime connection state.
    pub connection: ConnectionState,
    /// Human-readable execution backend, such as `Burn NdArray/F32`.
    pub backend: String,
    /// Number of resident models.
    pub resident_models: usize,
    /// Requests waiting for scheduling.
    pub queued_requests: usize,
    /// Requests currently executing.
    pub active_requests: usize,
    /// Total configured memory capacity.
    pub memory_capacity_bytes: u64,
    /// Total accounted memory usage.
    pub memory_used_bytes: u64,
    /// Memory currently reserved for KV cache.
    pub kv_cache_bytes: u64,
}

/// Origin of one chat transcript entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    /// Human input.
    User,
    /// Model output.
    Assistant,
    /// Runtime or connector information.
    System,
}

/// One message displayed in the chat transcript.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    /// Message origin.
    pub role: ChatRole,
    /// Rendered message text.
    pub content: String,
}

/// Lifecycle state of a benchmark or experiment entry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum RunState {
    /// The item can be launched.
    #[default]
    Ready,
    /// A run is currently active.
    Running,
    /// The last run completed successfully.
    Succeeded,
    /// The last run failed.
    Failed,
    /// No implementation is connected for this item.
    Unavailable,
}

impl RunState {
    pub(crate) const fn label(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Succeeded => "passed",
            Self::Failed => "failed",
            Self::Unavailable => "unavailable",
        }
    }
}

/// One launchable or observable benchmark/experiment entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkItem {
    /// Stable adapter-facing identifier.
    pub id: String,
    /// Short display name.
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Current lifecycle state.
    pub state: RunState,
    /// Optional compact result from the most recent run.
    pub last_result: Option<String>,
}

/// Complete UI data snapshot supplied by an integration backend.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkspaceSnapshot {
    /// Runtime and memory measurements.
    pub runtime: RuntimeSummary,
    /// Selected model name, if an LLM connector has selected one.
    pub selected_model: Option<String>,
    /// Current chat transcript.
    pub chat: Vec<ChatMessage>,
    /// Available or recently executed benchmarks.
    pub benchmarks: Vec<WorkItem>,
    /// Available or recently executed experiments.
    pub experiments: Vec<WorkItem>,
    /// Recent human-readable activity, newest last.
    pub activity: Vec<String>,
    /// Important connector notice shown in the overview.
    pub notice: Option<String>,
}
