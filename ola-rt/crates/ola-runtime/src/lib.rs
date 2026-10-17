#![doc = include_str!("../README.md")]

mod admission;
mod batch;
mod composed;
mod decision;
mod error;
mod executor;
mod kv_cache;
mod model_registry;
mod plan;
mod policy;
mod runtime;
mod scheduler;
mod snapshot;
mod state;

pub use admission::{
    AdmissionContext, AdmissionDecision, AdmissionPolicy, AdmissionReason, MemoryAwareAdmission,
    UnconstrainedAdmission,
};
pub use batch::{
    BatchCandidate, BatchContext, BatchExecutionMode, BatchPhase, BatchPolicy, BatchPressure,
    CostAwarePrefillBatcher, CostAwarePrefillDecision, ExecutionBatch, FifoTokenBudgetPolicy,
    MixedTokenBudgetPolicy, PowerOfTwoPrefillBatcher, PrefillBatcher, PrefillGroupingEstimate,
    PrefillGroupingStrategy, QueueAdaptiveTokenBudget, UnboundedBatchPolicy,
    UnbucketedPrefillBatcher,
};
pub use composed::{ComposedPolicy, HeadOfLine};
pub use decision::{
    AdmissionOutcome, AdmissionRecord, BatchHook, CapabilityFallback, DecisionRecord,
    GroupingDecision, GroupingRecord, RankingRecord, SelectionRecord, SelectionStage,
};
pub use error::RuntimeError;
pub use executor::{
    ExecutionPhase, ExecutionResult, Executor, ExecutorError, ExecutorEvent, ExecutorIteration,
};
pub use kv_cache::{
    KvBlockError, KvBlockId, KvBlockManager, KvBlockMetrics, KvRelease, KvSequenceId,
};
pub use model_registry::ModelRegistry;
pub use plan::{
    DecodeView, ExecutionPlan, IterationId, LifecycleAction, PlanDivergence, PlanError,
    PlanEstimates, PlannedBatch, PlannedDeferral, PlannedPhase, PlannedRow, PrefillView,
    ReleaseReason,
};
pub use policy::{ExecutionPolicy, PolicyDecision, PolicyDescriptor, PolicyError};
pub use runtime::Runtime;
pub use scheduler::{
    EarliestDeadlineFirst, FifoScheduler, JobCost, PriorityScheduler, Scheduler, ShortestJobFirst,
};
pub use snapshot::{
    DeviceSnapshot, ExecutorLimits, ModelSnapshot, PlanningSnapshot, PolicyMeasurements,
    QueuedRequest,
};
pub use state::{
    InferenceSession, MemoryState, RequestRecord, RequestStatus, RequestTiming, RuntimeState,
};
