#![doc = include_str!("../README.md")]

mod admission;
mod adaptive;
mod batch;
mod builder;
mod composed;
mod config;
mod decision;
mod error;
mod executor;
mod kv_cache;
mod model_registry;
mod plan;
mod policy;
mod preemption;
mod runtime;
mod scheduler;
mod snapshot;
mod state;

pub use admission::{
    AdmissionContext, AdmissionDecision, AdmissionPolicy, AdmissionReason, MemoryAwareAdmission,
    UnconstrainedAdmission, WatermarkAdmission,
};
pub use adaptive::{
    AdaptiveConstraints, AdaptiveIntegerDomain, AdaptiveLinearConstraint, AdaptiveMemoryConstraint,
    AdaptiveObjectiveMetrics,
    AdaptiveObjectiveWeights, AdaptiveOptimizationError, AdaptiveOptimizer, AdaptivePolicy,
    AdaptivePolicyFactory, AdaptiveRequestObservation, FiniteCandidateOptimizer,
    MilpCandidateOptimizer,
};
pub use batch::{
    BatchCandidate, BatchContext, BatchExecutionMode, BatchPhase, BatchPolicy, BatchPressure,
    CostAwarePrefillBatcher, CostAwarePrefillDecision, ExecutedKernel, ExecutionBatch,
    FifoTokenBudgetPolicy, MixedTokenBudgetPolicy, PowerOfTwoPrefillBatcher, PrefillBatcher,
    PrefillGroupingEstimate, PrefillGroupingStrategy, QueueAdaptiveTokenBudget,
    UnboundedBatchPolicy, UnbucketedPrefillBatcher,
};
pub use builder::ExecutorBuilder;
pub use composed::{ComposedPolicy, HeadOfLine};
pub use config::{BatchPolicyConfig, ExecutorConfig};
pub use decision::{
    AdmissionOutcome, AdmissionRecord, BatchHook, CapabilityFallback, DecisionRecord,
    GroupingDecision, GroupingRecord, PreemptionRecord, RankingRecord, SelectionRecord,
    SelectionStage,
};
pub use error::RuntimeError;
pub use executor::{
    ExecutionPhase, ExecutionResult, Executor, ExecutorError, ExecutorEvent, ExecutorIteration,
    ModelLifecycleEvent, ModelPlacement,
};
pub use kv_cache::{
    KvBlockError, KvBlockId, KvBlockManager, KvBlockMetrics, KvRelease, KvSequenceId, PrefixMatch,
};
pub use model_registry::ModelRegistry;
pub use plan::{
    DecodeView, ExecutionPlan, IterationId, LifecycleAction, PlanDivergence, PlanError,
    PlanEstimates, PlannedBatch, PlannedDeferral, PlannedKernel, PlannedPhase, PlannedRow,
    PreemptionReason, PrefillView, ReleaseReason,
};
pub use policy::{ExecutionPolicy, PolicyDecision, PolicyDescriptor, PolicyError};
pub use preemption::{
    LowestPriorityFirst, NewestFirst, NoPreemption, PreemptionCandidate, PreemptionPolicy,
};
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
