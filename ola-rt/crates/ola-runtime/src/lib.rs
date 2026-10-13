#![doc = include_str!("../README.md")]

mod admission;
mod batch;
mod error;
mod executor;
mod kv_cache;
mod model_registry;
mod runtime;
mod scheduler;
mod state;

pub use admission::{
    AdmissionContext, AdmissionDecision, AdmissionPolicy, AdmissionReason, MemoryAwareAdmission,
    UnconstrainedAdmission,
};
pub use batch::{
    BatchCandidate, BatchContext, BatchExecutionMode, BatchPhase, BatchPolicy, ExecutionBatch,
    FifoTokenBudgetPolicy, IterationPlan, PowerOfTwoPrefillBatcher, PrefillBatcher,
    UnboundedBatchPolicy, UnbucketedPrefillBatcher,
};
pub use error::RuntimeError;
pub use executor::{
    ExecutionPhase, ExecutionResult, Executor, ExecutorError, ExecutorEvent, ExecutorIteration,
};
pub use kv_cache::{
    KvBlockError, KvBlockId, KvBlockManager, KvBlockMetrics, KvRelease, KvSequenceId,
};
pub use model_registry::ModelRegistry;
pub use runtime::Runtime;
pub use scheduler::{FifoScheduler, Scheduler};
pub use state::{
    InferenceSession, MemoryState, RequestRecord, RequestStatus, RequestTiming, RuntimeState,
};
