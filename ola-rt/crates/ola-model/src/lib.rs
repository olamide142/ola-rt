#![doc = include_str!("../README.md")]

mod catalog;
mod definition;
mod error;
mod execution;
mod factory;
mod streaming;

pub use catalog::ModelCatalog;
pub use definition::{ModelDefinition, ModelSource};
pub use error::{ArchitectureLoadError, ModelError};
pub use execution::{
    BatchCapabilities, CachedPrefix, CausalLanguageModel, CausalModel, ExecutionStage,
    GeneratedToken, GenerationSession, KvCacheProfile, KvPoolStatus, ModelExecutionContext,
    ModelExecutionError, ModelMemoryProfile, PrefillOutcome, SessionMetrics, Tokenizer,
};
pub use factory::{ArchitectureFactory, ArchitectureRegistry, LoadedModel};
pub use streaming::IncrementalDecoder;
