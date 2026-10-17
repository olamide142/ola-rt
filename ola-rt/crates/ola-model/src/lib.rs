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
    BatchCapabilities, CausalLanguageModel, CausalModel, ExecutionStage, GeneratedToken,
    GenerationSession, KvCacheProfile, KvPoolStatus, ModelExecutionError, PrefillOutcome,
    SessionMetrics, Tokenizer,
};
pub use factory::{ArchitectureFactory, ArchitectureRegistry, LoadedModel};
pub use streaming::IncrementalDecoder;
