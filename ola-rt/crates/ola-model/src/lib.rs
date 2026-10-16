#![doc = include_str!("../README.md")]

mod catalog;
mod definition;
mod error;
mod execution;
mod factory;

pub use catalog::ModelCatalog;
pub use definition::{ModelDefinition, ModelSource};
pub use error::{ArchitectureLoadError, ModelError};
pub use execution::{
    CausalLanguageModel, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
    KvCacheProfile, ModelExecutionError, PrefillOutcome, SessionMetrics, Tokenizer,
};
pub use factory::{ArchitectureFactory, ArchitectureRegistry, LoadedModel};
