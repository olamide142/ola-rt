use std::{error::Error, fmt};

use ola_core::{ArchitectureId, ModelDescriptorError, ModelId};

/// Architecture-specific construction failure before registry context is added.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchitectureLoadError {
    message: String,
}

impl ArchitectureLoadError {
    /// Creates an architecture-specific diagnostic.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the factory-provided diagnostic text.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ArchitectureLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ArchitectureLoadError {}

/// Failure while defining, selecting, or constructing a model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    /// Descriptor violates a backend-neutral invariant.
    InvalidDescriptor(ModelDescriptorError),
    /// Source metadata is incomplete or malformed.
    InvalidSource {
        /// Model whose source is invalid.
        model: ModelId,
        /// Stable description of the violated invariant.
        reason: &'static str,
    },
    /// Catalog already contains the model identifier.
    DuplicateModel(ModelId),
    /// Catalog does not contain the selected model identifier.
    UnknownModel(ModelId),
    /// Architecture registry already contains this implementation identifier.
    DuplicateArchitecture(ArchitectureId),
    /// Factory reported an empty architecture identifier.
    InvalidArchitecture(ArchitectureId),
    /// No factory is registered for the selected model's architecture.
    UnsupportedArchitecture {
        /// Selected model.
        model: ModelId,
        /// Missing architecture implementation.
        architecture: ArchitectureId,
    },
    /// Selected factory could not construct the model.
    Load {
        /// Selected model.
        model: ModelId,
        /// Factory implementation that failed.
        architecture: ArchitectureId,
        /// Architecture-specific diagnostic.
        source: ArchitectureLoadError,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(error) => {
                write!(formatter, "invalid model descriptor: {error}")
            }
            Self::InvalidSource { model, reason } => {
                write!(formatter, "invalid source for model `{model}`: {reason}")
            }
            Self::DuplicateModel(model) => {
                write!(
                    formatter,
                    "model `{model}` is already present in the catalog"
                )
            }
            Self::UnknownModel(model) => write!(formatter, "model `{model}` is not in the catalog"),
            Self::DuplicateArchitecture(architecture) => write!(
                formatter,
                "architecture `{architecture}` already has a registered factory"
            ),
            Self::InvalidArchitecture(architecture) => {
                write!(
                    formatter,
                    "architecture id `{architecture}` cannot be empty"
                )
            }
            Self::UnsupportedArchitecture {
                model,
                architecture,
            } => write!(
                formatter,
                "model `{model}` requires unsupported architecture `{architecture}`"
            ),
            Self::Load {
                model,
                architecture,
                source,
            } => write!(
                formatter,
                "failed to load model `{model}` with architecture `{architecture}`: {source}"
            ),
        }
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidDescriptor(error) => Some(error),
            Self::Load { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<ModelDescriptorError> for ModelError {
    fn from(value: ModelDescriptorError) -> Self {
        Self::InvalidDescriptor(value)
    }
}
