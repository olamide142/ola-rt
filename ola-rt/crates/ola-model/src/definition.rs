use std::path::PathBuf;

use ola_core::ModelDescriptor;
use serde::{Deserialize, Serialize};

use crate::ModelError;

/// Location from which an architecture factory resolves model artifacts.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSource {
    /// Directory already materialized on the local filesystem.
    LocalDirectory(PathBuf),
    /// Hugging Face repository reference, optionally pinned to a revision.
    HuggingFace {
        /// Repository identifier such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
        repository: String,
        /// Optional immutable commit, tag, or branch reference.
        revision: Option<String>,
    },
}

/// Selectable model metadata and artifact source.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct ModelDefinition {
    /// Backend-neutral capabilities and architecture identity.
    pub descriptor: ModelDescriptor,
    /// Artifact location interpreted by the selected factory.
    pub source: ModelSource,
}

impl ModelDefinition {
    /// Creates a definition for artifacts already present in a local directory.
    pub fn local(descriptor: ModelDescriptor, directory: impl Into<PathBuf>) -> Self {
        Self {
            descriptor,
            source: ModelSource::LocalDirectory(directory.into()),
        }
    }

    /// Creates a definition for a Hugging Face repository reference.
    pub fn hugging_face(
        descriptor: ModelDescriptor,
        repository: impl Into<String>,
        revision: Option<String>,
    ) -> Self {
        Self {
            descriptor,
            source: ModelSource::HuggingFace {
                repository: repository.into(),
                revision,
            },
        }
    }

    /// Validates descriptor and source metadata without accessing artifacts.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.descriptor.validate()?;
        match &self.source {
            ModelSource::LocalDirectory(path) if path.as_os_str().is_empty() => {
                Err(ModelError::InvalidSource {
                    model: self.descriptor.id.clone(),
                    reason: "local directory cannot be empty",
                })
            }
            ModelSource::HuggingFace { repository, .. } if repository.trim().is_empty() => {
                Err(ModelError::InvalidSource {
                    model: self.descriptor.id.clone(),
                    reason: "Hugging Face repository cannot be empty",
                })
            }
            ModelSource::HuggingFace {
                revision: Some(revision),
                ..
            } if revision.trim().is_empty() => Err(ModelError::InvalidSource {
                model: self.descriptor.id.clone(),
                reason: "Hugging Face revision cannot be empty when present",
            }),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ola_core::ModelDescriptor;

    #[test]
    fn model_definitions_round_trip_through_json() {
        let definition = ModelDefinition::hugging_face(
            ModelDescriptor::language("tiny", "llama"),
            "org/model",
            Some("revision".into()),
        );
        let encoded = serde_json::to_string(&definition).unwrap();
        let decoded: ModelDefinition = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, definition);
    }
}
