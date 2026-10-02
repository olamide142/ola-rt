use std::{collections::BTreeSet, error::Error, fmt};

use crate::{ArchitectureId, Modality, ModelId};

/// Broad execution family for a registered model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    /// A text/token model, typically an autoregressive decoder.
    Language,
    /// A model combining a vision encoder with language generation.
    VisionLanguage,
    /// A vision-only model such as a detector or segmenter.
    Vision,
}

/// Runtime-visible model metadata, independent of any compute backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDescriptor {
    /// Stable identifier used by requests.
    pub id: ModelId,
    /// Architecture implementation family, such as `llama` or `qwen2`.
    pub architecture: ArchitectureId,
    /// Broad execution family used for admission and observability.
    pub kind: ModelKind,
    /// Input modalities accepted by this model.
    pub supported_modalities: Vec<Modality>,
    /// Optional total context-window limit in tokens.
    pub max_context_tokens: Option<usize>,
}

impl ModelDescriptor {
    /// Constructs metadata for a text/token language model.
    pub fn language(id: impl Into<ModelId>, architecture: impl Into<ArchitectureId>) -> Self {
        Self {
            id: id.into(),
            architecture: architecture.into(),
            kind: ModelKind::Language,
            supported_modalities: vec![Modality::Text, Modality::Tokens],
            max_context_tokens: None,
        }
    }

    /// Validates metadata required by catalogs and runtime admission.
    pub fn validate(&self) -> Result<(), ModelDescriptorError> {
        if self.id.is_empty() {
            return Err(ModelDescriptorError::EmptyModelId);
        }
        if self.architecture.is_empty() {
            return Err(ModelDescriptorError::EmptyArchitectureId);
        }
        if self.supported_modalities.is_empty() {
            return Err(ModelDescriptorError::NoSupportedModalities);
        }
        let unique_modalities = self
            .supported_modalities
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if unique_modalities.len() != self.supported_modalities.len() {
            return Err(ModelDescriptorError::DuplicateModality);
        }
        if self.max_context_tokens == Some(0) {
            return Err(ModelDescriptorError::ZeroContextLength);
        }
        Ok(())
    }
}

/// Invalid backend-neutral model metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelDescriptorError {
    /// Model identifier is empty or whitespace.
    EmptyModelId,
    /// Architecture identifier is empty or whitespace.
    EmptyArchitectureId,
    /// Model declares no accepted input modality.
    NoSupportedModalities,
    /// One accepted modality appears more than once.
    DuplicateModality,
    /// A present context-window limit is zero.
    ZeroContextLength,
}

impl fmt::Display for ModelDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyModelId => "model id cannot be empty",
            Self::EmptyArchitectureId => "architecture id cannot be empty",
            Self::NoSupportedModalities => "model must support at least one input modality",
            Self::DuplicateModality => "supported modalities must not contain duplicates",
            Self::ZeroContextLength => "model context length must be greater than zero",
        };
        formatter.write_str(message)
    }
}

impl Error for ModelDescriptorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_model_identity_and_capabilities() {
        let mut descriptor = ModelDescriptor::language("tiny", "llama");
        assert_eq!(descriptor.validate(), Ok(()));

        descriptor.supported_modalities.push(Modality::Text);
        assert_eq!(
            descriptor.validate(),
            Err(ModelDescriptorError::DuplicateModality)
        );
    }
}
