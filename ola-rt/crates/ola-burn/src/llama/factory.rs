use std::{
    error::Error,
    fmt,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use ola_core::ArchitectureId;
use ola_model::{
    ArchitectureFactory, ArchitectureLoadError, CausalLanguageModel, ModelDefinition, ModelSource,
};

use super::{
    HuggingFaceTokenizer, LlamaConfig, LlamaConfigError, LlamaForCausalLM, session::BurnLlamaModel,
};

/// Constructs Llama models on one selected Burn backend and device.
///
/// The factory is registered under an architecture ID (normally `llama`). Model
/// IDs and paths remain catalog data, so the same factory can load TinyLlama or
/// another compatible untied, unscaled Llama checkpoint.
pub struct LlamaFactory<B: Backend> {
    architecture: ArchitectureId,
    device: B::Device,
    backend: PhantomData<fn() -> B>,
}

impl<B: Backend> LlamaFactory<B> {
    /// Creates a factory registered as the canonical `llama` architecture.
    pub fn new(device: B::Device) -> Self {
        Self {
            architecture: "llama".into(),
            device,
            backend: PhantomData,
        }
    }

    /// Creates a factory under a custom architecture alias.
    ///
    /// Aliases are useful when experiments intentionally compare two Llama
    /// implementations through the same registry.
    pub fn with_architecture(device: B::Device, architecture: impl Into<ArchitectureId>) -> Self {
        Self {
            architecture: architecture.into(),
            device,
            backend: PhantomData,
        }
    }

    /// Loads a local Hugging Face artifact directory with detailed errors.
    pub fn load_local(
        &self,
        directory: impl AsRef<Path>,
    ) -> Result<CausalLanguageModel, LlamaLoadError>
    where
        B: 'static,
        B::Device: Send + Sync,
    {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            return Err(LlamaLoadError::NotDirectory(directory.to_path_buf()));
        }
        let config = LlamaConfig::from_file(directory.join("config.json"))?;
        let tokenizer = HuggingFaceTokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| LlamaLoadError::Tokenizer(error.to_string()))?;
        let weights = find_single_safetensors_file(directory)?;

        let mut module = LlamaForCausalLM::<B>::init(&config, &self.device);
        // Burn stores RMSNorm scale as `gamma`, while Hugging Face Llama uses
        // `weight`. The general PyTorch adapter cannot infer these nested
        // architecture-specific containers, so keep the mapping explicit and
        // narrow rather than allowing a partial checkpoint load.
        let mut store = SafetensorsStore::from_file(weights)
            .with_key_remapping(r"layernorm\.weight$", "layernorm.gamma")
            .with_key_remapping(r"^model\.norm\.weight$", "model.norm.gamma")
            .with_from_adapter(PyTorchToBurnAdapter);
        module
            .load_from(&mut store)
            .map_err(|error| LlamaLoadError::Weights(error.to_string()))?;

        Ok(CausalLanguageModel::new(
            tokenizer,
            BurnLlamaModel::new(module, config, self.device.clone()),
        ))
    }
}

impl<B> ArchitectureFactory<CausalLanguageModel> for LlamaFactory<B>
where
    B: Backend + 'static,
    B::Device: Send + Sync,
{
    fn architecture(&self) -> &ArchitectureId {
        &self.architecture
    }

    fn load(
        &self,
        definition: &ModelDefinition,
    ) -> Result<CausalLanguageModel, ArchitectureLoadError> {
        match &definition.source {
            ModelSource::LocalDirectory(directory) => self
                .load_local(directory)
                .map_err(|error| ArchitectureLoadError::new(error.to_string())),
            ModelSource::HuggingFace {
                repository,
                revision,
            } => Err(ArchitectureLoadError::new(format!(
                "Hugging Face source `{repository}`{} is not materialized; download it into a local directory before loading",
                revision
                    .as_deref()
                    .map(|value| format!(" at revision `{value}`"))
                    .unwrap_or_default()
            ))),
        }
    }
}

fn find_single_safetensors_file(directory: &Path) -> Result<PathBuf, LlamaLoadError> {
    let canonical = directory.join("model.safetensors");
    if canonical.is_file() {
        return Ok(canonical);
    }
    let mut candidates = std::fs::read_dir(directory)
        .map_err(|error| LlamaLoadError::ReadDirectory(error.to_string()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    candidates.sort();
    match candidates.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(LlamaLoadError::MissingWeights(directory.to_path_buf())),
        _ => Err(LlamaLoadError::ShardedWeights(candidates)),
    }
}

/// Failure while materializing a local Llama artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaLoadError {
    /// Artifact path is not a directory.
    NotDirectory(PathBuf),
    /// Llama configuration could not be parsed or is unsupported.
    Config(LlamaConfigError),
    /// Tokenizer construction failed.
    Tokenizer(String),
    /// Artifact directory could not be enumerated.
    ReadDirectory(String),
    /// No SafeTensors weights file was found.
    MissingWeights(PathBuf),
    /// More than one weights shard was found; shard indexes are not supported yet.
    ShardedWeights(Vec<PathBuf>),
    /// SafeTensors could not be mapped completely onto the Burn module.
    Weights(String),
}

impl fmt::Display for LlamaLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDirectory(path) => {
                write!(
                    formatter,
                    "artifact path `{}` is not a directory",
                    path.display()
                )
            }
            Self::Config(error) => error.fmt(formatter),
            Self::Tokenizer(message) => write!(formatter, "failed to load tokenizer: {message}"),
            Self::ReadDirectory(message) => {
                write!(formatter, "failed to inspect artifact directory: {message}")
            }
            Self::MissingWeights(path) => write!(
                formatter,
                "no `.safetensors` weights file found in `{}`",
                path.display()
            ),
            Self::ShardedWeights(paths) => write!(
                formatter,
                "found {} SafeTensors shards; indexed checkpoints are not implemented yet",
                paths.len()
            ),
            Self::Weights(message) => write!(formatter, "failed to load model weights: {message}"),
        }
    }
}

impl Error for LlamaLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LlamaConfigError> for LlamaLoadError {
    fn from(value: LlamaConfigError) -> Self {
        Self::Config(value)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    #[test]
    fn factory_is_registered_by_architecture_not_model_name() {
        let factory = LlamaFactory::<NdArray>::new(Default::default());

        assert_eq!(factory.architecture().as_str(), "llama");
    }
}
