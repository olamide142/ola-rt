use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

use burn::tensor::Device;
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter,
    SafetensorsStore,
};
use ola_core::ArchitectureId;
use ola_model::{
    ArchitectureFactory, ArchitectureLoadError, CausalLanguageModel, ModelDefinition, ModelSource,
};

use super::{
    HuggingFaceTokenizer, MistralConfig, MistralConfigError, MistralForCausalLM,
    MistralKvCacheConfig, session::BurnMistralModel,
};

/// Constructs Mistral models on one selected Burn backend and device.
///
/// The factory is registered under an architecture ID (normally `mistral`).
/// Model IDs and paths remain catalog data, so the same factory can load
/// any compatible untied, unscaled Mistral checkpoint.
pub struct MistralFactory {
    architecture: ArchitectureId,
    device: Device,
    kv_cache: Option<MistralKvCacheConfig>,
}

/// Native Burn Mistral module paired with the configuration that shaped it.
///
/// Most callers should use [`MistralFactory::load_local`] and the neutral
/// [`CausalLanguageModel`] interface. This representation exists for research
/// probes that need logits or hidden tensor outputs without widening the core
/// runtime API to include backend-specific tensors.
pub struct LoadedMistral {
    config: MistralConfig,
    module: MistralForCausalLM,
}

impl LoadedMistral {
    /// Returns the validated Hugging Face configuration.
    pub const fn config(&self) -> &MistralConfig {
        &self.config
    }

    /// Returns the native Burn module for explicit research measurements.
    pub const fn module(&self) -> &MistralForCausalLM {
        &self.module
    }

    /// Consumes the wrapper into configuration and module parts.
    pub fn into_parts(self) -> (MistralConfig, MistralForCausalLM) {
        (self.config, self.module)
    }
}

impl MistralFactory {
    /// Creates a factory registered as the canonical `mistral` architecture.
    pub fn new(device: Device) -> Self {
        Self {
            architecture: "mistral".into(),
            device,
            kv_cache: None,
        }
    }

    /// Creates a factory under a custom architecture alias.
    pub fn with_architecture(device: Device, architecture: impl Into<ArchitectureId>) -> Self {
        Self {
            architecture: architecture.into(),
            device,
            kv_cache: None,
        }
    }

    /// Overrides the shared KV-cache capacity and page geometry.
    ///
    /// By default, the factory reserves logical capacity for one complete
    /// model context with 16 token positions per physical block.
    pub fn with_kv_cache(mut self, config: MistralKvCacheConfig) -> Self {
        self.kv_cache = Some(config);
        self
    }

    /// Loads a local Hugging Face artifact directory with detailed errors.
    pub fn load_local(
        &self,
        directory: impl AsRef<Path>,
    ) -> Result<CausalLanguageModel, MistralLoadError> {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            return Err(MistralLoadError::NotDirectory(directory.to_path_buf()));
        }
        let tokenizer = HuggingFaceTokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| MistralLoadError::Tokenizer(error.to_string()))?;
        let (config, module) = self.load_native(directory)?.into_parts();

        let cache_config = self
            .kv_cache
            .unwrap_or_else(|| MistralKvCacheConfig::for_model(&config));
        let cache_profile = cache_config
            .profile(&config)
            .map_err(|error| MistralLoadError::KvCache(error.to_string()))?;
        let attention_profile = ola_kernel::PagedAttentionProfile::new(
            ola_kernel::burn_device_target(&self.device),
            ola_kernel::KernelDType::F32,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_head_dim(),
            cache_config.tokens_per_block(),
        )
        .expect("validated Mistral and KV-cache dimensions form a kernel profile");
        let model =
            BurnMistralModel::with_kv_cache(module, config, self.device.clone(), cache_config)
                .map_err(|error| MistralLoadError::KvCache(error.to_string()))?;
        Ok(CausalLanguageModel::new(tokenizer, model)
            .with_kv_cache_profile(cache_profile)
            .with_paged_attention_profile(attention_profile))
    }

    /// Loads configuration and weights while leaving tokenization to the caller.
    pub fn load_native(
        &self,
        directory: impl AsRef<Path>,
    ) -> Result<LoadedMistral, MistralLoadError> {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            return Err(MistralLoadError::NotDirectory(directory.to_path_buf()));
        }
        let config = MistralConfig::from_file(directory.join("config.json"))?;
        let weights = find_safetensors_files(directory)?;
        let mut module = MistralForCausalLM::init(&config, &self.device);
        load_safetensors_shards(&mut module, &weights)?;
        Ok(LoadedMistral { config, module })
    }
}

impl ArchitectureFactory<CausalLanguageModel> for MistralFactory {
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

fn find_safetensors_files(directory: &Path) -> Result<Vec<PathBuf>, MistralLoadError> {
    let canonical = directory.join("model.safetensors");
    if canonical.is_file() {
        return Ok(vec![canonical]);
    }
    for index_name in [
        "model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
    ] {
        let index = directory.join(index_name);
        if index.is_file() {
            return read_safetensors_index(&index);
        }
    }
    let mut candidates = std::fs::read_dir(directory)
        .map_err(|error| MistralLoadError::ReadDirectory(error.to_string()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    candidates.sort();
    match candidates.as_slice() {
        [only] => Ok(vec![only.clone()]),
        [] => Err(MistralLoadError::MissingWeights(directory.to_path_buf())),
        _ => Err(MistralLoadError::ShardedWeights(candidates)),
    }
}

fn read_safetensors_index(index: &Path) -> Result<Vec<PathBuf>, MistralLoadError> {
    let bytes =
        std::fs::read(index).map_err(|error| MistralLoadError::Weights(error.to_string()))?;
    let document: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        MistralLoadError::Weights(format!("invalid SafeTensors index: {error}"))
    })?;
    let weight_map = document
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| MistralLoadError::Weights("SafeTensors index has no weight_map".into()))?;
    let mut shard_names = weight_map
        .values()
        .map(|value| {
            value.as_str().ok_or_else(|| {
                MistralLoadError::Weights("SafeTensors index shard names must be strings".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    shard_names.sort_unstable();
    shard_names.dedup();
    let directory = index.parent().unwrap_or_else(|| Path::new("."));
    let paths = shard_names
        .into_iter()
        .map(|name| {
            let path = directory.join(name);
            if !path.is_file() {
                return Err(MistralLoadError::Weights(format!(
                    "SafeTensors index references missing shard `{}`",
                    path.display()
                )));
            }
            Ok(path)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if paths.is_empty() {
        return Err(MistralLoadError::Weights(
            "SafeTensors index contains an empty weight_map".into(),
        ));
    }
    Ok(paths)
}

fn load_safetensors_shards(
    module: &mut MistralForCausalLM,
    shards: &[PathBuf],
) -> Result<(), MistralLoadError> {
    let mut snapshots = BTreeMap::new();
    for shard in shards {
        let mut store = SafetensorsStore::from_file(shard)
            .with_key_remapping(r"layernorm\.weight$", "layernorm.gamma")
            .with_key_remapping(r"^model\.norm\.weight$", "model.norm.gamma");
        for (name, snapshot) in store
            .get_all_tensors()
            .map_err(|error| MistralLoadError::Weights(error.to_string()))?
        {
            if snapshots.insert(name.clone(), snapshot.clone()).is_some() {
                return Err(MistralLoadError::Weights(format!(
                    "duplicate tensor `{name}` across SafeTensors shards"
                )));
            }
        }
    }
    let result = module.apply(
        snapshots.into_values().collect(),
        None,
        // Checkpoints are usually bf16 or f16; the model runs in f32.
        Some(Box::new(
            PyTorchToBurnAdapter.chain(FloatCastAdapter::to(burn::tensor::DType::F32)),
        )),
        false,
    );
    if !result.errors.is_empty() || !result.missing.is_empty() {
        return Err(MistralLoadError::Weights(format!(
            "indexed SafeTensors did not fully load: {} missing tensors, {} errors",
            result.missing.len(),
            result.errors.len()
        )));
    }
    Ok(())
}

/// Failure while materializing a local Mistral artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MistralLoadError {
    /// Artifact path is not a directory.
    NotDirectory(PathBuf),
    /// Mistral configuration could not be parsed or is unsupported.
    Config(MistralConfigError),
    /// Tokenizer construction failed.
    Tokenizer(String),
    /// Artifact directory could not be enumerated.
    ReadDirectory(String),
    /// No SafeTensors weights file was found.
    MissingWeights(PathBuf),
    /// More than one weights shard was found without a supported index file.
    ShardedWeights(Vec<PathBuf>),
    /// SafeTensors could not be mapped completely onto the Burn module.
    Weights(String),
    /// KV-cache pool geometry could not be represented safely.
    KvCache(String),
}

impl fmt::Display for MistralLoadError {
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
                "found {} SafeTensors shards without a supported index file",
                paths.len()
            ),
            Self::Weights(message) => write!(formatter, "failed to load model weights: {message}"),
            Self::KvCache(message) => write!(formatter, "failed to configure KV cache: {message}"),
        }
    }
}

impl Error for MistralLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MistralConfigError> for MistralLoadError {
    fn from(value: MistralConfigError) -> Self {
        Self::Config(value)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;

    use super::*;

    #[test]
    fn factory_is_registered_by_architecture_not_model_name() {
        let factory = MistralFactory::new(Device::flex());

        assert_eq!(factory.architecture().as_str(), "mistral");
    }

    #[test]
    fn indexed_safetensors_files_are_resolved_once_in_stable_order() {
        let directory =
            std::env::temp_dir().join(format!("ola-mistral-index-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model-00002-of-00002.safetensors"), b"").unwrap();
        std::fs::write(directory.join("model-00001-of-00002.safetensors"), b"").unwrap();
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            r#"{"weight_map":{"z":"model-00002-of-00002.safetensors","a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let paths = find_safetensors_files(&directory).unwrap();
        assert_eq!(
            paths,
            vec![
                directory.join("model-00001-of-00002.safetensors"),
                directory.join("model-00002-of-00002.safetensors"),
            ]
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn indexed_safetensors_reject_missing_shards() {
        let directory =
            std::env::temp_dir().join(format!("ola-mistral-index-missing-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let error = find_safetensors_files(&directory).unwrap_err();
        assert!(error.to_string().contains("missing shard"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
