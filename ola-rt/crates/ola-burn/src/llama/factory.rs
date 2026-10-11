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
    HuggingFaceTokenizer, LlamaConfig, LlamaConfigError, LlamaForCausalLM, LlamaKvCacheConfig,
    session::BurnLlamaModel,
};

/// Constructs Llama models on one selected Burn backend and device.
///
/// The factory is registered under an architecture ID (normally `llama`). Model
/// IDs and paths remain catalog data, so the same factory can load TinyLlama or
/// another compatible untied, unscaled Llama checkpoint.
pub struct LlamaFactory<B: Backend> {
    architecture: ArchitectureId,
    device: B::Device,
    kv_cache: Option<LlamaKvCacheConfig>,
    backend: PhantomData<fn() -> B>,
}

/// Native Burn Llama module paired with the configuration that shaped it.
///
/// Most callers should use [`LlamaFactory::load_local`] and the neutral
/// [`CausalLanguageModel`] interface. This representation exists for research
/// probes that need logits or hidden tensor outputs without widening the core
/// runtime API to include backend-specific tensors.
pub struct LoadedLlama<B: Backend> {
    config: LlamaConfig,
    module: LlamaForCausalLM<B>,
}

impl<B: Backend> LoadedLlama<B> {
    /// Returns the validated Hugging Face configuration.
    pub const fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Returns the native Burn module for explicit research measurements.
    pub const fn module(&self) -> &LlamaForCausalLM<B> {
        &self.module
    }

    /// Consumes the wrapper into configuration and module parts.
    pub fn into_parts(self) -> (LlamaConfig, LlamaForCausalLM<B>) {
        (self.config, self.module)
    }
}

impl<B: Backend> LlamaFactory<B> {
    /// Creates a factory registered as the canonical `llama` architecture.
    pub fn new(device: B::Device) -> Self {
        Self {
            architecture: "llama".into(),
            device,
            kv_cache: None,
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
            kv_cache: None,
            backend: PhantomData,
        }
    }

    /// Overrides the shared KV-cache capacity and page geometry.
    ///
    /// By default, the factory reserves logical capacity for one complete
    /// model context with 16 token positions per physical block.
    pub fn with_kv_cache(mut self, config: LlamaKvCacheConfig) -> Self {
        self.kv_cache = Some(config);
        self
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
        let tokenizer = HuggingFaceTokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| LlamaLoadError::Tokenizer(error.to_string()))?;
        let (config, module) = self.load_native(directory)?.into_parts();

        let cache_config = self
            .kv_cache
            .unwrap_or_else(|| LlamaKvCacheConfig::for_model(&config));
        let model =
            BurnLlamaModel::with_kv_cache(module, config, self.device.clone(), cache_config)
                .map_err(|error| LlamaLoadError::KvCache(error.to_string()))?;
        Ok(CausalLanguageModel::new(tokenizer, model))
    }

    /// Loads configuration and weights while leaving tokenization to the caller.
    ///
    /// This is the narrow escape hatch used by logit-conformance tests. It
    /// preserves the native Burn type without making backend tensors part of
    /// `ola-model` or `ola-runtime`.
    pub fn load_native(&self, directory: impl AsRef<Path>) -> Result<LoadedLlama<B>, LlamaLoadError>
    where
        B: 'static,
    {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            return Err(LlamaLoadError::NotDirectory(directory.to_path_buf()));
        }
        let config = LlamaConfig::from_file(directory.join("config.json"))?;
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
        Ok(LoadedLlama { config, module })
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
    /// KV-cache pool geometry could not be represented safely.
    KvCache(String),
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
            Self::KvCache(message) => write!(formatter, "failed to configure KV cache: {message}"),
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
    #[cfg(feature = "wgpu")]
    use burn::backend::{Wgpu, wgpu::WgpuDevice};
    use burn::{
        backend::NdArray,
        tensor::{Int, Tensor, TensorData},
    };

    use super::*;

    #[test]
    fn factory_is_registered_by_architecture_not_model_name() {
        let factory = LlamaFactory::<NdArray>::new(Default::default());

        assert_eq!(factory.architecture().as_str(), "llama");
    }

    /// Compares Burn with the independent scalar oracle documented in
    /// `CONFORMANCE.md`. The external fixture is intentionally not downloaded
    /// by `cargo test`; run this explicitly after materializing the pinned
    /// artifact and setting `OLA_LLAMA_CONFORMANCE_DIR`.
    #[test]
    #[ignore = "requires the pinned external Hugging Face conformance fixture"]
    fn matches_pinned_reference_logits() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in CONFORMANCE.md");
        let weights = directory.join("model.safetensors");
        assert_eq!(
            std::fs::metadata(&weights).unwrap().len(),
            4_131_280,
            "fixture size differs; verify the pinned SHA-256 before using expected logits"
        );

        let factory = LlamaFactory::<NdArray<f32>>::new(Default::default());
        let loaded = factory.load_native(&directory).unwrap();
        let device = Default::default();
        let input: Tensor<NdArray<f32>, 2, Int> =
            Tensor::from_data(TensorData::new(vec![1_i64, 15_043, 1_789], [1, 3]), &device);
        let logits = loaded.module().forward(input);
        let actual = logits
            .clone()
            .slice([0..1, 0..3, 0..16])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        #[rustfmt::skip]
        let expected: [f64; 48] = [
            -0.042735109, -0.065533388, -0.100713878, -0.057570876,
             0.006277183,  0.019492461,  0.150525994,  0.102675800,
             0.045494971,  0.034240757,  0.127989488,  0.082428453,
            -0.012951005,  0.130573982,  0.073102357,  0.090223021,
             0.036621468,  0.086013101, -0.097030822, -0.045341135,
             0.131229144, -0.044916200, -0.051172383,  0.134220192,
            -0.121404217, -0.000155305, -0.081268123, -0.134603569,
             0.031301317, -0.104547035, -0.039362161, -0.027215165,
             0.055092733,  0.069419146, -0.048099038, -0.013597904,
            -0.029187137, -0.014970722, -0.002289230,  0.139541307,
            -0.141860866,  0.060625086, -0.199687625, -0.120067634,
             0.080293021, -0.112008740, -0.142695550, -0.041046643,
        ];
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let difference = (f64::from(*actual) - expected).abs();
            assert!(
                difference <= 2e-5,
                "logit {index}: actual {actual}, expected {expected}"
            );
        }

        let selected = logits.argmax(2).into_data().to_vec::<i64>().unwrap();
        assert_eq!(selected, vec![5_971, 1_789, 19_053]);
    }

    /// Validates the opt-in GPU backend against the CPU reference on identical
    /// checkpoint weights and input IDs.
    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires the pinned fixture and a working WGPU adapter"]
    fn wgpu_logits_match_cpu_logits() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in CONFORMANCE.md");
        let cpu_device = Default::default();
        let gpu_device = WgpuDevice::default();
        let cpu = LlamaFactory::<NdArray<f32>>::new(cpu_device)
            .load_native(&directory)
            .unwrap();
        let gpu = LlamaFactory::<Wgpu<f32, i32>>::new(gpu_device.clone())
            .load_native(&directory)
            .unwrap();
        let ids = vec![1_i64, 15_043, 1_789];
        let cpu_logits = cpu
            .module()
            .forward(Tensor::from_data(
                TensorData::new(ids.clone(), [1, 3]),
                &cpu_device,
            ))
            .slice([0..1, 0..3, 0..16])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let gpu_logits = gpu
            .module()
            .forward(Tensor::from_data(TensorData::new(ids, [1, 3]), &gpu_device))
            .slice([0..1, 0..3, 0..16])
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (index, (cpu, gpu)) in cpu_logits.into_iter().zip(gpu_logits).enumerate() {
            let difference = (cpu - gpu).abs();
            assert!(
                difference <= 2e-4,
                "logit {index}: CPU={cpu}, WGPU={gpu}, difference={difference}"
            );
        }
    }
}
