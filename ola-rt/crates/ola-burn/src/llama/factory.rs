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
    HuggingFaceTokenizer, LlamaConfig, LlamaConfigError, LlamaForCausalLM, LlamaKvCacheConfig,
    session::BurnLlamaModel,
};

/// Constructs Llama models on one selected Burn backend and device.
///
/// The factory is registered under an architecture ID (normally `llama`). Model
/// IDs and paths remain catalog data, so the same factory can load TinyLlama or
/// another compatible untied, unscaled Llama checkpoint.
pub struct LlamaFactory {
    architecture: ArchitectureId,
    device: Device,
    kv_cache: Option<LlamaKvCacheConfig>,
}

/// Native Burn Llama module paired with the configuration that shaped it.
///
/// Most callers should use [`LlamaFactory::load_local`] and the neutral
/// [`CausalLanguageModel`] interface. This representation exists for research
/// probes that need logits or hidden tensor outputs without widening the core
/// runtime API to include backend-specific tensors.
pub struct LoadedLlama {
    config: LlamaConfig,
    module: LlamaForCausalLM,
}

impl LoadedLlama {
    /// Returns the validated Hugging Face configuration.
    pub const fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Returns the native Burn module for explicit research measurements.
    pub const fn module(&self) -> &LlamaForCausalLM {
        &self.module
    }

    /// Consumes the wrapper into configuration and module parts.
    pub fn into_parts(self) -> (LlamaConfig, LlamaForCausalLM) {
        (self.config, self.module)
    }
}

impl LlamaFactory {
    /// Creates a factory registered as the canonical `llama` architecture.
    pub fn new(device: Device) -> Self {
        Self {
            architecture: "llama".into(),
            device,
            kv_cache: None,
        }
    }

    /// Creates a factory under a custom architecture alias.
    ///
    /// Aliases are useful when experiments intentionally compare two Llama
    /// implementations through the same registry.
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
    pub fn with_kv_cache(mut self, config: LlamaKvCacheConfig) -> Self {
        self.kv_cache = Some(config);
        self
    }

    /// Loads a local Hugging Face artifact directory with detailed errors.
    pub fn load_local(
        &self,
        directory: impl AsRef<Path>,
    ) -> Result<CausalLanguageModel, LlamaLoadError> {
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
        let cache_profile = cache_config
            .profile(&config)
            .map_err(|error| LlamaLoadError::KvCache(error.to_string()))?;
        let attention_profile = ola_kernel::PagedAttentionProfile::new(
            ola_kernel::burn_device_target(&self.device),
            ola_kernel::KernelDType::F32,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_head_dim(),
            cache_config.tokens_per_block(),
        )
        .expect("validated Llama and KV-cache dimensions form a kernel profile");
        let model =
            BurnLlamaModel::with_kv_cache(module, config, self.device.clone(), cache_config)
                .map_err(|error| LlamaLoadError::KvCache(error.to_string()))?;
        Ok(CausalLanguageModel::new(tokenizer, model)
            .with_kv_cache_profile(cache_profile)
            .with_paged_attention_profile(attention_profile))
    }

    /// Loads configuration and weights while leaving tokenization to the caller.
    ///
    /// This is the narrow escape hatch used by logit-conformance tests. It
    /// preserves the native Burn type without making backend tensors part of
    /// `ola-model` or `ola-runtime`.
    pub fn load_native(&self, directory: impl AsRef<Path>) -> Result<LoadedLlama, LlamaLoadError> {
        let directory = directory.as_ref();
        if !directory.is_dir() {
            return Err(LlamaLoadError::NotDirectory(directory.to_path_buf()));
        }
        let config = LlamaConfig::from_file(directory.join("config.json"))?;
        let weights = find_safetensors_files(directory)?;
        let mut module = LlamaForCausalLM::init(&config, &self.device);
        load_safetensors_shards(&mut module, &weights, config.tie_word_embeddings)?;
        Ok(LoadedLlama { config, module })
    }
}

impl ArchitectureFactory<CausalLanguageModel> for LlamaFactory {
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

fn find_safetensors_files(directory: &Path) -> Result<Vec<PathBuf>, LlamaLoadError> {
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
        [only] => Ok(vec![only.clone()]),
        [] => Err(LlamaLoadError::MissingWeights(directory.to_path_buf())),
        _ => Err(LlamaLoadError::ShardedWeights(candidates)),
    }
}

fn read_safetensors_index(index: &Path) -> Result<Vec<PathBuf>, LlamaLoadError> {
    let bytes = std::fs::read(index).map_err(|error| LlamaLoadError::Weights(error.to_string()))?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| LlamaLoadError::Weights(format!("invalid SafeTensors index: {error}")))?;
    let weight_map = document
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| LlamaLoadError::Weights("SafeTensors index has no weight_map".into()))?;
    let mut shard_names = weight_map
        .values()
        .map(|value| {
            value.as_str().ok_or_else(|| {
                LlamaLoadError::Weights("SafeTensors index shard names must be strings".into())
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
                return Err(LlamaLoadError::Weights(format!(
                    "SafeTensors index references missing shard `{}`",
                    path.display()
                )));
            }
            Ok(path)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if paths.is_empty() {
        return Err(LlamaLoadError::Weights(
            "SafeTensors index contains an empty weight_map".into(),
        ));
    }
    Ok(paths)
}

fn load_safetensors_shards(
    module: &mut LlamaForCausalLM,
    shards: &[PathBuf],
    tied_embeddings: bool,
) -> Result<(), LlamaLoadError> {
    let mut snapshots = BTreeMap::new();
    for shard in shards {
        let mut store = SafetensorsStore::from_file(shard)
            .with_key_remapping(r"layernorm\.weight$", "layernorm.gamma")
            .with_key_remapping(r"^model\.norm\.weight$", "model.norm.gamma");
        for (name, snapshot) in store
            .get_all_tensors()
            .map_err(|error| LlamaLoadError::Weights(error.to_string()))?
        {
            if snapshots.insert(name.clone(), snapshot.clone()).is_some() {
                return Err(LlamaLoadError::Weights(format!(
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
    let unexpected_missing = result
        .missing
        .iter()
        .filter(|(path, _)| !(tied_embeddings && path == "lm_head.weight"));
    if !result.errors.is_empty() || unexpected_missing.count() != 0 {
        return Err(LlamaLoadError::Weights(format!(
            "indexed SafeTensors did not fully load: {} missing tensors, {} errors",
            result.missing.len(),
            result.errors.len()
        )));
    }
    Ok(())
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
    /// More than one weights shard was found without a supported index file.
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
                "found {} SafeTensors shards without a supported index file",
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
    use burn::tensor::{Device, Int, Tensor, TensorData};

    use super::*;
    #[cfg(feature = "wgpu")]
    use ola_core::{GenerationConfig, ModelDescriptor, RequestSpec, Timestamp};
    #[cfg(feature = "wgpu")]
    use ola_runtime::{BatchExecutionMode, Executor, Runtime};

    #[test]
    fn factory_is_registered_by_architecture_not_model_name() {
        let factory = LlamaFactory::new(Device::flex());

        assert_eq!(factory.architecture().as_str(), "llama");
    }

    #[test]
    fn indexed_safetensors_files_are_resolved_once_in_stable_order() {
        let directory =
            std::env::temp_dir().join(format!("ola-llama-index-{}", std::process::id()));
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

    /// Hugging Face checkpoints usually store bf16 weights; the model runs in
    /// f32, so loading must cast rather than fail.
    #[test]
    fn bf16_checkpoints_load_into_the_f32_model() {
        use burn_store::BurnToPyTorchAdapter;

        let directory = std::env::temp_dir().join(format!("ola-llama-bf16-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let json = r#"{
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "rms_norm_eps": 0.00001,
            "eos_token_id": 2
        }"#;
        std::fs::write(directory.join("config.json"), json).unwrap();
        let config = LlamaConfig::from_json(json.as_bytes()).unwrap();
        let device = Device::flex();
        let original = LlamaForCausalLM::init(&config, &device);
        let mut store = SafetensorsStore::from_file(directory.join("model.safetensors"))
            .with_to_adapter(
                BurnToPyTorchAdapter.chain(FloatCastAdapter::to(burn::tensor::DType::BF16)),
            )
            .with_key_remapping(r"layernorm\.gamma$", "layernorm.weight")
            .with_key_remapping(r"^model\.norm\.gamma$", "model.norm.weight");
        original.save_into(&mut store).unwrap();

        let (_, loaded) = LlamaFactory::new(device)
            .load_native(&directory)
            .unwrap()
            .into_parts();
        std::fs::remove_dir_all(&directory).unwrap();
        // Name to (dtype, little-endian values) for every parameter.
        let tensors = |module: &LlamaForCausalLM| {
            module
                .collect(None, None, false)
                .into_iter()
                .map(|tensor| {
                    let bytes = tensor.to_bytes().unwrap();
                    let values = bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    (tensor.name, (tensor.dtype, values))
                })
                .collect::<BTreeMap<_, _>>()
        };
        let (expected, actual) = (tensors(&original), tensors(&loaded));
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>()
        );
        for (name, (dtype, values)) in &actual {
            assert_eq!(*dtype, burn::tensor::DType::F32, "{name}");
            // bf16 keeps 8 bits of mantissa: within 1% of the original.
            for (want, got) in expected[name].1.iter().zip(values) {
                assert!((want - got).abs() <= want.abs() * 0.01 + 1e-6, "{name}");
            }
        }
    }

    #[test]
    fn indexed_safetensors_reject_missing_shards() {
        let directory =
            std::env::temp_dir().join(format!("ola-llama-index-missing-{}", std::process::id()));
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

        let factory = LlamaFactory::new(Device::flex());
        let loaded = factory.load_native(&directory).unwrap();
        let device = Default::default();
        let input: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(vec![1_i64, 15_043, 1_789], [1, 3]), &device);
        let logits = loaded.module().forward(input);
        let actual = logits
            .clone()
            .slice([0..1, 0..3, 0..16])
            .into_data()
            .try_to_vec::<f32>()
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

        let selected = logits.argmax(2).into_data().try_to_vec_as::<i64>().unwrap();
        assert_eq!(selected, vec![5_971, 1_789, 19_053]);
    }

    /// Planned multi-request execution must emit exactly the tokens each
    /// request produces when run alone. The planned run exercises slot
    /// refill, chunked and continued prefill, power-of-two grouping, budget
    /// deferrals, and tensor-batched decode on real Llama weights.
    #[test]
    #[ignore = "requires the pinned external Hugging Face conformance fixture"]
    fn planned_multi_request_execution_matches_single_request_oracle() {
        use ola_core::{GenerationConfig, ModelDescriptor, RequestId, RequestSpec, Timestamp};
        use ola_runtime::{
            BatchExecutionMode, ComposedPolicy, Executor, FifoTokenBudgetPolicy, PlanDivergence,
            PlannedPhase, PowerOfTwoPrefillBatcher, Runtime,
        };

        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in CONFORMANCE.md");
        let artifact = LlamaFactory::new(Device::flex())
            .load_local(directory)
            .unwrap();
        let workload = [
            ("Hello", 3),
            ("Hello world", 5),
            ("The quick brown fox jumps", 2),
            ("a", 4),
            ("Once upon a time there was", 6),
        ];
        let request = |prompt: &str, max_new_tokens: usize| {
            let mut request = RequestSpec::text("tiny", prompt);
            request.generation = GenerationConfig {
                max_new_tokens,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            };
            request
        };
        let run_to_completion = |executor: &mut Executor, ids: &[RequestId]| {
            let mut iterations = Vec::new();
            for step in 1..=500 {
                if ids.iter().all(|id| executor.result(*id).is_some()) {
                    return iterations;
                }
                iterations.push(
                    executor
                        .step_iteration(Timestamp::from_micros(step))
                        .unwrap(),
                );
            }
            panic!("workload did not complete within 500 iterations");
        };

        let oracle = workload
            .iter()
            .map(|(prompt, max_new_tokens)| {
                let mut executor = Executor::default();
                executor
                    .register_model(ModelDescriptor::language("tiny", "llama"), artifact.clone())
                    .unwrap();
                let id = executor
                    .submit(request(prompt, *max_new_tokens), Timestamp::ZERO)
                    .unwrap();
                run_to_completion(&mut executor, &[id]);
                executor.result(id).unwrap().token_ids.clone()
            })
            .collect::<Vec<_>>();

        let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
            .unwrap()
            .with_prefill_chunk_size(2)
            .unwrap()
            .with_policy(
                ComposedPolicy::default()
                    .with_grouping(PowerOfTwoPrefillBatcher)
                    .with_batch(FifoTokenBudgetPolicy::new(4).unwrap()),
            );
        executor
            .register_model(ModelDescriptor::language("tiny", "llama"), artifact)
            .unwrap();
        let ids = workload
            .iter()
            .map(|(prompt, max_new_tokens)| {
                executor
                    .submit(request(prompt, *max_new_tokens), Timestamp::ZERO)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let iterations = run_to_completion(&mut executor, &ids);

        for ((id, expected), (prompt, _)) in ids.iter().zip(&oracle).zip(&workload) {
            assert_eq!(
                &executor.result(*id).unwrap().token_ids,
                expected,
                "planned output diverged from the single-request oracle for {prompt:?}"
            );
        }
        assert!(
            iterations
                .iter()
                .flat_map(|iteration| &iteration.execution_batches)
                .any(|batch| batch.mode == BatchExecutionMode::Tensor),
            "the workload should exercise at least one tensor invocation"
        );
        let plans = iterations
            .iter()
            .map(|iteration| &iteration.execution_plan)
            .collect::<Vec<_>>();
        assert!(
            plans
                .iter()
                .flat_map(|plan| &plan.batches)
                .any(|batch| batch.phase == PlannedPhase::ContinuedPrefill),
            "the workload should exercise continued prefill"
        );
        assert!(
            plans
                .iter()
                .any(|plan| plan.estimates().budget_deferrals > 0),
            "the workload should exercise token-budget deferral"
        );
        assert!(
            plans.iter().any(|plan| !plan.predicted_releases.is_empty()),
            "the workload should exercise predicted slot reuse"
        );
        assert!(
            iterations
                .iter()
                .flat_map(|iteration| &iteration.divergences)
                .all(|divergence| !matches!(divergence, PlanDivergence::ModeFallback { .. })),
            "Llama declares every batch hook and must not fall back"
        );
    }

    /// Every scheduling baseline must change only *when* requests run, never
    /// what they produce. Priorities, deadlines, and lengths are chosen so
    /// that each policy admits the queue in a different order.
    #[test]
    #[ignore = "requires the pinned external Hugging Face conformance fixture"]
    fn scheduling_policies_preserve_real_model_outputs() {
        use ola_core::{
            GenerationConfig, ModelDescriptor, Priority, RequestId, RequestSpec, Timestamp,
        };
        use ola_runtime::{
            ComposedPolicy, EarliestDeadlineFirst, ExecutionPolicy, Executor, HeadOfLine, JobCost,
            MemoryAwareAdmission, PriorityScheduler, Runtime, ShortestJobFirst,
        };

        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in CONFORMANCE.md");
        let artifact = LlamaFactory::new(Device::flex())
            .load_local(directory)
            .unwrap();
        // (prompt, output limit, priority, deadline)
        let workload = [
            ("Once upon a time there was", 6, 0, Some(900)),
            ("Hello world", 5, 5, Some(100)),
            ("The quick brown fox jumps", 2, 0, None),
            ("a", 4, 5, Some(50)),
            ("Hello", 3, 0, Some(400)),
        ];
        let request =
            |(prompt, max_new_tokens, priority, deadline): (&str, usize, i32, Option<u64>)| {
                let mut request = RequestSpec::text("tiny", prompt);
                request.generation = GenerationConfig {
                    max_new_tokens,
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: Some(0),
                };
                request.priority = Priority::new(priority);
                request.deadline = deadline.map(Timestamp::from_micros);
                request
            };
        let run = |executor: &mut Executor, ids: &[RequestId]| {
            let mut first_ranking = None;
            for step in 1..=500 {
                if ids.iter().all(|id| executor.result(*id).is_some()) {
                    return first_ranking.expect("at least one iteration ran");
                }
                let iteration = executor
                    .step_iteration(Timestamp::from_micros(step))
                    .unwrap();
                first_ranking.get_or_insert(iteration.decision.ranking.ranked);
            }
            panic!("workload did not complete within 500 iterations");
        };

        let oracle = workload
            .iter()
            .map(|entry| {
                let mut executor = Executor::default();
                executor
                    .register_model(ModelDescriptor::language("tiny", "llama"), artifact.clone())
                    .unwrap();
                let id = executor.submit(request(*entry), Timestamp::ZERO).unwrap();
                run(&mut executor, &[id]);
                executor.result(id).unwrap().token_ids.clone()
            })
            .collect::<Vec<_>>();

        let base = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
        let policies: Vec<Box<dyn ExecutionPolicy>> = vec![
            Box::new(base()),
            Box::new(base().with_scheduler(PriorityScheduler::new())),
            Box::new(base().with_scheduler(ShortestJobFirst::new(JobCost::PromptPlusOutputLimit))),
            Box::new(base().with_scheduler(EarliestDeadlineFirst)),
            Box::new(base().with_head_of_line(HeadOfLine::Backfill { max_skipped: 2 })),
        ];
        let mut rankings = Vec::new();
        for policy in policies {
            let name = policy.descriptor().to_string();
            let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
                .unwrap()
                .with_prefill_chunk_size(2)
                .unwrap()
                .with_policy(policy);
            executor
                .register_model(ModelDescriptor::language("tiny", "llama"), artifact.clone())
                .unwrap();
            let ids = workload
                .iter()
                .map(|entry| executor.submit(request(*entry), Timestamp::ZERO).unwrap())
                .collect::<Vec<_>>();
            let ranking = run(&mut executor, &ids)
                .into_iter()
                .map(|id| ids.iter().position(|candidate| *candidate == id).unwrap())
                .collect::<Vec<_>>();
            for ((id, expected), entry) in ids.iter().zip(&oracle).zip(&workload) {
                assert_eq!(
                    &executor.result(*id).unwrap().token_ids,
                    expected,
                    "{name} changed the output for {:?}",
                    entry.0
                );
            }
            rankings.push(ranking);
        }
        rankings.sort();
        rankings.dedup();
        assert!(
            rankings.len() >= 3,
            "policies should admit in different orders, got {rankings:?}"
        );
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
        let cpu_device = Device::flex();
        let gpu_device = Device::wgpu(Default::default());
        let cpu = LlamaFactory::new(cpu_device.clone())
            .load_native(&directory)
            .unwrap();
        let gpu = LlamaFactory::new(gpu_device.clone())
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
            .try_to_vec::<f32>()
            .unwrap();
        let gpu_logits = gpu
            .module()
            .forward(Tensor::from_data(TensorData::new(ids, [1, 3]), &gpu_device))
            .slice([0..1, 0..3, 0..16])
            .into_data()
            .try_to_vec::<f32>()
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

    /// Exercises the actual WGPU session path through runtime tensor grouping.
    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires the pinned fixture and a working WGPU adapter"]
    fn wgpu_executor_tensor_batches_prefill_and_decode() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in CONFORMANCE.md");
        let artifact = LlamaFactory::new(Device::wgpu(Default::default()))
            .load_local(directory)
            .unwrap();
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "llama"), artifact)
            .unwrap();
        let generation = GenerationConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        };
        for prompt in ["Hello", "Hello world"] {
            let mut request = RequestSpec::text("tiny", prompt);
            request.generation = generation.clone();
            executor.submit(request, Timestamp::ZERO).unwrap();
        }
        let admitted = executor.step_iteration(Timestamp::from_micros(1)).unwrap();

        assert_eq!(admitted.execution_batches.len(), 1);
        assert_eq!(
            admitted.execution_batches[0].mode,
            BatchExecutionMode::Tensor
        );
        assert_eq!(admitted.execution_batches[0].request_ids.len(), 2);

        let decoded = executor.step_iteration(Timestamp::from_micros(2)).unwrap();

        assert_eq!(decoded.execution_batches.len(), 1);
        assert_eq!(
            decoded.execution_batches[0].mode,
            BatchExecutionMode::Tensor
        );
        assert_eq!(decoded.execution_batches[0].request_ids.len(), 2);
    }
}
