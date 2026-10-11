use std::sync::{Arc, Mutex};

use burn::tensor::{ElementConversion, Int, Tensor, TensorData, backend::Backend};
use ola_core::GenerationConfig;
use ola_model::{
    CausalModel, ExecutionStage, GeneratedToken, GenerationSession, ModelExecutionError,
    SessionMetrics,
};
use ola_runtime::KvBlockError;

use super::{
    LlamaConfig, LlamaForCausalLM, LlamaKvCacheConfig,
    kv_cache::{LlamaKvCache, LlamaKvPool},
};

/// Shared Burn weights exposed through OlaRT's architecture-neutral model API.
pub(crate) struct BurnLlamaModel<B: Backend> {
    module: Arc<LlamaForCausalLM<B>>,
    config: Arc<LlamaConfig>,
    device: B::Device,
    cache_pool: Arc<Mutex<LlamaKvPool<B>>>,
}

impl<B: Backend> BurnLlamaModel<B> {
    pub(crate) fn with_kv_cache(
        module: LlamaForCausalLM<B>,
        config: LlamaConfig,
        device: B::Device,
        cache_config: LlamaKvCacheConfig,
    ) -> Result<Self, KvBlockError> {
        let cache_pool = LlamaKvPool::new(&config, cache_config, device.clone())?;
        Ok(Self {
            module: Arc::new(module),
            config: Arc::new(config),
            device,
            cache_pool: Arc::new(Mutex::new(cache_pool)),
        })
    }
}

impl<B> CausalModel for BurnLlamaModel<B>
where
    B: Backend + 'static,
    B::Device: Send + Sync,
{
    fn create_session(
        &self,
        generation: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
        if generation.max_new_tokens == 0 {
            return Err(ModelExecutionError::new(
                ExecutionStage::SessionCreation,
                "max_new_tokens must be greater than zero",
            ));
        }
        if !generation.top_p.is_finite() || generation.top_p <= 0.0 || generation.top_p > 1.0 {
            return Err(ModelExecutionError::new(
                ExecutionStage::SessionCreation,
                "top_p must be finite and in the interval (0, 1]",
            ));
        }
        if generation.temperature != 0.0 {
            return Err(ModelExecutionError::new(
                ExecutionStage::SessionCreation,
                "the initial Llama executor supports greedy decoding only; set temperature to 0",
            ));
        }
        let cache = LlamaKvCache::allocate(self.cache_pool.clone())
            .map_err(|error| cache_error(ExecutionStage::SessionCreation, error))?;
        Ok(Box::new(LlamaSession {
            module: self.module.clone(),
            config: self.config.clone(),
            device: self.device.clone(),
            generation,
            tokens: Vec::new(),
            cache: Some(cache),
            last_generated: None,
            metrics: SessionMetrics::default(),
            terminal: false,
        }))
    }
}

/// Request-local state for correctness-first cached Llama decoding.
struct LlamaSession<B: Backend> {
    module: Arc<LlamaForCausalLM<B>>,
    config: Arc<LlamaConfig>,
    device: B::Device,
    generation: GenerationConfig,
    tokens: Vec<u32>,
    cache: Option<LlamaKvCache<B>>,
    last_generated: Option<u32>,
    metrics: SessionMetrics,
    terminal: bool,
}

impl<B> LlamaSession<B>
where
    B: Backend + 'static,
    B::Device: Send + Sync,
{
    fn select_token(
        &mut self,
        logits: Tensor<B, 3>,
        stage: ExecutionStage,
    ) -> Result<GeneratedToken, ModelExecutionError> {
        let sequence_length = logits.dims()[1];
        let vocabulary = self.config.vocab_size;
        let last = logits
            .slice([0..1, sequence_length - 1..sequence_length, 0..vocabulary])
            .argmax(2)
            .into_data()
            .to_vec::<B::IntElem>()
            .map_err(|error| {
                ModelExecutionError::new(stage, format!("failed to read selected token: {error}"))
            })?;
        let selected = last[0].elem::<i64>();
        let id = u32::try_from(selected).map_err(|_| {
            ModelExecutionError::new(stage, "backend returned a token ID outside the u32 range")
        })?;
        let is_end_of_sequence = self.config.eos_token_id.contains(&id);
        self.last_generated = Some(id);
        self.metrics.generated_tokens += 1;
        self.terminal =
            is_end_of_sequence || self.metrics.generated_tokens >= self.generation.max_new_tokens;
        Ok(GeneratedToken {
            id,
            is_end_of_sequence,
        })
    }

    fn validate_token(&self, token: u32, stage: ExecutionStage) -> Result<(), ModelExecutionError> {
        if token as usize >= self.config.vocab_size {
            return Err(ModelExecutionError::new(
                stage,
                format!(
                    "token ID {token} is outside vocabulary size {}",
                    self.config.vocab_size
                ),
            ));
        }
        Ok(())
    }

    fn validate_context_length(
        &self,
        sequence_length: usize,
        stage: ExecutionStage,
    ) -> Result<(), ModelExecutionError> {
        if sequence_length > self.config.max_position_embeddings {
            return Err(ModelExecutionError::new(
                stage,
                format!(
                    "sequence length {sequence_length} exceeds model context length {}",
                    self.config.max_position_embeddings
                ),
            ));
        }
        Ok(())
    }
}

impl<B> GenerationSession for LlamaSession<B>
where
    B: Backend + 'static,
    B::Device: Send + Sync,
{
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<GeneratedToken, ModelExecutionError> {
        if !self.tokens.is_empty() || self.last_generated.is_some() {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prefill may be called exactly once per session",
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prompt must contain at least one token",
            ));
        }
        for &token in prompt_tokens {
            self.validate_token(token, ExecutionStage::Prefill)?;
        }
        self.validate_context_length(prompt_tokens.len(), ExecutionStage::Prefill)?;
        let token_values = prompt_tokens.iter().map(|&id| i64::from(id)).collect();
        let input_ids: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(token_values, [1, prompt_tokens.len()]),
            &self.device,
        );
        let cache = self
            .cache
            .as_mut()
            .expect("sessions allocate a cache at creation");
        let logits = self
            .module
            .prefill(input_ids, cache)
            .map_err(|error| cache_error(ExecutionStage::Prefill, error))?;
        self.metrics.kv_cache_bytes = cache.bytes();
        self.tokens.extend_from_slice(prompt_tokens);
        self.metrics.prompt_tokens = prompt_tokens.len();
        self.select_token(logits, ExecutionStage::Prefill)
    }

    fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
        let expected = self.last_generated.ok_or_else(|| {
            ModelExecutionError::new(
                ExecutionStage::Decode,
                "decode requires one successful prefill call",
            )
        })?;
        if self.terminal {
            return Err(ModelExecutionError::new(
                ExecutionStage::Decode,
                "generation already reached EOS or max_new_tokens",
            ));
        }
        if input_token != expected {
            return Err(ModelExecutionError::new(
                ExecutionStage::Decode,
                format!(
                    "decode input token {input_token} does not match the previous model output {expected}"
                ),
            ));
        }
        self.validate_token(input_token, ExecutionStage::Decode)?;
        self.validate_context_length(self.tokens.len() + 1, ExecutionStage::Decode)?;
        let input_ids: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(vec![i64::from(input_token)], [1, 1]),
            &self.device,
        );
        let cache = self.cache.as_mut().ok_or_else(|| {
            ModelExecutionError::new(
                ExecutionStage::Decode,
                "decode requires an initialized KV cache",
            )
        })?;
        let logits = self
            .module
            .decode(input_ids, cache)
            .map_err(|error| cache_error(ExecutionStage::Decode, error))?;
        self.metrics.kv_cache_bytes = cache.bytes();
        self.tokens.push(input_token);
        self.select_token(logits, ExecutionStage::Decode)
    }

    fn metrics(&self) -> SessionMetrics {
        self.metrics
    }
}

fn cache_error(stage: ExecutionStage, error: KvBlockError) -> ModelExecutionError {
    ModelExecutionError::new(stage, format!("KV cache: {error}"))
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type TestBackend = NdArray<f32>;

    fn tiny_model() -> BurnLlamaModel<TestBackend> {
        let config = LlamaConfig {
            model_type: "llama".into(),
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            head_dim: None,
            hidden_act: "silu".into(),
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
            rope_scaling: None,
            bos_token_id: Some(1),
            eos_token_id: Vec::new(),
        };
        let device = Default::default();
        TestBackend::seed(&device, 11);
        let module = LlamaForCausalLM::init(&config, &device);
        BurnLlamaModel::with_kv_cache(module, config, device, LlamaKvCacheConfig::new(16, 16))
            .unwrap()
    }

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        }
    }

    #[test]
    fn session_reports_paged_cache_bytes() {
        let model = tiny_model();
        let mut session = model.create_session(greedy(2)).unwrap();

        let first = session.prefill(&[1, 4]).unwrap();
        let _second = session.decode(first.id).unwrap();

        assert_eq!(
            session.metrics(),
            SessionMetrics {
                prompt_tokens: 2,
                generated_tokens: 2,
                // One 16-token physical block. Logical payload is 96 bytes;
                // the remaining page slots are internal fragmentation.
                kv_cache_bytes: 512,
            }
        );
    }

    #[test]
    fn rejects_sampling_until_a_sampler_is_implemented() {
        let model = tiny_model();
        let error = model
            .create_session(GenerationConfig::default())
            .err()
            .unwrap();

        assert_eq!(error.stage(), ExecutionStage::SessionCreation);
        assert!(error.message().contains("greedy"));
    }

    #[test]
    fn enforces_the_generation_limit_inside_the_session() {
        let model = tiny_model();
        let mut session = model.create_session(greedy(1)).unwrap();
        let first = session.prefill(&[1]).unwrap();

        let error = session.decode(first.id).unwrap_err();

        assert!(error.message().contains("max_new_tokens"));
    }

    #[test]
    fn rejects_a_zero_generation_limit_at_session_creation() {
        let model = tiny_model();
        let error = model.create_session(greedy(0)).err().unwrap();

        assert_eq!(error.stage(), ExecutionStage::SessionCreation);
        assert!(error.message().contains("greater than zero"));
    }
}
