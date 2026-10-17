use std::sync::{Arc, Mutex};

use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use ola_core::GenerationConfig;
use ola_model::{
    BatchCapabilities, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
    ModelExecutionError, PrefillOutcome, SessionMetrics,
};
use ola_runtime::KvBlockError;

use super::{
    LlamaConfig, LlamaForCausalLM, LlamaKvCacheConfig,
    kv_cache::{LlamaKvCache, LlamaKvPool},
};

/// Shared Burn weights exposed through OlaRT's architecture-neutral model API.
pub(crate) struct BurnLlamaModel {
    module: Arc<LlamaForCausalLM>,
    config: Arc<LlamaConfig>,
    device: burn::tensor::Device,
    cache_pool: Arc<Mutex<LlamaKvPool>>,
}

impl BurnLlamaModel {
    pub(crate) fn with_kv_cache(
        module: LlamaForCausalLM,
        config: LlamaConfig,
        device: burn::tensor::Device,
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

impl CausalModel for BurnLlamaModel {
    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    fn kv_pool_status(&self) -> Option<ola_model::KvPoolStatus> {
        Some(
            self.cache_pool
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .status(),
        )
    }

    fn batch_capabilities(&self) -> BatchCapabilities {
        BatchCapabilities::ALL
    }

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

    fn prefill_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_tokens: &[&[u32]],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        let batch_size = sessions.len();
        if prompt_tokens.len() != batch_size {
            return Some(batch_error(
                batch_size,
                ModelExecutionError::new(
                    ExecutionStage::Prefill,
                    format!(
                        "batch has {batch_size} sessions but {} prompts",
                        prompt_tokens.len()
                    ),
                ),
            ));
        }
        if batch_size == 0 {
            return Some(Vec::new());
        }
        let mut typed = Vec::with_capacity(batch_size);
        for session in sessions {
            let Some(session) = session.as_any_mut().downcast_mut::<LlamaSession>() else {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        "batched session was not created by this Llama backend",
                    ),
                ));
            };
            typed.push(session);
        }
        for (index, (session, prompt)) in typed.iter().zip(prompt_tokens).enumerate() {
            if let Err(error) = session.validate_prefill_input(prompt) {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        format!("batch request at index {index} is invalid: {error}"),
                    ),
                ));
            }
        }

        let sequence_lengths = prompt_tokens
            .iter()
            .map(|prompt| prompt.len())
            .collect::<Vec<_>>();
        let padded_length = sequence_lengths.iter().copied().max().unwrap_or(0);
        let mut token_values = Vec::with_capacity(batch_size * padded_length);
        for prompt in prompt_tokens {
            token_values.extend(prompt.iter().copied().map(i64::from));
            token_values.resize(token_values.len() + padded_length - prompt.len(), 0);
        }
        let input_ids: Tensor<2, Int> = Tensor::from_data(
            TensorData::new(token_values, [batch_size, padded_length]),
            &self.device,
        );
        let logits = {
            let mut caches = typed
                .iter_mut()
                .map(|session| {
                    session
                        .cache
                        .as_mut()
                        .expect("Llama sessions allocate a cache at creation")
                })
                .collect::<Vec<_>>();
            match self
                .module
                .prefill_batch(input_ids, caches.as_mut_slice(), &sequence_lengths)
            {
                Ok(logits) => logits,
                Err(error) => {
                    return Some(batch_error(
                        batch_size,
                        cache_error(ExecutionStage::Prefill, error),
                    ));
                }
            }
        };
        let final_rows = sequence_lengths
            .iter()
            .copied()
            .enumerate()
            .map(|(index, length)| {
                logits.clone().slice([
                    index..index + 1,
                    length - 1..length,
                    0..self.config.vocab_size,
                ])
            })
            .collect::<Vec<_>>();
        let selected = match Tensor::cat(final_rows, 0)
            .argmax(2)
            .into_data()
            .try_to_vec_as::<i64>()
        {
            Ok(selected) => selected,
            Err(error) => {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        format!("failed to read batched selected tokens: {error}"),
                    ),
                ));
            }
        };
        if selected.len() != batch_size {
            return Some(batch_error(
                batch_size,
                ModelExecutionError::new(
                    ExecutionStage::Prefill,
                    format!(
                        "backend returned {} token selections for batch size {batch_size}",
                        selected.len()
                    ),
                ),
            ));
        }

        Some(
            typed
                .into_iter()
                .zip(prompt_tokens.iter().copied())
                .zip(selected)
                .map(|((session, prompt), selected)| {
                    let selected = selected.elem::<i64>();
                    let id = u32::try_from(selected).map_err(|_| {
                        ModelExecutionError::new(
                            ExecutionStage::Prefill,
                            "backend returned a token ID outside the u32 range",
                        )
                    })?;
                    session.tokens.extend_from_slice(prompt);
                    session.metrics.prompt_tokens = prompt.len();
                    session.metrics.kv_cache_bytes = session
                        .cache
                        .as_ref()
                        .expect("a live Llama session owns a cache")
                        .bytes();
                    Ok(session.record_selected_token(id))
                })
                .collect(),
        )
    }

    fn prefill_chunk_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_chunks: &[&[u32]],
        is_final: &[bool],
    ) -> Option<Vec<Result<PrefillOutcome, ModelExecutionError>>> {
        let batch_size = sessions.len();
        if prompt_chunks.len() != batch_size || is_final.len() != batch_size {
            return Some(batch_error(
                batch_size,
                ModelExecutionError::new(
                    ExecutionStage::Prefill,
                    format!(
                        "chunk batch has {batch_size} sessions, {} chunks, and {} final flags",
                        prompt_chunks.len(),
                        is_final.len()
                    ),
                ),
            ));
        }
        if batch_size == 0 {
            return Some(Vec::new());
        }
        let mut typed = Vec::with_capacity(batch_size);
        for session in sessions {
            let Some(session) = session.as_any_mut().downcast_mut::<LlamaSession>() else {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        "chunked batch session was not created by this Llama backend",
                    ),
                ));
            };
            typed.push(session);
        }
        for (index, (session, chunk)) in typed.iter().zip(prompt_chunks).enumerate() {
            if let Err(error) = session.validate_prefill_chunk(chunk) {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        format!("chunk batch request at index {index} is invalid: {error}"),
                    ),
                ));
            }
        }

        let sequence_lengths = prompt_chunks
            .iter()
            .map(|chunk| chunk.len())
            .collect::<Vec<_>>();
        let padded_length = sequence_lengths.iter().copied().max().unwrap_or(0);
        let mut token_values = Vec::with_capacity(batch_size * padded_length);
        for chunk in prompt_chunks {
            token_values.extend(chunk.iter().copied().map(i64::from));
            token_values.resize(token_values.len() + padded_length - chunk.len(), 0);
        }
        let input_ids: Tensor<2, Int> = Tensor::from_data(
            TensorData::new(token_values, [batch_size, padded_length]),
            &self.device,
        );
        let logits = {
            let mut caches = typed
                .iter_mut()
                .map(|session| {
                    session
                        .cache
                        .as_mut()
                        .expect("Llama sessions allocate a cache at creation")
                })
                .collect::<Vec<_>>();
            match self.module.prefill_chunk_batch(
                input_ids,
                caches.as_mut_slice(),
                &sequence_lengths,
            ) {
                Ok(logits) => logits,
                Err(error) => {
                    return Some(batch_error(
                        batch_size,
                        cache_error(ExecutionStage::Prefill, error),
                    ));
                }
            }
        };
        let selected = match logits.argmax(2).into_data().try_to_vec_as::<i64>() {
            Ok(selected) => selected,
            Err(error) => {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        format!("failed to read chunked batch token selections: {error}"),
                    ),
                ));
            }
        };
        if selected.len() != batch_size {
            return Some(batch_error(
                batch_size,
                ModelExecutionError::new(
                    ExecutionStage::Prefill,
                    format!(
                        "backend returned {} chunk selections for batch size {batch_size}",
                        selected.len()
                    ),
                ),
            ));
        }

        Some(
            typed
                .into_iter()
                .zip(prompt_chunks.iter().copied())
                .zip(is_final.iter().copied())
                .zip(selected)
                .map(|(((session, chunk), is_final), selected)| {
                    session.tokens.extend_from_slice(chunk);
                    session.metrics.prompt_tokens = session.tokens.len();
                    session.metrics.kv_cache_bytes = session
                        .cache
                        .as_ref()
                        .expect("a live Llama session owns a cache")
                        .bytes();
                    if is_final {
                        let selected = selected.elem::<i64>();
                        let id = u32::try_from(selected).map_err(|_| {
                            ModelExecutionError::new(
                                ExecutionStage::Prefill,
                                "backend returned a token ID outside the u32 range",
                            )
                        })?;
                        Ok(PrefillOutcome::Token(session.record_selected_token(id)))
                    } else {
                        Ok(PrefillOutcome::Pending)
                    }
                })
                .collect(),
        )
    }

    fn decode_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        input_tokens: &[u32],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        let batch_size = sessions.len();
        if input_tokens.len() != batch_size {
            return Some(batch_error(
                batch_size,
                ModelExecutionError::new(
                    ExecutionStage::Decode,
                    format!(
                        "batch has {batch_size} sessions but {} input tokens",
                        input_tokens.len()
                    ),
                ),
            ));
        }
        let mut typed = Vec::with_capacity(batch_size);
        for session in sessions {
            let Some(session) = session.as_any_mut().downcast_mut::<LlamaSession>() else {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Decode,
                        "batched session was not created by this Llama backend",
                    ),
                ));
            };
            typed.push(session);
        }
        for (index, (session, input_token)) in
            typed.iter().zip(input_tokens.iter().copied()).enumerate()
        {
            if let Err(error) = session.validate_decode_input(input_token) {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Decode,
                        format!("batch request at index {index} is invalid: {error}"),
                    ),
                ));
            }
        }

        let token_values = input_tokens
            .iter()
            .copied()
            .map(i64::from)
            .collect::<Vec<_>>();
        let input_ids: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(token_values, [batch_size, 1]), &self.device);
        let logits = {
            let mut caches = typed
                .iter_mut()
                .map(|session| {
                    session
                        .cache
                        .as_mut()
                        .expect("Llama sessions allocate a cache at creation")
                })
                .collect::<Vec<_>>();
            match self.module.decode_batch(input_ids, caches.as_mut_slice()) {
                Ok(logits) => logits,
                Err(error) => {
                    return Some(batch_error(
                        batch_size,
                        cache_error(ExecutionStage::Decode, error),
                    ));
                }
            }
        };
        let selected = match logits
            .slice([0..batch_size, 0..1, 0..self.config.vocab_size])
            .argmax(2)
            .into_data()
            .try_to_vec_as::<i64>()
        {
            Ok(selected) => selected,
            Err(error) => {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Decode,
                        format!("failed to read batched selected tokens: {error}"),
                    ),
                ));
            }
        };
        if selected.len() != batch_size {
            return Some(batch_error(
                batch_size,
                ModelExecutionError::new(
                    ExecutionStage::Decode,
                    format!(
                        "backend returned {} token selections for batch size {batch_size}",
                        selected.len()
                    ),
                ),
            ));
        }

        Some(
            typed
                .into_iter()
                .zip(input_tokens.iter().copied())
                .zip(selected)
                .map(|((session, input_token), selected)| {
                    let selected = selected.elem::<i64>();
                    let id = u32::try_from(selected).map_err(|_| {
                        ModelExecutionError::new(
                            ExecutionStage::Decode,
                            "backend returned a token ID outside the u32 range",
                        )
                    })?;
                    session.tokens.push(input_token);
                    session.metrics.kv_cache_bytes = session
                        .cache
                        .as_ref()
                        .expect("a live Llama session owns a cache")
                        .bytes();
                    Ok(session.record_selected_token(id))
                })
                .collect(),
        )
    }
}

/// Request-local state for correctness-first cached Llama decoding.
struct LlamaSession {
    module: Arc<LlamaForCausalLM>,
    config: Arc<LlamaConfig>,
    device: burn::tensor::Device,
    generation: GenerationConfig,
    tokens: Vec<u32>,
    cache: Option<LlamaKvCache>,
    last_generated: Option<u32>,
    metrics: SessionMetrics,
    terminal: bool,
}

impl LlamaSession {
    fn select_token(
        &mut self,
        logits: Tensor<3>,
        stage: ExecutionStage,
    ) -> Result<GeneratedToken, ModelExecutionError> {
        let sequence_length = logits.dims()[1];
        let vocabulary = self.config.vocab_size;
        let last = logits
            .slice([0..1, sequence_length - 1..sequence_length, 0..vocabulary])
            .argmax(2)
            .into_data()
            .try_to_vec_as::<i64>()
            .map_err(|error| {
                ModelExecutionError::new(stage, format!("failed to read selected token: {error}"))
            })?;
        let selected = last[0].elem::<i64>();
        let id = u32::try_from(selected).map_err(|_| {
            ModelExecutionError::new(stage, "backend returned a token ID outside the u32 range")
        })?;
        Ok(self.record_selected_token(id))
    }

    fn record_selected_token(&mut self, id: u32) -> GeneratedToken {
        let is_end_of_sequence = self.config.eos_token_id.contains(&id);
        self.last_generated = Some(id);
        self.metrics.generated_tokens += 1;
        self.terminal =
            is_end_of_sequence || self.metrics.generated_tokens >= self.generation.max_new_tokens;
        GeneratedToken {
            id,
            is_end_of_sequence,
        }
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

    fn validate_decode_input(&self, input_token: u32) -> Result<(), ModelExecutionError> {
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
        self.validate_context_length(self.tokens.len() + 1, ExecutionStage::Decode)
    }

    fn validate_prefill_input(&self, prompt_tokens: &[u32]) -> Result<(), ModelExecutionError> {
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
        self.validate_context_length(prompt_tokens.len(), ExecutionStage::Prefill)
    }

    fn validate_prefill_chunk(&self, prompt_tokens: &[u32]) -> Result<(), ModelExecutionError> {
        if self.last_generated.is_some() || self.terminal {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prefill cannot continue after the first token was generated",
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prefill chunk must contain at least one token",
            ));
        }
        for &token in prompt_tokens {
            self.validate_token(token, ExecutionStage::Prefill)?;
        }
        self.validate_context_length(
            self.tokens.len().saturating_add(prompt_tokens.len()),
            ExecutionStage::Prefill,
        )
    }
}

impl GenerationSession for LlamaSession {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<GeneratedToken, ModelExecutionError> {
        self.validate_prefill_input(prompt_tokens)?;
        let token_values = prompt_tokens.iter().map(|&id| i64::from(id)).collect();
        let input_ids: Tensor<2, Int> = Tensor::from_data(
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

    fn prefill_chunk(
        &mut self,
        prompt_tokens: &[u32],
        is_final: bool,
    ) -> Result<PrefillOutcome, ModelExecutionError> {
        self.validate_prefill_chunk(prompt_tokens)?;
        let token_values = prompt_tokens.iter().map(|&id| i64::from(id)).collect();
        let input_ids: Tensor<2, Int> = Tensor::from_data(
            TensorData::new(token_values, [1, prompt_tokens.len()]),
            &self.device,
        );
        let cache = self
            .cache
            .as_mut()
            .expect("sessions allocate a cache at creation");
        let logits = self
            .module
            .prefill_chunk(input_ids, cache)
            .map_err(|error| cache_error(ExecutionStage::Prefill, error))?;
        self.metrics.kv_cache_bytes = cache.bytes();
        self.tokens.extend_from_slice(prompt_tokens);
        self.metrics.prompt_tokens = self.tokens.len();
        if is_final {
            self.select_token(logits, ExecutionStage::Prefill)
                .map(PrefillOutcome::Token)
        } else {
            Ok(PrefillOutcome::Pending)
        }
    }

    fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
        self.validate_decode_input(input_token)?;
        let input_ids: Tensor<2, Int> = Tensor::from_data(
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

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn cache_error(stage: ExecutionStage, error: KvBlockError) -> ModelExecutionError {
    let message = format!("KV cache: {error}");
    match error {
        KvBlockError::InsufficientCapacity { .. }
        | KvBlockError::BatchInsufficientCapacity { .. } => {
            ModelExecutionError::kv_capacity_exhausted(stage, message)
        }
        _ => ModelExecutionError::new(stage, message),
    }
}

fn batch_error<T>(
    batch_size: usize,
    error: ModelExecutionError,
) -> Vec<Result<T, ModelExecutionError>> {
    (0..batch_size).map(|_| Err(error.clone())).collect()
}

#[cfg(test)]
mod tests {
    use ola_core::{ModelDescriptor, RequestSpec, Timestamp};
    use ola_model::{CausalLanguageModel, Tokenizer};
    use ola_runtime::{BatchExecutionMode, BatchPhase, ExecutionBatch, Executor, Runtime};

    use super::*;

    #[derive(Clone, Copy)]
    struct NumericTokenizer;

    impl Tokenizer for NumericTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> Result<Vec<u32>, ModelExecutionError> {
            text.split_whitespace()
                .map(|piece| {
                    piece.parse().map_err(|_| {
                        ModelExecutionError::new(
                            ExecutionStage::Tokenize,
                            "expected numeric token IDs",
                        )
                    })
                })
                .collect()
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> Result<String, ModelExecutionError> {
            Ok(token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    fn tiny_model_config() -> LlamaConfig {
        LlamaConfig {
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
        }
    }

    fn tiny_model() -> BurnLlamaModel {
        let config = tiny_model_config();
        let device = Default::default();
        let module = LlamaForCausalLM::init(&config, &device);
        BurnLlamaModel::with_kv_cache(module, config, device, LlamaKvCacheConfig::new(64, 16))
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
    fn chunked_prefill_matches_complete_prefill_and_cached_decode() {
        let model = tiny_model();
        let mut complete = model.create_session(greedy(3)).unwrap();
        let mut chunked = model.create_session(greedy(3)).unwrap();
        let prompt = [1, 4, 5, 7, 3];

        let expected = complete.prefill(&prompt).unwrap();
        assert_eq!(
            chunked.prefill_chunk(&prompt[..2], false).unwrap(),
            PrefillOutcome::Pending
        );
        assert_eq!(
            chunked.metrics(),
            SessionMetrics {
                prompt_tokens: 2,
                generated_tokens: 0,
                kv_cache_bytes: 512,
            }
        );
        let actual = chunked.prefill_chunk(&prompt[2..], true).unwrap();

        assert_eq!(actual, PrefillOutcome::Token(expected));
        assert_eq!(chunked.metrics(), complete.metrics());
        assert_eq!(
            chunked.decode(expected.id).unwrap(),
            complete.decode(expected.id).unwrap()
        );
    }

    #[test]
    fn batched_chunks_match_complete_prefill_at_unequal_positions() {
        let model = tiny_model();
        let first_prompt = [1, 4, 5, 7, 3];
        let second_prompt = [1, 6, 8];
        let mut first = model.create_session(greedy(2)).unwrap();
        let mut second = model.create_session(greedy(2)).unwrap();
        let mut first_oracle = model.create_session(greedy(2)).unwrap();
        let mut second_oracle = model.create_session(greedy(2)).unwrap();
        let expected = [
            first_oracle.prefill(&first_prompt).unwrap(),
            second_oracle.prefill(&second_prompt).unwrap(),
        ];
        assert_eq!(
            first.prefill_chunk(&first_prompt[..2], false).unwrap(),
            PrefillOutcome::Pending
        );
        assert_eq!(
            second.prefill_chunk(&second_prompt[..1], false).unwrap(),
            PrefillOutcome::Pending
        );

        let outcomes = {
            let mut sessions: Vec<&mut dyn GenerationSession> =
                vec![first.as_mut(), second.as_mut()];
            model
                .prefill_chunk_batch(
                    &mut sessions,
                    &[&first_prompt[2..], &second_prompt[1..]],
                    &[true, true],
                )
                .expect("Burn Llama implements tensor-batched resumable prefill")
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };

        assert_eq!(outcomes, expected.map(PrefillOutcome::Token).to_vec());
        assert_eq!(first.metrics(), first_oracle.metrics());
        assert_eq!(second.metrics(), second_oracle.metrics());
        assert_eq!(
            first.decode(expected[0].id).unwrap(),
            first_oracle.decode(expected[0].id).unwrap()
        );
        assert_eq!(
            second.decode(expected[1].id).unwrap(),
            second_oracle.decode(expected[1].id).unwrap()
        );
    }

    #[test]
    fn chunk_reservation_failure_does_not_advance_the_session_cursor() {
        let mut config = tiny_model_config();
        config.max_position_embeddings = 8;
        let device = Default::default();
        let module = LlamaForCausalLM::init(&config, &device);
        let model =
            BurnLlamaModel::with_kv_cache(module, config, device, LlamaKvCacheConfig::new(2, 1))
                .unwrap();
        let mut session = model.create_session(greedy(2)).unwrap();
        assert_eq!(
            session.prefill_chunk(&[1], false).unwrap(),
            PrefillOutcome::Pending
        );
        let before = session.metrics();

        let status = model.kv_pool_status().unwrap();
        assert_eq!(
            (
                status.total_blocks,
                status.allocated_blocks,
                status.free_blocks
            ),
            (2, 1, 1)
        );

        let error = session.prefill_chunk(&[4, 5], true).unwrap_err();

        assert!(error.message().contains("KV cache"));
        assert!(error.is_kv_capacity_exhausted());
        assert_eq!(session.metrics(), before);
        assert!(matches!(
            session.prefill_chunk(&[4], true).unwrap(),
            PrefillOutcome::Token(GeneratedToken {
                is_end_of_sequence: false,
                ..
            })
        ));
        assert_eq!(session.metrics().prompt_tokens, 2);
        assert_eq!(session.metrics().generated_tokens, 1);
        drop(session);
        assert_eq!(model.kv_pool_status().unwrap().free_blocks, 2);
    }

    #[test]
    fn batched_prefill_matches_individual_sessions_with_different_lengths() {
        let model = tiny_model();
        let mut first = model.create_session(greedy(3)).unwrap();
        let mut second = model.create_session(greedy(3)).unwrap();
        let mut first_oracle = model.create_session(greedy(3)).unwrap();
        let mut second_oracle = model.create_session(greedy(3)).unwrap();
        let first_prompt = [1, 4];
        let second_prompt = [1, 5, 7];
        let expected = vec![
            first_oracle.prefill(&first_prompt).unwrap(),
            second_oracle.prefill(&second_prompt).unwrap(),
        ];

        let (batched, first_metrics, second_metrics) = {
            let mut sessions: Vec<&mut dyn GenerationSession> =
                vec![first.as_mut(), second.as_mut()];
            let batched = model
                .prefill_batch(&mut sessions, &[&first_prompt, &second_prompt])
                .expect("Burn Llama implements tensor-batched prefill")
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            (batched, sessions[0].metrics(), sessions[1].metrics())
        };

        assert_eq!(batched, expected);
        assert_eq!(first_metrics, first_oracle.metrics());
        assert_eq!(second_metrics, second_oracle.metrics());
        assert_eq!(
            first.decode(batched[0].id).unwrap(),
            first_oracle.decode(expected[0].id).unwrap()
        );
        assert_eq!(
            second.decode(batched[1].id).unwrap(),
            second_oracle.decode(expected[1].id).unwrap()
        );
    }

    #[test]
    fn batched_decode_matches_individual_sessions_with_different_positions() {
        let model = tiny_model();
        let mut first = model.create_session(greedy(3)).unwrap();
        let mut second = model.create_session(greedy(3)).unwrap();
        let mut first_oracle = model.create_session(greedy(3)).unwrap();
        let mut second_oracle = model.create_session(greedy(3)).unwrap();
        let first_token = first.prefill(&[1, 4]).unwrap();
        let second_token = second.prefill(&[1, 5, 7]).unwrap();
        let first_oracle_token = first_oracle.prefill(&[1, 4]).unwrap();
        let second_oracle_token = second_oracle.prefill(&[1, 5, 7]).unwrap();
        assert_eq!(first_token, first_oracle_token);
        assert_eq!(second_token, second_oracle_token);

        let (batched, first_metrics, second_metrics) = {
            let mut sessions: Vec<&mut dyn GenerationSession> =
                vec![first.as_mut(), second.as_mut()];
            let batched = model
                .decode_batch(&mut sessions, &[first_token.id, second_token.id])
                .expect("Burn Llama implements tensor-batched decode")
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            (batched, sessions[0].metrics(), sessions[1].metrics())
        };
        let expected = vec![
            first_oracle.decode(first_oracle_token.id).unwrap(),
            second_oracle.decode(second_oracle_token.id).unwrap(),
        ];

        assert_eq!(batched, expected);
        assert_eq!(first_metrics, first_oracle.metrics());
        assert_eq!(second_metrics, second_oracle.metrics());
    }

    /// Runs `prompts` together on `model` with newest-first preemption and
    /// returns each output and the number of preemptions.
    fn run_with_preemption(model: BurnLlamaModel, prompts: &[&str]) -> (Vec<Vec<u32>>, usize) {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), prompts.len())
            .unwrap()
            .with_policy(
                ola_runtime::ComposedPolicy::default().with_preemption(ola_runtime::NewestFirst),
            );
        executor
            .register_model(
                ModelDescriptor::language("tiny", "llama"),
                CausalLanguageModel::new(NumericTokenizer, model),
            )
            .unwrap();
        let ids = prompts
            .iter()
            .map(|prompt| {
                let mut spec = RequestSpec::text("tiny", *prompt);
                spec.generation = greedy(10);
                executor.submit(spec, Timestamp::ZERO).unwrap()
            })
            .collect::<Vec<_>>();
        let mut preemptions = 0;
        for step in 1..=200 {
            let iteration = executor
                .step_iteration(Timestamp::from_micros(step))
                .unwrap();
            for event in &iteration.events {
                match event {
                    ola_runtime::ExecutorEvent::Preempted { .. } => preemptions += 1,
                    ola_runtime::ExecutorEvent::Failed { message, .. } => panic!("{message}"),
                    _ => {}
                }
            }
            if iteration.is_idle() {
                break;
            }
        }
        let outputs = ids
            .iter()
            .map(|id| executor.result(*id).unwrap().token_ids.clone())
            .collect();
        (outputs, preemptions)
    }

    #[test]
    fn preempted_llama_requests_resume_with_identical_greedy_output() {
        let config = tiny_model_config();
        let device: burn::tensor::Device = Default::default();
        let module = LlamaForCausalLM::init(&config, &device);
        let model = |capacity_tokens| {
            BurnLlamaModel::with_kv_cache(
                module.clone(),
                config.clone(),
                device.clone(),
                LlamaKvCacheConfig::new(capacity_tokens, 4),
            )
            .unwrap()
        };
        let prompts = ["1 2 3", "4 5 6", "7 8 9"];

        // Alone, with room to spare, each request is the reference.
        let reference = prompts
            .iter()
            .map(|prompt| run_with_preemption(model(64), &[prompt]).0.remove(0))
            .collect::<Vec<_>>();
        // Together in seven 4-token blocks, three requests needing three
        // blocks each must be preempted and resumed by recompute.
        let (outputs, preemptions) = run_with_preemption(model(28), &prompts);

        assert!(preemptions > 0, "the pool must force preemption");
        assert_eq!(outputs, reference);
    }

    #[test]
    fn executor_uses_the_llama_tensor_decode_path_for_compatible_requests() {
        let artifact = CausalLanguageModel::new(NumericTokenizer, tiny_model());
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
        executor
            .register_model(ModelDescriptor::language("tiny", "llama"), artifact)
            .unwrap();
        let mut first_spec = RequestSpec::text("tiny", "1 4");
        first_spec.generation = greedy(3);
        let mut second_spec = RequestSpec::text("tiny", "1 5 7");
        second_spec.generation = greedy(3);
        let first = executor.submit(first_spec, Timestamp::ZERO).unwrap();
        let second = executor.submit(second_spec, Timestamp::ZERO).unwrap();
        let admitted = executor.step_iteration(Timestamp::from_micros(10)).unwrap();

        assert_eq!(
            admitted.execution_batches,
            vec![ExecutionBatch {
                phase: BatchPhase::Prefill,
                request_ids: vec![first, second],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 5,
                padding_tokens: 1,
                started_at: Timestamp::from_micros(10),
                finished_at: Timestamp::from_micros(10),
            }]
        );

        let decoded = executor.step_iteration(Timestamp::from_micros(20)).unwrap();

        assert_eq!(
            decoded.execution_batches,
            vec![ExecutionBatch {
                phase: BatchPhase::Decode,
                request_ids: vec![first, second],
                mode: BatchExecutionMode::Tensor,
                input_tokens: 2,
                padding_tokens: 0,
                started_at: Timestamp::from_micros(20),
                finished_at: Timestamp::from_micros(20),
            }]
        );
        assert_eq!(decoded.active_requests, 2);
        assert_eq!(decoded.execution_plan.estimates().input_tokens, 2);
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
