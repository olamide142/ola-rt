use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::tensor::{Int, Tensor, TensorData};
use ola_core::GenerationConfig;
use ola_kernel::{PagedAttentionKernel, PortablePagedAttention};
use ola_model::{
    BatchCapabilities, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
    ModelExecutionContext, ModelExecutionError, PrefillOutcome, SessionMetrics,
};
use ola_runtime::KvBlockError;

use super::{
    MistralConfig, MistralForCausalLM, MistralKvCacheConfig,
    kv_cache::{MistralKvCache, MistralKvPool},
};

/// Shared Burn weights exposed through OlaRT's architecture-neutral model API.
pub(crate) struct BurnMistralModel {
    module: Arc<MistralForCausalLM>,
    config: Arc<MistralConfig>,
    device: burn::tensor::Device,
    cache_pool: Arc<Mutex<MistralKvPool>>,
}

impl BurnMistralModel {
    pub(crate) fn with_kv_cache(
        module: MistralForCausalLM,
        config: MistralConfig,
        device: burn::tensor::Device,
        cache_config: MistralKvCacheConfig,
    ) -> Result<Self, KvBlockError> {
        let cache_pool = MistralKvPool::new(&config, cache_config, device.clone())?;
        Ok(Self {
            module: Arc::new(module),
            config: Arc::new(config),
            device,
            cache_pool: Arc::new(Mutex::new(cache_pool)),
        })
    }
}

fn paged_attention_kernel(context: ModelExecutionContext<'_>) -> Box<dyn PagedAttentionKernel> {
    match (context.paged_attention(), context.kernel_metrics()) {
        (Some(selection), Some(metrics)) => selection.create_measured(metrics),
        (Some(selection), None) => selection.create(),
        (None, _) => Box::new(PortablePagedAttention::new()),
    }
}

fn entropy_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64)
}

fn next_random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^= value >> 31;
    (value >> 11) as f64 * (1.0 / 9007199254740992.0)
}

fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    state: &mut u64,
) -> Result<usize, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("model returned non-finite or empty logits".to_owned());
    }
    if temperature == 0.0 {
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(left_id, left), (right_id, right)| {
                left.total_cmp(right).then_with(|| right_id.cmp(left_id))
            })
            .map(|(id, _)| id)
            .expect("non-empty logits have a maximum"));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|logit| ((*logit - maximum) / temperature).exp() as f64)
        .collect::<Vec<_>>();
    let normalizer = probabilities.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err("model logits could not produce a probability distribution".to_owned());
    }
    probabilities
        .iter_mut()
        .for_each(|probability| *probability /= normalizer);
    let mut ranked = (0..probabilities.len()).collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        probabilities[*right]
            .total_cmp(&probabilities[*left])
            .then_with(|| left.cmp(right))
    });
    let mut retained = Vec::new();
    let mut cumulative = 0.0;
    for index in ranked {
        retained.push(index);
        cumulative += probabilities[index];
        if cumulative >= f64::from(top_p) {
            break;
        }
    }
    let draw = next_random(state) * cumulative;
    let mut total = 0.0;
    for index in retained {
        total += probabilities[index];
        if draw < total {
            return Ok(index);
        }
    }
    Ok(logits.len() - 1)
}

impl CausalModel for BurnMistralModel {
    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    fn cached_prefix(&self, tokens: &[u32]) -> ola_model::CachedPrefix {
        self.cache_pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_prefix(tokens)
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
        if !generation.temperature.is_finite() || generation.temperature < 0.0 {
            return Err(ModelExecutionError::new(
                ExecutionStage::SessionCreation,
                "temperature must be finite and non-negative",
            ));
        }
        let cache = MistralKvCache::allocate(self.cache_pool.clone())
            .map_err(|error| cache_error(ExecutionStage::SessionCreation, error))?;
        let rng_state = generation.seed.unwrap_or_else(entropy_seed);
        Ok(Box::new(MistralSession {
            module: self.module.clone(),
            config: self.config.clone(),
            device: self.device.clone(),
            generation,
            tokens: Vec::new(),
            cache: Some(cache),
            rng_state,
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
            let Some(session) = session.as_any_mut().downcast_mut::<MistralSession>() else {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        "batched session was not created by this Mistral backend",
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
                        .expect("Mistral sessions allocate a cache at creation")
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
        Some(
            typed
                .into_iter()
                .zip(prompt_tokens.iter().copied())
                .enumerate()
                .map(|(index, (session, prompt))| {
                    let token =
                        session.select_token(final_rows[index].clone(), ExecutionStage::Prefill)?;
                    session.accept_tokens(prompt);
                    session.metrics.prompt_tokens = prompt.len();
                    session.metrics.kv_cache_bytes = session
                        .cache
                        .as_ref()
                        .expect("a live Mistral session owns a cache")
                        .bytes();
                    Ok(token)
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
        self.prefill_chunk_batch_with_context(
            sessions,
            prompt_chunks,
            is_final,
            ModelExecutionContext::new(),
        )
    }

    fn prefill_chunk_batch_with_context(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_chunks: &[&[u32]],
        is_final: &[bool],
        context: ModelExecutionContext<'_>,
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
            let Some(session) = session.as_any_mut().downcast_mut::<MistralSession>() else {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Prefill,
                        "chunked batch session was not created by this Mistral backend",
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
        let kernel = paged_attention_kernel(context);
        let logits = {
            let mut caches = typed
                .iter_mut()
                .map(|session| {
                    session
                        .cache
                        .as_mut()
                        .expect("Mistral sessions allocate a cache at creation")
                })
                .collect::<Vec<_>>();
            match self.module.prefill_chunk_batch(
                input_ids,
                caches.as_mut_slice(),
                &sequence_lengths,
                kernel.as_ref(),
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
        Some(
            typed
                .into_iter()
                .zip(prompt_chunks.iter().copied())
                .zip(is_final.iter().copied())
                .enumerate()
                .map(|(index, ((session, chunk), is_final))| {
                    let token = if is_final {
                        Some(session.select_token(
                            logits.clone().slice([
                                index..index + 1,
                                0..1,
                                0..self.config.vocab_size,
                            ]),
                            ExecutionStage::Prefill,
                        )?)
                    } else {
                        None
                    };
                    session.accept_tokens(chunk);
                    session.metrics.prompt_tokens = session.tokens.len();
                    session.metrics.kv_cache_bytes = session
                        .cache
                        .as_ref()
                        .expect("a live Mistral session owns a cache")
                        .bytes();
                    if is_final {
                        Ok(PrefillOutcome::Token(
                            token.expect("final chunks select a token"),
                        ))
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
        self.decode_batch_with_context(sessions, input_tokens, ModelExecutionContext::new())
    }

    fn decode_batch_with_context(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        input_tokens: &[u32],
        context: ModelExecutionContext<'_>,
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
            let Some(session) = session.as_any_mut().downcast_mut::<MistralSession>() else {
                return Some(batch_error(
                    batch_size,
                    ModelExecutionError::new(
                        ExecutionStage::Decode,
                        "batched session was not created by this Mistral backend",
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
        let kernel = paged_attention_kernel(context);
        let logits = {
            let mut caches = typed
                .iter_mut()
                .map(|session| {
                    session
                        .cache
                        .as_mut()
                        .expect("Mistral sessions allocate a cache at creation")
                })
                .collect::<Vec<_>>();
            match self
                .module
                .decode_batch(input_ids, caches.as_mut_slice(), kernel.as_ref())
            {
                Ok(logits) => logits,
                Err(error) => {
                    return Some(batch_error(
                        batch_size,
                        cache_error(ExecutionStage::Decode, error),
                    ));
                }
            }
        };
        Some(
            typed
                .into_iter()
                .zip(input_tokens.iter().copied())
                .enumerate()
                .map(|(index, (session, input_token))| {
                    let token = session.select_token(
                        logits
                            .clone()
                            .slice([index..index + 1, 0..1, 0..self.config.vocab_size]),
                        ExecutionStage::Decode,
                    )?;
                    session.accept_tokens(&[input_token]);
                    session.metrics.kv_cache_bytes = session
                        .cache
                        .as_ref()
                        .expect("a live Mistral session owns a cache")
                        .bytes();
                    Ok(token)
                })
                .collect(),
        )
    }
}

/// Request-local state for correctness-first cached Mistral decoding.
struct MistralSession {
    module: Arc<MistralForCausalLM>,
    config: Arc<MistralConfig>,
    device: burn::tensor::Device,
    generation: GenerationConfig,
    tokens: Vec<u32>,
    cache: Option<MistralKvCache>,
    rng_state: u64,
    last_generated: Option<u32>,
    metrics: SessionMetrics,
    terminal: bool,
}

impl MistralSession {
    fn accept_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        if let Some(cache) = &self.cache {
            cache.record_tokens(tokens);
        }
    }

    fn select_token(
        &mut self,
        logits: Tensor<3>,
        stage: ExecutionStage,
    ) -> Result<GeneratedToken, ModelExecutionError> {
        let sequence_length = logits.dims()[1];
        let vocabulary = self.config.vocab_size;
        let values = logits
            .slice([0..1, sequence_length - 1..sequence_length, 0..vocabulary])
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|error| {
                ModelExecutionError::new(stage, format!("failed to read selected token: {error}"))
            })?;
        let id = sample_token(
            &values,
            self.generation.temperature,
            self.generation.top_p,
            &mut self.rng_state,
        )
        .map_err(|error| ModelExecutionError::new(stage, error))?;
        let id = u32::try_from(id).map_err(|_| {
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

impl GenerationSession for MistralSession {
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
        self.accept_tokens(prompt_tokens);
        self.metrics.prompt_tokens = prompt_tokens.len();
        self.select_token(logits, ExecutionStage::Prefill)
    }

    fn prefill_chunk(
        &mut self,
        prompt_tokens: &[u32],
        is_final: bool,
    ) -> Result<PrefillOutcome, ModelExecutionError> {
        self.prefill_chunk_with_context(prompt_tokens, is_final, ModelExecutionContext::new())
    }

    fn prefill_chunk_with_context(
        &mut self,
        prompt_tokens: &[u32],
        is_final: bool,
        context: ModelExecutionContext<'_>,
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
        let kernel = paged_attention_kernel(context);
        let logits = self
            .module
            .prefill_chunk(input_ids, cache, kernel.as_ref())
            .map_err(|error| cache_error(ExecutionStage::Prefill, error))?;
        self.metrics.kv_cache_bytes = cache.bytes();
        self.accept_tokens(prompt_tokens);
        self.metrics.prompt_tokens = self.tokens.len();
        if is_final {
            self.select_token(logits, ExecutionStage::Prefill)
                .map(PrefillOutcome::Token)
        } else {
            Ok(PrefillOutcome::Pending)
        }
    }

    fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
        self.decode_with_context(input_token, ModelExecutionContext::new())
    }

    fn decode_with_context(
        &mut self,
        input_token: u32,
        context: ModelExecutionContext<'_>,
    ) -> Result<GeneratedToken, ModelExecutionError> {
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
        let kernel = paged_attention_kernel(context);
        let logits = self
            .module
            .decode(input_ids, cache, kernel.as_ref())
            .map_err(|error| cache_error(ExecutionStage::Decode, error))?;
        self.metrics.kv_cache_bytes = cache.bytes();
        self.accept_tokens(&[input_token]);
        self.select_token(logits, ExecutionStage::Decode)
    }

    fn attach_prefix(&mut self, tokens: &[u32]) -> Result<usize, ModelExecutionError> {
        if !self.tokens.is_empty() || self.last_generated.is_some() {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "a cached prefix can only start a fresh session",
            ));
        }
        let cache = self.cache.as_mut().ok_or_else(|| {
            ModelExecutionError::new(ExecutionStage::Prefill, "the session has no KV cache")
        })?;
        let attached = cache
            .attach_prefix(tokens)
            .map_err(|error| cache_error(ExecutionStage::Prefill, error))?;
        self.metrics.kv_cache_bytes = cache.bytes();
        self.tokens.extend_from_slice(&tokens[..attached]);
        self.metrics.prompt_tokens = attached;
        Ok(attached)
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
    use ola_runtime::{ComposedPolicy, Executor, NewestFirst, Runtime};

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

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        }
    }

    fn run_sequentially(model: BurnMistralModel, prompts: &[&str]) -> Vec<Vec<u32>> {
        let mut executor = Executor::default().with_prefill_chunk_size(64).unwrap();
        executor
            .register_model(
                ModelDescriptor::language("tiny", "mistral"),
                CausalLanguageModel::new(NumericTokenizer, model),
            )
            .unwrap();
        let mut step = 0;
        prompts
            .iter()
            .map(|prompt| {
                let mut spec = RequestSpec::text("tiny", *prompt);
                spec.generation = greedy(4);
                let id = executor.submit(spec, Timestamp::from_micros(step)).unwrap();
                loop {
                    step += 1;
                    if executor
                        .step_iteration(Timestamp::from_micros(step))
                        .unwrap()
                        .is_idle()
                    {
                        break;
                    }
                }
                executor.take_result(id).unwrap().token_ids
            })
            .collect()
    }

    /// Runs `prompts` together on `model` with newest-first preemption and
    /// returns each output and the number of preemptions.
    fn run_with_preemption(model: BurnMistralModel, prompts: &[&str]) -> (Vec<Vec<u32>>, usize) {
        let mut executor = Executor::with_max_active_requests(Runtime::default(), prompts.len())
            .unwrap()
            .with_policy(
                ComposedPolicy::default().with_preemption(NewestFirst),
            );
        executor
            .register_model(
                ModelDescriptor::language("tiny", "mistral"),
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

    fn tiny_config(sliding_window: Option<usize>) -> MistralConfig {
        MistralConfig {
            model_type: "mistral".into(),
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window,
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

    fn make_model(
        config: &MistralConfig,
        cache_config: MistralKvCacheConfig,
    ) -> BurnMistralModel {
        let device = burn::tensor::Device::default();
        let module = MistralForCausalLM::init(config, &device);
        BurnMistralModel::with_kv_cache(module, config.clone(), device, cache_config).unwrap()
    }

    #[test]
    fn enforces_the_generation_limit_inside_the_session() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::for_model(&config));
        let mut session = model
            .create_session(GenerationConfig {
                max_new_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let first = session.prefill(&[1, 5]).unwrap();
        let second = session.decode(first.id);

        assert!(matches!(second, Err(error) if error.stage() == ExecutionStage::Decode));
        assert_eq!(session.metrics().generated_tokens, 1);
    }

    #[test]
    fn rejects_a_zero_generation_limit_at_session_creation() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::for_model(&config));
        let error = model
            .create_session(GenerationConfig {
                max_new_tokens: 0,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .err()
            .unwrap();

        assert_eq!(error.stage(), ExecutionStage::SessionCreation);
        assert!(error.message().contains("greater than zero"));
    }

    #[test]
    fn seeded_sampling_is_accepted_and_reproducible() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::for_model(&config));
        let generate = |seed| {
            let mut session = model
                .create_session(GenerationConfig {
                    max_new_tokens: 4,
                    temperature: 0.7,
                    top_p: 0.9,
                    seed: Some(seed),
                })
                .unwrap();
            let first = session.prefill(&[1, 5, 7]).unwrap();
            let mut tokens = vec![first.id];
            let mut previous = first;
            for _ in 1..4 {
                if previous.is_end_of_sequence {
                    break;
                }
                previous = session.decode(previous.id).unwrap();
                tokens.push(previous.id);
            }
            tokens
        };

        assert_eq!(generate(42), generate(42));
    }

    #[test]
    fn sampler_respects_nucleus_cutoff_and_seed() {
        let logits = [0.0_f32, 10.0, 0.0, 0.0];
        let mut state = 7;
        let selected = sample_token(&logits, 0.7, 0.5, &mut state).unwrap();
        assert_eq!(selected, 1);
    }

    #[test]
    fn chunked_prefill_matches_complete_prefill_and_cached_decode() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::for_model(&config));
        let prompt = [1_u32, 5, 9, 13];

        let mut monolithic = model
            .create_session(GenerationConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let expected_first = monolithic.prefill(&prompt).unwrap();

        let mut chunked = model
            .create_session(GenerationConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        assert_eq!(
            chunked.prefill_chunk(&prompt[..2], false).unwrap(),
            PrefillOutcome::Pending
        );
        let actual_first = match chunked.prefill_chunk(&prompt[2..], true).unwrap() {
            PrefillOutcome::Token(token) => token,
            PrefillOutcome::Pending => panic!("final chunk must emit a token"),
        };
        assert_eq!(actual_first, expected_first);

        let expected_second = monolithic.decode(expected_first.id).unwrap();
        let actual_second = chunked.decode(actual_first.id).unwrap();
        assert_eq!(actual_second, expected_second);
    }

    #[test]
    fn batched_prefill_matches_individual_sessions_with_different_lengths() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::new(64, 4));
        let prompt_a = [1_u32, 5, 9];
        let prompt_b = [1_u32, 7];

        let mut solo_a = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let expected_a = solo_a.prefill(&prompt_a).unwrap();

        let mut solo_b = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let expected_b = solo_b.prefill(&prompt_b).unwrap();

        let mut batched_a = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let mut batched_b = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();

        let results = model
            .prefill_batch(
                &mut [batched_a.as_mut(), batched_b.as_mut()],
                &[&prompt_a, &prompt_b],
            )
            .unwrap();

        assert_eq!(results[0].as_ref().unwrap(), &expected_a);
        assert_eq!(results[1].as_ref().unwrap(), &expected_b);
    }

    #[test]
    fn batched_decode_matches_individual_sessions_with_different_positions() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::new(64, 4));
        let prompt_a = [1_u32, 5, 9];
        let prompt_b = [1_u32, 7];

        let mut session_a = model
            .create_session(GenerationConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let first_a = session_a.prefill(&prompt_a).unwrap();

        let mut session_b = model
            .create_session(GenerationConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let first_b = session_b.prefill(&prompt_b).unwrap();

        let expected_second_a = session_a.decode(first_a.id).unwrap();

        // Advance session_b by two decode steps so positions differ
        let b2 = session_b.decode(first_b.id).unwrap();
        let expected_third_b = session_b.decode(b2.id).unwrap();

        // Now run batched decode on a fresh pair advanced to the same cursors
        let mut batch_a = model
            .create_session(GenerationConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let ba_first = batch_a.prefill(&prompt_a).unwrap();

        let mut batch_b = model
            .create_session(GenerationConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let bb_first = batch_b.prefill(&prompt_b).unwrap();
        let bb_second = batch_b.decode(bb_first.id).unwrap();

        let outputs = model
            .decode_batch(
                &mut [batch_a.as_mut(), batch_b.as_mut()],
                &[ba_first.id, bb_second.id],
            )
            .unwrap();

        assert_eq!(outputs[0].as_ref().unwrap(), &expected_second_a);
        assert_eq!(outputs[1].as_ref().unwrap(), &expected_third_b);
    }

    #[test]
    fn batched_chunks_match_complete_prefill_at_unequal_positions() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::new(64, 4));
        let prompt_a = [1_u32, 5, 9, 13];
        let prompt_b = [1_u32, 7, 11];

        let mut solo_a = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let expected_a = solo_a.prefill(&prompt_a).unwrap();

        let mut solo_b = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let expected_b = solo_b.prefill(&prompt_b).unwrap();

        let mut batch_a = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let mut batch_b = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();

        // Round 1: unequal lengths
        let r1 = model
            .prefill_chunk_batch(
                &mut [batch_a.as_mut(), batch_b.as_mut()],
                &[&prompt_a[..2], &prompt_b[..1]],
                &[false, false],
            )
            .unwrap();
        assert_eq!(r1[0].as_ref().unwrap(), &PrefillOutcome::Pending);
        assert_eq!(r1[1].as_ref().unwrap(), &PrefillOutcome::Pending);

        // Round 2: final chunks
        let r2 = model
            .prefill_chunk_batch(
                &mut [batch_a.as_mut(), batch_b.as_mut()],
                &[&prompt_a[2..], &prompt_b[1..]],
                &[true, true],
            )
            .unwrap();
        assert_eq!(r2[0].as_ref().unwrap(), &PrefillOutcome::Token(expected_a));
        assert_eq!(r2[1].as_ref().unwrap(), &PrefillOutcome::Token(expected_b));
    }

    #[test]
    fn session_reports_paged_cache_bytes() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::new(16, 2));
        let mut session = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();

        assert_eq!(session.metrics().kv_cache_bytes, 0);
        let _ = session.prefill(&[1, 5, 9]).unwrap();
        assert!(session.metrics().kv_cache_bytes > 0);
    }

    #[test]
    fn a_cached_mistral_prefix_gives_the_same_greedy_output_as_recomputing_it() {
        let config = tiny_config(None);
        let device: burn::tensor::Device = Default::default();
        let module = MistralForCausalLM::init(&config, &device);
        let model = |cache: MistralKvCacheConfig| {
            BurnMistralModel::with_kv_cache(module.clone(), config.clone(), device.clone(), cache)
                .unwrap()
        };
        // Eight shared tokens: two full 4-token blocks.
        let prompts = ["1 2 3 4 5 6 7 8 9", "1 2 3 4 5 6 7 8 11 12"];

        let cached = model(MistralKvCacheConfig::new(64, 4).with_prefix_cache());
        let shared = cached.cache_pool.clone();
        let with_cache = run_sequentially(cached, &prompts);
        let plain = run_sequentially(model(MistralKvCacheConfig::new(64, 4)), &prompts);

        assert_eq!(with_cache, plain);
        let hits = shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .status()
            .prefix_hit_tokens;
        assert_eq!(hits, 8, "the second prompt reused the first's two blocks");
    }

    #[test]
    fn preempted_mistral_requests_resume_with_identical_greedy_output() {
        let config = tiny_config(None);
        let device: burn::tensor::Device = Default::default();
        let module = MistralForCausalLM::init(&config, &device);
        let model = |capacity_tokens| {
            BurnMistralModel::with_kv_cache(
                module.clone(),
                config.clone(),
                device.clone(),
                MistralKvCacheConfig::new(capacity_tokens, 4),
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
    fn chunk_reservation_failure_does_not_advance_the_session_cursor() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::new(2, 1));
        let mut session = model
            .create_session(GenerationConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();

        assert_eq!(
            session.prefill_chunk(&[1], false).unwrap(),
            PrefillOutcome::Pending
        );
        let before = session.metrics();

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
    }

    #[test]
    fn a_prefill_chunk_spanning_several_blocks_matches_whole_prompt_prefill() {
        let config = tiny_config(None);
        let model = make_model(&config, MistralKvCacheConfig::new(16, 2));
        let prompt = [1_u32, 5, 9, 13, 7, 11];

        let mut whole = model
            .create_session(GenerationConfig {
                max_new_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let expected = whole.prefill(&prompt).unwrap();

        let mut chunked = model
            .create_session(GenerationConfig {
                max_new_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: Some(0),
            })
            .unwrap();
        let actual = match chunked.prefill_chunk(&prompt, true).unwrap() {
            PrefillOutcome::Token(token) => token,
            PrefillOutcome::Pending => panic!("final chunk must emit a token"),
        };
        assert_eq!(actual, expected);
    }
}
