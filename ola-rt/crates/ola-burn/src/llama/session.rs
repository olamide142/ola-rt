use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::tensor::{Int, Tensor, TensorData};
use ola_core::GenerationConfig;
use ola_kernel::{PagedAttentionKernel, PortablePagedAttention};
use ola_model::{
    BatchCapabilities, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
    ModelExecutionContext, ModelExecutionError, PrefillOutcome, SessionMetrics, SessionState,
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

#[derive(serde::Deserialize)]
struct LlamaSessionState {
    tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    rng_state: u64,
    prompt_tokens: usize,
    cursor: usize,
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

    fn new_session(
        &self,
        generation: GenerationConfig,
    ) -> Result<LlamaSession, ModelExecutionError> {
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
        let cache = LlamaKvCache::allocate(self.cache_pool.clone())
            .map_err(|error| cache_error(ExecutionStage::SessionCreation, error))?;
        Ok(LlamaSession {
            module: self.module.clone(),
            config: self.config.clone(),
            device: self.device.clone(),
            generation: generation.clone(),
            tokens: Vec::new(),
            generated_tokens: Vec::new(),
            cache: Some(cache),
            rng_state: generation.seed.unwrap_or_else(entropy_seed),
            last_generated: None,
            metrics: SessionMetrics::default(),
            terminal: false,
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

impl CausalModel for BurnLlamaModel {
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
        Ok(Box::new(self.new_session(generation)?))
    }

    fn import_session(
        &self,
        generation: GenerationConfig,
        state: SessionState,
    ) -> Result<Option<Box<dyn GenerationSession>>, ModelExecutionError> {
        if state.format != "ola-burn/llama-recompute-v1" {
            return Ok(None);
        }
        let state: LlamaSessionState = serde_json::from_slice(&state.payload).map_err(|error| {
            ModelExecutionError::new(
                ExecutionStage::SessionCreation,
                format!("invalid Llama session state: {error}"),
            )
        })?;
        if state.tokens.is_empty()
            || state.generated_tokens.is_empty()
            || state.cursor != state.tokens.len()
            || state.prompt_tokens > state.tokens.len()
        {
            return Ok(None);
        }
        let mut session = self.new_session(generation)?;
        let _ = session.prefill(&state.tokens)?;
        session.tokens = state.tokens;
        session.generated_tokens = state.generated_tokens.clone();
        session.rng_state = state.rng_state;
        session.last_generated = state.generated_tokens.last().copied();
        session.metrics.prompt_tokens = state.prompt_tokens;
        session.metrics.generated_tokens = state.generated_tokens.len();
        session.terminal = session
            .last_generated
            .is_some_and(|token| session.config.eos_token_id.contains(&token))
            || session.metrics.generated_tokens >= session.generation.max_new_tokens;
        Ok(Some(Box::new(session)))
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
                        .expect("a live Llama session owns a cache")
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
        let kernel = paged_attention_kernel(context);
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
                        .expect("a live Llama session owns a cache")
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
        let kernel = paged_attention_kernel(context);
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
                        .expect("a live Llama session owns a cache")
                        .bytes();
                    Ok(token)
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
    generated_tokens: Vec<u32>,
    cache: Option<LlamaKvCache>,
    rng_state: u64,
    last_generated: Option<u32>,
    metrics: SessionMetrics,
    terminal: bool,
}

impl LlamaSession {
    /// Records tokens whose KV was just written, in the session and in the
    /// cache's prefix index.
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
        self.generated_tokens.push(id);
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

    fn export_state(&self) -> Result<Option<SessionState>, ModelExecutionError> {
        #[derive(serde::Serialize)]
        struct State<'a> {
            tokens: &'a [u32],
            generated_tokens: &'a [u32],
            rng_state: u64,
            prompt_tokens: usize,
            cursor: usize,
        }
        let payload = serde_json::to_vec(&State {
            tokens: &self.tokens,
            generated_tokens: &self.generated_tokens,
            rng_state: self.rng_state,
            prompt_tokens: self.metrics.prompt_tokens,
            cursor: self.tokens.len(),
        })
        .map_err(|error| {
            ModelExecutionError::new(
                ExecutionStage::SessionCreation,
                format!("could not serialize Llama session state: {error}"),
            )
        })?;
        Ok(Some(SessionState {
            format: "ola-burn/llama-recompute-v1".into(),
            cursor: self.tokens.len(),
            payload,
        }))
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use burn::tensor::Tensor;
    use ola_core::{ModelDescriptor, RequestSpec, Timestamp};
    use ola_kernel::{
        KernelCapability, KernelDType, KernelDescriptor, KernelOperation, KernelRegistry,
        KernelTarget, PagedAttentionInput, PagedAttentionProfile, PagedAttentionProvider,
    };
    use ola_model::{CausalLanguageModel, Tokenizer};
    use ola_runtime::{
        BatchExecutionMode, BatchPhase, ComposedPolicy, ExecutionBatch, Executor, Runtime,
    };

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

    struct CountingProvider {
        calls: Arc<AtomicUsize>,
    }

    struct CountingKernel {
        calls: Arc<AtomicUsize>,
    }

    const fn counting_descriptor() -> KernelDescriptor {
        KernelDescriptor::new(
            "test-counting-paged-attention",
            KernelOperation::PagedAttentionDecode,
            KernelTarget::Cpu,
        )
    }

    impl PagedAttentionProvider for CountingProvider {
        fn capability(&self) -> KernelCapability {
            KernelCapability::new(
                counting_descriptor(),
                [KernelTarget::Cpu],
                [KernelDType::F32],
            )
            .with_head_dimensions([4])
            .with_gqa_ratios([2])
        }

        fn priority(&self) -> u16 {
            10
        }

        fn create(&self) -> Box<dyn PagedAttentionKernel> {
            Box::new(CountingKernel {
                calls: Arc::clone(&self.calls),
            })
        }
    }

    impl PagedAttentionKernel for CountingKernel {
        fn descriptor(&self) -> KernelDescriptor {
            counting_descriptor()
        }

        fn execute(
            &self,
            input: PagedAttentionInput,
        ) -> Result<Tensor<4>, ola_kernel::KernelError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            PortablePagedAttention::new().execute(input)
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

    /// Runs `prompts` one after another on `model`, each to completion,
    /// with resumable prefill so cached prefixes can be reused.
    fn run_sequentially(model: BurnLlamaModel, prompts: &[&str]) -> Vec<Vec<u32>> {
        let mut executor = Executor::default().with_prefill_chunk_size(64).unwrap();
        executor
            .register_model(
                ModelDescriptor::language("tiny", "llama"),
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

    #[test]
    fn a_cached_llama_prefix_gives_the_same_greedy_output_as_recomputing_it() {
        let config = tiny_model_config();
        let device: burn::tensor::Device = Default::default();
        let module = LlamaForCausalLM::init(&config, &device);
        let model = |cache: LlamaKvCacheConfig| {
            BurnLlamaModel::with_kv_cache(module.clone(), config.clone(), device.clone(), cache)
                .unwrap()
        };
        // Eight shared tokens: two full 4-token blocks.
        let prompts = ["1 2 3 4 5 6 7 8 9", "1 2 3 4 5 6 7 8 11 12"];

        let cached = model(LlamaKvCacheConfig::new(64, 4).with_prefix_cache());
        let shared = cached.cache_pool.clone();
        let with_cache = run_sequentially(cached, &prompts);
        let plain = run_sequentially(model(LlamaKvCacheConfig::new(64, 4)), &prompts);

        assert_eq!(with_cache, plain);
        let hits = shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .status()
            .prefix_hit_tokens;
        assert_eq!(hits, 8, "the second prompt reused the first's two blocks");
    }

    #[test]
    fn a_prefill_chunk_spanning_several_blocks_matches_whole_prompt_prefill() {
        let config = tiny_model_config();
        let device: burn::tensor::Device = Default::default();
        let module = LlamaForCausalLM::init(&config, &device);
        // Four-token blocks: a 10-token chunk reserves three blocks at once.
        let model =
            BurnLlamaModel::with_kv_cache(module, config, device, LlamaKvCacheConfig::new(64, 4))
                .unwrap();
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

        let mut whole = model.create_session(greedy(1)).unwrap();
        let expected = whole.prefill(&prompt).unwrap();
        let mut chunked = model.create_session(greedy(1)).unwrap();
        assert_eq!(
            chunked.prefill_chunk(&prompt[..1], false).unwrap(),
            PrefillOutcome::Pending
        );
        let actual = chunked.prefill_chunk(&prompt[1..], true).unwrap();

        assert_eq!(actual, PrefillOutcome::Token(expected));
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
                kernel: None,
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
                kernel: decoded.execution_batches[0].kernel,
                started_at: Timestamp::from_micros(20),
                finished_at: Timestamp::from_micros(20),
            }]
        );
        assert_eq!(decoded.active_requests, 2);
        assert_eq!(decoded.execution_plan.estimates().input_tokens, 2);
    }

    #[test]
    fn executor_routes_the_planned_provider_into_tensor_decode() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut kernels = KernelRegistry::new();
        kernels
            .register(CountingProvider {
                calls: Arc::clone(&calls),
            })
            .unwrap();
        let profile =
            PagedAttentionProfile::new(KernelTarget::Cpu, KernelDType::F32, 2, 1, 4, 16).unwrap();
        let artifact = CausalLanguageModel::new(NumericTokenizer, tiny_model())
            .with_paged_attention_profile(profile);
        let policy = ComposedPolicy::default().with_kernel_registry(kernels);
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(policy)
            .with_clock(Arc::new(ola_core::MonotonicClock::start()));
        executor
            .register_model(ModelDescriptor::language("tiny", "llama"), artifact)
            .unwrap();
        for prompt in ["1 4", "1 5 7"] {
            let mut spec = RequestSpec::text("tiny", prompt);
            spec.generation = greedy(2);
            executor.submit(spec, Timestamp::ZERO).unwrap();
        }

        let prefill = executor
            .step_iteration(Timestamp::from_micros(1_000_000))
            .unwrap();
        assert!(prefill.execution_plan.batches[0].kernel.is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let decode = executor
            .step_iteration(Timestamp::from_micros(2_000_000))
            .unwrap();
        let planned = decode.execution_plan.batches[0]
            .kernel
            .as_ref()
            .expect("decode carries the selected provider");
        assert_eq!(planned.selection.descriptor(), counting_descriptor());
        assert_eq!(decode.execution_batches[0].mode, BatchExecutionMode::Tensor);
        let executed = decode.execution_batches[0]
            .kernel
            .expect("execution records the selected provider");
        assert_eq!(executed.descriptor, counting_descriptor());
        // The provider sees one batched operation even when its portable
        // fallback executes the two request-local inputs independently.
        assert_eq!(executed.calls, 1);
        assert!(executed.host_duration_nanos.is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn seeded_sampling_is_accepted_and_reproducible() {
        let model = tiny_model();
        let generation = GenerationConfig {
            max_new_tokens: 2,
            temperature: 0.8,
            top_p: 0.9,
            seed: Some(17),
        };
        let mut first = model.create_session(generation.clone()).unwrap();
        let mut second = model.create_session(generation).unwrap();

        let first_token = first.prefill(&[1, 4, 5]).unwrap();
        let second_token = second.prefill(&[1, 4, 5]).unwrap();
        assert_eq!(first_token.id, second_token.id);
        assert_eq!(
            first.decode(first_token.id).unwrap().id,
            second.decode(second_token.id).unwrap().id
        );
    }

    #[test]
    fn exports_versioned_recompute_safe_session_state() {
        let model = tiny_model();
        let mut session = model.create_session(greedy(3)).unwrap();
        let first = session.prefill(&[1, 4, 5]).unwrap();
        let _ = session.decode(first.id).unwrap();
        let state = session.export_state().unwrap().unwrap();
        assert_eq!(state.format, "ola-burn/llama-recompute-v1");
        assert_eq!(state.cursor, 4);
        assert!(!state.payload.is_empty());
    }

    #[test]
    fn imports_recompute_safe_session_state() {
        let model = tiny_model();
        let mut original = model.create_session(greedy(4)).unwrap();
        let first = original.prefill(&[1, 4, 5]).unwrap();
        let second = original.decode(first.id).unwrap();
        let state = original.export_state().unwrap().unwrap();

        let mut resumed = model
            .import_session(greedy(4), state)
            .unwrap()
            .expect("matching Llama state format is importable");
        assert_eq!(
            original.decode(second.id).unwrap().id,
            resumed.decode(second.id).unwrap().id
        );
    }

    #[test]
    fn sampler_respects_nucleus_cutoff_and_seed() {
        let logits = [4.0, 3.0, -20.0];
        let mut first_state = 23;
        let mut second_state = 23;
        let first = sample_token(&logits, 1.0, 0.5, &mut first_state).unwrap();
        let second = sample_token(&logits, 1.0, 0.5, &mut second_state).unwrap();

        assert_eq!(first, second);
        assert_eq!(first, 0);
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
