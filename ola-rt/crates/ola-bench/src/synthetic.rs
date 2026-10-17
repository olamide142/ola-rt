//! Deterministic stand-in model for scheduling simulations.
//!
//! Scheduling experiments measure *when* work runs, not what a network
//! computes, so they use a model whose outputs depend only on the request.
//! The first prompt token encodes an optional end-of-sequence step: `0` means
//! no early EOS, and `k` ends the request on its `k`-th generated token.

use std::sync::{Arc, Mutex, PoisonError};

use ola_core::GenerationConfig;
use ola_model::{
    BatchCapabilities, CachedPrefix, CausalLanguageModel, CausalModel, ExecutionStage,
    GeneratedToken, GenerationSession, KvCacheProfile, KvPoolStatus, ModelExecutionError,
    PrefillOutcome, SessionMetrics, Tokenizer,
};
use ola_runtime::{KvBlockError, KvBlockManager, KvSequenceId};

/// Physical KV-pool geometry published by the synthetic model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvGeometry {
    /// Bytes one cached token occupies.
    pub bytes_per_token: u64,
    /// Tokens per physical page.
    pub tokens_per_block: usize,
    /// Total pool capacity in bytes.
    pub capacity_bytes: u64,
}

impl KvGeometry {
    /// Returns the pool capacity in tokens.
    pub const fn capacity_tokens(self) -> usize {
        // Capacity and token size are small configuration values; the
        // quotient always fits in usize on supported targets.
        (self.capacity_bytes / self.bytes_per_token) as usize
    }
}

impl Default for KvGeometry {
    /// 16-token pages of 64-byte tokens in a 64 KiB pool (1024 tokens).
    fn default() -> Self {
        Self {
            bytes_per_token: 64,
            tokens_per_block: 16,
            capacity_bytes: 64 * 1024,
        }
    }
}

/// Builds the prompt that asks the synthetic model to stop after
/// `eos_after` generated tokens, or never early when `None`. `eos_after`
/// must stay below 1000, the floor of output tokens.
pub fn synthetic_prompt(prompt_tokens: usize, eos_after: Option<usize>) -> Vec<u32> {
    let marker = eos_after.map_or(0, |step| u32::try_from(step).unwrap_or(u32::MAX));
    let mut prompt = vec![1; prompt_tokens.max(1)];
    prompt[0] = marker;
    prompt
}

/// Builds a prompt of `prompt_tokens` whose first `shared_tokens` match
/// every other prompt built this way and whose remainder is derived from
/// `seed`. Prompt tokens stay below the output floor; the first token is the
/// no-early-stop marker.
pub fn synthetic_shared_prompt(prompt_tokens: usize, shared_tokens: usize, seed: u64) -> Vec<u32> {
    let mut prompt = synthetic_prompt(prompt_tokens, None);
    let mut state = seed;
    for token in prompt.iter_mut().skip(shared_tokens.max(1)) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 2..=999: distinct from the shared filler and below output tokens.
        *token = 2 + u32::try_from((state >> 33) % 998).unwrap_or(0);
    }
    prompt
}

/// Every output token is at least this; prompt tokens (and end-of-sequence
/// markers) stay below it, which lets a resumed request be recognized.
const SYNTHETIC_OUTPUT_FLOOR: u32 = 1_000;

/// Returns the token the synthetic model emits at `step` (1-based) for a
/// prompt of `prompt_tokens` tokens.
pub fn synthetic_token(prompt_tokens: usize, step: usize) -> u32 {
    u32::try_from(prompt_tokens.saturating_mul(1_000).saturating_add(step)).unwrap_or(u32::MAX)
}

/// Returns a deterministic executable with every batch hook and a KV profile.
///
/// Sessions allocate their cache from one shared block pool of the given
/// geometry, as a real model does: a prefill appends its prompt, each decode
/// appends the token fed back in, and growth the pool cannot satisfy fails
/// with a KV-capacity error. The pool reports its occupancy to the planner.
///
/// # Panics
///
/// Panics if `kv` has a zero field, which is a configuration error.
pub fn synthetic_artifact(kv: KvGeometry) -> CausalLanguageModel {
    build_artifact(kv, false)
}

/// Like [`synthetic_artifact`], with the pool's prefix cache enabled: full
/// blocks outlive their sessions and later sessions with the same prompt
/// prefix reuse them.
///
/// # Panics
///
/// Panics if `kv` has a zero field, which is a configuration error.
pub fn synthetic_artifact_with_prefix_cache(kv: KvGeometry) -> CausalLanguageModel {
    build_artifact(kv, true)
}

fn build_artifact(kv: KvGeometry, prefix_cache: bool) -> CausalLanguageModel {
    let profile = KvCacheProfile::new(kv.bytes_per_token, kv.tokens_per_block, kv.capacity_bytes)
        .expect("KV geometry fields must be positive");
    let mut pool = KvBlockManager::new(kv.capacity_bytes, kv.tokens_per_block, kv.bytes_per_token)
        .expect("KV geometry must hold at least one block");
    if prefix_cache {
        pool = pool.with_prefix_cache();
    }
    CausalLanguageModel::new(
        SyntheticTokenizer,
        SyntheticModel {
            bytes_per_token: kv.bytes_per_token,
            pool: Arc::new(Mutex::new(pool)),
        },
    )
    .with_kv_cache_profile(profile)
}

fn lock(pool: &Mutex<KvBlockManager>) -> std::sync::MutexGuard<'_, KvBlockManager> {
    pool.lock().unwrap_or_else(PoisonError::into_inner)
}

fn pool_error(stage: ExecutionStage, error: KvBlockError) -> ModelExecutionError {
    let message = format!("KV cache: {error}");
    match error {
        KvBlockError::InsufficientCapacity { .. }
        | KvBlockError::BatchInsufficientCapacity { .. } => {
            ModelExecutionError::kv_capacity_exhausted(stage, message)
        }
        _ => ModelExecutionError::new(stage, message),
    }
}

struct SyntheticTokenizer;

impl Tokenizer for SyntheticTokenizer {
    fn encode(
        &self,
        _text: &str,
        _add_special_tokens: bool,
    ) -> Result<Vec<u32>, ModelExecutionError> {
        Err(ModelExecutionError::new(
            ExecutionStage::Tokenize,
            "the synthetic model accepts token input only",
        ))
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

struct SyntheticModel {
    bytes_per_token: u64,
    pool: Arc<Mutex<KvBlockManager>>,
}

impl CausalModel for SyntheticModel {
    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    fn kv_pool_status(&self) -> Option<KvPoolStatus> {
        let pool = lock(&self.pool);
        let metrics = pool.metrics();
        Some(KvPoolStatus {
            total_blocks: metrics.total_blocks,
            free_blocks: metrics.free_blocks,
            allocated_blocks: metrics.allocated_blocks,
            cached_blocks: metrics.cached_blocks,
            tokens_per_block: pool.tokens_per_block(),
            live_sequences: metrics.live_sequences,
            evictions: metrics.evictions,
            prefix_hit_tokens: metrics.prefix_hit_tokens,
        })
    }

    fn cached_prefix(&self, tokens: &[u32]) -> CachedPrefix {
        let found = lock(&self.pool).lookup_prefix(tokens);
        CachedPrefix {
            tokens: found.tokens,
            evictable_blocks: found.evictable_blocks,
        }
    }

    fn batch_capabilities(&self) -> BatchCapabilities {
        BatchCapabilities::ALL
    }

    fn create_session(
        &self,
        _generation: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
        let sequence = lock(&self.pool)
            .allocate()
            .map_err(|error| pool_error(ExecutionStage::SessionCreation, error))?;
        Ok(Box::new(SyntheticSession {
            pool: Arc::clone(&self.pool),
            sequence,
            bytes_per_token: self.bytes_per_token,
            prompt: Vec::new(),
            resumed: 0,
            emitted: 0,
            ready: false,
        }))
    }

    fn prefill_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_tokens: &[&[u32]],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        Some(
            sessions
                .iter_mut()
                .zip(prompt_tokens)
                .map(|(session, prompt)| session.prefill(prompt))
                .collect(),
        )
    }

    fn prefill_chunk_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_chunks: &[&[u32]],
        is_final: &[bool],
    ) -> Option<Vec<Result<PrefillOutcome, ModelExecutionError>>> {
        Some(
            sessions
                .iter_mut()
                .zip(prompt_chunks)
                .zip(is_final)
                .map(|((session, chunk), is_final)| session.prefill_chunk(chunk, *is_final))
                .collect(),
        )
    }

    fn decode_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        input_tokens: &[u32],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        Some(
            sessions
                .iter_mut()
                .zip(input_tokens)
                .map(|(session, input)| session.decode(*input))
                .collect(),
        )
    }
}

struct SyntheticSession {
    pool: Arc<Mutex<KvBlockManager>>,
    sequence: KvSequenceId,
    bytes_per_token: u64,
    prompt: Vec<u32>,
    /// Output tokens already in `prompt` because a preempted request
    /// resumed by prefilling its prompt and earlier output together.
    resumed: usize,
    /// Tokens this session has emitted.
    emitted: usize,
    ready: bool,
}

impl SyntheticSession {
    /// Emits the next output token. Tokens depend on the original prompt
    /// length and the output position, so a resumed request continues
    /// exactly where it stopped, as a real model's would.
    fn emit(&mut self) -> GeneratedToken {
        self.emitted += 1;
        let generated = self.resumed + self.emitted;
        let eos_after = self.prompt.first().copied().unwrap_or(0);
        GeneratedToken {
            id: synthetic_token(self.prompt.len() - self.resumed, generated),
            is_end_of_sequence: eos_after != 0
                && usize::try_from(eos_after).is_ok_and(|step| step == generated),
        }
    }
}

impl GenerationSession for SyntheticSession {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<GeneratedToken, ModelExecutionError> {
        match self.prefill_chunk(prompt_tokens, true)? {
            PrefillOutcome::Token(token) => Ok(token),
            PrefillOutcome::Pending => unreachable!("a final chunk always selects a token"),
        }
    }

    fn prefill_chunk(
        &mut self,
        prompt_tokens: &[u32],
        is_final: bool,
    ) -> Result<PrefillOutcome, ModelExecutionError> {
        if self.ready {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prefill already completed",
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prompt chunk cannot be empty",
            ));
        }
        {
            let mut pool = lock(&self.pool);
            pool.append(self.sequence, prompt_tokens.len())
                .map_err(|error| pool_error(ExecutionStage::Prefill, error))?;
            pool.record_tokens(self.sequence, prompt_tokens)
                .map_err(|error| pool_error(ExecutionStage::Prefill, error))?;
        }
        self.prompt.extend_from_slice(prompt_tokens);
        if !is_final {
            return Ok(PrefillOutcome::Pending);
        }
        self.ready = true;
        // Output tokens are at least 1000 and prompt tokens are small, so a
        // prompt ending in output tokens is a resumed request.
        self.resumed = self
            .prompt
            .iter()
            .rev()
            .take_while(|token| **token >= SYNTHETIC_OUTPUT_FLOOR)
            .count();
        Ok(PrefillOutcome::Token(self.emit()))
    }

    fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
        if !self.ready {
            return Err(ModelExecutionError::new(
                ExecutionStage::Decode,
                "decode requires prefill",
            ));
        }
        {
            let mut pool = lock(&self.pool);
            pool.append(self.sequence, 1)
                .map_err(|error| pool_error(ExecutionStage::Decode, error))?;
            pool.record_tokens(self.sequence, &[input_token])
                .map_err(|error| pool_error(ExecutionStage::Decode, error))?;
        }
        Ok(self.emit())
    }

    fn metrics(&self) -> SessionMetrics {
        // The cache holds the prefilled sequence plus every token this
        // session emitted except the latest, which has not yet been fed
        // back as a decode input.
        let cached = self
            .prompt
            .len()
            .saturating_add(self.emitted.saturating_sub(1));
        SessionMetrics {
            prompt_tokens: self.prompt.len() - self.resumed,
            generated_tokens: self.resumed + self.emitted,
            kv_cache_bytes: (cached as u64).saturating_mul(self.bytes_per_token),
        }
    }

    fn attach_prefix(&mut self, tokens: &[u32]) -> Result<usize, ModelExecutionError> {
        self.attach(tokens)
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl SyntheticSession {
    fn attach(&mut self, tokens: &[u32]) -> Result<usize, ModelExecutionError> {
        if !self.prompt.is_empty() || self.ready {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "a cached prefix can only start a fresh session",
            ));
        }
        let attached = lock(&self.pool)
            .attach_prefix(self.sequence, tokens)
            .map_err(|error| pool_error(ExecutionStage::Prefill, error))?;
        self.prompt.extend_from_slice(&tokens[..attached]);
        Ok(attached)
    }
}

impl Drop for SyntheticSession {
    fn drop(&mut self) {
        let _ = lock(&self.pool).free(self.sequence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_sessions_emit_deterministic_tokens_and_honour_eos() {
        let model = synthetic_artifact(KvGeometry::default());
        let mut session = model.create_session(GenerationConfig::default()).unwrap();

        let first = session.prefill(&synthetic_prompt(3, Some(2))).unwrap();
        let second = session.decode(first.id).unwrap();

        assert_eq!(first.id, synthetic_token(3, 1));
        assert!(!first.is_end_of_sequence);
        assert_eq!(second.id, synthetic_token(3, 2));
        assert!(second.is_end_of_sequence);
        assert_eq!(session.metrics().kv_cache_bytes, 4 * 64);
    }

    #[test]
    fn sessions_share_one_block_pool_and_fail_typed_when_it_is_full() {
        // Four 4-token blocks.
        let model = synthetic_artifact(KvGeometry {
            bytes_per_token: 8,
            tokens_per_block: 4,
            capacity_bytes: 4 * 4 * 8,
        });
        let mut first = model.create_session(GenerationConfig::default()).unwrap();
        let mut second = model.create_session(GenerationConfig::default()).unwrap();

        let token = first.prefill(&synthetic_prompt(9, None)).unwrap();
        let status = model.kv_pool_status().unwrap();
        assert_eq!(
            (
                status.allocated_blocks,
                status.free_blocks,
                status.live_sequences
            ),
            (3, 1, 2)
        );
        let error = second.prefill(&synthetic_prompt(5, None)).unwrap_err();
        assert!(error.is_kv_capacity_exhausted(), "{error}");

        // Three decodes fill the third block; the fourth needs a new one.
        let mut input = token.id;
        for _ in 0..3 {
            input = first.decode(input).unwrap().id;
        }
        assert!(first.decode(input).is_ok());
        assert_eq!(model.kv_pool_status().unwrap().free_blocks, 0);

        drop(first);
        drop(second);
        let status = model.kv_pool_status().unwrap();
        assert_eq!((status.free_blocks, status.live_sequences), (4, 0));
    }

    /// Records the pool status each plan saw.
    struct Recording {
        inner: ola_runtime::ComposedPolicy,
        seen: Arc<Mutex<Vec<Option<KvPoolStatus>>>>,
    }

    impl ola_runtime::ExecutionPolicy for Recording {
        fn descriptor(&self) -> ola_runtime::PolicyDescriptor {
            self.inner.descriptor()
        }

        fn plan(
            &self,
            snapshot: &ola_runtime::PlanningSnapshot,
        ) -> Result<ola_runtime::PolicyDecision, ola_runtime::PolicyError> {
            self.seen
                .lock()
                .unwrap()
                .push(snapshot.model(&"synthetic".into()).kv_pool);
            self.inner.plan(snapshot)
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            self.inner.enforces_model_kv_capacity()
        }
    }

    #[test]
    fn running_out_of_kv_blocks_is_a_typed_divergence_the_planner_can_see() {
        use ola_core::{ModelDescriptor, RequestSpec, Timestamp};
        use ola_runtime::{Executor, ExecutorEvent, PlanDivergence, Runtime};

        let seen = Arc::new(Mutex::new(Vec::new()));
        // Unconstrained admission: nothing stops both prompts starting.
        let policy = Recording {
            inner: ola_runtime::ComposedPolicy::default(),
            seen: Arc::clone(&seen),
        };
        let mut executor = Executor::with_max_active_requests(Runtime::default(), 2)
            .unwrap()
            .with_policy(policy);
        executor
            .register_model(
                ModelDescriptor::language("synthetic", "synthetic"),
                synthetic_artifact(KvGeometry {
                    bytes_per_token: 8,
                    tokens_per_block: 4,
                    capacity_bytes: 4 * 4 * 8,
                }),
            )
            .unwrap();
        let request = || {
            let mut spec = RequestSpec::text("synthetic", "");
            spec.inputs = vec![ola_core::Input::Tokens(synthetic_prompt(9, None))];
            spec.generation.max_new_tokens = 2;
            spec
        };
        let first = executor.submit(request(), Timestamp::ZERO).unwrap();
        let second = executor.submit(request(), Timestamp::ZERO).unwrap();

        let iteration = executor.step_iteration(Timestamp::from_micros(1)).unwrap();

        assert!(matches!(
            iteration.events.as_slice(),
            [
                ExecutorEvent::Token { request_id, .. },
                ExecutorEvent::Failed { request_id: failed, .. },
            ] if *request_id == first && *failed == second
        ));
        assert!(
            iteration
                .divergences
                .contains(&PlanDivergence::KvCapacityExhausted { request_id: second })
        );
        executor.step_iteration(Timestamp::from_micros(2)).unwrap();
        let seen = seen.lock().unwrap();
        // The first plan saw an empty pool; the second saw the first
        // request's three blocks.
        assert_eq!(seen[0].map(|status| status.free_blocks), Some(4));
        assert_eq!(seen[1].map(|status| status.allocated_blocks), Some(3));
    }

    #[test]
    fn chunked_prefill_matches_whole_prompt_prefill() {
        let model = synthetic_artifact(KvGeometry::default());
        let prompt = synthetic_prompt(5, None);
        let mut whole = model.create_session(GenerationConfig::default()).unwrap();
        let mut chunked = model.create_session(GenerationConfig::default()).unwrap();

        let expected = whole.prefill(&prompt).unwrap();
        assert_eq!(
            chunked.prefill_chunk(&prompt[..2], false).unwrap(),
            PrefillOutcome::Pending
        );
        let actual = chunked.prefill_chunk(&prompt[2..], true).unwrap();

        assert_eq!(actual, PrefillOutcome::Token(expected));
        assert_eq!(KvGeometry::default().capacity_tokens(), 1_024);
    }
}
