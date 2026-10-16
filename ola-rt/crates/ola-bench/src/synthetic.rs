//! Deterministic stand-in model for scheduling simulations.
//!
//! Scheduling experiments measure *when* work runs, not what a network
//! computes, so they use a model whose outputs depend only on the request.
//! The first prompt token encodes an optional end-of-sequence step: `0` means
//! no early EOS, and `k` ends the request on its `k`-th generated token.

use ola_core::GenerationConfig;
use ola_model::{
    BatchCapabilities, CausalLanguageModel, CausalModel, ExecutionStage, GeneratedToken,
    GenerationSession, KvCacheProfile, ModelExecutionError, PrefillOutcome, SessionMetrics,
    Tokenizer,
};

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
/// `eos_after` generated tokens, or never early when `None`.
pub fn synthetic_prompt(prompt_tokens: usize, eos_after: Option<usize>) -> Vec<u32> {
    let marker = eos_after.map_or(0, |step| u32::try_from(step).unwrap_or(u32::MAX));
    let mut prompt = vec![1; prompt_tokens.max(1)];
    prompt[0] = marker;
    prompt
}

/// Returns the token the synthetic model emits at `step` (1-based) for a
/// prompt of `prompt_tokens` tokens.
pub fn synthetic_token(prompt_tokens: usize, step: usize) -> u32 {
    u32::try_from(prompt_tokens.saturating_mul(1_000).saturating_add(step)).unwrap_or(u32::MAX)
}

/// Returns a deterministic executable with every batch hook and a KV profile.
///
/// # Panics
///
/// Panics if `kv` has a zero field, which is a configuration error.
pub fn synthetic_artifact(kv: KvGeometry) -> CausalLanguageModel {
    let profile = KvCacheProfile::new(kv.bytes_per_token, kv.tokens_per_block, kv.capacity_bytes)
        .expect("KV geometry fields must be positive");
    CausalLanguageModel::new(
        SyntheticTokenizer,
        SyntheticModel {
            bytes_per_token: kv.bytes_per_token,
        },
    )
    .with_kv_cache_profile(profile)
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
}

impl CausalModel for SyntheticModel {
    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    fn batch_capabilities(&self) -> BatchCapabilities {
        BatchCapabilities::ALL
    }

    fn create_session(
        &self,
        _generation: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
        Ok(Box::new(SyntheticSession {
            bytes_per_token: self.bytes_per_token,
            prompt: Vec::new(),
            generated: 0,
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
    bytes_per_token: u64,
    prompt: Vec<u32>,
    generated: usize,
    ready: bool,
}

impl SyntheticSession {
    fn emit(&mut self) -> GeneratedToken {
        self.generated += 1;
        let eos_after = self.prompt.first().copied().unwrap_or(0);
        GeneratedToken {
            id: synthetic_token(self.prompt.len(), self.generated),
            is_end_of_sequence: eos_after != 0
                && usize::try_from(eos_after).is_ok_and(|step| step == self.generated),
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
        self.prompt.extend_from_slice(prompt_tokens);
        if !is_final {
            return Ok(PrefillOutcome::Pending);
        }
        self.ready = true;
        Ok(PrefillOutcome::Token(self.emit()))
    }

    fn decode(&mut self, _input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
        if !self.ready {
            return Err(ModelExecutionError::new(
                ExecutionStage::Decode,
                "decode requires prefill",
            ));
        }
        Ok(self.emit())
    }

    fn metrics(&self) -> SessionMetrics {
        // The cache holds the prompt plus every generated token except the
        // latest, which has not yet been fed back as a decode input.
        let cached = self
            .prompt
            .len()
            .saturating_add(self.generated.saturating_sub(1));
        SessionMetrics {
            prompt_tokens: self.prompt.len(),
            generated_tokens: self.generated,
            kv_cache_bytes: (cached as u64).saturating_mul(self.bytes_per_token),
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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
