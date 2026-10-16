use std::{any::Any, error::Error, fmt, sync::Arc};

use ola_core::GenerationConfig;

/// Stage at which model execution failed.
///
/// Stable stages make failures aggregatable in telemetry without parsing error
/// messages emitted by different architecture implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionStage {
    /// Converting application text into model token IDs.
    Tokenize,
    /// Converting generated token IDs back into text.
    Detokenize,
    /// Allocating request-local execution state.
    SessionCreation,
    /// Processing all prompt tokens and producing the first generated token.
    Prefill,
    /// Processing one generated token and selecting the next token.
    Decode,
}

/// Backend- and architecture-neutral execution failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelExecutionError {
    stage: ExecutionStage,
    message: String,
}

impl ModelExecutionError {
    /// Creates an error with a stable stage and implementation-specific detail.
    pub fn new(stage: ExecutionStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
        }
    }

    /// Returns the failed execution stage.
    pub const fn stage(&self) -> ExecutionStage {
        self.stage
    }

    /// Returns the implementation-specific diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ModelExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} failed: {}", self.stage, self.message)
    }
}

impl Error for ModelExecutionError {}

/// Text/token conversion associated with one loaded model artifact.
///
/// Tokenizer implementations must be safe to share across concurrent requests.
/// Architecture factories are responsible for ensuring that tokenizer files
/// and model weights belong to the same artifact revision.
pub trait Tokenizer: Send + Sync {
    /// Converts input text into model token IDs.
    ///
    /// When `add_special_tokens` is true, the implementation applies the
    /// artifact's configured BOS/EOS or other required wrapper tokens.
    fn encode(&self, text: &str, add_special_tokens: bool)
    -> Result<Vec<u32>, ModelExecutionError>;

    /// Converts token IDs into text.
    ///
    /// `skip_special_tokens` should normally be true for user-visible output
    /// and false for tokenizer conformance tests.
    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, ModelExecutionError>;
}

/// Shared weights and immutable configuration for a causal language model.
///
/// Implementations may internally share device buffers. Mutable request state
/// belongs exclusively to the [`GenerationSession`] returned by
/// [`create_session`](Self::create_session).
pub trait CausalModel: Send + Sync {
    /// Creates isolated state for one generation request.
    ///
    /// The session owns this generation configuration for its full lifetime,
    /// including any seeded random-number generator used during sampling.
    fn create_session(
        &self,
        generation: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError>;

    /// Returns whether sessions created by this model can resume prefill.
    ///
    /// The runtime only divides prompts when this returns true. The default
    /// keeps existing architecture implementations on the complete-prompt
    /// compatibility path.
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    /// Declares which tensor-batched hooks below this model implements.
    ///
    /// Planners read this before execution to record whether work is
    /// expected to share a tensor invocation. A declared hook may still return
    /// `None`; the runtime then falls back to individual execution and
    /// records the divergence. The default declares no batched paths.
    fn batch_capabilities(&self) -> BatchCapabilities {
        BatchCapabilities::NONE
    }

    /// Attempts one tensor-batched prefill across compatible sessions.
    ///
    /// `prompt_tokens` contains one complete, non-empty prompt per session.
    /// Prompts may have different lengths. Returning `None` declares that this
    /// implementation has no batched path; the runtime then calls
    /// [`GenerationSession::prefill`] independently. Returning `Some` must
    /// produce exactly one result per input session in the same order.
    /// Implementations must validate the complete batch before mutating any
    /// request state whenever failure can be detected in advance.
    fn prefill_batch(
        &self,
        _sessions: &mut [&mut dyn GenerationSession],
        _prompt_tokens: &[&[u32]],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        None
    }

    /// Attempts one tensor-batched resumable prefill across compatible sessions.
    ///
    /// Each non-empty chunk continues its corresponding session at that
    /// session's current absolute position. `is_final` identifies chunks that
    /// end their complete prompts. Returning `None` requests the individual
    /// [`GenerationSession::prefill_chunk`] fallback. Implementations must
    /// validate and reserve the complete batch before mutating any session.
    fn prefill_chunk_batch(
        &self,
        _sessions: &mut [&mut dyn GenerationSession],
        _prompt_chunks: &[&[u32]],
        _is_final: &[bool],
    ) -> Option<Vec<Result<PrefillOutcome, ModelExecutionError>>> {
        None
    }

    /// Attempts one tensor-batched decode across compatible sessions.
    ///
    /// Returning `None` declares that this implementation has no batched path;
    /// the runtime then calls [`GenerationSession::decode`] independently.
    /// Returning `Some` must produce exactly one result per input session in
    /// the same order. Implementations must validate the complete batch before
    /// mutating session state whenever failure can be detected in advance.
    fn decode_batch(
        &self,
        _sessions: &mut [&mut dyn GenerationSession],
        _input_tokens: &[u32],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        None
    }
}

/// Tensor-batched execution paths implemented by a [`CausalModel`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchCapabilities {
    /// [`CausalModel::prefill_batch`] packs complete prompts.
    pub prefill: bool,
    /// [`CausalModel::prefill_chunk_batch`] packs resumable prompt chunks.
    pub chunked_prefill: bool,
    /// [`CausalModel::decode_batch`] packs single-token decodes.
    pub decode: bool,
}

impl BatchCapabilities {
    /// No batched paths; every operation uses its session individually.
    pub const NONE: Self = Self {
        prefill: false,
        chunked_prefill: false,
        decode: false,
    };

    /// Every batched hook is implemented.
    pub const ALL: Self = Self {
        prefill: true,
        chunked_prefill: true,
        decode: true,
    };
}

/// One generated token and its terminal status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratedToken {
    /// Token selected by the architecture implementation's sampler.
    pub id: u32,
    /// Whether this token is an end-of-sequence token for the loaded artifact.
    pub is_end_of_sequence: bool,
}

/// Result of processing one resumable prompt segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillOutcome {
    /// More prompt tokens remain, so no output token is available yet.
    Pending,
    /// The final prompt segment selected the request's first output token.
    Token(GeneratedToken),
}

/// Observable snapshot of request-local model state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SessionMetrics {
    /// Number of prompt tokens accepted during prefill.
    pub prompt_tokens: usize,
    /// Number of tokens generated by prefill and decode calls.
    pub generated_tokens: usize,
    /// Physical bytes currently attributed to this request's KV cache.
    ///
    /// Paged implementations include reserved but unoccupied token slots.
    pub kv_cache_bytes: u64,
}

/// Physical layout and capacity of one model's shared paged KV pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheProfile {
    bytes_per_token: u64,
    tokens_per_block: usize,
    capacity_bytes: u64,
}

impl KvCacheProfile {
    /// Creates a valid non-empty paged-cache profile.
    pub const fn new(
        bytes_per_token: u64,
        tokens_per_block: usize,
        capacity_bytes: u64,
    ) -> Option<Self> {
        if bytes_per_token == 0 || tokens_per_block == 0 || capacity_bytes == 0 {
            return None;
        }
        Some(Self {
            bytes_per_token,
            tokens_per_block,
            capacity_bytes,
        })
    }

    /// Returns bytes occupied by one token position before page rounding.
    pub const fn bytes_per_token(self) -> u64 {
        self.bytes_per_token
    }

    /// Returns token positions reserved together in one physical page.
    pub const fn tokens_per_block(self) -> usize {
        self.tokens_per_block
    }

    /// Returns total physical bytes shared by this model's live sessions.
    pub const fn capacity_bytes(self) -> u64 {
        self.capacity_bytes
    }

    /// Conservatively estimates page-rounded bytes for a token count.
    pub fn estimate_bytes(self, tokens: usize) -> Option<u64> {
        let blocks = tokens.div_ceil(self.tokens_per_block);
        (blocks as u64)
            .checked_mul(self.tokens_per_block as u64)?
            .checked_mul(self.bytes_per_token)
    }
}

/// Mutable, request-local causal-generation state.
///
/// The required call order is exactly one `prefill`, followed by zero or more
/// `decode` calls until EOS or the caller's token limit. Implementations must
/// reject decode-before-prefill and repeated-prefill calls with a
/// [`ModelExecutionError`] rather than silently resetting state.
pub trait GenerationSession: Send {
    /// Processes the complete prompt and returns the first generated token.
    ///
    /// Implementations should reject an empty token slice. Tokenization and
    /// chat-template application happen before this boundary.
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<GeneratedToken, ModelExecutionError>;

    /// Processes one non-empty prompt segment and optionally finishes prefill.
    ///
    /// `is_final` identifies the segment containing the end of the prompt.
    /// Implementations supporting resumable prefill must retain KV state after
    /// [`PrefillOutcome::Pending`] and must not sample or increment generated
    /// token metrics until the final segment. The compatibility default only
    /// accepts a final segment and delegates to [`Self::prefill`].
    fn prefill_chunk(
        &mut self,
        prompt_tokens: &[u32],
        is_final: bool,
    ) -> Result<PrefillOutcome, ModelExecutionError> {
        if !is_final {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "this model does not support resumable prefill",
            ));
        }
        self.prefill(prompt_tokens).map(PrefillOutcome::Token)
    }

    /// Consumes the previously generated token and returns the next token.
    ///
    /// The caller passes `input_token` explicitly so execution traces can
    /// distinguish model output from the next decode input.
    fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError>;

    /// Returns a cheap snapshot suitable for telemetry and policy state.
    fn metrics(&self) -> SessionMetrics;

    /// Exposes concrete request state to its owning model's optional batched
    /// execution implementation.
    ///
    /// Runtime code must treat the returned value as opaque. Downcasting is
    /// reserved for the same architecture implementation that created the
    /// session.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Executable causal-language artifact produced by an architecture factory.
///
/// Both members are reference counted so a loaded artifact can serve many
/// requests without duplicating tokenizer data or model weights.
#[derive(Clone)]
pub struct CausalLanguageModel {
    tokenizer: Arc<dyn Tokenizer>,
    model: Arc<dyn CausalModel>,
    kv_cache_profile: Option<KvCacheProfile>,
}

impl CausalLanguageModel {
    /// Combines a tokenizer and compatible causal-model implementation.
    pub fn new<T, M>(tokenizer: T, model: M) -> Self
    where
        T: Tokenizer + 'static,
        M: CausalModel + 'static,
    {
        Self {
            tokenizer: Arc::new(tokenizer),
            model: Arc::new(model),
            kv_cache_profile: None,
        }
    }

    /// Combines already shared tokenizer and model implementations.
    pub const fn from_shared(tokenizer: Arc<dyn Tokenizer>, model: Arc<dyn CausalModel>) -> Self {
        Self {
            tokenizer,
            model,
            kv_cache_profile: None,
        }
    }

    /// Attaches physical KV-pool metadata for admission planning.
    pub const fn with_kv_cache_profile(mut self, profile: KvCacheProfile) -> Self {
        self.kv_cache_profile = Some(profile);
        self
    }

    /// Returns physical KV-pool metadata when the architecture supplied it.
    pub const fn kv_cache_profile(&self) -> Option<KvCacheProfile> {
        self.kv_cache_profile
    }

    /// Returns whether request sessions can process prompts incrementally.
    pub fn supports_chunked_prefill(&self) -> bool {
        self.model.supports_chunked_prefill()
    }

    /// Returns the tensor-batched paths the model declares.
    pub fn batch_capabilities(&self) -> BatchCapabilities {
        self.model.batch_capabilities()
    }

    /// Returns the artifact-compatible tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Returns the shared causal-model implementation.
    pub fn model(&self) -> &dyn CausalModel {
        self.model.as_ref()
    }

    /// Creates isolated generation state for one request.
    pub fn create_session(
        &self,
        generation: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
        self.model.create_session(generation)
    }

    /// Attempts a model-specific tensor-batched prefill.
    pub fn prefill_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_tokens: &[&[u32]],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        self.model.prefill_batch(sessions, prompt_tokens)
    }

    /// Attempts a model-specific tensor-batched resumable prefill.
    pub fn prefill_chunk_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        prompt_chunks: &[&[u32]],
        is_final: &[bool],
    ) -> Option<Vec<Result<PrefillOutcome, ModelExecutionError>>> {
        self.model
            .prefill_chunk_batch(sessions, prompt_chunks, is_final)
    }

    /// Attempts a model-specific tensor-batched decode.
    pub fn decode_batch(
        &self,
        sessions: &mut [&mut dyn GenerationSession],
        input_tokens: &[u32],
    ) -> Option<Vec<Result<GeneratedToken, ModelExecutionError>>> {
        self.model.decode_batch(sessions, input_tokens)
    }
}

impl fmt::Debug for CausalLanguageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CausalLanguageModel")
            .field("tokenizer", &"dyn Tokenizer")
            .field("model", &"dyn CausalModel")
            .field("kv_cache_profile", &self.kv_cache_profile)
            .finish()
    }
}

#[cfg(test)]
mod tests {
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
                            "expected space-separated integers",
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

    struct IncrementModel;

    impl CausalModel for IncrementModel {
        fn create_session(
            &self,
            _generation: GenerationConfig,
        ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
            Ok(Box::new(IncrementSession::default()))
        }
    }

    #[derive(Default)]
    struct IncrementSession {
        metrics: SessionMetrics,
        prefill_complete: bool,
    }

    impl GenerationSession for IncrementSession {
        fn prefill(
            &mut self,
            prompt_tokens: &[u32],
        ) -> Result<GeneratedToken, ModelExecutionError> {
            if self.prefill_complete {
                return Err(ModelExecutionError::new(
                    ExecutionStage::Prefill,
                    "prefill may only run once",
                ));
            }
            let last = prompt_tokens.last().copied().ok_or_else(|| {
                ModelExecutionError::new(ExecutionStage::Prefill, "prompt cannot be empty")
            })?;
            self.prefill_complete = true;
            self.metrics.prompt_tokens = prompt_tokens.len();
            self.metrics.generated_tokens = 1;
            Ok(GeneratedToken {
                id: last + 1,
                is_end_of_sequence: false,
            })
        }

        fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
            if !self.prefill_complete {
                return Err(ModelExecutionError::new(
                    ExecutionStage::Decode,
                    "decode requires prefill",
                ));
            }
            self.metrics.generated_tokens += 1;
            Ok(GeneratedToken {
                id: input_token + 1,
                is_end_of_sequence: input_token >= 4,
            })
        }

        fn metrics(&self) -> SessionMetrics {
            self.metrics
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn tokenizer_and_model_are_shared_but_sessions_are_isolated() {
        let artifact = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let prompt = artifact.tokenizer().encode("1 2", true).unwrap();
        let mut first = artifact
            .create_session(GenerationConfig::default())
            .unwrap();
        let second = artifact
            .create_session(GenerationConfig::default())
            .unwrap();

        let first_token = first.prefill(&prompt).unwrap();

        assert_eq!(first_token.id, 3);
        assert_eq!(first.metrics().generated_tokens, 1);
        assert_eq!(second.metrics(), SessionMetrics::default());
    }

    #[test]
    fn models_declare_no_batched_paths_by_default() {
        let artifact = CausalLanguageModel::new(NumericTokenizer, IncrementModel);

        assert_eq!(artifact.batch_capabilities(), BatchCapabilities::NONE);
    }

    #[test]
    fn session_rejects_decode_before_prefill() {
        let artifact = CausalLanguageModel::new(NumericTokenizer, IncrementModel);
        let mut session = artifact
            .create_session(GenerationConfig::default())
            .unwrap();

        let error = session.decode(1).unwrap_err();

        assert_eq!(error.stage(), ExecutionStage::Decode);
        assert_eq!(error.message(), "decode requires prefill");
    }
}
