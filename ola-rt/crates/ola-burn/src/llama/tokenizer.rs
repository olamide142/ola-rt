use std::path::Path;

use ola_model::{ExecutionStage, ModelExecutionError, Tokenizer};

/// Adapter from a Hugging Face `tokenizer.json` to OlaRT's tokenizer boundary.
#[derive(Clone)]
pub struct HuggingFaceTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HuggingFaceTokenizer {
    /// Loads a self-contained Hugging Face tokenizer JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelExecutionError> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|error| {
            ModelExecutionError::new(
                ExecutionStage::Tokenize,
                format!("failed to load tokenizer `{}`: {error}", path.display()),
            )
        })?;
        Ok(Self { inner })
    }

    /// Wraps an already constructed tokenizer.
    ///
    /// This constructor is useful for conformance tests and applications that
    /// obtain tokenizer bytes from a source other than the filesystem.
    pub const fn new(inner: tokenizers::Tokenizer) -> Self {
        Self { inner }
    }

    /// Returns the underlying tokenizer for tokenizer-specific inspection.
    pub const fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

impl std::fmt::Debug for HuggingFaceTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HuggingFaceTokenizer")
            .field("vocabulary_size", &self.inner.get_vocab_size(true))
            .finish()
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, ModelExecutionError> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| {
                ModelExecutionError::new(
                    ExecutionStage::Tokenize,
                    format!("Hugging Face tokenizer rejected input: {error}"),
                )
            })
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, ModelExecutionError> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                ModelExecutionError::new(
                    ExecutionStage::Detokenize,
                    format!("Hugging Face tokenizer rejected token IDs: {error}"),
                )
            })
    }
}
