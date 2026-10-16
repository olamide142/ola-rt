//! Incremental detokenization for streamed generation.

use crate::{ModelExecutionError, Tokenizer};

/// Converts a growing token sequence into text deltas.
///
/// Decoding tokens one at a time is wrong for most tokenizers: a character
/// can span several byte-level tokens, and SentencePiece-style tokenizers
/// render a leading space differently at the start of a sequence. The
/// decoder therefore decodes a short window twice, once ending before and
/// once after the newest tokens, and emits only the difference. Text ending
/// in U+FFFD is held back, because it usually means a multi-byte character
/// is still incomplete; [`finish`](Self::finish) releases whatever remains.
///
/// Each step decodes only the tokens since the last emitted delta, so the
/// cost per token does not grow with the output length. Special tokens are
/// skipped, matching the text the executor reports in its final result.
#[derive(Clone, Debug, Default)]
pub struct IncrementalDecoder {
    tokens: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl IncrementalDecoder {
    /// Starts an empty decoder.
    pub const fn new() -> Self {
        Self {
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    /// Appends one generated token and returns any newly complete text.
    pub fn push(
        &mut self,
        tokenizer: &dyn Tokenizer,
        token: u32,
    ) -> Result<String, ModelExecutionError> {
        self.tokens.push(token);
        let prefix = tokenizer.decode(&self.tokens[self.prefix_offset..self.read_offset], true)?;
        let full = tokenizer.decode(&self.tokens[self.prefix_offset..], true)?;
        if full.len() <= prefix.len() || full.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        let Some(delta) = full.get(prefix.len()..) else {
            // The window re-rendered earlier text differently; wait for the
            // next token rather than emit a split character.
            return Ok(String::new());
        };
        let delta = delta.to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        Ok(delta)
    }

    /// Returns text held back by [`push`](Self::push), including an
    /// incomplete trailing character, and resets the window.
    pub fn finish(&mut self, tokenizer: &dyn Tokenizer) -> Result<String, ModelExecutionError> {
        let prefix = tokenizer.decode(&self.tokens[self.prefix_offset..self.read_offset], true)?;
        let full = tokenizer.decode(&self.tokens[self.prefix_offset..], true)?;
        self.prefix_offset = self.tokens.len();
        self.read_offset = self.tokens.len();
        Ok(full.get(prefix.len()..).unwrap_or_default().to_owned())
    }

    /// Returns every token pushed so far.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per UTF-8 byte; 0 is a special token that decodes to
    /// nothing when special tokens are skipped.
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
        fn encode(&self, text: &str, _: bool) -> Result<Vec<u32>, ModelExecutionError> {
            Ok(text.bytes().map(|byte| u32::from(byte) + 1).collect())
        }

        fn decode(
            &self,
            tokens: &[u32],
            skip_special: bool,
        ) -> Result<String, ModelExecutionError> {
            let bytes = tokens
                .iter()
                .filter(|token| !(skip_special && **token == 0))
                .map(|token| u8::try_from(token.saturating_sub(1)).unwrap())
                .collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    /// Word tokens; a word renders with a leading space except at the start
    /// of a decoded sequence, like SentencePiece.
    struct WordTokenizer;

    const WORDS: [&str; 3] = ["hello", "streaming", "world"];

    impl Tokenizer for WordTokenizer {
        fn encode(&self, _: &str, _: bool) -> Result<Vec<u32>, ModelExecutionError> {
            unreachable!()
        }

        fn decode(&self, tokens: &[u32], _: bool) -> Result<String, ModelExecutionError> {
            Ok(tokens
                .iter()
                .map(|token| WORDS[*token as usize])
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    fn stream(tokenizer: &dyn Tokenizer, tokens: &[u32]) -> Vec<String> {
        let mut decoder = IncrementalDecoder::new();
        let mut deltas = tokens
            .iter()
            .map(|token| decoder.push(tokenizer, *token).unwrap())
            .collect::<Vec<_>>();
        deltas.push(decoder.finish(tokenizer).unwrap());
        deltas
    }

    #[test]
    fn multi_byte_characters_are_held_until_complete() {
        let text = "héllo 世界";
        let tokens = ByteTokenizer.encode(text, false).unwrap();

        let deltas = stream(&ByteTokenizer, &tokens);

        assert_eq!(deltas.concat(), text);
        assert!(deltas.iter().all(|delta| !delta.contains('\u{FFFD}')));
        // "é" is two bytes, so its first byte emits nothing.
        assert_eq!(deltas[..3], ["h".to_owned(), String::new(), "é".to_owned()]);
    }

    #[test]
    fn deltas_reproduce_full_decoding_with_context_dependent_spaces() {
        let tokens = [0, 1, 2, 1];
        let deltas = stream(&WordTokenizer, &tokens);

        assert_eq!(
            deltas.concat(),
            WordTokenizer.decode(&tokens, true).unwrap()
        );
        assert_eq!(deltas[1], " streaming");
    }

    #[test]
    fn finish_releases_an_incomplete_trailing_character() {
        let mut tokens = ByteTokenizer.encode("a界", false).unwrap();
        tokens.pop();

        let deltas = stream(&ByteTokenizer, &tokens);

        assert_eq!(deltas.concat(), "a\u{FFFD}");
        assert_eq!(deltas.last().unwrap(), "\u{FFFD}");
    }

    #[test]
    fn special_tokens_produce_no_text() {
        let deltas = stream(&ByteTokenizer, &[105, 0]);
        assert_eq!(deltas.concat(), "h");
    }
}
