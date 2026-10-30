use std::{error::Error, fmt, path::Path};

use serde::Deserialize;

/// Hugging Face configuration fields required by the initial Mistral executor.
///
/// Unknown JSON fields are deliberately accepted. Hugging Face configurations
/// contain provenance and training fields that do not affect inference. In
/// contrast, known architectural features that this implementation cannot yet
/// execute are rejected by [`validate`](Self::validate).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct MistralConfig {
    /// Hugging Face architecture family discriminator.
    pub model_type: String,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Width of the gated feed-forward intermediate representation.
    pub intermediate_size: usize,
    /// Number of decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads used by grouped-query attention.
    pub num_key_value_heads: usize,
    /// Maximum context length described by the artifact.
    pub max_position_embeddings: usize,
    /// Numerical-stability epsilon used by RMSNorm.
    pub rms_norm_eps: f64,
    /// Base frequency used by rotary position embeddings.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Optional sliding window attention size.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Optional explicit head width used by Mistral configurations.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Activation name used by the MLP.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Whether attention projections contain bias vectors.
    #[serde(default)]
    pub attention_bias: bool,
    /// Whether MLP projections contain bias vectors.
    #[serde(default)]
    pub mlp_bias: bool,
    /// Whether the output projection shares storage with token embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Optional RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// Beginning-of-sequence token, when declared by the artifact.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// One or more end-of-sequence tokens.
    #[serde(default, deserialize_with = "deserialize_token_ids")]
    pub eos_token_id: Vec<u32>,
}

impl MistralConfig {
    /// Parses and validates a Hugging Face `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, MistralConfigError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| MistralConfigError::Read {
            path: path.to_path_buf(),
            source: source.to_string(),
        })?;
        Self::from_json(&bytes)
    }

    /// Parses and validates Hugging Face configuration JSON.
    pub fn from_json(json: &[u8]) -> Result<Self, MistralConfigError> {
        let config: Self = serde_json::from_slice(json)
            .map_err(|error| MistralConfigError::Json(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Returns the dimension of one attention head.
    pub fn attention_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Verifies shape invariants and rejects unsupported Mistral variants.
    pub fn validate(&self) -> Result<(), MistralConfigError> {
        if self.model_type != "mistral" {
            return Err(MistralConfigError::Invalid(format!(
                "model_type must be `mistral`, found `{}`",
                self.model_type
            )));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                return Err(MistralConfigError::Invalid(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(MistralConfigError::Invalid(
                "hidden_size must be divisible by num_attention_heads".into(),
            ));
        }
        let inferred_head_dim = self.hidden_size / self.num_attention_heads;
        if self
            .head_dim
            .is_some_and(|head_dim| head_dim != inferred_head_dim)
        {
            return Err(MistralConfigError::Unsupported(
                "head_dim different from hidden_size / num_attention_heads".into(),
            ));
        }
        if !inferred_head_dim.is_multiple_of(2) {
            return Err(MistralConfigError::Invalid(
                "attention head dimension must be even for rotary embeddings".into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(MistralConfigError::Invalid(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(MistralConfigError::Invalid(
                "rms_norm_eps must be finite and positive".into(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(MistralConfigError::Invalid(
                "rope_theta must be finite and positive".into(),
            ));
        }
        if self.sliding_window == Some(0) {
            return Err(MistralConfigError::Invalid(
                "sliding_window must be greater than zero".into(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(MistralConfigError::Unsupported(format!(
                "hidden_act `{}`; only `silu` is implemented",
                self.hidden_act
            )));
        }
        if self.attention_bias {
            return Err(MistralConfigError::Unsupported(
                "attention projection biases".into(),
            ));
        }
        if self.mlp_bias {
            return Err(MistralConfigError::Unsupported(
                "MLP projection biases".into(),
            ));
        }
        if self.tie_word_embeddings {
            return Err(MistralConfigError::Unsupported(
                "tied input/output embeddings".into(),
            ));
        }
        if self.rope_scaling.is_some() {
            return Err(MistralConfigError::Unsupported("RoPE scaling".into()));
        }
        Ok(())
    }
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_hidden_act() -> String {
    "silu".into()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<OneOrManyTokenIds>::deserialize(deserializer).map(|ids| match ids {
        None => Vec::new(),
        Some(OneOrManyTokenIds::One(id)) => vec![id],
        Some(OneOrManyTokenIds::Many(ids)) => ids,
    })
}

/// Failure to read, parse, or support one Mistral configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MistralConfigError {
    /// Configuration file could not be read.
    Read {
        /// Attempted path.
        path: std::path::PathBuf,
        /// Operating-system diagnostic.
        source: String,
    },
    /// File was not valid JSON or did not contain required fields.
    Json(String),
    /// Shape or numeric invariant was violated.
    Invalid(String),
    /// Configuration is valid Mistral but requests an unimplemented variant.
    Unsupported(String),
}

impl fmt::Display for MistralConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "failed to read `{}`: {source}", path.display())
            }
            Self::Json(message) => write!(formatter, "invalid Mistral config JSON: {message}"),
            Self::Invalid(message) => write!(formatter, "invalid Mistral configuration: {message}"),
            Self::Unsupported(feature) => {
                write!(
                    formatter,
                    "unsupported Mistral configuration feature: {feature}"
                )
            }
        }
    }
}

impl Error for MistralConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "model_type": "mistral",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 0.00001,
        "sliding_window": 16,
        "eos_token_id": [2, 3],
        "unrelated_training_field": true
    }"#;

    #[test]
    fn parses_required_fields_and_ignores_non_execution_metadata() {
        let config = MistralConfig::from_json(MINIMAL.as_bytes()).unwrap();

        assert_eq!(config.attention_head_dim(), 4);
        assert_eq!(config.eos_token_id, vec![2, 3]);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.sliding_window, Some(16));
    }

    #[test]
    fn parses_optional_sliding_window_when_null_or_absent() {
        let json = MINIMAL.replace("\"sliding_window\": 16,", "\"sliding_window\": null,");
        let config = MistralConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.sliding_window, None);

        let json = MINIMAL.replace("\"sliding_window\": 16,", "");
        let config = MistralConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.sliding_window, None);
    }

    #[test]
    fn accepts_single_eos_token_representation() {
        let json = MINIMAL.replace("[2, 3]", "2");
        let config = MistralConfig::from_json(json.as_bytes()).unwrap();

        assert_eq!(config.eos_token_id, vec![2]);
    }

    #[test]
    fn rejects_non_divisible_grouped_query_attention() {
        let json = MINIMAL.replace("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3");
        let error = MistralConfig::from_json(json.as_bytes()).unwrap_err();

        assert!(error.to_string().contains("must be divisible"));
    }

    #[test]
    fn rejects_zero_sliding_window() {
        let json = MINIMAL.replace("\"sliding_window\": 16,", "\"sliding_window\": 0,");
        let error = MistralConfig::from_json(json.as_bytes()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("sliding_window must be greater than zero")
        );
    }

    #[test]
    fn rejects_features_instead_of_silently_misexecuting_them() {
        let json = MINIMAL.replace(
            "\"unrelated_training_field\": true",
            "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0}",
        );
        let error = MistralConfig::from_json(json.as_bytes()).unwrap_err();

        assert_eq!(
            error,
            MistralConfigError::Unsupported("RoPE scaling".into())
        );
    }

    #[test]
    fn rejects_a_mislabeled_architecture() {
        let json = MINIMAL.replace("\"model_type\": \"mistral\"", "\"model_type\": \"llama\"");
        let error = MistralConfig::from_json(json.as_bytes()).unwrap_err();

        assert!(error.to_string().contains("model_type must be `mistral`"));
    }
}
