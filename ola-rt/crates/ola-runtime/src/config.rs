use serde::{Deserialize, Serialize};

/// Serializable batch-policy choices supported by the runtime builder.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatchPolicyConfig {
    /// Preserve the default unbounded batch selection.
    Unbounded,
    /// Select a FIFO prefix under one input-token budget.
    FifoTokenBudget {
        /// Maximum input-token positions per iteration.
        max_tokens: usize,
    },
    /// Prioritize decodes while bounding prompt work and total input tokens.
    MixedTokenBudget {
        /// Maximum input-token positions per iteration.
        max_tokens: usize,
        /// Maximum prompt-token positions per iteration.
        max_prefill_tokens: usize,
    },
}

impl Default for BatchPolicyConfig {
    fn default() -> Self {
        Self::Unbounded
    }
}

/// Serializable limits and device-memory settings for an executor.
///
/// Policy components and model artifacts remain runtime-owned objects; this
/// configuration covers the stable scalar boundary that can be loaded before
/// those components are selected.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExecutorConfig {
    /// Maximum number of concurrent live requests.
    #[serde(default = "default_max_active_requests")]
    pub max_active_requests: usize,
    /// Maximum prompt positions processed per resumable prefill iteration.
    #[serde(default)]
    pub prefill_chunk_size: Option<usize>,
    /// Total device-memory budget visible to admission, in bytes.
    #[serde(default)]
    pub memory_capacity_bytes: u64,
    /// Optional stable device label exposed to planning and telemetry.
    #[serde(default)]
    pub device_label: Option<String>,
    /// Batch-policy selection applied by [`Executor::from_config`].
    #[serde(default)]
    pub batch_policy: BatchPolicyConfig,
}

const fn default_max_active_requests() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalar_executor_limits() {
        let config: ExecutorConfig = serde_json::from_str(
            r#"{"max_active_requests":4,"prefill_chunk_size":32,"memory_capacity_bytes":4096}"#,
        )
        .unwrap();

        assert_eq!(config.max_active_requests, 4);
        assert_eq!(config.prefill_chunk_size, Some(32));
        assert_eq!(config.memory_capacity_bytes, 4096);
        assert_eq!(config.device_label, None);
    }
}
