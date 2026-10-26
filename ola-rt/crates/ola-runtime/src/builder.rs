use crate::{BatchPolicyConfig, ExecutionPolicy, Executor, ExecutorConfig, ExecutorError};

/// High-level constructor for an executor and its scalar runtime settings.
pub struct ExecutorBuilder {
    config: ExecutorConfig,
    policy: Option<Box<dyn ExecutionPolicy>>,
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutorBuilder {
    /// Creates a builder with the default executor configuration.
    pub const fn new() -> Self {
        Self {
            config: ExecutorConfig {
                max_active_requests: 1,
                prefill_chunk_size: None,
                memory_capacity_bytes: 0,
                device_label: None,
                batch_policy: BatchPolicyConfig::Unbounded,
            },
            policy: None,
        }
    }

    /// Creates a builder from a JSON-encoded [`ExecutorConfig`].
    ///
    /// This keeps configuration loading at the builder boundary while model
    /// attachment and policy construction remain explicit Rust operations.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json).map(|config| Self::new().with_config(config))
    }

    /// Replaces the scalar configuration loaded from a file or service.
    pub fn with_config(mut self, config: ExecutorConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the maximum number of concurrent live requests.
    pub const fn with_max_active_requests(mut self, limit: usize) -> Self {
        self.config.max_active_requests = limit;
        self
    }

    /// Sets the resumable prefill chunk limit.
    pub const fn with_prefill_chunk_size(mut self, size: Option<usize>) -> Self {
        self.config.prefill_chunk_size = size;
        self
    }

    /// Sets the device-memory capacity visible to admission.
    pub const fn with_memory_capacity_bytes(mut self, bytes: u64) -> Self {
        self.config.memory_capacity_bytes = bytes;
        self
    }

    /// Sets the stable device label exposed to planning and telemetry.
    pub fn with_device_label(mut self, label: impl Into<String>) -> Self {
        self.config.device_label = Some(label.into());
        self
    }

    /// Replaces the execution policy used by the built executor.
    pub fn with_policy<P>(mut self, policy: P) -> Self
    where
        P: ExecutionPolicy + 'static,
    {
        self.policy = Some(Box::new(policy));
        self
    }

    /// Validates the scalar settings and constructs the executor.
    pub fn build(self) -> Result<Executor, ExecutorError> {
        let mut executor = Executor::from_config(self.config)?;
        if let Some(policy) = self.policy {
            executor = executor.with_policy(policy);
        }
        Ok(executor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComposedPolicy, FifoTokenBudgetPolicy};

    #[test]
    fn builds_configured_executor_and_policy() {
        let executor = ExecutorBuilder::new()
            .with_max_active_requests(3)
            .with_prefill_chunk_size(Some(16))
            .with_memory_capacity_bytes(1024)
            .with_policy(
                ComposedPolicy::default().with_batch(FifoTokenBudgetPolicy::new(8).unwrap()),
            )
            .build()
            .unwrap();

        assert_eq!(executor.runtime().state().memory.capacity_bytes, 1024);
        assert_eq!(executor.policy_descriptor().name, "composed");
    }

    #[test]
    fn builds_from_json_configuration() {
        let executor = ExecutorBuilder::from_json(
            r#"{
                "max_active_requests": 2,
                "memory_capacity_bytes": 4096,
                "device_label": "test-device",
                "batch_policy": {"kind": "fifo_token_budget", "max_tokens": 8}
            }"#,
        )
        .unwrap()
        .build()
        .unwrap();

        assert_eq!(executor.runtime().state().memory.capacity_bytes, 4096);
        assert_eq!(executor.max_active_requests(), 2);
    }

    #[test]
    fn rejects_invalid_json_configuration() {
        assert!(ExecutorBuilder::from_json("{not-json").is_err());
    }
}
