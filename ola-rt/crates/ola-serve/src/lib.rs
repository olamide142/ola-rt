#![doc = include_str!("../README.md")]

use std::collections::BTreeMap;

use ola_core::{
    GenerationConfig, Input, ModelDescriptor, ModelId, Priority, RequestId, RequestSpec, Timestamp,
};
use ola_runtime::{Runtime, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Coarse service readiness reported to a future health endpoint.
pub enum HealthStatus {
    /// Runtime is constructed and able to accept valid requests.
    Ready,
}

#[derive(Clone, Debug, PartialEq)]
/// Transport-neutral generation request accepted by the service boundary.
pub struct GenerateRequest {
    /// Registered model to invoke.
    pub model: ModelId,
    /// Ordered text, token, or image inputs.
    pub inputs: Vec<Input>,
    /// Autoregressive generation controls.
    pub generation: GenerationConfig,
    /// Relative service priority.
    pub priority: Priority,
    /// Optional deadline in the runtime clock epoch.
    pub deadline: Option<Timestamp>,
    /// Labels preserved for experiments and telemetry.
    pub attributes: BTreeMap<String, String>,
}

impl GenerateRequest {
    /// Constructs a text request using normal priority and default sampling.
    pub fn text(model: impl Into<ModelId>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            inputs: vec![Input::Text(prompt.into())],
            generation: GenerationConfig::default(),
            priority: Priority::default(),
            deadline: None,
            attributes: BTreeMap::new(),
        }
    }
}

#[derive(Debug)]
/// Application facade owning one configured runtime.
pub struct InferenceService {
    runtime: Runtime,
}

impl InferenceService {
    /// Wraps a configured runtime.
    pub const fn new(runtime: Runtime) -> Self {
        Self { runtime }
    }

    /// Returns service readiness without performing device work.
    pub const fn health(&self) -> HealthStatus {
        HealthStatus::Ready
    }

    /// Lists registered model metadata in stable identifier order.
    pub fn models(&self) -> impl ExactSizeIterator<Item = &ModelDescriptor> {
        self.runtime.state().models().iter()
    }

    /// Translates and submits a generation request.
    pub fn submit(
        &mut self,
        request: GenerateRequest,
        arrival_at: Timestamp,
    ) -> Result<RequestId, RuntimeError> {
        self.runtime.submit(
            RequestSpec {
                model: request.model,
                inputs: request.inputs,
                generation: request.generation,
                priority: request.priority,
                deadline: request.deadline,
                attributes: request.attributes,
            },
            arrival_at,
        )
    }

    /// Borrows the underlying runtime for observation.
    pub const fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Mutably borrows the runtime for executor-loop lifecycle operations.
    pub const fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// Consumes the facade and returns its runtime.
    pub fn into_runtime(self) -> Runtime {
        self.runtime
    }
}

#[cfg(test)]
mod tests {
    use ola_core::ModelDescriptor;
    use ola_runtime::Runtime;

    use super::*;

    #[test]
    fn submits_transport_requests_to_the_runtime() {
        let mut runtime = Runtime::default();
        runtime
            .register_model(ModelDescriptor::language("tiny", "tiny-decoder"))
            .unwrap();
        let mut service = InferenceService::new(runtime);

        let id = service
            .submit(GenerateRequest::text("tiny", "hello"), Timestamp::ZERO)
            .unwrap();

        assert_eq!(id.get(), 1);
        assert_eq!(service.runtime().state().queued_len(), 1);
    }
}
