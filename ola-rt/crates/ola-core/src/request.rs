use std::{collections::BTreeMap, error::Error, fmt};

use crate::{ModelId, RequestId, Timestamp};

#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
/// Kind of data supplied as one request input.
pub enum Modality {
    /// Untokenized natural-language text.
    Text,
    /// Token IDs prepared by a compatible tokenizer.
    Tokens,
    /// Encoded image bytes plus decoded shape metadata.
    Image,
}

/// Image payload and shape presented to a future preprocessing stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageInput {
    /// Encoded or raw image bytes; interpretation belongs to the model frontend.
    pub bytes: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of color channels.
    pub channels: u8,
}

/// One typed input in a potentially multimodal request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Input {
    /// Untokenized text.
    Text(String),
    /// Token IDs whose tokenizer compatibility is checked by the model frontend.
    Tokens(Vec<u32>),
    /// Image data and shape metadata.
    Image(ImageInput),
}

impl Input {
    /// Returns the input's modality without inspecting its payload.
    pub const fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Tokens(_) => Modality::Tokens,
            Self::Image(_) => Modality::Image,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Relative service priority; larger values represent higher priority.
///
/// FIFO ignores this field. Later priority-aware policies may use it while the
/// request representation remains unchanged.
pub struct Priority(i32);

impl Priority {
    /// Default priority for requests without service-class information.
    pub const NORMAL: Self = Self(0);

    /// Constructs a relative priority.
    pub const fn new(value: i32) -> Self {
        Self(value)
    }

    /// Returns the signed priority value.
    pub const fn get(self) -> i32 {
        self.0
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Backend-neutral controls for autoregressive generation.
pub struct GenerationConfig {
    /// Maximum number of tokens the runtime may generate.
    pub max_new_tokens: usize,
    /// Sampling temperature; zero permits deterministic greedy selection.
    pub temperature: f32,
    /// Nucleus-sampling probability mass in the interval `(0, 1]`.
    pub top_p: f32,
    /// Sampling seed used to reproduce token selection.
    ///
    /// `None` permits implementation-selected entropy and should be avoided in
    /// experiments that claim deterministic replay.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            seed: Some(0),
        }
    }
}

/// A request before the runtime assigns identity and arrival time.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestSpec {
    /// Registered model that should execute the request.
    pub model: ModelId,
    /// Ordered inputs presented to the model frontend.
    pub inputs: Vec<Input>,
    /// Output-generation controls.
    pub generation: GenerationConfig,
    /// Relative scheduling priority.
    pub priority: Priority,
    /// Optional absolute deadline in the same epoch as request arrival.
    pub deadline: Option<Timestamp>,
    /// Experiment- or application-defined labels preserved for telemetry.
    pub attributes: BTreeMap<String, String>,
}

impl RequestSpec {
    /// Constructs a normal-priority text generation request with default
    /// generation settings.
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

    /// Validates backend-independent request invariants.
    pub fn validate(&self) -> Result<(), RequestError> {
        if self.model.is_empty() {
            return Err(RequestError::EmptyModelId);
        }
        if self.inputs.is_empty() {
            return Err(RequestError::NoInputs);
        }
        if self.generation.max_new_tokens == 0 {
            return Err(RequestError::ZeroMaxNewTokens);
        }
        if !self.generation.temperature.is_finite() || self.generation.temperature < 0.0 {
            return Err(RequestError::InvalidTemperature);
        }
        if !self.generation.top_p.is_finite()
            || self.generation.top_p <= 0.0
            || self.generation.top_p > 1.0
        {
            return Err(RequestError::InvalidTopP);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Validated request after runtime identity and arrival time are assigned.
pub struct Request {
    /// Runtime-local request identifier.
    pub id: RequestId,
    /// Admission time relative to the runtime or trace epoch.
    pub arrival_at: Timestamp,
    /// Original caller intent.
    pub spec: RequestSpec,
}

impl Request {
    /// Validates and constructs an admitted request.
    pub fn new(
        id: RequestId,
        arrival_at: Timestamp,
        spec: RequestSpec,
    ) -> Result<Self, RequestError> {
        spec.validate()?;
        if spec.deadline.is_some_and(|deadline| deadline < arrival_at) {
            return Err(RequestError::DeadlineBeforeArrival);
        }
        Ok(Self {
            id,
            arrival_at,
            spec,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Failure of a backend-independent request invariant.
pub enum RequestError {
    /// Model identifier is empty or whitespace.
    EmptyModelId,
    /// No input was supplied.
    NoInputs,
    /// Generation was requested with a zero token limit.
    ZeroMaxNewTokens,
    /// Temperature was negative, infinite, or not a number.
    InvalidTemperature,
    /// Nucleus probability was outside `(0, 1]` or not finite.
    InvalidTopP,
    /// Deadline precedes the request's arrival timestamp.
    DeadlineBeforeArrival,
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyModelId => "model id cannot be empty",
            Self::NoInputs => "request must contain at least one input",
            Self::ZeroMaxNewTokens => "max_new_tokens must be greater than zero",
            Self::InvalidTemperature => "temperature must be finite and non-negative",
            Self::InvalidTopP => "top_p must be finite and in the interval (0, 1]",
            Self::DeadlineBeforeArrival => "deadline cannot be earlier than arrival",
        };
        formatter.write_str(message)
    }
}

impl Error for RequestError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_deadline_before_arrival() {
        let mut spec = RequestSpec::text("tiny", "hello");
        spec.deadline = Some(Timestamp::from_micros(9));

        let error = Request::new(RequestId::new(1), Timestamp::from_micros(10), spec).unwrap_err();

        assert_eq!(error, RequestError::DeadlineBeforeArrival);
    }
}
