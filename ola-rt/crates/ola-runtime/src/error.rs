use std::{error::Error, fmt};

use ola_core::{Modality, ModelDescriptorError, ModelId, RequestError, RequestId, Timestamp};

#[derive(Debug, Eq, PartialEq)]
/// Error raised while admitting or transitioning runtime work.
pub enum RuntimeError {
    /// Model metadata violates a backend-neutral invariant.
    InvalidModel(ModelDescriptorError),
    /// Request fields violate a backend-independent invariant.
    InvalidRequest(RequestError),
    /// Request refers to a model absent from the registry.
    UnknownModel(ModelId),
    /// Model does not accept one of the request's input modalities.
    UnsupportedModality {
        /// Selected model.
        model: ModelId,
        /// Unsupported request-input modality.
        modality: Modality,
    },
    /// A model identifier was registered more than once.
    DuplicateModel(ModelId),
    /// A request identifier collided with an existing record.
    DuplicateRequest(RequestId),
    /// No request record exists for the supplied identifier.
    UnknownRequest(RequestId),
    /// Scheduler returned an identifier that is not currently queued.
    InvalidSchedule(RequestId),
    /// Requested lifecycle operation is invalid for the current status.
    InvalidTransition {
        /// Request on which the operation was attempted.
        request: RequestId,
        /// Human-readable operation name.
        operation: &'static str,
    },
    /// A lifecycle event was reported before its causal predecessor.
    TimestampRegression {
        /// Request containing the invalid measurement.
        request: RequestId,
        /// Event whose timestamp moved backward.
        event: &'static str,
        /// Earliest valid timestamp for the event.
        previous: Timestamp,
        /// Timestamp supplied by the caller.
        observed: Timestamp,
    },
    /// Monotonic request identifier space was exhausted.
    RequestIdExhausted,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(error) => write!(formatter, "invalid model: {error}"),
            Self::InvalidRequest(error) => write!(formatter, "invalid request: {error}"),
            Self::UnknownModel(model) => write!(formatter, "model `{model}` is not registered"),
            Self::UnsupportedModality { model, modality } => {
                write!(
                    formatter,
                    "model `{model}` does not support {modality:?} input"
                )
            }
            Self::DuplicateModel(model) => {
                write!(formatter, "model `{model}` is already registered")
            }
            Self::DuplicateRequest(request) => {
                write!(formatter, "request `{request}` already exists")
            }
            Self::UnknownRequest(request) => {
                write!(formatter, "request `{request}` does not exist")
            }
            Self::InvalidSchedule(request) => {
                write!(
                    formatter,
                    "scheduler selected non-queued request `{request}`"
                )
            }
            Self::InvalidTransition { request, operation } => {
                write!(
                    formatter,
                    "cannot {operation} request `{request}` in its current state"
                )
            }
            Self::TimestampRegression {
                request,
                event,
                previous,
                observed,
            } => write!(
                formatter,
                "{event} timestamp {observed} for request `{request}` precedes {previous}"
            ),
            Self::RequestIdExhausted => formatter.write_str("request id space is exhausted"),
        }
    }
}

impl Error for RuntimeError {}

impl From<RequestError> for RuntimeError {
    fn from(value: RequestError) -> Self {
        Self::InvalidRequest(value)
    }
}

impl From<ModelDescriptorError> for RuntimeError {
    fn from(value: ModelDescriptorError) -> Self {
        Self::InvalidModel(value)
    }
}
