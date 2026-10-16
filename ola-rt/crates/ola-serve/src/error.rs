use std::{error::Error, fmt};

use ola_core::{ModelId, RequestId};
use ola_runtime::{AdmissionReason, RuntimeError};

/// Why a serving operation or a request did not succeed.
///
/// Variants keep validation, capacity, admission, and execution failures
/// apart so a transport can map each to its own status.
#[derive(Debug, PartialEq)]
pub enum ServeError {
    /// The request violates a runtime invariant and was never queued.
    Invalid(RuntimeError),
    /// The request names a model that is not registered.
    UnknownModel(ModelId),
    /// No request with this identifier is live or retained.
    UnknownRequest(RequestId),
    /// Too many requests are unfinished; retry later.
    Overloaded {
        /// Configured limit on unfinished requests.
        limit: usize,
    },
    /// Admission determined that the request can never fit.
    Rejected(AdmissionReason),
    /// The model failed while preparing or executing the request.
    Execution(String),
    /// The request was cancelled before it finished.
    Cancelled,
    /// The driver is running but cannot serve: no model is executable.
    NotReady,
    /// The driver is stopping or has stopped and accepts no work.
    ShuttingDown,
    /// The executor reported an internal failure, such as an invalid plan.
    Internal(String),
}

impl ServeError {
    pub(crate) fn from_submission(error: RuntimeError) -> Self {
        match error {
            RuntimeError::UnknownModel(model) => Self::UnknownModel(model),
            error => Self::Invalid(error),
        }
    }
}

impl fmt::Display for ServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => write!(formatter, "{error}"),
            Self::UnknownModel(model) => write!(formatter, "model `{model}` is not registered"),
            Self::UnknownRequest(request) => {
                write!(formatter, "request `{request}` is not known")
            }
            Self::Overloaded { limit } => write!(
                formatter,
                "the server is at its limit of {limit} unfinished requests"
            ),
            Self::Rejected(reason) => {
                write!(formatter, "request can never be admitted: {reason:?}")
            }
            Self::Execution(message) => write!(formatter, "execution failed: {message}"),
            Self::Cancelled => formatter.write_str("request was cancelled"),
            Self::NotReady => formatter.write_str("no model is ready to execute requests"),
            Self::ShuttingDown => formatter.write_str("the server is shutting down"),
            Self::Internal(message) => write!(formatter, "internal error: {message}"),
        }
    }
}

impl Error for ServeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalid(error) => Some(error),
            _ => None,
        }
    }
}
