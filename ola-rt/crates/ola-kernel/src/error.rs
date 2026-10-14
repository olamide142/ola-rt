use std::{error::Error, fmt};

/// Failure to configure or execute a kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelError {
    message: String,
}

impl KernelError {
    /// Creates a kernel diagnostic.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the diagnostic text.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for KernelError {}
