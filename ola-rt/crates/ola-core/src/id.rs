use std::fmt;

/// Uniquely identifies a request within one runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates an identifier from its runtime-local numeric representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the runtime-local numeric representation.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Stable, human-readable identifier for a registered model.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a model identifier.
    ///
    /// Validation occurs when a descriptor or request is admitted, allowing
    /// parsers to construct values without hidden allocation or failure paths.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns whether the identifier contains no non-whitespace characters.
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl From<&str> for ModelId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ModelId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Stable identifier for a model implementation family.
///
/// Examples include `llama`, `qwen2`, and `vision-transformer`. Architecture
/// identity is intentionally separate from [`ModelId`]: many concrete model
/// artifacts may share one implementation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArchitectureId(String);

impl ArchitectureId {
    /// Creates an architecture identifier.
    ///
    /// Validation occurs at model or factory registration boundaries.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns whether the identifier contains no non-whitespace characters.
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl fmt::Display for ArchitectureId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl From<&str> for ArchitectureId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ArchitectureId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}
