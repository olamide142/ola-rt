#![doc = include_str!("../README.md")]

pub mod cluster;
mod driver;
mod error;
pub mod http;

/// A serving backend that can expose the native HTTP contract.
///
/// Local and distributed backends retain their own request lifecycles, while
/// the transport and endpoint selection remain behind this small boundary.
pub trait InferenceBackend: Clone + Send + Sync + 'static {
    /// Builds a router for this backend.
    fn router(&self) -> axum::Router;
}

/// Builds a router through the backend abstraction.
pub fn backend_router<B: InferenceBackend>(backend: B) -> axum::Router {
    backend.router()
}

pub use driver::{
    Driver, DriverConfig, FinishReason, GenerationOutput, Health, Instruments, ModelStatus, Reply,
    RequestEvent, RequestHandle, RequestSnapshot, RuntimeHandle, ServiceState, Submission,
};
pub use error::ServeError;
