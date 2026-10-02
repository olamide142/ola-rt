use std::fmt::Debug;

/// Minimal boundary between the OlaRT runtime and a compute implementation.
///
/// Tensor execution methods will be added only when the first model path makes
/// their required shape clear.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Backend-specific device handle owned by the adapter.
    type Device: Clone + Debug + Send + Sync + 'static;

    /// Returns a stable diagnostic name for traces and benchmark metadata.
    fn name(&self) -> &'static str;

    /// Returns the selected backend device.
    fn device(&self) -> &Self::Device;
}
