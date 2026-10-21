//! Backend-neutral device telemetry.

/// Allocator view of one device's memory.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeviceMemory {
    /// Bytes held by live allocations.
    pub bytes_in_use: u64,
    /// Bytes the allocator has reserved from the device, including space not
    /// currently in use.
    pub bytes_reserved: u64,
}

/// Reads a device's state for health reporting.
///
/// Implementations live with their backend (for example CubeCL's allocator
/// in `ola-kernel`). Readings may wait for the device's work queue, so callers
/// sample on the thread that drives execution rather than on request paths.
pub trait DeviceTelemetry: Send + Sync {
    /// Stable description of the device, such as `wgpu:0`.
    fn label(&self) -> String;

    /// Current memory, or `None` when the backend cannot report it.
    fn memory(&self) -> Option<DeviceMemory>;

    /// Percent of the counter's last period the device spent running work,
    /// from 0 to 100, or `None` when no counter answers.
    fn busy_percent(&self) -> Option<f32> {
        None
    }
}
