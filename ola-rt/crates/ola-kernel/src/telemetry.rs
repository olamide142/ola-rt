//! Device memory and utilization through CubeCL.

use burn_cubecl::cubecl::{Device, MemoryScope};
use ola_observe::{DeviceMemory, DeviceTelemetry};

/// Reports one CubeCL device's allocator memory and utilization.
///
/// Memory covers every stream on the device and reflects allocations by
/// anything sharing CubeCL's per-device client, including Burn tensors on
/// the same device. Utilization comes from whatever counter the device's
/// driver or platform keeps (NVML for NVIDIA cards, processor times for the
/// CPU); some counters measure between readings, so the first may be
/// missing. Readings wait for the device, so sample them from the thread
/// that drives execution.
pub struct CubeclTelemetry {
    device: Device,
    label: String,
}

impl CubeclTelemetry {
    /// Reports on `device`, described as `label`.
    pub fn new(device: impl Into<Device>, label: impl Into<String>) -> Self {
        Self {
            device: device.into(),
            label: label.into(),
        }
    }
}

impl DeviceTelemetry for CubeclTelemetry {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn memory(&self) -> Option<DeviceMemory> {
        let usage = self
            .device
            .client()
            .memory_report(MemoryScope::Device)
            .usage();
        Some(DeviceMemory {
            bytes_in_use: usage.bytes_in_use,
            bytes_reserved: usage.bytes_reserved,
        })
    }

    fn busy_percent(&self) -> Option<f32> {
        self.device
            .utilization()
            .ok()
            .map(|utilization| utilization.busy_percent)
    }
}

/// Telemetry for the default WGPU device, the one
/// `Device::wgpu(Default::default())` selects.
#[cfg(feature = "wgpu")]
pub fn wgpu_telemetry() -> CubeclTelemetry {
    CubeclTelemetry::new(
        burn_cubecl::cubecl::wgpu::WgpuDevice::default(),
        "wgpu:default",
    )
}

/// Telemetry for CUDA device `index`.
#[cfg(feature = "cuda")]
pub fn cuda_telemetry(index: usize) -> Option<CubeclTelemetry> {
    Device::cuda(index)
        .ok()
        .map(|device| CubeclTelemetry::new(device, format!("cuda:{index}")))
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;

    #[test]
    fn allocations_show_up_in_the_reported_memory() {
        let device = Device::cpu().unwrap();
        let telemetry = CubeclTelemetry::new(device.clone(), "cpu:0");
        let before = telemetry.memory().unwrap();

        let _buffer = device.client().empty(1 << 20);
        let after = telemetry.memory().unwrap();

        assert_eq!(telemetry.label(), "cpu:0");
        assert!(
            after.bytes_in_use >= before.bytes_in_use + (1 << 20),
            "{before:?} → {after:?}"
        );
        assert!(after.bytes_reserved >= after.bytes_in_use);
        // The CPU counter measures between readings; the second exists.
        let _ = telemetry.busy_percent();
        let busy = telemetry.busy_percent().expect("a second CPU reading");
        assert!((0.0..=100.0).contains(&busy));
    }
}
