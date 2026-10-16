#![doc = include_str!("../README.md")]

use burn::tensor::Device;
use ola_core::ComputeBackend;

pub mod llama;

/// Associates a Burn backend and device with the backend-neutral runtime.
#[derive(Clone, Debug)]
pub struct BurnComputeBackend {
    device: Device,
}

impl BurnComputeBackend {
    /// Creates an adapter that owns the selected Burn device.
    pub const fn new(device: Device) -> Self {
        Self { device }
    }

    /// Returns the native Burn device for backend-specific model setup.
    pub fn burn_device(&self) -> &Device {
        &self.device
    }
}

impl ComputeBackend for BurnComputeBackend {
    type Device = Device;

    fn name(&self) -> &'static str {
        "burn"
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;
    use ola_core::ComputeBackend;

    use super::BurnComputeBackend;

    #[test]
    fn exposes_burn_through_the_compute_boundary() {
        let adapter = BurnComputeBackend::new(Device::ndarray());

        assert_eq!(adapter.name(), "burn");
        let _ = adapter.device();
    }
}
