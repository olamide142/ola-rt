use burn::{backend::DispatchDevice, tensor::Device};

#[cfg(feature = "cubecl")]
use burn_cubecl::CubeDevice;

use crate::KernelTarget;

/// Classifies a Burn device for backend-neutral kernel planning.
///
/// Backends without a dedicated optimized target retain the portable path.
pub fn burn_device_target(device: &Device) -> KernelTarget {
    #[allow(unreachable_patterns)]
    match device.as_dispatch() {
        DispatchDevice::Flex(_) => KernelTarget::Cpu,
        #[cfg(feature = "cubecl-cpu")]
        DispatchDevice::Cube(CubeDevice::Cpu(_)) => KernelTarget::Cpu,
        #[cfg(feature = "wgpu")]
        DispatchDevice::Cube(CubeDevice::Wgpu(_)) => KernelTarget::Wgpu,
        #[cfg(feature = "cuda")]
        DispatchDevice::Cube(CubeDevice::Cuda(_)) => KernelTarget::Cuda,
        _ => KernelTarget::Portable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flex_is_a_cpu_execution_target() {
        assert_eq!(burn_device_target(&Device::flex()), KernelTarget::Cpu);
    }
}
