#![doc = include_str!("../README.md")]

use std::{any::type_name, marker::PhantomData};

use burn::tensor::backend::Backend;
use ola_core::ComputeBackend;

pub mod llama;

/// Associates a Burn backend and device with the backend-neutral runtime.
#[derive(Clone, Debug)]
pub struct BurnComputeBackend<B: Backend> {
    device: B::Device,
    backend: PhantomData<fn() -> B>,
}

impl<B: Backend> BurnComputeBackend<B> {
    /// Creates an adapter that owns the selected Burn device.
    pub const fn new(device: B::Device) -> Self {
        Self {
            device,
            backend: PhantomData,
        }
    }

    /// Returns the native Burn device for backend-specific model setup.
    pub fn burn_device(&self) -> &B::Device {
        &self.device
    }
}

impl<B> ComputeBackend for BurnComputeBackend<B>
where
    B: Backend + 'static,
    B::Device: Clone + std::fmt::Debug + Send + Sync + 'static,
{
    type Device = B::Device;

    fn name(&self) -> &'static str {
        type_name::<B>()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use ola_core::ComputeBackend;

    use super::BurnComputeBackend;

    #[test]
    fn exposes_burn_through_the_compute_boundary() {
        let adapter = BurnComputeBackend::<NdArray>::new(Default::default());

        assert!(adapter.name().contains("NdArray"));
        let _ = adapter.device();
    }
}
