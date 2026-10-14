/// Stable operation family implemented by a kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum KernelOperation {
    /// Single-token attention over logical KV-cache pages.
    PagedAttentionDecode,
}

/// Hardware family targeted by a kernel implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum KernelTarget {
    /// Backend-portable tensor operations used as a correctness baseline.
    Portable,
    /// Host CPU implementation.
    Cpu,
    /// WebGPU implementation, potentially authored with CubeCL.
    Wgpu,
    /// Native CUDA implementation.
    Cuda,
}

/// Stable metadata describing one kernel implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelDescriptor {
    name: &'static str,
    operation: KernelOperation,
    target: KernelTarget,
}

impl KernelDescriptor {
    /// Creates a descriptor.
    pub const fn new(name: &'static str, operation: KernelOperation, target: KernelTarget) -> Self {
        Self {
            name,
            operation,
            target,
        }
    }

    /// Returns the stable implementation name used in diagnostics and reports.
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Returns the operation family.
    pub const fn operation(self) -> KernelOperation {
        self.operation
    }

    /// Returns the target hardware family.
    pub const fn target(self) -> KernelTarget {
        self.target
    }
}
