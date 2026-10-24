#![doc = include_str!("../README.md")]

mod capability;
mod device;
mod error;
mod metadata;
mod metrics;
mod paged_attention;
mod registry;

#[cfg(feature = "cubecl")]
mod cubecl_extension;
#[cfg(feature = "cubecl")]
mod telemetry;

pub use capability::{
    KernelCapability, KernelDType, KernelIncompatibility, KernelRequirements, PagedAttentionProfile,
};
pub use device::burn_device_target;
pub use error::KernelError;
pub use metadata::{KernelDescriptor, KernelOperation, KernelTarget};
pub use metrics::{KernelExecutionMetrics, KernelExecutionSnapshot};
pub use paged_attention::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvBlockTable,
    PagedKvBlocks, PagedKvPage, PortablePagedAttention,
};
pub use registry::{
    KernelFallbackReason, KernelRegistry, KernelRejection, KernelSelection, PagedAttentionProvider,
};

#[cfg(feature = "cubecl")]
pub use cubecl_extension::{CubePagedAttention, PagedAttentionOps};
#[cfg(feature = "cubecl")]
pub use telemetry::CubeclTelemetry;
#[cfg(feature = "cuda")]
pub use telemetry::cuda_telemetry;
#[cfg(feature = "wgpu")]
pub use telemetry::wgpu_telemetry;
