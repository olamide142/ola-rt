#![doc = include_str!("../README.md")]

mod error;
mod metadata;
mod paged_attention;

#[cfg(feature = "cubecl")]
mod cubecl_extension;
#[cfg(feature = "cubecl")]
mod telemetry;

pub use error::KernelError;
pub use metadata::{KernelDescriptor, KernelOperation, KernelTarget};
pub use paged_attention::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvPage,
    PortablePagedAttention,
};

#[cfg(feature = "cubecl")]
pub use cubecl_extension::{CubePagedAttention, PagedAttentionOps};
#[cfg(feature = "cubecl")]
pub use telemetry::CubeclTelemetry;
#[cfg(feature = "cuda")]
pub use telemetry::cuda_telemetry;
#[cfg(feature = "wgpu")]
pub use telemetry::wgpu_telemetry;
