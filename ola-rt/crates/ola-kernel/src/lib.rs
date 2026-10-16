#![doc = include_str!("../README.md")]

mod error;
mod metadata;
mod paged_attention;

#[cfg(feature = "cubecl")]
mod cubecl_extension;

pub use error::KernelError;
pub use metadata::{KernelDescriptor, KernelOperation, KernelTarget};
pub use paged_attention::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvPage,
    PortablePagedAttention,
};

#[cfg(feature = "cubecl")]
pub use cubecl_extension::{CubePagedAttention, PagedAttentionOps};
