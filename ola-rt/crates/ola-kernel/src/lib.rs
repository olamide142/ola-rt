#![doc = include_str!("../README.md")]

mod error;
mod metadata;
mod paged_attention;

pub use error::KernelError;
pub use metadata::{KernelDescriptor, KernelOperation, KernelTarget};
pub use paged_attention::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvPage,
    PortablePagedAttention,
};
