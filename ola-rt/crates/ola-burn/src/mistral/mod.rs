//! Correctness-first Burn implementation of the Hugging Face Mistral format.
//!
//! This module provides a complete execution path for Mistral models, including
//! grouped-query attention, rotary embeddings, SwiGLU feed-forward networks,
//! optional sliding-window attention, and a block-paged KV cache.

mod config;
mod factory;
mod kv_cache;
mod model;
mod session;
mod tokenizer;

pub use config::{MistralConfig, MistralConfigError};
pub use factory::{LoadedMistral, MistralFactory, MistralLoadError};
pub use kv_cache::MistralKvCacheConfig;
pub use model::MistralForCausalLM;
pub use tokenizer::HuggingFaceTokenizer;
