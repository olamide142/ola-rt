//! Correctness-first Burn implementation of the Hugging Face Llama format.
//!
//! This module intentionally starts with a narrow, inspectable execution path:
//! greedy decoding and a block-paged KV cache. Individual sessions remain the
//! correctness oracle, while compatible variable-length prefill and cached
//! decode operations can share model tensor work without sharing request-local
//! cache state.

mod config;
mod factory;
mod kv_cache;
mod model;
mod session;
mod tokenizer;

pub use config::{LlamaConfig, LlamaConfigError};
pub use factory::{LlamaFactory, LlamaLoadError, LoadedLlama};
pub use kv_cache::LlamaKvCacheConfig;
pub use model::LlamaForCausalLM;
pub use tokenizer::HuggingFaceTokenizer;
