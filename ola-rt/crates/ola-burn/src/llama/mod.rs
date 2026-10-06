//! Correctness-first Burn implementation of the Hugging Face Llama format.
//!
//! This module intentionally starts with a narrow, inspectable execution path:
//! one request at a time, greedy decoding, and a request-local contiguous KV
//! cache. The original full-sequence path remains available as a correctness
//! oracle while cached decoding processes only the newest token.

mod config;
mod factory;
mod model;
mod session;
mod tokenizer;

pub use config::{LlamaConfig, LlamaConfigError};
pub use factory::{LlamaFactory, LlamaLoadError, LoadedLlama};
pub use model::LlamaForCausalLM;
pub use tokenizer::HuggingFaceTokenizer;
