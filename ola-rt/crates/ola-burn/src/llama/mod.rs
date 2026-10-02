//! Correctness-first Burn implementation of the Hugging Face Llama format.
//!
//! This module intentionally starts with a narrow, inspectable execution path:
//! one request at a time, full-sequence recomputation, and greedy decoding. The
//! absence of a KV cache is represented honestly in session metrics instead of
//! hiding an experimental cache behind the first implementation.

mod config;
mod factory;
mod model;
mod session;
mod tokenizer;

pub use config::{LlamaConfig, LlamaConfigError};
pub use factory::{LlamaFactory, LlamaLoadError};
pub use model::LlamaForCausalLM;
pub use tokenizer::HuggingFaceTokenizer;
