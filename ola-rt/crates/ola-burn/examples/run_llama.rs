//! Loads one local Hugging Face Llama artifact and performs greedy generation.
//!
//! This is an executable research probe, not a production CLI. It exercises
//! the same catalog-independent architecture registry and execution traits that
//! a future server will use while keeping the artifact path and prompt explicit.

use std::{env, error::Error, io};

use burn::backend::NdArray;
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor};
use ola_model::{ArchitectureRegistry, CausalLanguageModel, ModelDefinition};

type CpuBackend = NdArray<f32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let artifact_directory = arguments.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: cargo run -p ola-burn --example run_llama -- <artifact-directory> [prompt] [max-new-tokens]",
        )
    })?;
    let prompt = arguments.next().unwrap_or_else(|| "Hello".into());
    let max_new_tokens = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);

    let definition = ModelDefinition::local(
        ModelDescriptor::language("selected-llama", "llama"),
        artifact_directory,
    );
    let mut architectures = ArchitectureRegistry::<CausalLanguageModel>::default();
    architectures.register(LlamaFactory::<CpuBackend>::new(Default::default()))?;
    let loaded = architectures.load(&definition)?;

    let prompt_tokens = loaded.inner().tokenizer().encode(&prompt, true)?;
    let mut session = loaded.inner().create_session(GenerationConfig {
        max_new_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(0),
    })?;
    let first = session.prefill(&prompt_tokens)?;
    let mut generated = vec![first.id];
    let mut previous = first;
    for _ in 1..max_new_tokens {
        if previous.is_end_of_sequence {
            break;
        }
        previous = session.decode(previous.id)?;
        generated.push(previous.id);
    }

    let text = loaded.inner().tokenizer().decode(&generated, true)?;
    println!("prompt_tokens={prompt_tokens:?}");
    println!("generated_tokens={generated:?}");
    println!("metrics={:?}", session.metrics());
    println!("text={text:?}");
    Ok(())
}
