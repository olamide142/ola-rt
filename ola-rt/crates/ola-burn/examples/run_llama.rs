//! Loads one local Hugging Face Llama artifact and performs greedy generation.
//!
//! This is an executable research probe, not a production CLI. It exercises
//! the same catalog-independent architecture registry and execution traits that
//! a future server will use while keeping the artifact path and prompt explicit.

use std::{env, error::Error, io};

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use burn::{backend::NdArray, tensor::backend::Backend};
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor};
use ola_model::{ArchitectureRegistry, CausalLanguageModel, ModelDefinition};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let mut backend = "cpu".to_owned();
    let first = arguments.next();
    let artifact_directory = if first.as_deref() == Some("--backend") {
        backend = arguments.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "--backend requires cpu or cuda",
            )
        })?;
        arguments.next()
    } else {
        first
    };
    let artifact_directory = artifact_directory.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: cargo run -p ola-burn --example run_llama [--features wgpu|cuda] -- [--backend cpu|wgpu|cuda] <artifact-directory> [prompt] [max-new-tokens]",
        )
    })?;
    let prompt = arguments.next().unwrap_or_else(|| "Hello".into());
    let max_new_tokens = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);

    match backend.as_str() {
        "cpu" => run::<NdArray<f32>>(
            Default::default(),
            artifact_directory,
            prompt,
            max_new_tokens,
        ),
        #[cfg(feature = "cuda")]
        "cuda" => run::<Cuda<f32, i32>>(
            Default::default(),
            artifact_directory,
            prompt,
            max_new_tokens,
        ),
        #[cfg(feature = "wgpu")]
        "wgpu" => run::<Wgpu<f32, i32>>(
            Default::default(),
            artifact_directory,
            prompt,
            max_new_tokens,
        ),
        #[cfg(not(feature = "cuda"))]
        "cuda" => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "CUDA support is not compiled; rerun with --features cuda",
        )
        .into()),
        #[cfg(not(feature = "wgpu"))]
        "wgpu" => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "WGPU support is not compiled; rerun with --features wgpu",
        )
        .into()),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown backend `{other}`; expected cpu, wgpu, or cuda"),
        )
        .into()),
    }
}

fn run<B>(
    device: B::Device,
    artifact_directory: String,
    prompt: String,
    max_new_tokens: usize,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + 'static,
    B::Device: Send + Sync,
{
    let definition = ModelDefinition::local(
        ModelDescriptor::language("selected-llama", "llama"),
        artifact_directory,
    );
    let mut architectures = ArchitectureRegistry::<CausalLanguageModel>::default();
    architectures.register(LlamaFactory::<B>::new(device))?;
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
    println!("backend={}", std::any::type_name::<B>());
    println!("text={text:?}");
    Ok(())
}
