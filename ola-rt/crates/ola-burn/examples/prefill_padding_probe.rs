//! Diagnoses differences between unpadded and right-padded batched prefill.
//!
//! For each prompt length it reports, at the last real position, the greedy
//! token of an unpadded single-row forward pass and of the same row inside a
//! right-padded batch, the top-1/top-2 logit margin, and the largest logit
//! difference. It then replays the `llama-prefill-strategies` flow (prefill
//! plus one decode) per request and as one padded batch, and prints any
//! request whose tokens differ.
//!
//! ```text
//! cargo run --release -p ola-burn --features cuda --example prefill_padding_probe -- \
//!     --backend cuda <artifact-directory> [lengths] [token]
//! ```

use std::{env, error::Error, io};

use burn::tensor::{Device, Int, Tensor, TensorData};
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor};
use ola_model::{ArchitectureRegistry, CausalLanguageModel, GenerationSession, ModelDefinition};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1).collect::<Vec<_>>();
    let mut backend = "cpu".to_owned();
    if arguments.first().map(String::as_str) == Some("--backend") {
        arguments.remove(0);
        backend = arguments.remove(0);
    }
    let directory = arguments.first().cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: prefill_padding_probe [--backend cpu|cuda] <artifact-directory> [lengths] [token]",
        )
    })?;
    let lengths = arguments
        .get(1)
        .map(String::as_str)
        .unwrap_or("5,7,9,15,17,31,33,63")
        .split(',')
        .map(|length| length.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let token = arguments
        .get(2)
        .map(|token| token.parse::<u32>())
        .transpose()?
        .unwrap_or(1);
    let device = match backend.as_str() {
        "cpu" => Device::flex(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported backend `{other}`"),
            )
            .into());
        }
    };
    let prompts = lengths
        .iter()
        .map(|&length| vec![token; length])
        .collect::<Vec<_>>();
    println!("backend={backend} lengths={lengths:?} token={token}");
    logits(&device, &directory, &prompts)?;
    generation(&device, &directory, &prompts)?;
    Ok(())
}

/// Compares last-position logits of each prompt alone against the same row in
/// one right-padded batch of every prompt.
fn logits(device: &Device, directory: &str, prompts: &[Vec<u32>]) -> Result<(), Box<dyn Error>> {
    let (config, module) = LlamaFactory::new(device.clone())
        .load_native(directory)?
        .into_parts();
    let vocabulary = config.vocab_size;
    let padded = prompts.iter().map(Vec::len).max().unwrap_or(0);
    let mut values = Vec::with_capacity(prompts.len() * padded);
    for prompt in prompts {
        values.extend(prompt.iter().map(|&token| i64::from(token)));
        values.resize(values.len() + padded - prompt.len(), 0);
    }
    let batch: Tensor<2, Int> =
        Tensor::from_data(TensorData::new(values, [prompts.len(), padded]), device);
    let batched = module.forward(batch);

    println!("\n[logits] length  single  batched  margin       max|diff|");
    for (index, prompt) in prompts.iter().enumerate() {
        let length = prompt.len();
        let input: Tensor<2, Int> = Tensor::from_data(
            TensorData::new(
                prompt.iter().map(|&token| i64::from(token)).collect(),
                [1, length],
            ),
            device,
        );
        let single = last_row(module.forward(input), 0, length, vocabulary);
        let row = last_row(batched.clone(), index, length, vocabulary);
        let (single_top, margin) = top_two(&single);
        let (row_top, _) = top_two(&row);
        let difference = single
            .iter()
            .zip(&row)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "[logits] {length:6}  {single_top:6}  {row_top:7}  {margin:10.6}  {difference:10.6}{}",
            if single_top == row_top {
                ""
            } else {
                "  <-- argmax differs"
            }
        );
    }
    Ok(())
}

/// Replays prefill plus one decode, per request and as one padded batch.
fn generation(
    device: &Device,
    directory: &str,
    prompts: &[Vec<u32>],
) -> Result<(), Box<dyn Error>> {
    let definition = ModelDefinition::local(
        ModelDescriptor::language("probe-llama", "llama"),
        directory.to_owned(),
    );
    let mut architectures = ArchitectureRegistry::<CausalLanguageModel>::default();
    architectures.register(LlamaFactory::new(device.clone()))?;
    let loaded = architectures.load(&definition)?;
    let model = loaded.inner();
    let generation = GenerationConfig {
        max_new_tokens: 2,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(0),
    };

    let mut individual = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mut session = model.create_session(generation.clone())?;
        let first = session.prefill(prompt)?;
        let second = session.decode(first.id)?;
        individual.push([first.id, second.id]);
    }

    let mut sessions = prompts
        .iter()
        .map(|_| model.create_session(generation.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut handles = sessions
        .iter_mut()
        .map(|session| session.as_mut() as &mut dyn GenerationSession)
        .collect::<Vec<_>>();
    let slices = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let firsts = model
        .prefill_batch(&mut handles, &slices)
        .ok_or("model does not support batched prefill")?
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let first_ids = firsts.iter().map(|token| token.id).collect::<Vec<_>>();
    let seconds = model
        .decode_batch(&mut handles, &first_ids)
        .ok_or("model does not support batched decode")?
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    println!("\n[tokens] length  individual        padded batch");
    let mut mismatches = 0;
    for (index, prompt) in prompts.iter().enumerate() {
        let batched = [first_ids[index], seconds[index].id];
        let differs = batched != individual[index];
        mismatches += usize::from(differs);
        println!(
            "[tokens] {:6}  {:16}  {:16}{}",
            prompt.len(),
            format!("{:?}", individual[index]),
            format!("{batched:?}"),
            if differs { "  <-- differs" } else { "" }
        );
    }
    println!("\nmismatched requests: {mismatches}/{}", prompts.len());
    Ok(())
}

fn last_row(logits: Tensor<3>, row: usize, length: usize, vocabulary: usize) -> Vec<f32> {
    logits
        .slice([row..row + 1, length - 1..length, 0..vocabulary])
        .into_data()
        .try_to_vec::<f32>()
        .expect("logits are f32")
}

/// Returns the argmax and the gap between the two largest logits.
fn top_two(values: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best.1 {
            second = best.1;
            best = (index, value);
        } else if value > second {
            second = value;
        }
    }
    (best.0, best.1 - second)
}
