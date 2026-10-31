# Models

OlaRT runs **Llama- and Mistral-architecture** models from Hugging Face
directories (`config.json`, `tokenizer.json`, SafeTensors weights). Weights
are cast to **f32** on load, whatever the checkpoint stores, so a model needs
about **4 bytes per parameter** of memory: RAM for `--backend cpu`, GPU
memory for `wgpu` and `cuda`.

```bash
lab/lab fetch HuggingFaceTB/SmolLM2-135M-Instruct    # → ~/models/smollm2-135m-instruct
```

`fetch` downloads only what OlaRT loads, then reports the architecture, the
f32 size, and whether the config passes the loaders' checks.

## Tested

| Model | Arch | f32 size | Runs with | Notes |
| --- | --- | ---: | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | llama | 0.5 GiB | `ola`, `run_llama`; CPU and WGPU | bf16 checkpoint, tied embeddings. Identical greedy text on CPU and WGPU. |
| `Locutusque/TinyMistral-248M` | mistral | 0.9 GiB | `run_mistral`; CPU and WGPU | 32-token sliding window. Identical greedy text on CPU and WGPU. Small and repetitive; good for testing, not for text quality. |

## Should work, untested

| Model | Arch | f32 size | Fits on |
| --- | --- | ---: | --- |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | llama | ~1.4 GiB | CPU, 4 GB GPU |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama | ~4.1 GiB | CPU only on a 4 GB GPU machine |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | llama | ~6.8 GiB | CPU, or a GPU with 8 GB or more |

Run `lab/lab fetch` and read its verdict before relying on any of these.

## Will not load

- **Other architectures** (Qwen, Phi, Gemma, GPT-2, …): only `llama` and
  `mistral` are implemented.
- **Llama 3.x** (`rope_scaling` of type `llama3`): only linear RoPE scaling
  is supported; the loader rejects other kinds rather than run them wrongly.
  The official Llama repositories are also gated (they need a Hugging Face
  login, which `fetch` does not do).
- **Mistral with tied embeddings, biases, or RoPE scaling**: rejected.
- **`.bin`-only checkpoints**: SafeTensors only.
- **Mistral 7B and other 7B models**: about 29 GiB in f32; more than this
  machine's RAM or GPU. They need fp16/bf16 execution, which OlaRT does not
  have yet.

## Which tool loads which

| Tool | Llama | Mistral |
| --- | --- | --- |
| `ola serve`, `ola experiment run --model` (and `make lab-new MODEL=`) | yes | no |
| `run_llama` example (`make llama MODEL=…`) | yes | no |
| `run_mistral` example | no | yes |

```bash
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-burn --example run_mistral -- \
    ~/models/tinymistral-248m "The capital of France is" 24
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-burn --features wgpu \
    --example run_mistral -- --backend wgpu ~/models/tinymistral-248m "The capital of France is" 24
```

## Backends on this machine

The development laptop has a GTX 1650 (4 GB, Turing, compute 7.5).

- `cpu` always works.
- `wgpu` works through Vulkan; the first run of a process compiles shaders,
  so time a warmed-up run, not the first.
- `cuda` needs the CUDA toolkit with NVRTC, which is not installed (no
  `nvcc` or `libnvrtc`); install it to use `--features cuda`.
