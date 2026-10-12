# ola-burn

`ola-burn` is the adapter between OlaRT's backend-neutral runtime and Burn.

The foundation adapter owns a Burn device and exposes it through
`ola_core::ComputeBackend`. Concrete tensor execution stays in this crate; the
runtime, scheduler, serving boundary, and model catalog do not depend on Burn
tensor types.

## Initial Llama implementation

`ola_burn::llama` is the first architecture implementation. It consumes a local
directory in the common Hugging Face layout:

```text
artifact-directory/
├── config.json
├── tokenizer.json
└── model.safetensors
```

`LlamaFactory<B>` implements
`ArchitectureFactory<CausalLanguageModel>` for any compatible Burn backend
`B`. Registering that factory under `llama` does not register a particular
model. A `ModelDefinition` separately selects the artifact directory, allowing
TinyLlama and other compatible Llama checkpoints to share the implementation.

The load path is deliberately strict:

1. parse `config.json` into `LlamaConfig`;
2. validate all dimensions and implemented architectural features;
3. load `tokenizer.json` with Hugging Face Tokenizers;
4. initialize a Burn module whose names mirror Hugging Face checkpoint keys;
5. remap Hugging Face RMSNorm `weight` names to Burn `gamma`, then load
   `model.safetensors` with Burn's PyTorch adapter;
6. return a backend-neutral tokenizer/model bundle.

Strict validation matters for research. A non-Llama `model_type`, checkpoint
using RoPE scaling, tied embeddings, projection biases, a non-SiLU MLP, or
multiple SafeTensors shards is currently rejected rather than executed with
approximately similar semantics.

## Execution semantics

The decoder implements token embeddings, pre-normalized decoder blocks,
grouped-query causal self-attention, Llama's split-half rotary embeddings,
SwiGLU MLPs, final RMSNorm, and the language-model head.

The baseline is intentionally simple and easy to inspect:

- batch size is one at the `GenerationSession` boundary;
- only greedy decoding (`temperature = 0`) is accepted;
- prefill writes compact keys and values into fixed-size per-layer pages;
- runtime block tables map each sequence's logical blocks to physical pages;
- decode processes only the newest token and writes into its assigned page;
- request metrics report physical page bytes attributed to the sequence,
  including internal fragmentation;
- capacity reservation is atomic, and released physical block IDs are reused;
- decode attention traverses pages directly and combines their scores and
  values with a numerically stable online softmax;
- the session enforces call order, vocabulary bounds, context length, EOS, and
  `max_new_tokens`;
- Hub references are descriptive only and must first be materialized locally.

The unchanged full-sequence path is the cache's numerical oracle. Unit tests
compare page-native decode logits with complete recomputation at every
generated position and across several page sizes. This gives later batching,
sampling, quantization, fusion, and accelerator work a small reference
implementation to compare against. In particular, Llama-specific RoPE is
implemented explicitly because Burn's general rotary module pairs adjacent
dimensions, while Hugging Face Llama pairs the two halves of each attention
head.

The default page pool has capacity for one full model context and uses 16-token
blocks. Experiments can override both values before loading an artifact:

```rust,no_run
use burn::backend::NdArray;
use ola_burn::llama::{LlamaFactory, LlamaKvCacheConfig};

let factory = LlamaFactory::<NdArray<f32>>::new(Default::default())
    .with_kv_cache(LlamaKvCacheConfig::new(8_192, 32));
# let _ = factory;
```

Capacity is shared by the live sessions for one loaded model. Physical tensors
are materialized lazily and retained at the pool's high-water mark so released
block IDs can be reused without changing their physical identity.

Loaded Llama artifacts publish the same bytes-per-token, page size, and total
pool capacity as a backend-neutral `KvCacheProfile`. The runtime's strict
admission policy therefore reserves against the physical Burn pool before
creating a session rather than discovering overcommit during prefill.

## Running a local artifact

From the implementation workspace:

```text
cargo run -p ola-burn --example run_llama -- \
  /path/to/hugging-face-llama \
  "Hello" \
  8
```

The default is Burn's CPU `NdArray<f32>` backend. WGPU is an opt-in path that
uses the platform graphics API and is the simplest GPU route on a machine
without a CUDA toolkit:

```text
cargo run -p ola-burn --features wgpu --example run_llama -- \
  --backend wgpu /path/to/hugging-face-llama "Hello" 8
```

Native CUDA is also feature-gated and requires both a compatible NVIDIA driver
and a local CUDA toolkit/NVRTC installation:

```text
cargo run -p ola-burn --features cuda --example run_llama -- \
  --backend cuda /path/to/hugging-face-llama "Hello" 8
```

The example selects the `llama` factory through `ArchitectureRegistry`,
tokenizes the prompt, performs greedy prefill and decode, and prints token IDs,
session metrics, the concrete backend type, and decoded text. A random test
checkpoint validates mechanics but produces meaningless text; a trained
checkpoint such as TinyLlama is required for meaningful output and needs enough
memory for its full-precision representation.

The WGPU path has been checked on the development GTX 1650 with the pinned
fixture: its selected tokens match CPU exactly, and sampled logits remain
within the documented accelerator tolerance. This is correctness evidence, not
a throughput claim; first-run compilation and shader warm-up must be separated
from steady-state measurements.

## Evidence expectations

Changes to this implementation should retain a correctness comparison against a
known reference implementation for fixed inputs and checkpoint revision. New
performance paths must record backend, device, dtype, context length, generated
length, warm-up, repetitions, and whether compilation/autotuning is included.
Throughput results must state whether they use full-sequence or page-native
cached decoding. The current attention path uses portable Burn operations; it
must not be presented as a fused accelerator kernel.

The pinned fixture, independent scalar oracle, numerical tolerance, and manual
test command are specified in [`CONFORMANCE.md`](CONFORMANCE.md). That protocol
is the reference baseline for future KV-cache work.
