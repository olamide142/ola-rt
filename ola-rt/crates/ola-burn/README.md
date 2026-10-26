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

`LlamaFactory` implements
`ArchitectureFactory<CausalLanguageModel>` for a runtime-selected Burn device.
Burn 0.22 dispatches tensors using `Device`, so the same model type can run on
CPU, WGPU, or CUDA without a backend type parameter. Registering that factory
under `llama` does not register a particular
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

- the direct `GenerationSession` API processes one request at a time;
- the shared model can tensor-batch compatible variable-length prefills and
  one-token decode operations;
- only greedy decoding (`temperature = 0`) is accepted;
- prefill writes compact keys and values into fixed-size per-layer pages;
- resumable prefill atomically reserves each bounded segment, advances its
  request-local cursor, and selects a token only after the final segment;
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
use burn::tensor::Device;
use ola_burn::llama::{LlamaFactory, LlamaKvCacheConfig};

let factory = LlamaFactory::new(Device::flex())
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

## Tensor-batched prefill and decode

Variable-length prompt batches are right-padded for dense transformer work.
Causal attention ensures that a real prompt position cannot observe its padded
suffix. Each row's final real-token logits are selected independently, and
only real K/V positions are written into that request's physical pages. Cache
reservation uses the request-specific prompt lengths and succeeds or fails
atomically for the complete batch.

The optional model-level batch path processes one cached token for several
sessions sharing the loaded artifact. It batches embeddings, Q/K/V projections,
attention-output projection, normalization, MLP layers, vocabulary projection,
argmax, and the host token read. RoPE application and page traversal remain
request-local because live sequences may occupy different absolute positions
and different numbers of physical pages.

Each session keeps its own logical block table and generation state. KV growth
uses `KvBlockManager::append_batch`, which validates every sequence and combined
capacity before mutating any allocation. Tests compare batched output and
session metrics with independent prefill and decode, including prompts with
different lengths and decode at different positions. Executor integration
tests verify that compatible Llama requests report real `Tensor` batches for
both phases. The individual paths remain numerical and lifecycle oracles.

## Resumable prefill

Llama sessions accept bounded prompt segments through the architecture-neutral
`prefill_chunk` contract. A segment reserves all of its pages before writing
any layer data. Attention then advances one prompt position at a time over the
populated page prefix, preserving exact causal position and allowing the
runtime to yield between segments. Capacity failure leaves the accepted-token
cursor and session metrics unchanged. Tests compare final tokens, metrics, and
the following cached decode against complete-prompt prefill.

Compatible sessions can share this path at unequal absolute positions and with
different segment lengths. The implementation atomically reserves all rows,
then tensor-batches the active row subset at each segment offset while each row
attends only through its newly populated absolute position.

## Mistral implementation

`ola_burn::mistral` implements the Mistral architecture family. It follows the same
Hugging Face directory layout and strict validation principles as Llama, with
support for:

- `model_type: "mistral"`;
- Grouped-query attention with arbitrary valid head ratios;
- Split-half rotary position embeddings;
- SwiGLU feed-forward networks with SiLU activation;
- Optional sliding-window attention (`sliding_window`), enforcing attention spans
  in both full-sequence forward passes and block-paged decode attention;
- Block-paged KV cache pools (`MistralKvPool`, `MistralKvCache`) with prefix caching,
  sub-window reads, and atomic variable-length multi-request reservations;
- Resumable prompt chunking (`prefill_chunk`) and tensor-batched decode (`decode_batch`);
- An executable probe via `examples/run_mistral.rs`.

```rust,no_run
use burn::tensor::Device;
use ola_burn::mistral::{MistralFactory, MistralKvCacheConfig};

let factory = MistralFactory::new(Device::flex())
    .with_kv_cache(MistralKvCacheConfig::new(8_192, 32));
# let _ = factory;
```

## Running a local artifact

From the implementation workspace:

```text
cargo run -p ola-burn --example run_llama -- \
  /path/to/hugging-face-llama \
  "Hello" \
  8
```

The default is Burn's pure-Rust CPU Flex device. WGPU is an opt-in path that
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
session metrics, and decoded text. A random test
checkpoint validates mechanics but produces meaningless text; a trained
checkpoint such as TinyLlama is required for meaningful output and needs enough
memory for its full-precision representation.

The WGPU path has been checked on the development GTX 1650 with the pinned
fixture: its selected tokens match CPU exactly, and sampled logits remain
within the documented accelerator tolerance. This is correctness evidence, not
a throughput claim; first-run compilation and shader warm-up must be separated
from steady-state measurements.

With `--features wgpu` or `--features cuda`, single-token decode attempts the
CubeCL paged-attention implementation through Burn's CubeCL/Fusion extension.
It falls back to portable Burn attention on other devices and unsupported
dtypes. The current CubeCL kernel is a bring-up implementation, not a tuned
FlashAttention kernel; benchmark it before treating it as a performance win.

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
