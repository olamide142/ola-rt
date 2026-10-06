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
- prefill stores one compact key/value tensor per decoder layer;
- decode processes only the newest token and appends its keys and values;
- request metrics report the cache's actual element storage in bytes;
- cache growth currently concatenates contiguous tensors, so it still copies
  existing cache contents and has no capacity reservation or paging;
- the session enforces call order, vocabulary bounds, context length, EOS, and
  `max_new_tokens`;
- Hub references are descriptive only and must first be materialized locally.

The unchanged full-sequence path is the cache's numerical oracle. Unit tests
compare cached decode logits with complete recomputation at every generated
position. This gives later paged allocation, batching, sampling, quantization,
and accelerator work a small reference implementation to compare against. In
particular, Llama-specific RoPE is implemented explicitly because Burn's
general rotary module pairs adjacent dimensions, while Hugging Face Llama pairs
the two halves of each attention head.

## Running a local artifact

From the implementation workspace:

```text
cargo run -p ola-burn --example run_llama -- \
  /path/to/hugging-face-llama \
  "Hello" \
  8
```

The example uses Burn's CPU `NdArray<f32>` backend, selects the `llama` factory
through `ArchitectureRegistry`, tokenizes the prompt, performs greedy prefill
and decode, and prints token IDs, session metrics, and decoded text. A random
test checkpoint validates mechanics but produces meaningless text; a trained
checkpoint such as TinyLlama is required for meaningful output and needs enough
memory for its full-precision representation.

## Evidence expectations

Changes to this implementation should retain a correctness comparison against a
known reference implementation for fixed inputs and checkpoint revision. New
performance paths must record backend, device, dtype, context length, generated
length, warm-up, repetitions, and whether compilation/autotuning is included.
Throughput results must state whether they use full-sequence or cached decoding;
the current contiguous cache must not be presented as paged allocation.

The pinned fixture, independent scalar oracle, numerical tolerance, and manual
test command are specified in [`CONFORMANCE.md`](CONFORMANCE.md). That protocol
is the reference baseline for future KV-cache work.
