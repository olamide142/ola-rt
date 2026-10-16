# Llama conformance protocol

This document defines the first numerical conformance check for OlaRT's Burn
Llama implementation. It distinguishes three different claims:

- unit tests establish local invariants and tensor shapes;
- the external fixture test establishes checkpoint-to-logit compatibility;
- generation from a random model establishes mechanics, not language quality.

Passing this protocol does not establish performance, numerical equivalence on
every backend or dtype, tokenizer chat-template compatibility, or support for
all configurations in the Llama family.

## Pinned artifact

The fixture is the public Hugging Face repository
`hf-internal-testing/tiny-random-LlamaForCausalLM` at immutable revision:

```text
9fb191250dd56d0ba7ec9785a025ed29c03d5998
```

Files used by the loader and their SHA-256 digests are:

```text
5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382  config.json
9cf973f7249a063159ed9c5e7eb05e5d15289b7c5571ae4e1d7315b2189b5831  tokenizer.json
49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992  model.safetensors
```

The checkpoint has 1,032,272 F32 parameters. It is small enough for a scalar
oracle and is deliberately random: it tests architecture mechanics without
making a claim about useful generated text.

Materialize it outside the repository so third-party model data is not silently
vendored into source control:

```text
revision=9fb191250dd56d0ba7ec9785a025ed29c03d5998
fixture=/tmp/ola-rt-tiny-random-llama
mkdir -p "$fixture"
curl -fL -o "$fixture/config.json" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$revision/config.json"
curl -fL -o "$fixture/tokenizer.json" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$revision/tokenizer.json"
curl -fL -o "$fixture/model.safetensors" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$revision/model.safetensors"
sha256sum "$fixture/config.json" "$fixture/tokenizer.json" "$fixture/model.safetensors"
```

The shell variables above are documentation conveniences, not requirements of
the runtime. Resolve them carefully before deleting or overwriting anything.

## Independent oracle

`tools/llama_scalar_reference.py` reads F32 SafeTensors using only Python's
standard library. It implements unbatched Llama with scalar loops and F64
accumulation:

- Hugging Face `[output, input]` linear weights;
- pre-normalized residual blocks;
- split-half rotary embeddings;
- grouped-query causal attention;
- SwiGLU feed-forward layers;
- final RMSNorm and language-model head.

It does not import Burn, PyTorch, NumPy, Transformers, or the Rust
implementation. This independence is useful because a second tensor framework
could reproduce the same orientation or reshape mistake. The oracle is slow by
design and is only appropriate for tiny fixtures.

Generate the committed observations with:

```text
python3 tools/llama_scalar_reference.py \
  /tmp/ola-rt-tiny-random-llama \
  1 15043 1789
```

The input is the tokenizer output for `Hello` followed by the first expected
greedy token. The oracle records the first sixteen logits at all three input
positions and selects token IDs `5971`, `1789`, and `19053`. Checking every
position exercises causal behavior; checking later positions exercises rotary
position handling rather than only the position-zero special case.

## Running the Rust comparison

The test is ignored in the default suite because it depends on an external
artifact. After verifying all three hashes, run:

```text
OLA_LLAMA_CONFORMANCE_DIR=/tmp/ola-rt-tiny-random-llama \
  cargo test -p ola-burn \
  llama::factory::tests::matches_pinned_reference_logits -- --ignored --exact
```

Burn's F32 `Flex` CPU output (NdArray before the Burn 0.22 migration) is
compared with the scalar F64 oracle using an absolute tolerance of `2e-5`.
The tolerance covers accumulation-order and elementary-function differences
while remaining small relative to the observed logit magnitudes. Argmax token IDs must match exactly.

## Change-control rule

Do not update expected logits merely because the test fails. First identify
whether the change is due to an intended equation, backend, dtype, fixture, or
tolerance change. Any accepted update must record:

1. the exact artifact revision and hashes;
2. the oracle command and input token IDs;
3. the backend, dtype, and tolerance;
4. why the old expectation was no longer authoritative.

KV-cache work must additionally compare cached one-token decode logits with
this full-sequence baseline at every generated position. Matching only final
text is insufficient because argmax can hide significant logit drift.

## Planned multi-request execution

`llama::factory::tests::planned_multi_request_execution_matches_single_request_oracle`
runs five prompts of different lengths and output limits through one planned
executor with three slots, two-token prefill chunks, power-of-two grouping,
and a four-token iteration budget. Every request's generated token IDs must
equal those it produces when run alone. The test also asserts that it
exercised tensor invocations, continued prefill, budget deferral, and
predicted slot reuse, and that Llama never fell back from a declared tensor
hook. It runs with the other fixture tests (`make test-fixture`).
