# ola-model

`ola-model` selects model artifacts and architecture implementations without
depending on a tensor backend.

The crate distinguishes four concepts that are often accidentally coupled:

1. `ModelId` identifies a concrete artifact visible to users and requests.
2. `ArchitectureId` identifies the code capable of interpreting that artifact.
3. `ModelSource` records where configuration, tokenizer, and weights originate.
4. `ArchitectureFactory<M>` constructs backend- or experiment-specific type
   `M` from a selected definition.

## Selection flow

```text
ModelId
  ↓
ModelCatalog::get
  ↓
ModelDefinition { descriptor, source }
  ↓ descriptor.architecture
ArchitectureRegistry::load
  ↓
ArchitectureFactory<M>::load
  ↓
LoadedModel<M>
```

The registry is generic over `M`. A future `LlamaFactory<BurnBackend>` can
produce an executable Burn model, while tests can produce a small stub and a
simulator can produce a cost model. No backend tensor types enter `ola-core`.
When different factories produce different concrete Rust types, `M` should be a
shared execution enum or a boxed inference trait defined at the execution
boundary; the selection mechanism itself does not require type-specific cases.

## Causal-language execution contract

The first shared execution type is `CausalLanguageModel`:

```text
CausalLanguageModel
  ├── Tokenizer                 shared, immutable, thread-safe
  └── CausalModel               shared weights, thread-safe
         │
         └── create_session()
                  ↓
            GenerationSession   one per request, mutable
              ├── prefill()
              ├── prefill_chunk() → Pending | Token
              ├── decode()
              └── metrics()
```

The separation encodes an important ownership rule. Model weights and tokenizer
state may be shared across requests; sequence position, generated-token count,
and KV-cache allocations may not. A concrete Burn Llama implementation can
store weights behind an `Arc` and clone that handle into each session.

The contract exchanges token IDs and generated tokens, not backend tensors.
Architecture implementations retain control over device-resident logits,
sampling, and cache representation. This avoids forcing large vocabulary logits
through host memory merely to satisfy a generic interface. `GenerationConfig`
is bound when a session is created, so sampling behavior and RNG seed cannot
change accidentally in the middle of one sequence. The default seed is fixed
for reproducible foundation tests; production callers may explicitly request
implementation-selected entropy.

`SessionMetrics` exposes the minimum state needed by later scheduling and
observability work: prompt tokens, generated tokens, and KV-cache bytes. It is a
snapshot, not a promise about a particular cache layout.

Models explicitly advertise resumable prefill support. The compatibility
default accepts only a final segment and delegates to complete-prompt
`prefill`; this prevents a runtime policy from silently splitting prompts for
an architecture whose session state cannot resume safely. A pending segment
updates accepted prompt/KV metrics but cannot emit or count an output token.
An optional model-level chunk-batch hook accepts different chunk lengths,
absolute positions, and final-segment flags while retaining isolated sessions.

An architecture may also attach a `KvCacheProfile` to the loaded model. Unlike
live metrics, this describes physical planning facts known before a session
starts: bytes per token, token positions per page, and total shared pool
capacity. The runtime uses that profile for conservative admission without
depending on Burn tensors or Llama-specific configuration types.

## Availability versus residency

Catalog membership means an artifact is configured and selectable. It does not
mean its files have been downloaded, its weights fit in memory, or an instance
is loaded. `ola-runtime::ModelRegistry` continues to represent models that are
resident and available for request admission. Loading code should register a
runtime descriptor only after successful construction and resource allocation.

## Source semantics

`ModelSource` is descriptive. This crate performs no filesystem access and no
network access. A factory decides how to resolve its source, and future policy
can govern downloading, caching, revision pinning, or offline-only operation.
For reproducible experiments, remote definitions should pin an immutable
revision rather than relying on a moving default branch.

## Extension rule

Adding another model using an existing architecture adds a catalog definition.
Adding a new architecture adds one factory implementation. Neither operation
requires modifying the registry or runtime scheduler.

See `examples/select_model.rs` for an executable specification in which two
Llama artifacts share one factory and a Qwen artifact selects another.
