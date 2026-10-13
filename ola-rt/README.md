# OlaRT implementation workspace

This directory contains the executable portion of OlaRT. The repository root
contains the broader research notes and paper planning; this workspace contains
the interfaces and implementations on which experiments will depend.

The timestamped, repository-wide implementation snapshot is maintained in
[`../progress_report.md`](../progress_report.md).

OlaRT asks one systems question:

> Given the current workload, model, hardware, memory, cache state, and service
> objectives, what should an inference runtime do next?

The implementation establishes stable semantics for requests, runtime state,
lifecycle transitions, scheduling, serving, compute adapters, and repeatable
workloads. A narrow Llama path now provides the first real model execution
workload against which those abstractions can be tested.

## Crate boundaries

| Crate | Responsibility | Must not own |
| --- | --- | --- |
| `ola-core` | Backend-neutral domain types and identifiers | Queues, policy, Burn tensors |
| `ola-model` | Model catalog, artifact sources, architecture selection | Tensor execution, scheduling, downloads |
| `ola-runtime` | Model registry, request lifecycle, state, scheduling | HTTP, model kernels |
| `ola-burn` | Adapting Burn concepts to OlaRT interfaces | Scheduling or serving |
| `ola-observe` | Runtime event measurements and latency derivation | Lifecycle mutation, policy decisions |
| `ola-serve` | Transport-neutral service boundary | HTTP-framework details, execution policy |
| `ola-bench` | Deterministic workload construction, replay, summaries | Runtime production logic |
| `ola-ui` | Ratatui workbench and integration contracts | Model kernels, scheduling policy |

Dependencies point inward. `ola-core` knows nothing about the other crates.
`ola-model` describes available artifacts and selects an architecture factory.
The runtime tracks models that are actually resident and available for requests.
Serving, observation, and benchmarking call the runtime. The Burn adapter
implements a core interface without leaking Burn types into the runtime.

## Model selection

Model identity and architecture identity are deliberately separate:

```text
ModelCatalog
  │
  │ select ModelId
  ▼
ModelDefinition
  ├── ModelDescriptor
  │     ├── ModelId          e.g. tinyllama-chat
  │     └── ArchitectureId   e.g. llama
  └── ModelSource            e.g. local directory or Hub reference
        │
        ▼
ArchitectureRegistry
  │
  │ select by ArchitectureId
  ▼
ArchitectureFactory<M>
  │
  ▼
LoadedModel<M>
```

Consequently, `tinyllama-chat` and `llama-3.2-1b` can share one `llama`
factory, while `qwen2.5-0.5b` selects a different factory. Adding a model is a
catalog operation; adding a new architecture is an implementation operation.
The generic loaded type `M` lets Burn, a simulator, or a future backend reuse
the same selection machinery without type-erased tensors in `ola-core`.

For causal language models, factories can return `CausalLanguageModel`, a
backend-neutral bundle of tokenizer and shared model implementation. Each
request obtains a separate `GenerationSession`; mutable decode state such as a
KV cache therefore belongs to the request rather than the shared weights.

## Request lifecycle

The initial state machine is deliberately explicit:

```text
submit
  │
  ▼
Queued ───── cancel ────► Cancelled
  │
  │ scheduler selects
  ▼
Running ──── complete ──► Completed
  │
  ├───────── fail ──────► Failed
  │
  └───────── cancel ────► Cancelled
```

Terminal states cannot transition again. A scheduler only selects an ID from a
read-only `RuntimeState`; the runtime validates the selection and performs the
transition. This separation is important for research: experimental policies
may be replaced without granting them permission to corrupt runtime state.

## Time and repeatability

Core APIs do not read the wall clock. Callers supply a monotonic `Timestamp`
measured in microseconds relative to a runtime or workload epoch. This makes a
trace replay reproducible and permits simulated time without changing runtime
code. A future server will translate its monotonic clock into these timestamps.

## Foundation invariants

- Request IDs are assigned monotonically by one runtime.
- A request references a registered model before it can enter the queue.
- Every admitted input modality is supported by the selected model.
- Request inputs and generation parameters are validated at admission.
- Queue membership, active membership, status, and timing are changed together
  by runtime lifecycle methods.
- Lifecycle timestamps are monotonic; impossible negative intervals are rejected
  instead of contaminating experiment data.
- FIFO is a baseline, not a privileged scheduling implementation.
- Workload event order is validated before replay.
- Percentiles use the documented nearest-rank definition.
- Backend-specific devices remain behind `ComputeBackend`.
- Model IDs select artifacts; architecture IDs select implementations.
- Catalog membership does not imply that a model is loaded or runtime-resident.

## Current scope

Implemented:

- request and multimodal input representation;
- model metadata and registration;
- model catalog and backend-neutral architecture-factory selection;
- tokenizer, causal-model, and per-request generation-session contracts;
- queued/running/terminal lifecycle state;
- FIFO scheduling through a replaceable trait;
- explicit queue, active-set, memory, and timing state;
- deterministic fixed-size KV block allocation and fragmentation metrics;
- a synchronous multi-request executor with iteration-level stepping and
  aggregate live KV-memory accounting;
- page-rounded, memory-aware admission against model KV pools and optional
  device-wide budgets;
- replaceable iteration batch composition with auditable plans and a
  starvation-safe FIFO input-token budget;
- model-grouped prefill and decode execution with explicit tensor/individual
  records and real Burn Llama variable-length tensor paths;
- replaceable prompt-length bucketing with explicit dense-padding accounting;
- backend-neutral observation of iteration work, interval rates, cumulative
  counters, and terminal request latency boundaries;
- a transport-neutral generation service;
- a Ratatui workbench with background local Llama generation, KV benchmarks,
  and FIFO runtime experiments;
- deterministic fixed-workload generation and replay;
- basic latency summaries;
- a Burn device adapter;
- opt-in WGPU and native CUDA Burn backends, with WGPU inference numerically
  validated on the development GPU;
- strict parsing of compatible Hugging Face Llama configurations;
- Hugging Face tokenizer and single-file SafeTensors loading;
- a correctness-first Burn Llama decoder with grouped-query attention;
- configurable physical KV page storage backed by runtime block tables;
- isolated greedy generation sessions using cached single-token decode;
- executor-driven local UI generation with retained token and text results;
- a pinned external checkpoint and independent scalar conformance oracle.

Deliberately deferred:

- tensors in the public runtime API;
- fused accelerator attention, eviction, preemption, and prefix reuse;
- fused multi-request paged attention and preemption;
- probabilistic sampling;
- sharded and remotely materialized checkpoints;
- RoPE-scaled and tied-embedding Llama variants;
- asynchronous serving and HTTP;
- GPU telemetry and accelerator-specific fused kernels;
- persistence and distributed execution.

## Small Llama probe

The `ola-burn` example loads a local Hugging Face directory through the same
architecture registry used by future model selection:

```text
cargo run -p ola-burn --example run_llama -- \
  /path/to/model \
  "Hello" \
  8

# Opt-in WGPU execution
cargo run -p ola-burn --features wgpu --example run_llama -- \
  --backend wgpu /path/to/model "Hello" 8
```

See `crates/ola-burn/README.md` for supported artifact features, baseline
execution semantics, and the limits of its first paged KV cache. Its
`CONFORMANCE.md` pins the external checkpoint and independent scalar logit
oracle used as the numerical baseline for future optimized paths.

## Research extension rule

New policies should consume observable state and return decisions. They should
not embed experimental branches directly into the runtime. New measurements
must define their units and clock boundaries. New benchmark claims must record
the workload, model, hardware, software version, warm-up, repetitions, and
latency distribution.

## Verification

From the repository root, `make help` lists the maintained developer commands.
The complete default quality gate is:

```text
make verify
```

The underlying commands, when run directly from this directory, are:

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```
