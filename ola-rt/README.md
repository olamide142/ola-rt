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
| `ola-core` | Backend-neutral domain types, identifiers, timestamps, configs | Queues, policy, Burn tensors |
| `ola-model` | Model catalog, artifact sources, architecture selection | Tensor execution, scheduling, downloads |
| `ola-runtime` | Execution engine, lifecycle state, execution policies, plans, paged KV pools | Sockets, HTTP, model weights |
| `ola-kernel` | Replaceable tensor-kernel contracts, portable references, CubeCL kernels | Scheduling, model weights, KV allocation |
| `ola-burn` | Adapting Burn primitives to OlaRT (Llama, Mistral, devices) | Scheduling, admission, serving |
| `ola-observe` | Runtime event measurements, traces, latency derivation | Lifecycle mutation, policy decisions |
| `ola-serve` | HTTP service boundary (SSE streaming, OpenAI-compatible chat completions) | Execution policy, tensor manipulation |
| `ola-bench` | Deterministic workload construction, virtual-time simulation, bootstrap statistics | Runtime production state |
| `ola-cluster` | Sans-IO cluster coordination, worker summaries, routing policies, MILP router | Raw sockets, wall-clock time, kernel execution |
| `ola-net` | Tokio byte stream wire protocol, control and binary session frames | Cluster policy, model weights |
| `ola-cli` | Unified CLI (`ola experiment run`, `compare`, `serve`) | Direct runtime mutation |
| `ola-ui` | Ratatui terminal workbench and experiment inspector | Model kernels, scheduling policy |

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
- queued/running/terminal lifecycle state machine;
- pluggable schedulers: FIFO, Priority, Earliest Deadline First (EDF), Shortest Job First (SJF), and conservative Backfill;
- memory-aware admission: worst-case KV reservation and quantile-based headroom admission under remaining-output beliefs;
- preemption and recovery under KV memory pressure with watermark admission and recompute semantics;
- explicit queue, active-set, memory, and timing state via immutable `PlanningSnapshot`;
- auditable `ExecutionPlan` with strict `validate` invariant checking before iteration execution;
- deterministic fixed-size KV block allocation, page tables, and fragmentation metrics;
- synchronous multi-request executor with iteration-level stepping and aggregate live KV-memory accounting;
- model-grouped prefill and decode execution with explicit tensor/individual records and real Burn Llama & Mistral variable-length tensor paths;
- bounded resumable prefill with request-local cursors, shared prefill/decode token budgets, progress events, atomic segment reservation, and cancellation-safe KV release;
- tensor-batched resumable chunks with variable lengths and unequal absolute positions;
- replaceable prompt-length bucketing with explicit dense-padding accounting and a workload-aware cost policy balancing padding against invocation count;
- switching cost modeling, inertia penalty, and dual-hysteresis `ReplayOptimizer` for online adaptive token budget optimization;
- backend-neutral observation of iteration work, interval rates, cumulative counters, and terminal request latency boundaries;
- replaceable kernel contracts with validated paged-attention inputs, portable online-softmax reference, and CubeCL paged attention;
- opt-in WGPU and native CUDA Burn backends, with WGPU inference numerically validated on development GPUs;
- strict parsing of compatible Hugging Face Llama and Mistral configurations;
- Hugging Face tokenizer and single-file SafeTensors loading;
- correctness-first Burn Llama and Mistral decoders with grouped-query attention;
- configurable physical KV page storage backed by runtime block tables;
- isolated greedy generation sessions using cached single-token decode;
- transport-neutral generation service and asynchronous HTTP serving with SSE streaming (`ola-serve`);
- sans-IO cluster coordinator (`ola-cluster`) with routing baselines (Round Robin, Least Loaded, Power of Two Choices, KV Headroom, Prefix Affinity) and small-batch MILP router (`optimize_batch_routing`);
- TCP wire protocol and framing (`ola-net`) with control frames and binary session state handoff;
- Ratatui workbench (`ola-ui`) with background local generation, KV benchmarks, and runtime experiments;
- unified command-line tool (`ola-cli`) supporting experiment execution, report comparison, and multi-role HTTP serving (standalone, worker, gateway);
- 14 reproducible experiment studies and probes across virtual-time, host-time, and untimed domains (`ola-bench`), with paired trials, bootstrap 95% confidence intervals, and automated regression detection.

Deliberately deferred:

- tensors in the public runtime API;
- probabilistic sampling (beyond greedy decoding);
- multi-node RDMA / NCCL tensor parallelism;
- sharded and remotely materialized checkpoints;
- RoPE-scaled and tied-embedding Llama variants;
- hardware-specific fused FlashAttention kernels for non-standard memory layouts.

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
