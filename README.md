# OlaRT

> **An opinionated Rust runtime and research framework for adaptive, scalable AI inference.**

OlaRT is a long-term AI systems research project focused on the execution, optimization, scheduling, scaling, and serving of modern machine-learning workloads.

It begins as a practical runtime built on top of existing Rust ML infrastructure—primarily Burn and CubeCL—and gradually grows into a research platform for:

- LLM inference
- VLM and multimodal inference
- computer vision and OlaDetection workloads
- GPU execution
- request scheduling
- KV-cache management
- continuous batching
- distributed inference
- adaptive load balancing
- heterogeneous compute
- runtime cost modelling
- online optimization
- learned scheduling
- spatial and world-model inference

OlaRT is intended to remain useful throughout a multi-year PhD program: first as a learning platform, then as an inference engine, then as an experimental systems runtime, and finally as a vehicle for original research.

---

## Why OlaRT?

Modern AI applications often look deceptively simple:

```text
input → model → output
```

The systems underneath are not.

```text
Request
  ↓
Preprocessing
  ↓
Model execution
  ↓
Tensor operations
  ↓
Kernel selection
  ↓
Memory management
  ↓
GPU execution
  ↓
Caching
  ↓
Scheduling
  ↓
Batching
  ↓
Distributed placement
  ↓
Response
```

OlaRT exists to make these decisions explicit, observable, replaceable, and researchable.

The long-term question is:

> **How should modern AI workloads execute when latency, throughput, memory, modality, hardware, cost, and service-level objectives all matter at once?**

---

## Philosophy

OlaRT follows a simple architectural philosophy:

> **Own the abstractions that matter to inference research. Reuse everything else until there is a reason not to.**

Burn and CubeCL already provide excellent primitives for tensors, kernels, backends, autodiff, and GPU execution.

OlaRT does not initially try to replace them.

Instead:

```text
Application
    ↓
OlaRT API
    ↓
OlaRT Runtime
    ↓
OlaRT Scheduler / Memory / Cache / Serving
    ↓
Compute Adapter
    ↓
Burn
    ↓
CubeCL
    ↓
CPU / CUDA / Metal / Vulkan / other backends
```

Over time, any layer may be replaced when doing so becomes useful for research.

This is similar in spirit to how higher-level frameworks compose lower-level ecosystems into a coherent developer experience.

OlaRT should feel like its own system even when Burn powers parts of the compute stack.

---

# Goals

OlaRT aims to become:

- an understandable inference runtime
- a batteries-included Rust framework for AI serving experiments
- a research platform for adaptive inference
- a benchmark environment for scheduling and load-balancing algorithms
- a shared execution layer for LLM, VLM, vision, and detection workloads
- a platform where papers can be reproduced and extended
- a long-lived PhD research laboratory

OlaRT is **not** initially trying to become:

- a PyTorch replacement
- a Burn replacement
- a complete training framework
- the fastest inference engine in the world
- a production cloud platform
- a dependency-free ML stack

---

# Core Architecture

```text
                         OlaRT

┌──────────────────────────────────────────────────────────────┐
│                         Public API                           │
│                                                              │
│ Runtime │ Model │ Request │ Session │ Device │ Deployment    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Inference Runtime                        │
│                                                              │
│ lifecycle │ batching │ scheduling │ admission │ observability │
└──────────────────────────────┬───────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        Memory Manager     Cache Manager   Cost Model
                │              │              │
                └──────────────┼──────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Execution Layer                          │
│                                                              │
│ LLM │ VLM │ Vision │ OlaDetection │ Future Modalities        │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Compute Abstraction                       │
│                                                              │
│ Burn Adapter │ Native Ola Backend │ Future Backends          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                     Burn / CubeCL / CUDA
                               │
                               ▼
                            Hardware
```

---

# Major Subsystems

## Runtime

The runtime owns the lifecycle of inference work.

Responsibilities may include:

- model registration
- model loading
- request admission
- execution sessions
- batching
- queueing
- scheduling
- resource accounting
- cancellation
- timeout handling
- streaming
- instrumentation

Example future API:

```rust
let runtime = OlaRuntime::builder()
    .model("llama", model)
    .scheduler(Scheduler::Adaptive)
    .continuous_batching(true)
    .metrics(true)
    .build()?;

runtime.serve("0.0.0.0:8080").await?;
```

---

## Compute

Initially, Burn is expected to provide much of the tensor and model execution machinery.

Potential architecture:

```text
OlaRT
  ↓
ComputeBackend trait
  ↓
BurnBackend
  ↓
Burn
  ↓
CubeCL
  ↓
GPU
```

Later:

```text
ComputeBackend
   ├── BurnBackend
   ├── OlaCpuBackend
   ├── OlaCudaBackend
   └── ExperimentalBackend
```

The public OlaRT API should avoid leaking backend-specific types whenever possible.

---

## LLM Runtime

OlaRT will support the full LLM serving lifecycle:

```text
Prompt
  ↓
Tokenization
  ↓
Prefill
  ↓
KV Cache
  ↓
Decode
  ↓
Sampling
  ↓
Streaming Response
```

Research and implementation areas include:

- transformer execution
- prefill/decode separation
- KV-cache allocation
- paged KV caches
- prefix caching
- quantized KV caches
- continuous batching
- speculative decoding
- sequence scheduling
- token streaming
- multi-GPU execution

---

## VLM Runtime

VLM workloads introduce different scheduling and memory behaviour.

```text
Image ─→ Vision Encoder ─→ Visual Tokens ─┐
                                          ├→ Multimodal Model
Text ───────────────→ Text Tokens ────────┘
```

OlaRT will investigate:

- visual token generation
- image preprocessing
- dynamic image resolution
- multimodal prompt execution
- encoder caching
- visual token caching
- multimodal KV cache
- mixed text/image batches
- VLM-aware scheduling
- VLM-aware cost estimation
- multimodal SLOs

---

## OlaDetection

OlaDetection is the vision/detection research track built on OlaRT.

Potential workloads include:

- object detection
- segmentation
- SAM-style inference
- document segmentation
- OCR pipelines
- image-region understanding
- exam-document processing
- vision-transformer inference

The main systems questions are not only model quality.

They include:

- how to batch images of different resolutions
- when to tile large images
- how to cache image encodings
- how to schedule mixed vision workloads
- how to route vision jobs across accelerators
- how to trade resolution against latency
- how to compose detection, OCR, and language stages efficiently

---

# Scheduling

Scheduling is a first-class research surface.

Baseline strategies:

- FIFO
- round robin
- shortest-job-first
- priority scheduling
- deadline-aware scheduling

Future strategies may consider:

- sequence length
- modality
- image resolution
- predicted compute cost
- KV-cache footprint
- queue length
- memory pressure
- GPU utilization
- SLO deadline
- model residency
- cache locality
- energy/cost
- batch compatibility

The runtime should make schedulers pluggable.

```rust
trait Scheduler {
    fn schedule(&mut self, state: &RuntimeState) -> Schedule;
}
```

---

# Adaptive Load Balancing

OlaRT eventually targets clusters rather than single GPUs.

```text
                    Requests
                       │
                       ▼
                 Ola Router
                       │
             Adaptive Placement
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        GPU 0        GPU 1        GPU 2
```

A placement decision may depend on:

- GPU memory
- compute utilization
- current queue
- active model
- model residency
- cache locality
- network topology
- request modality
- deadline
- predicted runtime

Future techniques may include:

- heuristics
- queueing theory
- online optimization
- stochastic optimization
- contextual bandits
- reinforcement learning
- learned cost models

---

# Optimization

Runtime decisions can be viewed as optimization problems.

A runtime state might be represented as:

\[
s_t = (Q_t, M_t, U_t, C_t, H_t)
\]

where:

- \(Q_t\): request queues
- \(M_t\): memory state
- \(U_t\): utilization
- \(C_t\): cache state
- \(H_t\): hardware state

A runtime action might assign requests to resources:

\[
a_t = \text{placement / batching / scheduling decision}
\]

Possible objectives:

\[
\min \alpha L + \beta C + \gamma M + \delta V
\]

where:

- \(L\): latency
- \(C\): compute or monetary cost
- \(M\): memory pressure
- \(V\): SLO violations

This creates a direct bridge between inference engineering and optimization research.

---

# Observability

Every serious experiment needs observability.

OlaRT should eventually expose:

- request latency
- queue latency
- TTFT
- TPOT
- tokens/sec
- requests/sec
- batch size
- cache hit rate
- KV-cache occupancy
- GPU memory
- GPU utilization
- kernel latency
- scheduler decisions
- placement decisions
- request migration
- model residency
- error rates

Example trace:

```text
Request 842
  │
  ├── queue       3.2 ms
  ├── batch       1.1 ms
  ├── prefill    21.7 ms
  ├── decode     78.4 ms
  └── response    0.8 ms
```

---

# Repository Structure

The exact structure will evolve.

```text
ola-rt/
│
├── crates/
│   ├── ola-core/
│   ├── ola-runtime/
│   ├── ola-model/
│   ├── ola-compute/
│   ├── ola-burn/
│   ├── ola-memory/
│   ├── ola-cache/
│   ├── ola-scheduler/
│   ├── ola-serve/
│   ├── ola-observe/
│   ├── ola-llm/
│   ├── ola-vlm/
│   ├── ola-detection/
│   ├── ola-distributed/
│   └── ola-bench/
│
├── experiments/
├── benchmarks/
├── examples/
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── papers/
│   └── notes/
│
├── IMPLEMENTATION.md
└── README.md
```

Do not create every crate immediately.

The repository should grow only when boundaries become real.

---

# Dependency Strategy

OlaRT is intentionally pragmatic.

Likely dependencies may include:

- Burn
- CubeCL
- SafeTensors
- Hugging Face tokenizers
- Tokio
- Axum
- Serde
- Tracing
- OpenTelemetry
- Criterion
- Rayon
- CUDA ecosystem crates where useful

Dependencies are divided into three categories.

## Infrastructure

Usually safe to depend on:

- HTTP
- serialization
- logging
- configuration
- CLI parsing
- tokenizers
- model formats

## Compute infrastructure

Initially reused:

- Burn tensors
- Burn modules
- Burn backends
- CubeCL kernels
- vendor libraries

## Research-critical abstractions

OlaRT should increasingly own:

- inference lifecycle
- scheduler
- admission control
- batching
- KV-cache policy
- memory accounting
- cost modelling
- load balancing
- distributed placement
- adaptive execution policy
- research instrumentation

---

# Roadmap

## Phase 0 — Foundations

**September–December 2026**

Goal: understand the stack while establishing OlaRT's architecture.

- [ ] micrograd-style scalar autograd exercise
- [ ] tiny educational tensor implementation
- [ ] understand Burn tensor/backend/device abstractions
- [ ] implement transformer components in Burn
- [ ] understand CubeCL
- [ ] CPU/SIMD experiments
- [ ] GPU kernel experiments
- [ ] benchmark Burn against custom kernels
- [ ] define OlaRT public API
- [ ] implement request abstraction
- [ ] model registry
- [ ] runtime state
- [ ] basic inference session
- [ ] basic LLM execution
- [ ] prefill/decode split
- [ ] simple KV cache
- [ ] metrics
- [ ] FIFO scheduler
- [ ] multi-request simulation

End-of-phase target:

```text
clients
  ↓
OlaRT
  ↓
scheduler
  ↓
LLM runtime
  ↓
Burn
  ↓
GPU
```

---

## Phase 1 — Inference Engine

**PhD Year 1**

Goal: turn OlaRT into a credible single-node inference research engine.

Areas:

- LLM execution
- continuous batching
- KV-cache management
- quantization
- kernel profiling
- model loading
- streaming
- SLO metrics
- scheduler benchmarks
- GPU memory accounting
- reproducible workloads

Research baseline:

```text
FIFO
vs
SJF
vs
memory-aware
vs
deadline-aware
```

Expected outputs:

- stable OlaRT runtime
- benchmark suite
- technical report
- reproducible scheduler experiments
- first workshop/paper-quality research question

---

## Phase 2 — Multimodal and Adaptive Runtime

**PhD Year 2**

Goal: make OlaRT modality-aware and adaptive.

Areas:

- VLM runtime
- OlaDetection
- image/video workloads
- multimodal batching
- dynamic resolution
- encoder caching
- modality-aware cost models
- adaptive scheduling
- multi-GPU placement
- heterogeneous hardware
- learned workload prediction

Research questions:

- Should LLM and VLM requests share the same scheduling policy?
- Can multimodal request cost be predicted before execution?
- Can image resolution be adapted to meet SLOs?
- Can runtime state determine model or kernel choices?

Expected outputs:

- multimodal benchmark suite
- VLM scheduler
- OlaDetection execution pipeline
- adaptive runtime paper
- open-source release suitable for external reproduction

---

## Phase 3 — Distributed Optimization

**PhD Year 3**

Goal: investigate distributed, adaptive inference as an optimization problem.

Areas:

- cluster scheduling
- adaptive load balancing
- distributed KV cache
- request migration
- cache-aware placement
- model replication
- autoscaling
- online optimization
- RL or bandit schedulers
- heterogeneous clusters
- robustness and fault tolerance

Target research loop:

```text
observe
  ↓
predict
  ↓
optimize
  ↓
schedule
  ↓
execute
  ↓
measure
  └────→ observe
```

Expected outputs:

- thesis-level system
- multiple publishable experiments
- distributed OlaRT prototype
- final comparative evaluation
- thesis chapters derived directly from the runtime

---

# Experiment Standard

Every serious experiment should live in:

```text
experiments/
```

Recommended structure:

```text
experiments/012_memory_aware_scheduler/
├── README.md
├── config/
├── src/
├── data/
├── results/
└── plots/
```

Each experiment should document:

1. research question
2. hypothesis
3. baseline
4. hardware
5. workload
6. metrics
7. implementation
8. results
9. interpretation
10. limitations
11. next experiment

Negative results stay in the repository.

---

# Research Notes

Use:

```text
docs/research/
```

Suggested format:

```text
2026-09-burn-backends.md
2026-10-gpu-memory.md
2026-11-kv-cache.md
2026-12-scheduling.md
2027-03-continuous-batching.md
2027-08-vlm-serving.md
```

Every note should answer:

- What did I investigate?
- What did I expect?
- What happened?
- Why?
- What remains unclear?
- What experiment comes next?

---

# Benchmarking Rules

Every performance claim must include:

- hardware
- software versions
- model
- input distribution
- warmup
- repetitions
- latency percentiles
- throughput
- memory usage
- baseline
- configuration
- failure cases

No optimization is considered successful merely because one run was faster.

---

# Long-Term Vision

OlaRT should eventually support:

```text
                Models

       LLM   VLM   Vision   Spatial
          \    |      |       /
           \   |      |      /
               OlaRT
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
     Memory   Scheduler  Cache
        │        │        │
        └────────┼────────┘
                 ▼
            Cost Model
                 │
                 ▼
            Optimizer
                 │
                 ▼
             Hardware
                 │
                 ▼
             Feedback
                 │
                 └──────────────→ Runtime
```

The end goal is not simply to execute models.

The end goal is to study **adaptive AI execution**:

> Given a workload, model, hardware state, cache state, resource budget, and service objective, what should the runtime do next?

---

# Status

The authoritative implementation snapshot is the timestamped
[`progress_report.md`](progress_report.md).

**Current phase:** foundations transitioning into the first inference-engine
vertical slice — October 2026.

Current engineering focus:

- define explicit per-iteration batch-composition and token-budget policy;
- turn deterministic multi-session iterations into tensor-level batches;
- optimize page-native attention with accelerator-specific fusion.

---

# Final Principle

OlaRT should evolve according to one rule:

> **Do not rebuild a lower-level component merely because it exists. Rebuild or replace it when doing so enables understanding, control, performance, or research that the current abstraction prevents.**

OlaRT begins as a framework built on Burn.

It should grow into an inference engine.

Then into a multimodal runtime.

Then into a distributed systems laboratory.

And finally into a platform for original research in adaptive, scalable AI inference.
