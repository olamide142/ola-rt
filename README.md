# Ola-RT

> **A Compositional Rust Runtime and Research Platform for Inference Execution Policies in AI Serving**

[![Rust](https://img.shields.io/badge/rust-2024%20%7C%201.95-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![PhD Research](https://img.shields.io/badge/phd-Nile%20University%20of%20Nigeria-8A2BE2.svg)]()

Ola-RT is an inference runtime and systems research platform engineered in Rust, developed as the experimental core of doctoral research by **Victor O. Olowofeso** at **Nile University of Nigeria**.

Modern AI serving systems (such as vLLM, TensorRT-LLM, and SGLang) make dozens of interdependent operational decisions per iteration: request scheduling, memory-aware admission, batch construction, chunked prefill, KV-cache preemption, kernel selection, and cluster routing. In existing runtimes, these choices are typically embedded within tight, monolithic heuristics that are difficult to formalize, compare, or replace.

Ola-RT asks a foundational systems question:

> **Given a dynamic workload, model architecture, physical KV cache pressure, hardware capabilities, and latency SLOs, what should an inference runtime do next?**

Ola-RT answers this by formalizing and executing **Inference Execution Policies (IEP)**: treating runtime decisions as modular, compositional, and strictly validated mathematical functions over observable system state.

---

## Key Highlights

- **Formal Policy Composition:** Decomposes runtime control into explicit stages (Scheduler, Admission, Batching, Prefill Chunking, Preemption, Kernel Selection, Routing) with formal plan validation.
- **Physical KV Cache & Paged Attention:** Paged KV management with block tables, zero-copy block allocation, memory fragmentation metrics, and both portable online-softmax and CubeCL GPU attention kernels.
- **Continuous Batching & Chunked Prefill:** Iteration-level scheduling, resumable prefill chunks, shared token budgets, and cost-calibrated prompt bucketing balancing padding overhead against kernel invocation costs.
- **Memory-Aware Admission & Preemption:** Worst-case KV reservation alongside quantile-based headroom admission ($q_\alpha$) driven by Kaplan-Meier remaining-output length beliefs, backed by watermark preemption and recompute semantics.
- **Switching Costs & Adaptive Optimization:** Explicit transition penalty modeling ($c_{\text{switch}}$) and dual-hysteresis `ReplayOptimizer` to prevent policy thrashing during online workload adaptation.
- **Distributed Cluster Coordination:** Sans-IO `Coordinator` supporting Round Robin, Least Loaded, Power-of-Two, KV Headroom, and Prefix Affinity routing, small-batch MILP optimization, and TCP session state handoff via `ola-net`.
- **Reproducible Experimental Science:** 14 registered experiment studies and probes across virtual-time, host-time, and untimed domains (`ola-bench`), evaluated with paired trials, bootstrap 95% confidence intervals, and automated regression detection.
- **Full Model Serving:** Strict Hugging Face checkpoint loading (SafeTensors + Tokenizers) for Llama (e.g. SmolLM2, TinyLlama, Llama-3) and Mistral architectures on CPU and WGPU/CUDA, with an OpenAI-compatible SSE streaming HTTP server (`ola-serve`).

---

## Theoretical Foundation: Inference Execution Policies (IEP)

In Ola-RT, serving is modeled as a discrete-time closed-loop control system:

```text
       ┌────────────────────────────────────────────────────────┐
       │                                                        │
       ▼                                                        │
  PlanningSnapshot (Sₙ)                                         │ Telemetry (Tₙ)
       │                                                        │ (TTFT, TBT,
       ▼                                                        │  KV pressure,
┌──────────────┐     ExecutionPlan (Aₙ)    ┌─────────────────┐  │  padding waste)
│ Policy (π)   │ ────────────────────────► │ Validator (Γ)   │  │
└──────────────┘                           └────────┬────────┘  │
                                                    │           │
                                           Admitted │           │
                                           Plan (GP)│           │
                                                    ▼           │
                                           ┌─────────────────┐  │
                                           │ Engine Executor │ ─┘
                                           └─────────────────┘
```

$$\mathcal{S}_n \xrightarrow{\pi} \mathcal{A}_n \xrightarrow{\Gamma} G_P \xrightarrow{\text{execute}} \mathcal{T}_n \to \mathcal{S}_{n+1}$$

1. **State Observation ($\mathcal{S}_n$):** At iteration $n$, the runtime constructs an immutable `PlanningSnapshot` containing the queued requests $Q_n$, active execution set $\mathcal{A}_n$, memory/block table occupancy $M_n$, and remaining-output length beliefs $B_n$.
2. **Policy Evaluation ($\pi$):** The active policy proposes an `ExecutionPlan` $\mathcal{A}_n = \pi(\mathcal{S}_n)$, specifying which requests to admit, which active sequences to advance, how many prefill tokens to chunk, and whether to preempt work.
3. **Plan Validation ($\Gamma$):** Before any tensor executes, the plan is checked against strict runtime safety invariants:
   $$\Gamma(\mathcal{S}_n, \mathcal{A}_n) \in \{0, 1\}$$
   Validation guarantees candidate membership, request uniqueness, token budget legality, physical KV page availability, and device concurrency bounds. If $\Gamma = 0$, the plan is rejected and the engine falls back to a deterministic, starvation-safe baseline $G_{\text{safe}}$.
4. **Execution & Telemetry ($\mathcal{T}_n$):** The validated plan $G_P$ is dispatched to the model and compute layer. Observed iteration metrics (latencies, token counts, padding waste, cache utilization) feed back to update runtime beliefs and online optimizers.

### Modular Policy Decomposition

Instead of a black-box monolithic scheduler, policies compose through explicit interfaces:

| Policy Component | Interface Input | Output Decision | Formal Constraint / Role |
| :--- | :--- | :--- | :--- |
| **Scheduler** | Queued requests $Q_n$ | Ranked request order | Defines queue priority (FIFO, Priority, EDF, SJF, Backfill) |
| **Admission Control** | Ranked queue & KV state | Admitted / deferred subset | Ensures KV headroom (worst-case or belief quantile $q_\alpha$) |
| **Batch Policy** | Admitted & active requests | Selected iteration candidates | Enforces continuous batching & token budgets |
| **Prefill Bucketer** | Selected prefill chunks | Tensor bucket grouping | Minimizes padding waste vs kernel launch overhead |
| **Preemption Policy** | Active set & KV exhaustion | Preemption victims | Recovers physical blocks with recompute/swap semantics |
| **Kernel Selector** | Tensor shapes & device | Kernel provider dispatch | Selects between portable reference and fused CubeCL kernels |
| **Cluster Router** | Ingress requests & workers | Worker placement / deferral | Multi-node load balancing & small-batch MILP routing |

---

## Workspace Architecture

Ola-RT is organized as a modular, 12-crate Rust workspace. Dependencies point strictly inward to preserve clean research boundaries:

```text
                                [ Clients / CLI / UI ]
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  ola-serve   │  (OpenAI-compatible HTTP / SSE streaming)
                                   └──────┬───────┘
                                          │
                                          ▼
   ┌──────────────┐                ┌──────────────┐                ┌──────────────┐
   │   ola-cli    │ ─────────────► │ ola-cluster  │ ◄────────────► │   ola-net    │
   │  (CLI tools) │                │  (Sans-IO    │                │ (TCP framing │
   └──────┬───────┘                │  coordinator)│                │  & handoff)  │
          │                        └──────┬───────┘                └──────────────┘
          │                               │
          ▼                               ▼
   ┌──────────────┐                ┌──────────────┐                ┌──────────────┐
   │    ola-ui    │ ─────────────► │ ola-runtime  │ ◄────────────► │  ola-kernel  │
   │   (Ratatui   │                │ (Executor,   │                │  (Paged-attn │
   │  workbench)  │                │  policies,   │                │   contracts) │
   └──────────────┘                │  validation) │                └──────────────┘
          │                        └──────┬───────┘
          │                               │
          ▼                               ▼
   ┌──────────────┐                ┌──────────────┐
   │  ola-bench   │ ─────────────► │   ola-burn   │  (Llama & Mistral architectures,
   │ (Simulation, │                └──────┬───────┘   Burn / CubeCL adapters)
   │  14 studies) │                       │
   └──────────────┘                       ▼
          │                        ┌──────────────┐
          │                        │  ola-model   │  (Catalog & architecture factories)
          │                        └──────┬───────┘
          │                               │
          ▼                               ▼
   ┌──────────────┐                ┌──────────────┐
   │ ola-observe  │ ─────────────► │   ola-core   │  (Domain types, IDs, monotonic time)
   │   (Traces)   │                └──────────────┘
   └──────────────┘
```

### Crate Boundaries

| Crate | Responsibility | Must Not Own |
| :--- | :--- | :--- |
| [`ola-core`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-core) | Backend-neutral domain types, request identities, timestamps, generation configs | Queues, policy logic, Burn tensors |
| [`ola-model`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-model) | Model catalog, artifact sources, architecture factories | Tensor execution, scheduling, downloads |
| [`ola-runtime`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime) | Execution engine, lifecycle state machine, policy traits, `ExecutionPlan`, paged KV cache | Sockets, HTTP, model weights |
| [`ola-kernel`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-kernel) | Replaceable attention kernel contracts, portable reference, CubeCL GPU kernels | Scheduling, model weights, KV allocation |
| [`ola-burn`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn) | Adapting Burn primitives to Ola-RT (Llama, Mistral, CPU/WGPU/CUDA devices) | Scheduling, admission, serving |
| [`ola-observe`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-observe) | Monotonic event measurements, latency derivation, JSON Lines execution traces | Lifecycle mutation, policy decisions |
| [`ola-serve`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-serve) | OpenAI-compatible HTTP service, SSE token streaming, graceful draining | Execution policy, tensor manipulation |
| [`ola-bench`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench) | Virtual-time simulation, workload traces, cost models, percentile bootstrap statistics | Runtime production state |
| [`ola-cluster`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-cluster) | Sans-IO cluster coordinator, worker summaries, routing baselines, MILP batch router | Raw sockets, wall-clock time, kernel execution |
| [`ola-net`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-net) | Tokio byte stream wire framing, control messages, binary session transfer frames | Cluster policy, model weights |
| [`ola-cli`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-cli) | Unified command-line interface (`ola experiment run`, `compare`, `serve`) | Direct runtime mutation |
| [`ola-ui`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-ui) | Ratatui terminal workbench, local generation, live KV benchmarks | Model kernels, scheduling policy |

---

## Supported Models and Hardware

### Architectures
- **Llama:** Grouped-Query Attention (GQA), SwiGLU activations, split-half Rotary Position Embeddings (RoPE), RMSNorm, and unshared output projections.
- **Mistral:** Grouped-Query Attention with support for sliding window attention patterns.
- *Tested Checkpoints:* SmolLM2-135M, TinyLlama-1.1B, Llama-3.2-1B, TinyMistral-248M.

### Execution Backends
- **Portable CPU:** Pure Rust compute path via Burn. Requires zero GPU dependencies or drivers—ideal for CI and running virtual-time experiments.
- **WGPU:** Cross-platform GPU execution (Vulkan/Metal/DirectX), numerically validated on consumer hardware (tested on NVIDIA GeForce GTX 1650).
- **Native CUDA:** Direct NVIDIA hardware execution via CubeCL/CUDA.
- **Strict Conformance:** Includes an independent scalar logit reference oracle (`CONFORMANCE.md`) to verify zero numerical drift across optimizations.

---

## Quickstart

### Prerequisites
- Rust 1.95+ (2024 edition)
- `make` and standard build tools

### 1. Build and Verify
Run the complete CPU quality gate (formatting, type checking, unit tests, clippy with warnings denied, and documentation links):

```bash
make verify
```

Or run tests directly with Cargo:
```bash
cargo test --manifest-path ola-rt/Cargo.toml --workspace
```

### 2. Run Virtual-Time Experiments (No GPU Required)
All theoretical policy evaluations run deterministically in virtual time on CPU without needing a GPU:

```bash
# List all registered experiments, clock domains, and parameters
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment list

# 1. Compare schedulers (FIFO, Priority, EDF, SJF, Backfill) under overload
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
  experiment run scheduler-comparison --set trace=overload seeds=10

# 2. Evaluate prefill bucketing strategies (padding waste vs kernel invocations)
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
  experiment run prefill-bucketing

# 3. Benchmark KV memory pressure with watermark admission and preemption
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
  experiment run kv-pressure --set trace=early_stopping seeds=10

# 4. Compare baseline report against a candidate with bootstrap CI separation
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
  experiment compare baseline.json candidate.json --threshold 0.05
```

### 3. Serve Models over HTTP (OpenAI-Compatible)
Launch a local Hugging Face model checkpoint as an HTTP inference service:

```bash
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
  serve --model /path/to/smollm2-135m-instruct --addr 127.0.0.1:8080
```

Query the server using standard OpenAI-compatible Server-Sent Events (SSE) streaming:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2",
    "messages": [{"role": "user", "content": "What is an inference execution policy?"}],
    "stream": true
  }'
```

### 4. Interactive Terminal Workbench
Launch the interactive Ratatui TUI to inspect real-time generation, KV block allocation, and scheduling replays:

```bash
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-ui
```

---

## Experimental Rigor & Research Methodology

Ola-RT enforces strict scientific protocols across all benchmarks and studies:

1. **Explicit Clock Domains:**
   - **Virtual Time:** Driven by a deterministic simulation clock and calibrated cost model. Results are 100% reproducible across different hardware.
   - **Host Time:** High-resolution monotonic CPU wall-clock timestamps for end-to-end execution.
   - **Device Time:** GPU kernel profiling timestamps (Nsight Compute / CubeCL).
   - **Untimed:** Pure algorithmic metrics (padding overhead, token counts, eviction counts).
2. **Statistical Significance:** Studies execute paired, order-rotated trials across multiple seeds with warmups. All reported summaries include means and **95% bootstrap confidence intervals** ($B = 2{,}000$).
3. **Automated Regression Gating:** `ola experiment compare` compares candidate runs against baselines and flags regressions only when confidence intervals separate by more than a configurable threshold (e.g. 5%).
4. **Lab Notebook System:** The [`lab/`](file:///home/olamide/dev/ola-rt/lab/README.md) workflow tracks hypotheses, configuration digests, versioned JSON reports, and traces in auditable notebook directories (`make lab-new`, `make lab-run`, `make lab-compare`).

---

## Research Trajectory & Papers

Ola-RT serves as the experimental vehicle for a multi-stage PhD research program:

- **Working Paper:**
  > *A Compositional Formalization of Inference Execution Policies in Ola-RT*
  > Victor O. Olowofeso, Nile University of Nigeria.
  > Working draft reference: [`papers/OLA_RT_FORMALIZATION_REFERENCE.md`](file:///home/olamide/dev/ola-rt/papers/OLA_RT_FORMALIZATION_REFERENCE.md).
- **Core Research Questions:**
  1. **Policy Composition:** Can LLM execution policies be formalized compositionally with executable boundary invariants?
  2. **Memory Risk & Quantile Admission:** How does quantile-based KV cache headroom reservation ($q_\alpha$) mitigate preemption overhead compared to worst-case static allocation?
  3. **Switching Costs & Adaptive Batching:** How do switching cost penalties ($c_{\text{switch}}$) and inertia guards prevent thrashing in online token-budget optimization?
  4. **Dynamic Kernel Dispatch:** When does accelerator-specific fused attention outperform portable references across dynamic batch sizes?
  5. **Disaggregated Serving:** How do small-batch MILP routers coordinate heterogeneous prefill and decode workers under network transfer costs?

---

## Project Roadmap

- [x] **Phase 0 — Foundations:** Pure Rust request lifecycles, memory accounting, FIFO baseline, Burn compute adapters, and determinism.
- [x] **Phase 1 — Single-Node Inference Engine:** Paged KV cache, continuous batching, chunked prefill, pluggable schedulers (Priority/EDF/SJF/Backfill), watermark preemption, quantile admission, calibrated cost models, and SSE streaming HTTP serving.
- [ ] **Phase 2 — Multimodal & Adaptive Runtime:** Vision-Language Model (VLM) execution, dynamic image tiling, multimodal prompt scheduling, and learned cost estimation.
- [ ] **Phase 3 — Distributed Cluster Optimization:** Scaling the sans-IO coordinator to multi-node clusters, dynamic session migration, and cross-node speculative decoding.

---

## Citation

If you use Ola-RT in your academic research, please cite:

```bibtex
@misc{olowofeso2026olart,
  author       = {Victor O. Olowofeso},
  title        = {Ola-RT: A Compositional Rust Runtime and Research Platform for Inference Execution Policies in AI Serving},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/voolowofeso/ola-rt}},
  institution  = {Nile University of Nigeria}
}
```

---

## License

Ola-RT is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
