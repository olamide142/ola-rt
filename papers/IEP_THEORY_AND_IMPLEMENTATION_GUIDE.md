# Inference Execution Policies (IEP): Theory-to-Code Companion

> **A Comprehensive Reference Guide for Doctoral Research, Systems Architecture, and Paper Writing**  
> **Author:** Victor O. Olowofeso  
> **Affiliation:** Department of Computer Science, Nile University of Nigeria  
> **Primary Paper Reference:** *Formalizing Inference Execution Policies: A Unified Control Framework for Adaptive Generative Model Serving* (Draft – October 2026)  
> **Implementation Codebase:** `ola-rt` (OlaRT Reference Inference Runtime)

---

## 1. Executive Purpose & Scope

Modern large-scale generative model serving (LLMs, VLMs) exposes an increasingly complex set of execution choices: request admission, routing, batching, scheduling, preemption, chunked prefill, KV-cache paging and eviction, kernel selection, precision modes, and resource scaling. While systems such as Orca, vLLM, DistServe, Sarathi-Serve, and SGLang have achieved breakthrough performance by optimizing specific subsets of these mechanisms, they generally treat them as isolated, independently engineered heuristics.

Your doctoral dissertation formalizes the foundational abstraction: the **Inference Execution Policy (IEP)**. Under this paradigm, generative inference is modeled as an **event-driven, constrained stochastic control problem** over a multi-timescale action hierarchy:

$$\text{State } S_n \longrightarrow \text{Policy } \pi(S_n) \longrightarrow \text{Action } A_n \longrightarrow \text{Physical Plan } G_P \longrightarrow \text{Execution} \longrightarrow \text{Telemetry } T_n \longrightarrow S_{n+1}$$

The `ola-rt` repository is the explicit reference implementation of this theoretical framework. Every mathematical symbol, feasibility constraint, and control construct in your paper corresponds to a typed, verifiable component in Rust.

This guide provides an exhaustive mapping between the paper's formal theory and the codebase's architecture, walking through the end-to-end execution loop, detailing the underlying subsystems, and providing an exact experimental blueprint to populate the empirical sections (Sections 40–43) of your papers.

---

## 2. Master Theory-to-Code Mapping

| Paper Concept & Symbol | Paper Section | Rust Crate & Path | Core Rust Types & Functions | Systems Invariants & Contracts |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Request**<br>$r_i = (a_i, x_i, \ell_i^{\text{in}}, L_i^{\text{out}}, \phi_i, c_i, q_i, d_i)$ | §5, Def 1 | `ola-core` | [`RequestSpec`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-core/src/request.rs), [`RequestId`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-core/src/id.rs), [`GenerationConfig`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-core/src/request.rs), [`Timestamp`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-core/src/time.rs) | Immutable client request specification. Distinguishes known input length $\ell_i^{\text{in}}$ from unknown stochastic output length $L_i^{\text{out}}$. |
| **Request Progress**<br>$z_i(t) = (p_i(t), d_i(t), k_i(t))$ | §5.2 | `ola-runtime` | [`RequestProgress`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/state.rs), [`RequestState`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/state.rs) | Tracks prompt processed $p_i$, output tokens generated $d_i$, and KV block residency $k_i$. Enforces valid state transitions (`Queued` $\to$ `Running` $\to$ `Completed`). |
| **System State & Observation**<br>$S_n = (Q_n, R_n, Z_n, K_n, H_n, X_n, T_n, B_n)$,<br>$O_n = \psi(S_n)$ | §7, §11 | `ola-runtime` | [`PlanningSnapshot`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/snapshot.rs), [`RuntimeState`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/state.rs) | Immutable snapshot projected to the policy. Contains queue queues $Q_n$, active requests $R_n$, memory/pressure states $K_n$, deadlines, and arrival timestamps. |
| **Execution Feasibility Predicate**<br>$\Gamma(s, a; \mathcal{C}) \in \{0, 1\}$ | §9, Def 2 | `ola-runtime` | [`ExecutionPlan::validate`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/plan.rs) | Verifies that candidate actions do not violate physical memory ceilings, token budgets, or model architectural limits before committing to hardware. |
| **Execution Action**<br>$A_n \in \mathcal{A}_{\mathcal{C}}(S_n)$ | §8 | `ola-runtime` | [`ExecutionPlan`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/plan.rs), [`LifecycleAction`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/plan.rs), [`PlannedRow`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/plan.rs) | Pure-data plan describing admissions, preemptions, chunk boundaries, token budgets, and kernel bindings. |
| **Inference Execution Policy**<br>$\pi : \mathcal{H} \to \Delta(\mathcal{A})$ | §11, Def 8 | `ola-runtime` | [`ExecutionPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/policy.rs) | `plan(&self, &PlanningSnapshot) -> Result<PolicyDecision, PolicyError>` and `observe(&mut self, &ExecutorIteration)`. Pure functional planning + sequential learning. |
| **Hierarchical Policy Decomposition**<br>$\pi = \pi_D \circ \pi_R \circ \pi_I \circ \pi_O$ | §16.5 | `ola-runtime` | [`ComposedPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/composed.rs) | Composes deployment/admission ($\pi_R$), scheduling ($\pi_I$), batching ($\pi_I$), and preemption ($\pi_I$) into a coherent single planner. |
| **Adaptive Policy & Parameter Tuning**<br>$\pi(S_n; \theta)$, $\min [c(s, a) + \lambda_{\text{sw}} C_{\text{switch}}]$ | §11.2, §15, §27 | `ola-runtime` | [`AdaptivePolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/adaptive.rs), [`AdaptiveOptimizer`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/adaptive.rs) | Maintains parameter vector $\theta$, formulates LP/MILP constraints via `good_lp`, and falls back safely if candidates fail validation. |
| **Safe Fallback Policy**<br>$\pi_{\text{safe}}(S_n)$ | §35 | `ola-runtime` | `with_fallback` in [`AdaptivePolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/adaptive.rs) | Guaranteed fallback to a verified heuristic (e.g. FIFO) if optimization times out, fails feasibility, or drifts out of distribution. |
| **Logical Model**<br>$\mathcal{M} = (\mathcal{G}_L, \theta, \Phi)$ | §4.1 | `ola-model`, `ola-burn` | [`CausalModel`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-model/src/execution.rs), [`LlamaForCausalLM`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/llama/model.rs), [`MistralForCausalLM`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/mistral/model.rs) | Mathematical representation of model layers, weights, and embeddings, independent of hardware placement or batching mechanisms. |
| **Physical Execution Graph**<br>$G_P(s, a) = (V_P, E_P, \mu, \kappa, \rho, \sigma)$ | §4.2 | `ola-runtime`, `ola-burn` | [`ExecutionBatch`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/batch.rs), [`BurnLlamaModel`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/llama/session.rs), [`BurnMistralModel`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/mistral/session.rs) | Concrete materialized computation: padded input tensors, device block tables, and kernel invocations. |
| **Paged KV Cache & Memory Actions**<br>$A_n^{\text{kv}}$ | §8.6, §10.1 | `ola-runtime`, `ola-burn` | [`KvBlockPool`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/kv_cache.rs), `LlamaKvPool`, `MistralKvPool` | Physical block pools, prefix caching, sub-window reads for sliding-window attention, and atomic multi-sequence allocation. |
| **Kernel & Operator Execution**<br>$A_n^{\text{kernel}}$, $\kappa(v) \in \mathcal{K}(v, s)$ | §8.10, §21 | `ola-kernel` | [`KernelRegistry`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-kernel/src/registry.rs), [`PagedAttentionKernel`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-kernel/src/paged_attention.rs), [`CubeClPagedAttention`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-kernel/src/cubecl_extension.rs) | Exposes kernel capabilities, profiles, and runtime selection between portable reference and fused accelerator kernels. |
| **Performance Cost Model**<br>$\hat{T}_{\text{iter}} = f_{\theta}(S_n, A_n)$ | §10.5, §31.5, §34 | `ola-bench` | [`fit_cost_model`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/calibration.rs), [`CostFit`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/calibration.rs) | Ordinary least-squares regression with bootstrap confidence intervals predicting iteration overhead from token and batch dimensions. |
| **Decision Auditability & Plan Reporting** | §31.3, §31.4 | `ola-bench`, `ola-runtime` | [`DecisionRecord`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/decision.rs), [`PlanReport`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/plan_report.rs) | Detailed record of candidate evaluation, $\theta$ values, kernel provider identity, and microsecond-level planning durations. |
| **Empirical Evaluation & Simulation** | §36–§39 | `ola-bench` | [`SimulationConfig`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/simulation.rs), [`SchedulingTrace`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/trace.rs), [`SchedulingSummary`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/scheduling_summary.rs) | Virtual-time simulation, reproducible trace replays, bootstrap resampling, Jain fairness, and Pareto comparison. |

---

## 3. The End-to-End Control Loop

In Section 2.2 of your paper, serving is formulated as an event-driven constrained stochastic control problem. The runtime does not merely advance queues; it evaluates a continuous feedback loop:

```
[Request Ingestion]
       │  r_i = (prompt, sampling_params, priority, deadline)
       ▼
[State Construction: PlanningSnapshot] ────► S_n = (Q_n, R_n, Z_n, K_n, H_n, X_n)
       │
       ▼
[Policy Planning: ExecutionPolicy::plan] ──► π(S_n; θ)
       │
       ▼
[Candidate Plan: PolicyDecision] ──────────► A_n = (LifecycleAction, PlannedInvocation, PlannedRow)
       │
       ▼
[Feasibility Validation: validate] ───────► Γ(S_n, A_n; C) ∈ {0, 1}
       ├─ If 0 (Infeasible) ───────────────► Trigger Safe Fallback π_safe
       └─ If 1 (Feasible)
               │
               ▼
[Plan Commit & Batch Materialization] ─────► G_P(S_n, A_n)
       │
       ▼
[Physical Model Execution] ────────────────► Burn / CubeCL (prefill/decode forward pass)
       │
       ▼
[Telemetry & Event Logging] ───────────────► T_n (TTFT, TBT, memory delta, provider timing)
       │
       ▼
[Policy Observation: ExecutionPolicy::observe] ──► Update θ, accumulate rewards, advance state
```

### 3.1 Step 1: Snapshot Construction ($S_n \to O_n$)
At the start of an iteration, the executor freezes an immutable [`PlanningSnapshot`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/snapshot.rs).
- **Queued Requests ($Q_n$):** Prompts waiting for initial admission, including priority $q_i$, arrival timestamp $a_i$, and deadline $d_i$.
- **Active Requests ($R_n$):** Requests currently resident in the execution system, segmented into prefilling and decoding.
- **Generation Progress ($Z_n$):** Exact token counts processed ($p_i$) and generated ($d_i$).
- **Memory & Cache State ($K_n$):** Available KV blocks, allocated blocks, and cache pressure ($\text{allocated} / \text{total}$).

*Core Invariant:* The snapshot is strictly read-only. Schedulers and policies take `&self` and receive `&PlanningSnapshot`, preventing any experimental or buggy policy from corrupting runtime state.

### 3.2 Step 2: Policy Planning ($\pi(S_n) \to A_n$)
The installed [`ExecutionPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/policy.rs) executes its `plan()` method.
- **Hierarchical Composition:** In [`ComposedPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/composed.rs), decisions are resolved sequentially:
  1. *Admission Policy:* Evaluates watermark thresholds against model KV pools.
  2. *Request Scheduler:* Orders requests using algorithms like FIFO, SJF, EDF, or Priority.
  3. *Batch Policy:* Allocates prompt chunks and decode steps within token budget $B_n$.
  4. *Preemption Policy:* Identifies victim requests to requeue if memory pressure exceeds safe bounds.
  5. *Kernel Provider:* Binds exact kernel implementations (e.g. CubeCL fused paged attention vs portable CPU fallback).

### 3.3 Step 3: Feasibility Validation ($\Gamma(S_n, A_n; \mathcal{C}) = 1$)
Before committing an action, the runtime runs [`ExecutionPlan::validate`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/plan.rs#L696). It verifies:
1. **Single-Role Invariant:** No request appears multiple times in contradictory roles (e.g., cannot be both preempted and decoded).
2. **Phase Admissibility:** Decode steps are only assigned to requests that completed prefill.
3. **Chunking Bounds:** Prompt chunks do not exceed configured ceilings or process beyond prompt length.
4. **Memory Feasibility:** Active reservations do not exceed the physical capacity of the model's KV pool.

*Adaptive Safety Fallback:* In [`AdaptivePolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/adaptive.rs), candidate parameter evaluations are checked against `validate`. If the candidate plan fails feasibility, the system logs the divergence and automatically runs $\pi_{\text{safe}}$ (e.g. strict FIFO with bounded tokens), ensuring the serving engine never crashes or stalls.

### 3.4 Step 4: Physical Graph Materialization ($G_P(S_n, A_n)$)
The abstract plan is compiled into physical [`ExecutionBatch`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/batch.rs) structures:
- Memory is atomically reserved in the model's KV pool ([`MistralKvPool`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/mistral/kv_cache.rs) or [`LlamaKvPool`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/llama/kv_cache.rs)).
- Variable-length prompts are right-padded into 2D tensor matrices, while prompt masks ensure tokens attend only to real sequence positions.
- In decode mode, single query tokens are batched across active requests, with block tables passed directly to the paged attention kernel.

### 3.5 Step 5: Telemetry, Observability, and Feedback
When model forward passes finish:
- Output tokens, generated IDs, and stop tokens are emitted.
- Detailed timing telemetry is stamped via [`ola-observe`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-observe/src/lib.rs): Time-To-First-Token (TTFT), Inter-Token Latency (Time-Between-Tokens / TBT), and synchronous host durations.
- The executor calls `policy.observe(&iteration)`.
- The adaptive policy inspects performance rewards (e.g., SLO compliance, throughput, latency penalties) and updates its parameter vector $\theta$ for the next iteration.

---

## 4. Deep-Dives into Core Subsystems

### 4.1 Subsystem A: Scheduling and Batching Policies ($A^{\text{sched}}, A^{\text{batch}}$)

In your paper (§8.3, §8.4, §19), batching and scheduling control how the token budget $B_n$ is partitioned between prompt tokens and decode steps:

$$\sum_{i \in P_n} x_{i,n} + \sum_{j \in D_n} y_{j,n} \le B_n$$

In `ola-runtime`, this is implemented across modular components:

1. **Request Schedulers ([`ola_runtime::scheduler`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/scheduler.rs)):**
   - [`FirstInFirstOut`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/scheduler.rs#L40): Standard arrival-order queueing.
   - [`ShortestJobFirst`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/scheduler.rs#L85): Prioritizes requests with shorter prompts/expected generation to minimize mean waiting time.
   - [`EarliestDeadlineFirst`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/scheduler.rs#L130): Ranks by absolute SLO deadline ($a_i + d_i$), directly optimizing deadline slack.
   - [`PriorityScheduler`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/scheduler.rs#L180): Multi-tier priority queueing with configurable aging intervals to prevent lower-priority starvation.

2. **Batching Strategies ([`ola_runtime::batch`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/batch.rs)):**
   - **Continuous Batching:** Requests transition independently between prefill and decode phases; batch membership changes at every iteration.
   - **Chunked Prefill:** Long prompts are partitioned into chunks $c_i^{\text{pf}}(n) \le C_{\text{chunk}}$. Decode requests run concurrently with prefill chunks, preventing prefill bursts from stalling latency-sensitive decoding.
   - **Cost-Aware Batching:** Evaluates non-linear grouping overheads when deciding whether to combine prefill chunks into a single right-padded matrix or run them sequentially.

### 4.2 Subsystem B: Paged KV-Cache & Sliding-Window Memory ($A^{\text{kv}}$)

In Section 8.6, KV management is formalized as:
$$A_n^{\text{kv}} = (\text{alloc}, \text{retain}, \text{evict}, \text{offload}, \text{migrate}, \text{reuse}, \text{recompute})$$

`ola-burn` and `ola-runtime` implement this through paged physical storage:

```
Logical Request Sequence:   [Tok 0] [Tok 1] [Tok 2] [Tok 3] | [Tok 4] [Tok 5] [Tok 6] [Tok 7]
                                      │                                       │
Logical Page Table:              Page 0 (Block 14)                       Page 1 (Block 3)
                                      │                                       │
Physical Pool Memory:     [Block 0] ... [Block 3] ... [Block 14] ... [Free Blocks]
```

1. **Physical Block Paging ([`KvBlockPool`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/kv_cache.rs)):**
   - Key/Value tensors are allocated in discrete blocks (e.g. 16 or 32 tokens per block).
   - Eliminates external memory fragmentation. Logical request addresses are dynamically mapped to non-contiguous physical pages via runtime block tables.

2. **Prefix Caching & Radix Sharing:**
   - Completed blocks are indexed by content hash.
   - When requests arrive with common prefixes (e.g. few-shot system prompts), the runtime attaches existing blocks directly into the new request's page table, avoiding prefill recomputation entirely.

3. **Sliding-Window Attention (Mistral Support):**
   - In [`MistralKvCache::read_layer_window`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-burn/src/mistral/kv_cache.rs), when sliding window $W$ is configured, decode attention slices only the most recent $\min(\text{tokens}, W)$ cached positions across non-contiguous physical blocks.
   - Bounds the memory footprint and attention complexity to $\mathcal{O}(W)$ rather than $\mathcal{O}(L_{\text{seq}})$.

4. **Preemption & Recompute Recovery ([`PreemptionPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/preemption.rs)):**
   - When memory pressure triggers a shortfall, the runtime preempts live requests (using [`NewestFirst`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/preemption.rs)), freeing their physical pages.
   - When resumed, the request recomputes prompt + generated tokens in a single unified prefill pass, guaranteeing identical output to uninterrupted execution.

### 4.3 Subsystem C: Cross-Layer Kernel Optimization ($A^{\text{kernel}}$)

In Section 21 and Section 26.1, your paper presents a critical cross-layer insight:
$$\arg\min_k T_{\text{kernel}} \ne \arg\min_k J_{\text{system}}$$

A locally faster attention kernel that requires a large scratchpad workspace $M_{\text{workspace}}$ can reduce the memory available for KV blocks, shrinking the feasible batch size $B_n$ and degrading global system goodput.

In `ola-kernel`:
- [`PagedAttentionKernel`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-kernel/src/paged_attention.rs) abstracts attention implementations behind a validated contract: query shapes, block tables, page sizes, and KV head ratios.
- [`KernelRegistry`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-kernel/src/registry.rs) tracks available providers:
  1. *Portable Reference (`PortablePagedAttention`):* Pure Burn tensor implementation, verified on CPU and GPU.
  2. *Fused Kernel (`CubeClPagedAttention`):* Fused CubeCL kernel that traverses physical block tables, computes attention scores, performs online softmax, and accumulates values in a single dispatch.
- The planning policy can explicitly select providers based on memory/latency trade-offs, and every iteration report records the exact provider invoked.

### 4.4 Subsystem D: Adaptive Policy & Optimization Engine ($\pi(S_n; \theta)$)

In Sections 11.2, 15, 27, and Appendix E, you define the adaptive controller:

$$A_n = \arg\min_{a \in \hat{\mathcal{A}}_n} \left[ \hat{J}_n(a) + \lambda_{\text{sw}} C_{\text{switch}}(A_{n-1}, a) \right]$$

In [`ola-runtime::adaptive`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/adaptive.rs):
- **Parameter Vector $\theta$:** Represents tunable runtime knobs:
  - `token_budget`: Maximum batch token ceiling.
  - `chunk_size`: Maximum prompt segment size for chunked prefill.
  - `scheduler`: Active scheduling discipline (`fifo`, `sjf`, `edf`, `priority`).
  - `overhead_penalty`: Cost model multiplier for batched prefill padding.
- **LP/MILP Formulation (`good_lp`):** Solves constrained optimization over candidate parameters, ensuring that chosen settings satisfy memory and active slot limits.
- **Observation Attribution:** [`DecisionRecord`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/decision.rs) records the exact $\theta$ parameters and observation counters for every iteration, enabling offline policy evaluation and gradient estimation.

---

## 5. Structural Propositions & Implementation Proofs

Your paper establishes several formal structural propositions in Section 17. The test suite in `ola-rt` serves as empirical and programmatic proofs of these theorems:

### Proposition 1: Static Configurations are Execution Policies ($\Pi_{\text{static}} \subseteq \Pi_{\text{IEP}}$)
- **Paper Statement:** A fixed static configuration (e.g. constant batch size, fixed chunk size) is simply an IEP where $\pi(H_n) = a^*$ for all states.
- **Code Realization:** The executor accepts any [`ExecutionPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/policy.rs). A static policy is simply a [`ComposedPolicy`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/composed.rs) initialized with fixed parameters. It shares the exact same planning and validation boundaries as adaptive controllers.

### Proposition 2: Static Non-Universality
- **Paper Statement:** If state $s_1$ (e.g. low-load, latency-sensitive) uniquely prefers action $a_1^*$ and state $s_2$ (e.g. high-load, throughput-sensitive) uniquely prefers action $a_2^*$ with $a_1^* \ne a_2^*$, then no static configuration can be optimal across both states.
- **Code Realization:** In `ola-bench/src/scheduling_summary.rs`, the `adaptive_budget_recovers_throughput_when_overhead_dominates` test proves that when fixed-overhead regimes change, a fixed token budget suffers substantial goodput loss compared to a budget that adjusts with batch composition.

### Proposition 3: Finite-Memory Controllers as Markov Policies
- **Paper Statement:** Any deterministic controller with internal memory $m_n$ (e.g. moving averages, MLFQ levels) is Markovian over an augmented state $\tilde{S}_n = (S_n, m_n)$.
- **Code Realization:** In [`PriorityScheduler`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/scheduler.rs), requests track waiting age; priority promotion occurs at discrete thresholds. The policy state is completely captured in [`PlanningSnapshot`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-runtime/src/snapshot.rs) via arrival timestamps, maintaining strict functional repeatability without hidden external state.

### Proposition 4: Output-Length Uncertainty Induces Partial Observability
- **Paper Statement:** Because true generation length $L_i^{\text{out}}$ is unknown at arrival, serving is partially observed. Memory allocation based on worst-case bounds underutilizes cache, while underestimation risks out-of-memory preemption.
- **Code Realization:** In `ola-runtime/tests/preemption.rs` and `ola-burn/src/llama/session.rs`, preemption tests demonstrate that when requests exceed allocated cache margins, the runtime detects typed KV exhaustion and triggers recovery, proving that memory allocation without length beliefs requires dynamic preemption mechanisms.

---

## 6. Empirical Blueprint: Generating Results for Section 43

Section 43 of your paper contains the placeholder:
> `[RESULT TO INSERT: This section must be written only after experiments are performed. Do not manufacture quantitative claims.]`

The benchmarking suite in [`ola-bench`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/lib.rs) and the CLI in [`ola-cli`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-cli/src/lib.rs) were engineered specifically to generate the tables and figures required to complete this section.

### 6.1 Section 43.1: Static Configuration Sensitivity & Optimality Gap
- **Goal:** Show that the optimal static configuration varies across workload regimes, proving Proposition 2 and quantifying $\Delta_{\text{static}}$.
- **Execution Command:**
  ```bash
  cargo run --release -p ola-cli -- experiment run scheduler-comparison --set requests=100
  ```
- **Analysis:** Compares FIFO, SJF, and EDF across varying arrival intensities ($\lambda$). Extract the P99 TTFT and goodput metrics from the resulting JSON report.
- **Paper Contribution:** Produces data for **Figure 4 (Static Optimality Gap)** and **Table 4 (Main Results)**.

### 6.2 Section 43.2: End-to-End Adaptive Policy Evaluation
- **Goal:** Compare the adaptive policy against the best static baseline on non-stationary traces.
- **Execution Command:**
  ```bash
  cargo run --release -p ola-cli -- experiment run adaptive-prefill-cost
  ```
- **Harness:** [`SimulationConfig`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/simulation.rs) runs virtual-time simulation with exact reproducibility across random seeds.
- **Metrics Extracted:**
  - Goodput: Successful requests per second meeting both TTFT and TPOT SLOs.
  - P95 / P99 TTFT: Time to first token distributions.
  - P95 / P99 TPOT: Time per output token distributions.
  - Jain Fairness Index: Normalized service equality across request classes.

### 6.3 Section 43.3: Cross-Layer Coordination
- **Goal:** Demonstrate that jointly controlling batching, KV caching, and kernel selection outperforms isolated subsystem tuning.
- **Execution Commands:**
  ```bash
  # Kernel & Workspace Impact
  cargo run --release -p ola-cli -- experiment run paged-attention-kernel
  
  # Prefix Caching & KV Reuse
  cargo run --release -p ola-cli -- experiment run prefix-cache
  
  # Chunked Prefill vs Whole Prefill
  cargo run --release -p ola-cli -- experiment run chunked-prefill
  ```
- **Paper Contribution:** Directly populates **Figure 7 (Cross-Layer Effect)** and **Table 3 (Controlled Dimensions)**.

### 6.4 Section 43.4: Workload Shift & Transition Behavior
- **Goal:** Evaluate policy resilience during step shifts, burst arrivals, and prompt-length transitions (§37.5–§37.8).
- **Implementation Mechanism:** [`SchedulingTrace::overload`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/trace.rs) and `SchedulingTrace::priority_classes` simulate sharp load spikes and sudden arrivals of long prompts.
- **Paper Contribution:** Generates the timeline plots for **Figure 5 (Dynamic Workload)**, showing detection delay, adaptation delay, and recovery time.

### 6.5 Section 43.6: Controller & Optimization Overhead
- **Goal:** Report the execution time of the policy engine to prove adaptation benefit exceeds control cost ($J_{\text{ctrl}} \ll J_{\text{op}}$).
- **Data Source:** Every run creates a [`PlanReport`](file:///home/olamide/dev/ola-rt/ola-rt/crates/ola-bench/src/plan_report.rs).
- **Inspection:** Inspect microsecond planning latencies:
  ```rust
  let report = PlanReport::from_json(&report_json).unwrap();
  // Records plan construction, validation duration, and provider resolution
  ```

---

## 7. Writing & Defense Strategy: High-Impact Positioning

When writing conference papers (e.g., OSDI, SOSP, EuroSys, ASPLOS, MLSys, SIGMETRICS) or preparing your PhD dissertation defense, use the following rhetorical and methodological strategies:

### 1. Frame the Problem Beyond Scheduling
*Do not say:* "We propose a new scheduler for LLM inference." (Reviewers will cite Orca, FastServe, and vLLM and claim the space is crowded.)  
*Do say:* "Existing systems treat scheduling, KV paging, chunking, and kernel dispatch as disconnected heuristics. We formalize the **joint execution policy** governing these coupled decisions and provide an explicit admissibility contract that prevents performance optimization from silently sacrificing model semantics."

### 2. Emphasize Semantic Equivalence Contracts
Highlight Section 9 of your paper. Many serving benchmarks report throughput gains achieved by quantization or approximate kernels without making quality degradation explicit. By formally partitioning actions into:
$$\mathcal{A} = \mathcal{A}_{\text{exact}} \cup \mathcal{A}_{\text{approx}}$$
OlaRT guarantees that system-level optimizations (like continuous batching, prefix reuse, and chunking) strictly preserve bitwise or numerical equivalence before exploring approximate trade-offs.

### 3. Leverage Rust-Enforced Systems Invariants
In systems venues, reviewers look for proof that an architecture is not just a simulator, but an operable runtime:
- **Zero-Unchecked-Plan Invariant:** The policy produces an immutable candidate plan; the runtime validates feasibility ($\Gamma = 1$) before state changes.
- **Decoupled Learning Invariant:** `plan(&self)` is deterministic and non-mutating; adaptation occurs strictly in `observe(&mut self)`, preventing race conditions and non-deterministic planning anomalies.
- **Atomic Multi-Sequence Reservation:** KV memory reservations for batches are atomic, eliminating partial-allocation deadlocks.

---

## 8. Summary Checklist for Paper Milestones

- [ ] **Paper 1 (Formalization & Architecture):**
  - [ ] Finalize mathematical definitions (§5–§16) using the notation in Appendix A.
  - [ ] Incorporate architecture diagram of OlaRT (§31) mirroring Section 2 of this guide.
  - [ ] Populate Table 1 & Table 3 (Taxonomy of prior systems expressed as restricted IEP projections).
  - [ ] Embed the static non-universality proof (Proposition 2).
- [ ] **Paper 2 (Empirical Evaluation & Workload Characterization):**
  - [ ] Run `scheduler-comparison`, `prefix-cache`, and `chunked-prefill` benchmark suites.
  - [ ] Extract bootstrap confidence intervals from `ola-bench` JSON reports.
  - [ ] Generate Figures 4, 5, 6, and 7 from benchmark traces.
  - [ ] Populate Table 4 with empirical Goodput, P99 TTFT, and TPOT numbers.
- [ ] **Paper 3 (Adaptive Control & Non-Stationary Learning):**
  - [ ] Evaluate `AdaptivePolicy` against fixed baselines on non-stationary traces.
  - [ ] Report switching costs, hysteresis stability, and controller execution overhead ($J_{\text{ctrl}}$).
