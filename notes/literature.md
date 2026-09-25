For your PhD direction, I would build the literature review around **inference execution as a decision/optimization problem**, rather than treating “LLM inference” as one giant topic. Your central thread is roughly:

**workload + model + hardware state → execution policy → runtime decisions → latency/throughput/cost/energy/SLO outcomes**

That puts your work at the intersection of **AI systems, runtime systems, scheduling, Operations Research, online optimization, compilers, and inference algorithms**.

A very good starting point is the 2025 *A Survey of LLM Inference Systems*. It explicitly organizes inference around operators, execution, batching, scheduling, memory management, quantization, caching, adaptive mechanisms, and distributed/disaggregated systems—almost exactly the landscape you need to position Ola-RT. ([arXiv][1])

## 1. Start with these 3 surveys

Read these before going deeply into individual papers.

| Priority | Paper                                                                                                                        | Why                                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| ⭐⭐⭐⭐⭐    | **A Survey of LLM Inference Systems** — Pan & Li, 2025                                                                       | Probably your best high-level map. Covers kernels → batching → scheduling → memory → distributed inference. ([arXiv][1]) |
| ⭐⭐⭐⭐⭐    | **LLM Inference Serving: Survey of Recent Advances and Opportunities** — Li et al., 2024                                     | Strong systems perspective on serving developments since 2023. ([arXiv][2])                                              |
| ⭐⭐⭐⭐     | **A Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency** — Park et al., 2025 | Compares 25 inference engines and their optimization techniques. ([arXiv][3])                                            |

These can help you construct the taxonomy for your literature review.

---

# 2. Core inference-runtime papers

These are the papers I would consider **mandatory reading** for Ola-RT.

### Orca — iteration-level scheduling

**Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models”, OSDI 2022**

This is foundational.

The important idea is **iteration-level scheduling** rather than request-level batching.

Instead of:

$$
\text{batch requests}\rightarrow \text{run until all finish}
$$

Orca effectively makes scheduling decisions around generation iterations:

$$
B_t = \pi(Q_t,S_t)
$$

where \(Q_t\) is the request queue and \(S_t\) is runtime state.

That concept is extremely close to what you mean by an **inference execution policy**.

**Read this very carefully.**

---

### vLLM / PagedAttention

**Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023**

PagedAttention treats KV-cache storage somewhat like virtual-memory paging, allowing non-contiguous cache blocks and reducing fragmentation. vLLM reported 2–4× throughput improvements over contemporary systems in its evaluation. ([DOI][4])

This paper should probably have an entire section in your literature review:

> **Inference Memory Management**

Study:

* KV-cache allocation
* block management
* fragmentation
* scheduling interaction
* cache sharing
* eviction
* preemption

Your eventual policy representation should probably include **memory-management actions**.

---

### FlexGen

**Sheng et al., “FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU”, 2023**

Very relevant to your OR direction.

FlexGen decides where tensors should live across:

$$
GPU \leftrightarrow CPU \leftrightarrow Disk
$$

and formulates part of the configuration problem using **linear programming**. ([arXiv][5])

This is one of the strongest papers connecting:

> inference systems ↔ mathematical optimization

Read its optimization formulation closely.

---

### FastServe

**Wu et al., “Fast Distributed Inference Serving for Large Language Models”, 2023**

Introduces token-level preemption and a modified **Multi-Level Feedback Queue** scheduler. ([arXiv][6])

Very relevant because your policy space might contain:

$$
\text{preempt}(r)
$$

$$
\text{resume}(r)
$$

$$
\text{priority}(r)
$$

$$
\text{queue}(r)
$$

This starts looking much more like an operating-system scheduler.

---

# 3. Scheduling and execution-policy papers

This cluster is probably the **closest to your first two papers**.

### Llumnix

**Sun et al., “Llumnix: Dynamic Scheduling for Large Language Model Serving”, 2024**

One of your most important papers.

Llumnix addresses heterogeneous and unpredictable workloads by dynamically rescheduling requests between model instances, including live migration of inference state. ([arXiv][7])

Conceptually:

$$
\pi(s_t)\rightarrow
\{\text{execute},\text{wait},\text{migrate},\text{preempt},...\}
$$

That is very close to your execution-policy abstraction.

⭐⭐⭐⭐⭐

---

### Sarathi / Sarathi-Serve

Look for:

**“Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve”**

Key idea:

**chunked prefill**

Rather than allowing a huge prompt to block decoding:

$$
P = P_1 + P_2 + \dots + P_k
$$

and interleave those chunks with decode batches.

This introduces another policy dimension:

$$
\text{prefill chunk size}
$$

Your execution-policy formalism absolutely should model this.

⭐⭐⭐⭐⭐

---

### DistServe

**Zhong et al., “DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving”, 2024**

Extremely relevant.

Separates:

$$
\text{Prefill GPUs}
$$

from

$$
\text{Decode GPUs}
$$

because the two stages have very different resource characteristics.

The system jointly considers TTFT and TPOT constraints and optimizes resource allocation and parallelism for the two phases. ([arXiv Science][8])

This gives you another policy variable:

$$
\text{placement}(phase,device)
$$

⭐⭐⭐⭐⭐

---

### Splitwise

**“Splitwise: Efficient Generative LLM Inference Using Phase Splitting”**

Same general family: separate prompt/prefill from token generation.

Useful to compare against DistServe.

---

### P/D-Serve

**“P/D-Serve: Serving Disaggregated Large Language Model at Scale”, 2024**

Covers practical prefill/decode infrastructure including KV-cache transfer and dynamically organizing prefill/decode instances. ([arXiv][9])

Especially useful when you go from mathematical policy to **cluster implementation**.

---

### Preble

**“Preble: Efficient Distributed Prompt Scheduling for LLM Serving”, 2024**

Important for:

* distributed scheduling
* prefix-cache locality
* prompt reuse
* global vs local scheduling
* load balancing

It uses a two-level architecture with a global request-level scheduler and per-instance iteration-level schedulers. ([arXiv][10])

This is particularly relevant to a hierarchical Ola-RT policy:

$$
\pi =
(\pi_{\text{cluster}},\pi_{\text{device}},\pi_{\text{iteration}})
$$

---

# 4. The Operations Research literature

This is where your PhD can become substantially different from a normal ML-systems thesis.

## LLM Serving Optimization with Variable Prefill and Decode Lengths

**Wang, Ye & Zhou, 2025**

⭐⭐⭐⭐⭐ for you.

They formulate scheduling heterogeneous prefill/decode workloads and show important variants of the problem are NP-hard. They investigate scheduling algorithms, dynamic programming, local search, and LP-based scheduling. ([arXiv][11])

This paper is very close to your intended bridge:

$$
\text{LLM Runtime}
\longleftrightarrow
\text{Operations Research}
$$

Read the mathematical formulation carefully.

---

## Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization

**Pang et al., 2025**

Another particularly relevant paper.

It formulates inference scheduling using a **mixed-integer programming model**, then combines offline optimization with online scheduling and uses a Lagrangian method for prefill/decode decisions. ([arXiv][12])

This should definitely appear in your literature review.

It also hints at exactly where your work can go:

$$
\text{MILP}
\rightarrow
\text{online optimization}
\rightarrow
\text{RL / learned policy}
$$

---

# 5. Older systems literature you should not skip

Your thesis shouldn't make it appear as if inference scheduling began with LLMs.

There is a lineage.

### Clipper

**Crankshaw et al., “Clipper: A Low-Latency Online Prediction Serving System”, NSDI 2017**

Uses batching, caching and adaptive model selection for online inference. ([arXiv][13])

Important historically.

---

### INFaaS

**Romero et al., “INFaaS: A Model-less and Managed Inference Serving System”, 2019/ATC 2021**

Very relevant to your policy idea.

Given performance/accuracy requirements, INFaaS dynamically selects:

* model variant
* hardware
* model optimizations
* autoscaling strategy

as workloads change. ([arXiv][14])

In abstract terms:

$$
\pi(s)=
(\text{model},\text{device},\text{optimization},\text{replicas})
$$

That's remarkably close to an execution-policy representation.

⭐⭐⭐⭐⭐

---

### Clockwork

**“Serving DNNs like Clockwork: Performance Predictability from the Bottom Up”**

Read for:

* deterministic execution
* predictable latency
* scheduling
* resource control
* SLOs.

Very useful when defining what an execution policy actually controls.

---

### Nexus

**“Nexus: A GPU Cluster Engine for Accelerating DNN-based Video Analysis”**

Useful for:

* batching
* GPU allocation
* latency-aware scheduling
* heterogeneous workloads.

---

### AlpaServe

**“AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving”**

Important for the interaction between:

$$
\text{model parallelism}
+
\text{replication}
+
\text{placement}
+
\text{workload distribution}
$$

---

# 6. KV-cache / memory-management literature

Make this a major subsection.

Read:

1. **PagedAttention / vLLM**
2. **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
3. **Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression**
4. **StreamingLLM: Efficient Streaming Language Models with Attention Sinks**
5. **SnapKV**
6. **PyramidKV**
7. **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**
8. **CacheGen**
9. **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**
10. **LMCache-related work**

Why these matter:

KV cache becomes a runtime resource:

$$
M_{\text{KV}}(r,t)
$$

and your scheduler decides things such as:

$$
\{\text{retain},\text{evict},\text{compress},
\text{quantize},\text{offload},\text{migrate}\}
$$

That's a rich optimization problem in its own right.

---

# 7. Attention and kernel papers

These are necessary because Ola-RT eventually needs to understand that a policy decision can happen below the scheduler.

### FlashAttention

**Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”**

Mandatory.

The crucial concept is **IO-aware algorithms**:

$$
\text{performance}\neq \text{FLOPs alone}
$$

Instead:

$$
T \approx f(
\text{compute},
\text{HBM traffic},
\text{SRAM},
\text{occupancy}
)
$$

---

### FlashAttention-2

**Dao, 2023**

Improves work partitioning between thread blocks and warps and reduces non-matmul work. The paper reports roughly 2× improvement over FlashAttention in its tested configurations and 50–73% of theoretical A100 FLOP/s. ([arXiv][15])

⭐⭐⭐⭐⭐

Then look at:

* **FlashAttention-3**
* **FlashInfer**
* **FlashDecoding**
* **FlashDecoding++**
* **Hydragen**

These expose another decision layer:

$$
\text{kernel} =
\pi(
shape,
dtype,
sequence\ length,
batch,
GPU
)
$$

which connects directly with the Burn/CubeCL autotuning discussion we had.

---

# 8. Compiler and kernel-selection literature

This area becomes critical when Ola-RT progresses from scheduler → complete inference runtime.

Read:

### TVM

**Chen et al., “TVM: An Automated End-to-End Optimizing Compiler for Deep Learning”, OSDI 2018**

Mandatory.

---

### Ansor

**“Ansor: Generating High-Performance Tensor Programs for Deep Learning”, OSDI 2020**

Very relevant to search-based optimization.

---

### AutoTVM

Read alongside TVM.

---

### Tensor Comprehensions

**“Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions”**

---

### Triton

**Tillet et al., “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations”**

Essential for understanding kernel abstraction.

---

### MLIR

**Lattner et al., “MLIR: Scaling Compiler Infrastructure for Domain Specific Computation”**

Important when you design Ola-RT's IR.

---

### XLA

Study XLA/HLO architecture and related papers.

Your future architecture may look like:

$$
Model
\rightarrow Graph
\rightarrow OlaIR
\rightarrow ExecutionPlan
\rightarrow KernelIR
\rightarrow Device
$$

---

# 9. Quantization literature

You already started studying this in Burn, so formalize it into the review.

### LLM.int8()

**Dettmers et al., 2022**

### GPTQ

**Frantar et al., “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers”**

### SmoothQuant

**Xiao et al., 2022/2023**

Training-free W8A8 quantization that moves quantization difficulty from activation outliers into weights. ([arXiv][16])

### AWQ

**Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”**

### QuIP / QuIP#

### AQLM

### SpQR

### KIVI

### QQQ

QQQ is especially interesting because it combines quantization technique with specialized W4A8 kernels, making the connection between numerical representation and actual runtime performance explicit. ([arXiv][17])

Your execution policy could eventually select:

$$
precision_t\in
\{FP16,BF16,FP8,INT8,INT4,\ldots\}
$$

subject to:

$$
AccuracyLoss \leq \epsilon.
$$

That turns precision into another optimization variable.

---

# 10. Speculative decoding

Another important runtime policy dimension.

Read:

### Speculative Decoding

**Leviathan et al., “Fast Inference from Transformers via Speculative Decoding”**

and

**Chen et al., “Accelerating Large Language Model Decoding with Speculative Sampling”**

Then:

* **Medusa**
* **EAGLE**
* **EAGLE-2**
* **EAGLE-3**
* **Lookahead Decoding**
* **REST**
* **SpecInfer**

Policy decisions now include:

$$
\text{draft model}
$$

$$
k=\text{number of speculative tokens}
$$

$$
\text{verification strategy}
$$

and even whether speculation should be enabled at all.

That becomes particularly interesting under non-stationary workloads.

---

# 11. Parallelism literature

You need this for distributed Ola-RT.

Read the major lineage:

### Megatron-LM

**“Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism”**

### GPipe

### PipeDream

### DeepSpeed / ZeRO

Although these originated mostly around training, they establish:

* tensor parallelism
* pipeline parallelism
* data parallelism
* memory partitioning.

Then study inference-specific parallelism in:

* TensorRT-LLM
* DeepSpeed-FastGen
* vLLM
* DistServe
* Splitwise
* Megatron inference.

Your policy variables may eventually include:

$$
TP\in \{1,2,4,8\}
$$

$$
PP\in \{1,2,4,\ldots\}
$$

$$
DP\in \{1,2,\ldots\}
$$

with the constraint

$$
TP\times PP\times DP \le N_{\text{GPU}}.
$$

That is an obvious combinatorial-optimization problem.

---

# 12. Non-stationary workloads and adaptive policies

This is the theoretical literature you need for your proposed:

> **Learning Inference Execution Policies under Non-Stationary Workloads**

Do **not** restrict yourself to LLM papers here.

You need the theory from online decision-making.

### Multi-Armed Bandits

Read:

**Auer et al. — Finite-time Analysis of the Multiarmed Bandit Problem**

Then:

* UCB
* Thompson Sampling
* contextual bandits
* non-stationary bandits
* sliding-window UCB
* discounted UCB
* change-point bandits.

---

### Reinforcement Learning

Obviously:

**Sutton & Barto** as background.

Papers/topics:

* Q-learning
* DQN
* PPO
* actor-critic
* constrained RL
* safe RL
* offline RL.

But I wouldn't jump directly to deep RL for your scheduler.

Start with OR/online optimization.

---

# 13. Online optimization

This could become one of the theoretical foundations of the PhD.

Study:

### Online Convex Optimization

**Zinkevich, “Online Convex Programming and Generalized Infinitesimal Gradient Ascent”, 2003**

Then:

* Online Gradient Descent
* Follow-The-Regularized-Leader
* Mirror Descent
* dynamic regret
* adaptive regret.

Especially study **dynamic regret**:

$$
R_T^{dynamic}
=
\sum_{t=1}^{T}
f_t(x_t)
-
\sum_{t=1}^{T}
f_t(x_t^*)
$$

because your optimum changes as:

* arrival rate changes
* sequence lengths change
* GPU availability changes
* model mix changes.

That is almost exactly a non-stationary inference workload.

---

# 14. Queueing theory

This may become surprisingly important to your thesis.

Read the literature around:

### M/M/1

$$
L=\frac{\rho}{1-\rho}
$$

### M/G/1

### Processor Sharing

### Priority Queues

### SRPT — Shortest Remaining Processing Time

### Join-the-Shortest-Queue

### Power-of-Two Choices

### Heavy-traffic scheduling

### Queueing networks

Then relate these to:

$$
Arrival\ Process
\rightarrow Queue
\rightarrow Batch
\rightarrow GPU
\rightarrow Response
$$

FastServe's scheduling work is already effectively touching this domain. ([arXiv][6])

---

# 15. Lyapunov optimization

This is something I would specifically recommend you investigate.

It is widely used for online resource allocation under uncertainty.

Core source:

**Michael Neely — Stochastic Network Optimization with Application to Communication and Queueing Systems**

You get formulations like:

$$
\min \lim_{T\rightarrow\infty}
\frac1T
\sum_t E[C_t]
$$

subject to queues remaining stable.

Interestingly, very recent work is already applying Lyapunov optimization to dynamic LLM inference scheduling. For example, a September 2026 paper studies edge LLM scheduling with latency and load-balancing constraints using Lyapunov optimization and learning components. ([arXiv][18])

This is a direction worth watching.

---

# 16. VLM / multimodal inference

Since Ola-RT eventually needs VLM inference, add a dedicated literature-review section.

First read architecture papers:

* **CLIP**
* **ViT**
* **Flamingo**
* **BLIP-2**
* **LLaVA**
* **LLaVA-1.5**
* **Qwen-VL**
* **Qwen2-VL**
* **InternVL**
* **Gemma multimodal / PaliGemma**
* **SAM / SAM 2 / SAM 3**, where applicable to your vision-runtime experiments.

But your research question shouldn't mainly be about architectures.

Look at the **systems consequences**:

$$
Image
\rightarrow VisionEncoder
\rightarrow Projection
\rightarrow LLM Prefill
\rightarrow Decode
$$

Now the runtime has heterogeneous stages:

$$
GPU_{vision}
\rightarrow GPU_{LLM}
$$

and different resource demands.

Eventually you get an optimization problem like:

$$
\pi(s_t)=
(
device_{vision},
batch_{vision},
device_{llm},
batch_{prefill},
batch_{decode},
precision,
kernel,
cache\ policy
)
$$

This is an extremely natural extension of your thesis.

---

# 17. Mixture-of-Experts papers

Since we just discussed Cursor's “Mixture of Kittens”-style optimization ideas, MoE is also worth including.

Read:

### Switch Transformer

**Fedus et al.**

### GShard

### DeepSpeed-MoE

### MegaBlocks

**“MegaBlocks: Efficient Sparse Training with Mixture-of-Experts”**

### Tutel

### Expert Choice Routing

### DeepSeekMoE

### DeepSeek-V2 / V3 technical reports

For inference, MoE adds another execution-policy problem:

$$
token
\rightarrow experts
\rightarrow GPU placement
\rightarrow communication
$$

So:

$$
\pi_{\text{expert}}
:
(token,state)
\rightarrow
(device,expert)
$$

creates routing, placement and load-balancing problems.

---

# 18. Benchmarking papers/systems

Because we've also discussed benchmarking, your literature review needs a methodology section.

Read/study:

* **MLPerf Inference**
* **MLPerf Tiny**
* **MLPerf LoadGen**
* **HELM**
* **LLMPerf**
* **GenAI-Perf**
* **vLLM benchmark methodology**
* **TensorRT-LLM benchmark methodology**
* **SGLang benchmarking**

Metrics:

$$
TTFT
$$

$$
TPOT
$$

$$
ITL
$$

$$
P50,\ P95,\ P99
$$

$$
tokens/sec
$$

$$
requests/sec
$$

$$
goodput
$$

$$
GPU\ utilization
$$

$$
HBM\ utilization
$$

$$
energy/token
$$

$$
cost/1M\ tokens
$$

Eventually Ola-RT should have a benchmark harness around all of these.

---

# 19. The papers most directly connected to **your actual thesis**

If I reduce everything to about **20 papers you should understand extremely deeply**, I would choose:

| #  | Paper                                                                 | Thesis connection                     |
| -- | --------------------------------------------------------------------- | ------------------------------------- |
| 1  | **Orca**                                                              | iteration-level execution policy      |
| 2  | **vLLM / PagedAttention**                                             | memory policy                         |
| 3  | **FlexGen**                                                           | LP + inference optimization           |
| 4  | **FastServe**                                                         | preemption + scheduling               |
| 5  | **Llumnix**                                                           | dynamic scheduling                    |
| 6  | **Sarathi-Serve**                                                     | chunked prefill                       |
| 7  | **DistServe**                                                         | prefill/decode placement              |
| 8  | **Splitwise**                                                         | phase disaggregation                  |
| 9  | **Preble**                                                            | distributed scheduling/cache locality |
| 10 | **INFaaS**                                                            | adaptive inference configuration      |
| 11 | **Clockwork**                                                         | predictable execution                 |
| 12 | **LLM Serving Optimization with Variable Prefill and Decode Lengths** | scheduling theory/OR                  |
| 13 | **Hybrid Offline-online Scheduling for LLM Inference Optimization**   | MIP + online optimization             |
| 14 | **FlashAttention**                                                    | hardware-aware execution              |
| 15 | **FlashAttention-2**                                                  | kernel optimization                   |
| 16 | **TVM**                                                               | compiler/runtime                      |
| 17 | **Ansor**                                                             | automatic schedule search             |
| 18 | **Triton**                                                            | kernel abstraction                    |
| 19 | **SmoothQuant**                                                       | precision policy                      |
| 20 | **Speculative Decoding**                                              | decoding-policy optimization          |

Those twenty form a very coherent research story.

---

# 20. I would structure your literature review like this

Your eventual PhD literature review could have roughly:

```text
1. Deep Learning and Transformer Inference
   1.1 Transformer execution
   1.2 Prefill and decode
   1.3 Compute and memory characteristics

2. Large-Scale Model Serving
   2.1 Early ML serving systems
       Clipper
       Clockwork
       INFaaS
       Nexus

   2.2 LLM serving systems
       Orca
       vLLM
       FastServe
       Sarathi

3. Memory Management for Generative Inference
   3.1 KV cache
   3.2 PagedAttention
   3.3 Cache compression
   3.4 Cache offloading
   3.5 Prefix caching

4. Scheduling and Resource Management
   4.1 Continuous batching
   4.2 Preemption
   4.3 Chunked prefill
   4.4 Request migration
   4.5 Load balancing

5. Distributed and Disaggregated Inference
   5.1 Tensor/model parallelism
   5.2 Prefill/decode disaggregation
   5.3 Distributed KV caches
   5.4 Cluster scheduling

6. Inference Optimization
   6.1 LP/MILP
   6.2 Combinatorial optimization
   6.3 Queueing theory
   6.4 Online optimization
   6.5 Stochastic optimization

7. Adaptive Execution
   7.1 Non-stationary workloads
   7.2 Contextual bandits
   7.3 Reinforcement learning
   7.4 Dynamic regret
   7.5 Learned scheduling

8. Model and Numerical Optimizations
   8.1 Quantization
   8.2 Speculative decoding
   8.3 Sparsity
   8.4 Mixture-of-Experts

9. Compiler and Kernel Optimization
   9.1 Graph compilers
   9.2 IR
   9.3 Fusion
   9.4 Kernel generation
   9.5 Autotuning
   9.6 Hardware-aware selection

10. Multimodal Inference Systems
    10.1 Vision encoders
    10.2 VLM execution
    10.3 Heterogeneous pipelines
    10.4 Multimodal scheduling

11. Benchmarking and Evaluation
    11.1 Latency
    11.2 Throughput
    11.3 Goodput
    11.4 SLO attainment
    11.5 Cost
    11.6 Energy

12. Research Gap
    Existing systems implement specific policies.
    There is no sufficiently general representation of
    inference execution policies enabling systematic
    analysis, optimization and adaptation across
    workloads, models and hardware.
```

And **that final section is where your PhD starts**.

The important distinction is that your thesis shouldn't simply be:

> “I built another LLM scheduler.”

The stronger research thesis is:

$$
\boxed{
\text{Can inference execution itself be represented as a formal,
optimizable policy?}
}
$$

Then:

$$
\boxed{
\pi^*
=
\arg\min_{\pi\in\Pi}
J(\pi,W,H,M)
}
$$

where:

* \(\Pi\) = valid execution policies
* \(W\) = workload
* \(H\) = hardware
* \(M\) = model
* \(J\) = latency/cost/energy/SLO objective.

And eventually:

$$
\pi_t^*
=
f(W_t,H_t,M,S_t)
$$

for a changing environment.

That naturally connects **Orca → vLLM → Llumnix → DistServe → OR → online optimization → learned policies → Ola-RT**.

I would make **Orca, vLLM, FlexGen, Llumnix, Sarathi-Serve, DistServe, the two OR scheduling papers, FlashAttention, TVM/Ansor and INFaaS** your first deep-reading batch rather than trying to read 100 papers linearly. ([arXiv][5])

[1]: https://arxiv.org/abs/2506.21901?utm_source=chatgpt.com "A Survey of LLM Inference Systems"
[2]: https://arxiv.org/abs/2407.12391?utm_source=chatgpt.com "LLM Inference Serving: Survey of Recent Advances and Opportunities"
[3]: https://arxiv.org/abs/2505.01658?utm_source=chatgpt.com "A Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency"
[4]: https://doi.org/10.1145/3600006.3613165?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention | Proceedings of the 29th Symposium on Operating Systems Principles"
[5]: https://arxiv.org/abs/2303.06865?utm_source=chatgpt.com "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"
[6]: https://arxiv.org/abs/2305.05920?utm_source=chatgpt.com "Fast Distributed Inference Serving for Large Language Models"
[7]: https://arxiv.org/abs/2406.03243?utm_source=chatgpt.com "Llumnix: Dynamic Scheduling for Large Language Model Serving"
[8]: https://arxiv.science/abs/2401.09670?utm_source=chatgpt.com "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | arXiv Science"
[9]: https://arxiv.org/pdf/2408.08147?utm_source=chatgpt.com "P/D-Serve: Serving Disaggregated Large Language Model at Scale"
[10]: https://arxiv.org/pdf/2407.00023?utm_source=chatgpt.com "Preble: Efficient Distributed Prompt Scheduling for LLM Serving"
[11]: https://arxiv.org/abs/2508.06133?utm_source=chatgpt.com "LLM Serving Optimization with Variable Prefill and Decode Lengths"
[12]: https://arxiv.org/abs/2502.15763?utm_source=chatgpt.com "Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization"
[13]: https://arxiv.org/abs/1612.03079?utm_source=chatgpt.com "Clipper: A Low-Latency Online Prediction Serving System"
[14]: https://arxiv.org/abs/1905.13348?utm_source=chatgpt.com "INFaaS: A Model-less and Managed Inference Serving System"
[15]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
[16]: https://arxiv.org/abs/2211.10438?utm_source=chatgpt.com "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
[17]: https://arxiv.org/pdf/2406.09904?utm_source=chatgpt.com "QQQ: Quality Quattuor-Bit Quantization for Large Language Models"
[18]: https://arxiv.org/abs/2609.17193?utm_source=chatgpt.com "End-to-End Latency-Minimizing and Load-Balanced Request Scheduling for Edge LLM Inference in Agentic AI Services"

