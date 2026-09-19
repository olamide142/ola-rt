For the PhD direction we’ve been shaping — **AI inference systems → runtime architecture → formalized execution policies → adaptive scheduling under non-stationary workloads → distributed/multimodal inference** — I would build your literature programme in layers rather than reading papers randomly.

The key idea is:

> **Year 1:** understand how inference systems work and formalize them.
> **Year 2:** optimize and learn execution policies.
> **Year 3:** generalize the runtime to dynamic, distributed and multimodal workloads.

Recent inference-system surveys strongly support this decomposition: modern serving research spans kernels, execution, batching, scheduling, memory/KV-cache management, placement, disaggregation, load prediction and adaptive mechanisms. ([arXiv][1])

I would therefore make the following your **PhD paper-reading spine**.

---

# Phase 0 — Neural-network and Transformer execution foundations

### Months 0–3

You don't need to become a model-training researcher, but you need to know exactly **what the runtime is executing**.

### Core papers

| Paper                                                                                                           | Why it belongs                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Attention Is All You Need** — Vaswani et al., 2017                                                            | Transformer computational graph. Everything later depends on understanding attention, Q/K/V, heads, FFNs and autoregressive execution. |
| **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al., 2018                                 | Useful contrast between encoder-style inference and autoregressive decoder inference.                                                  |
| **Language Models are Few-Shot Learners** — Brown et al., 2020                                                  | Helps understand how model scale changed inference requirements.                                                                       |
| **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al., 2023                                 | Modern decoder-only architecture close to what you will actually run.                                                                  |
| **Llama 2** — Touvron et al., 2023                                                                              | Useful practical architecture reference.                                                                                               |
| **Mistral 7B** — Jiang et al., 2023                                                                             | Sliding-window attention, grouped-query attention and practical inference-oriented architecture.                                       |
| **Fast Transformer Decoding: One Write-Head is All You Need** — Shazeer, 2019                                   | Multi-query attention. Important because KV-cache size is partly an architectural problem.                                             |
| **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — Ainslie et al., 2023 | Bridges MHA and MQA and explains a major modern inference optimization.                                                                |

### What you should be able to explain afterwards

You should be able to draw:

```text
tokens
  ↓
embedding
  ↓
transformer block
  ├── QKV projection
  ├── attention
  ├── KV cache
  ├── output projection
  └── FFN
  ↓
logits
  ↓
sampling
  ↓
next token
```

and distinguish:

```text
PREFILL
prompt = 3000 tokens
        ↓
large parallel matrix operations

DECODE
token 3001
token 3002
token 3003
...
        ↓
many tiny sequential iterations
```

That distinction eventually becomes one of the most important abstractions in your runtime.

---

# Phase 1 — GPU execution and the memory hierarchy

### Months 2–6

This phase is critical for `ola-rt`.

You want to stop thinking:

> GPU = device that performs matrix multiplication.

and start thinking:

> GPU = hierarchical parallel machine whose performance is often determined by data movement.

## Must-read papers

### 1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**

Dao et al., 2022.

Probably one of the most important systems papers for you.

The key insight is that FLOPs are not the only relevant complexity measure. Moving tensors between HBM and SRAM can dominate execution. FlashAttention restructures exact attention around the GPU memory hierarchy. ([arXiv][2])

**Why read it**

Because it teaches the philosophy:

> Optimize **movement**, not merely arithmetic.

That philosophy should eventually influence your runtime scheduler.

---

### 2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**

Read immediately after FlashAttention.

It examines:

* thread blocks
* warps
* occupancy
* work partitioning
* non-matmul FLOPs
* synchronization

and demonstrates how an algorithm that is theoretically efficient can still leave GPU capability unused. ([arXiv][3])

---

### 3. **Roofline: An Insightful Visual Performance Model for Multicore Architectures**

Williams et al., 2009.

Extremely important.

Understand:

$$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

You'll repeatedly classify operations as:

```text
compute-bound
memory-bound
bandwidth-bound
communication-bound
```

---

### 4. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**

Despite being primarily a training paper, it is a foundational explanation of tensor/model parallelism. ([arXiv][4])

Understand:

* tensor parallelism
* communication between GPUs
* partitioning matrices
* all-reduce

These mechanisms appear again in distributed inference.

---

# Phase 2 — ML compilers and runtime architecture

### Months 4–9

This phase directly feeds the architecture of `ola-rt`.

The big question becomes:

> How do we turn a model definition into an efficient hardware execution plan?

---

## 5. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning**

Essential.

TVM separates graph-level optimization from operator/kernel optimization and targets multiple hardware backends. ([arXiv][5])

Pay particular attention to:

```text
model
 ↓
computational graph
 ↓
graph optimization
 ↓
operator lowering
 ↓
schedule
 ↓
generated kernel
 ↓
hardware
```

This will help you decide what belongs in:

```text
ola-graph
ola-ir
ola-compiler
ola-runtime
ola-scheduler
ola-kernels
```

---

## 6. **Ansor: Generating High-Performance Tensor Programs for Deep Learning**

Very relevant to your optimization interests.

Ansor searches a large tensor-program optimization space using program sampling, evolutionary search and learned cost models. ([USENIX][6])

This paper contains a seed for a later OlaRuntime question:

> Why manually select an execution strategy if the runtime can search or learn it?

---

## 7. **Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines**

This is important conceptually because Halide separates:

$$
\text{what computation means}
$$

from

$$
\text{how computation executes}
$$

That separation is extremely relevant to your proposed **inference execution-policy representation**.

---

## 8. **MLIR: Scaling Compiler Infrastructure for Domain-Specific Computation**

Read to understand:

* intermediate representations
* dialects
* progressive lowering
* reusable compiler infrastructure.

Your inference policy representation may eventually look like an IR or declarative scheduling dialect.

---

## 9. **GSPMD: General and Scalable Parallelization for ML Computation Graphs**

Very important later.

It demonstrates compiler-based automatic parallelization from tensor partitioning annotations. ([arXiv][7])

Think:

```text
logical model
       +
execution constraints
       ↓
parallel execution plan
```

That is conceptually close to what you're trying to achieve with inference policies.

---

# Phase 3 — Classical ML inference serving

### Months 6–10

Before studying vLLM, study the problem **before LLMs**.

Otherwise you won't understand exactly what autoregressive inference changed.

---

## 10. **Clipper: A Low-Latency Online Prediction Serving System**

Crankshaw et al., NSDI 2017.

Study:

* model serving abstraction
* batching
* latency targets
* model selection
* caching.

---

## 11. **Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis**

Shen et al., SOSP 2019.

Important for:

* GPU scheduling
* batch sizing
* SLO-aware serving
* multi-model workloads.

---

## 12. **Serving DNNs Like Clockwork: Performance Predictability from the Bottom Up**

One of the papers I especially want you to study.

Clockwork approaches inference as a **scheduling/predictability problem**, exploiting predictable execution times to provide tight SLOs. ([USENIX][8])

Your future research might challenge its assumptions when workloads become:

```text
variable context lengths
variable output lengths
multimodal inputs
heterogeneous models
bursty arrivals
changing hardware conditions
```

---

## 13. **INFaaS: Automated Model-less Inference Serving**

Important because execution becomes an optimization problem over:

* model variant
* hardware
* batching
* resource assignment.

---

## 14. **InferLine: Latency-Aware Provisioning and Scaling for Prediction Serving Pipelines**

Useful for studying:

* queueing
* pipelines
* tail latency
* autoscaling
* workload bursts.

---

# Phase 4 — The beginning of modern LLM serving

### Months 8–12

This is where your core PhD literature begins.

---

# 15. Orca — **A Distributed Serving System for Transformer-Based Generative Models**

Yu et al., OSDI 2022.

**Mandatory reading.**

Orca recognized that autoregressive generation requires fundamentally different scheduling.

Its major idea was:

> **iteration-level scheduling**

rather than treating a complete request as the scheduling unit. ([USENIX][9])

Old model:

```text
Request A ──────────────────────
Request B ──────────────────────
```

Orca:

```text
iteration 1 → schedule
iteration 2 → schedule
iteration 3 → schedule
...
```

This is directly relevant to your first paper.

### Question to ask

What exactly is Orca's **execution policy**?

Represent it formally.

---

# 16. **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention**

Kwon et al.

Another absolutely mandatory paper.

Core idea:

> Treat KV-cache memory somewhat like virtual memory.

Instead of requiring contiguous KV allocation:

```text
request
[........................]
```

PagedAttention allows:

```text
logical KV
 ↓

page → block 7
page → block 31
page → block 12
```

Read especially for:

* KV-cache fragmentation
* memory allocation
* request scheduling
* continuous batching.

---

# 17. **DeepSpeed Inference**

Important comparison to vLLM.

DeepSpeed tackles:

* multi-GPU inference
* kernel fusion
* tensor parallelism
* heterogeneous memory
* CPU/NVMe offloading. ([arXiv][10])

---

# 18. **FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU**

This one is particularly interesting for you because **execution placement is formulated as an optimization problem**.

FlexGen uses linear programming to determine where tensors reside/move across:

```text
GPU
CPU
disk
```

under hardware constraints. ([arXiv][11])

That is very close philosophically to:

$$
\underset{\pi}{\operatorname{argmin}}
\quad
\text{execution cost}(\pi)
$$

where \(\pi\) is an execution policy.

---

# Phase 5 — KV-cache systems

### Around end of Year 1

KV cache deserves an entire reading phase.

A 2026 survey now describes the design space along three particularly useful dimensions:

* **temporal:** execution/scheduling
* **spatial:** placement/migration
* **structural:** representation/retention. ([arXiv][12])

That taxonomy maps extremely well onto OlaRuntime.

---

## Read

### 19. **PagedAttention / vLLM**

Again, from the KV-memory perspective.

### 20. **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs**

Selective KV retention.

### 21. **StreamingLLM: Efficient Streaming Language Models with Attention Sinks**

Important for persistent long-running inference.

### 22. **Scissorhands: Exploiting Persistence of Importance Hypothesis for LLM KV Cache Compression**

KV eviction.

### 23. **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**

KV quantization.

### 24. **CacheGen: KV Cache Compression and Streaming for Fast LLM Serving**

Especially useful when compute becomes distributed.

### 25. **Prompt Cache: Modular Attention Reuse for Low-Latency Inference**

Cache reuse.

### 26. **RadixAttention / SGLang**

Read the SGLang paper and implementation.

The important concept is reuse of KV state across requests with shared prefixes.

---

# Phase 6 — Scheduling, batching and latency/throughput trade-offs

### Early Year 2

This is likely your **main research domain**.

---

## 27. **Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**

Important idea:

> Don't let huge prefills monopolize execution.

Break prefill into chunks.

---

## 28. **Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**

Must read.

Think carefully about:

```text
prefill
decode
chunking
stalling
batch composition
latency
throughput
```

This is essentially execution-policy design.

---

## 29. **FastServe: Fast Distributed Inference Serving for Large Language Models**

Study preemption and scheduling.

---

## 30. **DeepSpeed-FastGen / Dynamic SplitFuse**

Interesting because it dynamically combines pieces of prompt processing and decoding.

Compare it against Sarathi.

---

## 31. **Online Scheduling for LLM Inference with KV Cache Constraints**

Jaillet et al., 2025.

This one should be especially high priority for you because it explicitly treats LLM inference scheduling as a mathematical online scheduling problem.

It considers KV-memory constraints and analyzes semi-online and stochastic fully online settings. ([arXiv][13])

This paper sits almost perfectly at the intersection of:

```text
AI systems
+
operations research
+
online algorithms
+
inference
```

---

## 32. **Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints**

Ao, Luo, Simchi-Levi, Wang.

**Extremely relevant to your PhD.**

The formulation includes:

* multi-stage online scheduling
* endogenous KV-cache growth
* memory constraints
* online arrivals. ([DOI][14])

Given your interest in OR, I would probably place this in your **top 10 papers overall**.

---

# Phase 7 — Prefill/decode disaggregation

### Year 2

One of the dominant changes in inference architecture has been recognizing that:

```text
prefill ≠ decode
```

in hardware behavior.

---

# 33. **Splitwise: Efficient Generative LLM Inference Using Phase Splitting**

Mandatory.

The paper characterizes prefill as relatively compute intensive while decode is relatively memory intensive, motivating separate hardware pools. ([arXiv][15])

---

# 34. **DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving**

Mandatory.

Read carefully around:

* goodput
* TTFT
* TPOT
* interference
* resource allocation.

---

# 35. **Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving**

Very important modern system.

Mooncake separates prefill and decoding and makes distributed KV-cache infrastructure central to the serving architecture. Its scheduler targets throughput while respecting latency SLOs. ([arXiv][16])

This moves architecture toward:

```text
        scheduler
           │
      ┌────┴─────┐
      ↓          ↓
prefill pool   decode pool
      │          │
      └── KV ────┘
          fabric
```

---

# 36. **P/D-Serve: Serving Disaggregated Large Language Models at Scale**

Useful for understanding the engineering problems that appear once P/D separation leaves the conceptual stage.

---

# Phase 8 — Quantization and compressed inference

This should not be your main research area, but you need to understand it because execution policy depends on numerical representation.

---

## 37. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**

Foundational practical quantization work.

---

## 38. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**

Understand second-order weight quantization and why reducing weight precision reduces memory pressure and bandwidth. ([arXiv][17])

---

## 39. **SmoothQuant**

Very important.

Its key trick is moving quantization difficulty from activations to weights using a mathematically equivalent transformation, enabling W8A8 inference. ([arXiv][18])

---

## 40. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**

Read after GPTQ.

Compare:

```text
GPTQ
SmoothQuant
AWQ
```

from the perspective of runtime implications rather than accuracy.

---

## 41. **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression**

Useful when you eventually study mixed representations.

---

# Phase 9 — Speculative execution

### Mid Year 2

This should interest you a lot because it introduces another **execution policy decision**.

Instead of:

```text
large model → token
large model → token
large model → token
```

we do:

```text
small model:
   guess A B C D

large model:
   verify A B C D
```

---

## 42. **Fast Inference from Transformers via Speculative Decoding**

Leviathan et al.

Foundational speculative decoding.

---

## 43. **Accelerating Large Language Model Decoding with Speculative Sampling**

Chen et al.

Read alongside the previous one.

---

## 44. **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**

Multiple candidate tokens from extra heads.

---

## 45. **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**

Important modern speculative approach.

---

## 46. **EAGLE-2**

Dynamic draft trees.

---

## 47. **Lookahead Decoding**

Useful alternative that does not rely on a conventional draft model.

There is now a substantial speculative-decoding literature covering draft models, self-speculation, tree verification, retrieval/n-gram methods, multimodal applications and more. ([GitHub][19])

### Runtime question

Your scheduler might eventually choose:

$$
\pi =
\begin{cases}
\text{normal decoding} \\
\text{speculative decoding} \\
\text{n-gram speculation} \\
\text{small-draft model}
\end{cases}
$$

depending on the workload.

That makes speculative decoding a **policy decision**, not merely an algorithm.

---

# Phase 10 — Multi-model and adapter serving

You will need this for realistic serving environments.

---

## 48. **S-LoRA: Serving Thousands of Concurrent LoRA Adapters**

Important.

S-LoRA jointly manages dynamic LoRA weights and KV tensors and introduces unified paging. ([arXiv][20])

Think about a workload like:

```text
Llama
 ├── adapter customer-A
 ├── adapter customer-B
 ├── adapter customer-C
 └── adapter customer-D
```

instead of four independent models.

---

## 49. **Punica: Multi-Tenant LoRA Serving**

Read alongside S-LoRA.

Focus on batching requests with heterogeneous adapters.

---

# Phase 11 — Distributed inference and parallel execution

### Year 2

Your runtime eventually needs to move from:

```text
1 GPU
```

to:

```text
GPU cluster
```

---

## Read/revisit

### 50. Megatron-LM

Tensor parallelism.

### 51. DeepSpeed Inference

Large-scale inference.

### 52. GSPMD

Compiler-driven partitioning.

### 53. **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning**

Very important idea:

Automatically choose combinations of:

* operator parallelism
* pipeline parallelism
* data parallelism.

### 54. **GPipe**

Pipeline parallelism.

### 55. **PipeDream**

Pipeline scheduling.

### 56. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**

Training oriented, but crucial for understanding distributed memory partitioning.

---

# Phase 12 — Formalizing inference execution policies

This is where your **first PhD paper becomes distinct from "another inference runtime."**

You need literature outside pure LLM inference.

Your question is essentially:

> What is an execution policy, how should it be represented, and what properties can we reason about?

Think:

$$
\pi(s) \rightarrow a
$$

where:

* \(s\) = runtime state
* \(\pi\) = execution policy
* \(a\) = scheduling/execution decision.

For example:

```text
state:
    queue_depth = 42
    free_kv_blocks = 31
    prefill_tokens = 4200
    decode_requests = 17
    gpu_util = 0.72

policy:
    chunk_prefill = 512
    max_batch = 24
    priority = decode
```

---

## Papers/areas to read

### 57. **Halide**

Again, because of separation between algorithm and schedule.

### 58. **TVM**

Again, because schedules are representations of execution choices.

### 59. **Ansor**

Execution schedule search.

### 60. **GSPMD**

Declarative parallelization.

### 61. **Building Scalable and Flexible Cluster Managers Using Declarative Programming**

Worth studying for the idea that resource-management decisions can be specified declaratively rather than buried in imperative scheduler logic. The paper appeared at OSDI 2020 alongside several systems papers on scheduling and runtime management. ([USENIX][21])

### 62. **Specification and Verification in the Field: Applying Formal Methods to BPF JIT Compilers in the Linux Kernel**

Not directly ML.

But very relevant to your goal of moving from:

```text
scheduler code
```

toward:

```text
formal semantics
+
properties
+
verification
```

---

# Phase 13 — Learning scheduling policies

### Major Year 2 → Year 3 transition

Now you're moving from:

$$
\text{human-designed policy}
$$

to

$$
\text{optimized/learned policy}.
$$

---

# 63. **DeepRM: Resource Management with Deep Reinforcement Learning**

One of the early systems papers framing scheduling as RL.

Read it historically rather than as the solution you should copy.

---

# 64. **Decima: Learning Scheduling Algorithms for Data Processing Clusters**

Mao et al., SIGCOMM 2019.

**Mandatory for you.**

Decima learns scheduling policies for DAG jobs using reinforcement learning instead of handcrafted heuristics. It had to develop representations for dependency graphs and handle stochastic job arrivals. ([SIGCOMM Conferences][22])

The conceptual connection to your work is obvious:

```text
Decima:
cluster state → learned scheduler

OlaRuntime:
inference state → learned execution policy
```

---

# 65. **NeuroCuts: Learning Packet Classification Trees**

Another beautiful systems + RL example.

Not inference.

But it shows how RL can replace difficult handcrafted systems heuristics.

---

# Phase 14 — Non-stationary workloads

This is potentially where your **second major paper** lives.

Your runtime environment is not stationary.

$$
P(X_t) \neq P(X_{t+10000})
$$

Morning:

```text
short prompts
interactive workload
latency-sensitive
```

Night:

```text
large batches
long contexts
throughput-sensitive
```

Then suddenly:

```text
VLM traffic spike
```

Your optimal scheduler has changed.

---

## Read

### 66. **Achieving Optimal Dynamic Regret for Non-stationary Bandits without Prior Information**

Auer et al.

Important introduction to decision-making where the best action itself changes over time. ([Proceedings of Machine Learning Research][23])

---

### 67. **Hedging the Drift: Learning to Optimize Under Nonstationarity**

Especially relevant.

It studies data-driven decisions under changing environments through dynamic regret. ([DOI][24])

---

### Then study concepts rather than individual papers only:

* sliding-window UCB
* discounted UCB
* change-point detection
* contextual bandits
* dynamic regret
* variation budgets
* switching environments
* concept drift
* online optimization.

Your inference policy problem may eventually look like:

$$
\pi_t
=
f(s_t,\theta_t)
$$

where workload characteristics \(\theta_t\) change over time.

---

# Phase 15 — Queueing theory and scheduling theory

This section should be treated as seriously as the ML systems papers.

Because many "AI scheduling" problems are actually old scheduling/queueing problems in new clothes.

Read about:

### 68. **SRPT — Shortest Remaining Processing Time**

### 69. Processor Sharing

### 70. Weighted Fair Queueing

### 71. Multi-class queueing networks

### 72. Little's Law

$$
L=\lambda W
$$

### 73. M/M/1 and M/G/1 queues

### 74. Pollaczek–Khinchine formula

### 75. Online scheduling and competitive analysis

### 76. MaxWeight scheduling

You will eventually need to recognize when an apparent inference problem is equivalent to:

```text
online scheduling
knapsack
bin packing
queueing
resource allocation
MDP
bandit
```

That recognition could become one of your strongest research advantages.

---

# Phase 16 — Multimodal/VLM inference

### Late Year 2 → Year 3

This is where your **VLM inference engine** work starts.

VLM serving introduces additional heterogeneous stages beyond LLM prefill/decode.

A modern system may look like:

```text
image/video
    ↓
preprocessing
    ↓
vision encoder
    ↓
projector
    ↓
LLM prefill
    ↓
LLM decode
```

These stages have very different compute and memory profiles.

---

# 77. **CLIP: Learning Transferable Visual Models From Natural Language Supervision**

Foundation for image-text representations.

---

# 78. **An Image is Worth 16×16 Words — Vision Transformer**

You need to understand why an image becomes hundreds/thousands of vision tokens.

---

# 79. **Flamingo: A Visual Language Model for Few-Shot Learning**

Important multimodal architecture.

---

# 80. **BLIP-2**

Important because it introduces an intermediate bridge between frozen image encoders and language models.

---

# 81. **LLaVA: Visual Instruction Tuning**

Very important for understanding the canonical:

```text
vision encoder
     ↓
projector
     ↓
LLM
```

architecture.

---

# 82. **ModServe: Scalable and Resource-Efficient Large Multimodal Model Serving**

This one is **very important for your eventual VLM inference research**.

ModServe analyzes multimodal inference as heterogeneous stages and proposes stage decoupling, modality-aware scheduling and adaptive scaling. Microsoft reports production traces with variable, heavy-tailed requests and bursty traffic. ([Microsoft][25])

Notice how closely that connects to your non-stationary execution-policy research.

---

# 83. **Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization**

Another high-priority paper.

Nova treats:

```text
vision encode
LLM prefill
LLM decode
```

as heterogeneous stages and adaptively reallocates GPU resources between them under changing workloads. ([arXiv][26])

This is almost tailor-made reading for your thesis direction.

---

# 84. **ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism**

Read for:

* stage disaggregation
* multimodal parallelism
* load balancing
* dynamic allocation.

---

# 85. **HydraInfer: Hybrid Disaggregated Scheduling for Multimodal LLM Serving**

Useful for studying multimodal stage-aware scheduling.

---

# Phase 17 — Vision-token optimization

A large part of VLM inference cost comes from sending enormous numbers of visual tokens into the LLM.

Read:

### 86. **FastV: Efficient Vision-Language Models via Visual Token Pruning**

### 87. **SparseVLM**

### 88. **VisionZip**

### 89. **LVPruning: Language-Guided Vision Token Pruning**

LVPruning uses language-conditioned importance to remove vision tokens and reports large reductions in token counts and inference computation. ([arXiv][27])

### 90. **Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?**

I particularly like this one for PhD reading because it questions assumptions behind existing token-pruning methods rather than merely proposing another algorithm. ([arXiv][28])

---

# Phase 18 — Dynamic inference clusters

By Year 3, move from:

```text
scheduler inside one runtime
```

to:

```text
scheduler across inference replicas
```

Study:

* load balancing
* autoscaling
* heterogeneous GPUs
* admission control
* overload control
* SLO-aware routing
* KV migration.

### Particularly relevant families

```text
Llumnix
Mooncake
DistServe
Splitwise
DynamoLLM
SGLang
vLLM
Ray Serve
Clockwork
INFaaS
```

At this point, don't merely read the papers.

Implement competing policies.

---

# The especially important papers for **your first PhD paper**

If the proposed paper remains:

> **Formalizing Inference Execution Policies**

then I'd prioritize these approximately in this conceptual order:

1. **Orca**
2. **vLLM / PagedAttention**
3. **Sarathi**
4. **Sarathi-Serve**
5. **DistServe**
6. **Splitwise**
7. **Mooncake**
8. **Clockwork**
9. **FlexGen**
10. **TVM**
11. **Halide**
12. **Ansor**
13. **GSPMD**
14. **Online Scheduling for LLM Inference with KV Cache Constraints**
15. **Fluid-Guided Online Scheduling with Memory Constraints**

The central exercise should be:

> **Extract the implicit policy implemented by each system.**

For example:

### Orca

```text
Scheduling granularity:
    iteration

Admission:
    continuous

Batching:
    selective

Memory policy:
    relatively static

Prefill/decode:
    colocated
```

### vLLM

```text
Scheduling:
    iteration/continuous

Memory:
    paged KV blocks

Batching:
    continuous

Preemption:
    supported

Prefix reuse:
    possible
```

### Sarathi

```text
prefill:
    chunked

decode priority:
    high

batch construction:
    decode + prefill chunk

objective:
    reduce decode stalls
```

### DistServe

```text
placement:
    prefill ≠ decode

resource allocation:
    separate pools

objective:
    maximize goodput

constraints:
    TTFT + TPOT SLO
```

Then you start constructing something like:

```rust
ExecutionPolicy {
    admission_policy,
    batching_policy,
    prefill_policy,
    decode_policy,
    memory_policy,
    kv_policy,
    preemption_policy,
    placement_policy,
    parallelism_policy,
    routing_policy,
    speculation_policy,
}
```

That may eventually become the conceptual heart of `ola-rt`.

---

# Papers for your **second paper**

Assuming something like:

> **Learning Inference Execution Policies under Non-Stationary Workloads**

your reading should converge on:

```text
Decima
DeepRM
non-stationary bandits
dynamic regret
online optimization
change-point detection
contextual bandits
online scheduling
queueing theory
LLM scheduling
```

Especially:

* Decima
* Online Scheduling for LLM Inference with KV Cache Constraints
* Fluid-Guided Online Scheduling
* Hedging the Drift
* Auer et al. non-stationary bandits
* Clockwork
* Sarathi-Serve
* Mooncake
* ModServe
* Nova.

That's an unusually interesting intersection of **MLSys + OR + online learning**. ([SIGCOMM Conferences][22])

---

# Your third-paper neighbourhood

Something like:

> **Adaptive Distributed Inference under Heterogeneous Resources**

Then:

```text
DistServe
Splitwise
Mooncake
Llumnix
Alpa
GSPMD
Megatron
DeepSpeed
S-LoRA
heterogeneous cluster scheduling
load balancing
online resource allocation
```

---

# Your fourth-paper neighbourhood

Potentially:

> **Policy-Driven Multimodal Inference**

Now extend:

$$
\pi_{\text{LLM}}
$$

to

$$
\pi_{\text{multimodal}}
$$

with:

```text
video decoding
image preprocessing
vision encoding
vision-token pruning
projection
prefill
decode
KV placement
```

Then your central literature becomes:

```text
ModServe
Nova
ElasticMM
HydraInfer
LLaVA
BLIP-2
Flamingo
FastV
SparseVLM
LVPruning
```

Recent multimodal systems work is explicitly moving toward stage-aware disaggregation and adaptive resource allocation, which makes this a natural extension of the policy abstraction rather than a separate PhD topic. ([Microsoft][25])

---

# Your complete PhD reading phases

I would therefore organize the three years approximately like this:

| Phase       | Topic                                                          |    Approx. timing |
| ----------- | -------------------------------------------------------------- | ----------------: |
| 0           | Transformer mechanics                                          |        Months 1–2 |
| 1           | GPU architecture / memory                                      |        Months 1–4 |
| 2           | ML compilers / runtimes                                        |        Months 3–6 |
| 3           | Classical inference serving                                    |        Months 4–7 |
| 4           | LLM serving fundamentals                                       |        Months 5–9 |
| 5           | KV-cache systems                                               |       Months 6–10 |
| 6           | Scheduling / batching                                          |       Months 7–12 |
| **Paper 1** | **Formalizing Inference Execution Policies**                   |   **Months 9–15** |
| 7           | Prefill/decode disaggregation                                  |      Months 10–15 |
| 8           | Quantization                                                   |      Months 10–14 |
| 9           | Speculative decoding                                           |      Months 12–16 |
| 10          | LoRA/multi-model serving                                       |      Months 13–17 |
| 11          | Distributed inference                                          |      Months 14–20 |
| 12          | Formal policy representation                                   |      Months 10–18 |
| 13          | Learned scheduling                                             |      Months 15–21 |
| 14          | Non-stationary workloads                                       |      Months 16–24 |
| 15          | Queueing / online optimization                                 | Throughout Year 2 |
| **Paper 2** | **Learning Execution Policies under Non-Stationary Workloads** |  **Months 18–26** |
| 16          | VLM architecture                                               |      Months 20–24 |
| 17          | VLM serving                                                    |      Months 22–28 |
| 18          | Vision-token optimization                                      |      Months 24–30 |
| **Paper 3** | **Distributed/adaptive inference**                             |  **Months 24–31** |
| **Paper 4** | **Multimodal policy-driven inference**                         |  **Months 28–36** |

---

## The really important part

Don't measure PhD progress by:

> "I read 150 papers."

For every major systems paper, your lab notebook should contain **seven things**:

1. **Problem:** what bottleneck exists?
2. **State:** what information does the system observe?
3. **Actions:** what decisions can it make?
4. **Policy:** what rule determines those decisions?
5. **Objective:** latency, throughput, goodput, cost, utilization, energy?
6. **Constraints:** memory, SLO, GPUs, bandwidth, fairness?
7. **Implementation:** how would I reproduce this decision mechanism in `ola-rt`?

For example, after reading 30 inference papers you should have a table looking like:

| System         | State                     | Actions                         | Policy                      | Objective          | Dynamic?     |
| -------------- | ------------------------- | ------------------------------- | --------------------------- | ------------------ | ------------ |
| Orca           | request iterations        | batch membership                | iteration scheduling        | throughput/latency | partly       |
| vLLM           | requests + KV blocks      | admit/preempt/batch             | continuous scheduler        | utilization        | partly       |
| Sarathi        | decode + prefill work     | chunk size/batch                | stall-free                  | latency/throughput | limited      |
| DistServe      | P/D queues                | placement/routing               | disaggregated               | goodput            | limited      |
| Mooncake       | queues + distributed KV   | routing/cache placement         | KV-centric                  | SLO throughput     | yes          |
| ModServe       | modality/stage loads      | scale/place/schedule            | modality-aware              | SLO/cost           | yes          |
| **OlaRuntime** | **general runtime state** | **general execution decisions** | **explicit policy \(\pi\)** | **user-defined**   | **adaptive** |

And **that last row is your research thesis**.

A useful starting survey before going paper-by-paper is *A Survey of LLM Inference Systems* (2025); for KV-specific work, the 2026 *System-Aware KV Cache Optimization* survey is particularly current. ([arXiv][1])

I would treat roughly **90 papers above as your master reading universe, but only ~30–35 as mandatory deep reads**. The rest should be read selectively when they become relevant to an experiment or paper. The goal by the end of Year 1 is not merely to know vLLM, Orca and Sarathi; it is to be able to look at any inference engine and say: **“I can identify its state space, decisions, objective, constraints and implicit execution policy, and I can represent that policy independently of its implementation.”** That is a much stronger foundation for the rest of the PhD.

[1]: https://arxiv.org/abs/2506.21901?utm_source=chatgpt.com "A Survey of LLM Inference Systems"
[2]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[3]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
[4]: https://arxiv.org/abs/1909.08053?utm_source=chatgpt.com "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
[5]: https://arxiv.org/abs/1802.04799?utm_source=chatgpt.com "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"
[6]: https://www.usenix.org/conference/osdi20/presentation/zheng?utm_source=chatgpt.com "Ansor: Generating High-Performance Tensor Programs for Deep Learning | USENIX"
[7]: https://arxiv.org/abs/2105.04663?utm_source=chatgpt.com "GSPMD: General and Scalable Parallelization for ML Computation Graphs"
[8]: https://www.usenix.org/conference/osdi20/presentation/gujarati?utm_source=chatgpt.com "Serving DNNs like Clockwork: Performance Predictability from the Bottom Up | USENIX"
[9]: https://www.usenix.org/conference/osdi22/presentation/yu?utm_source=chatgpt.com "Orca: A Distributed Serving System for Transformer-Based Generative Models | USENIX"
[10]: https://arxiv.org/abs/2207.00032?utm_source=chatgpt.com "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale"
[11]: https://arxiv.org/abs/2303.06865?utm_source=chatgpt.com "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"
[12]: https://arxiv.org/abs/2607.08057?utm_source=chatgpt.com "Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization"
[13]: https://arxiv.org/abs/2502.07115?utm_source=chatgpt.com "Online Scheduling for LLM Inference with KV Cache Constraints"
[14]: https://doi.org/10.48550/arXiv.2504.11320?utm_source=chatgpt.com "[2504.11320] Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints"
[15]: https://arxiv.org/abs/2311.18677?utm_source=chatgpt.com "Splitwise: Efficient generative LLM inference using phase splitting"
[16]: https://arxiv.org/abs/2407.00079?utm_source=chatgpt.com "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
[17]: https://arxiv.org/abs/2210.17323?utm_source=chatgpt.com "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
[18]: https://arxiv.org/abs/2211.10438?utm_source=chatgpt.com "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
[19]: https://github.com/NeuraLiying/awesome-speculative-decoding?utm_source=chatgpt.com "GitHub - NeuraLiying/awesome-speculative-decoding: A curated list of speculative decoding papers, code, and resources for efficient LLM/MLLM inference. · GitHub"
[20]: https://arxiv.org/abs/2311.03285?utm_source=chatgpt.com "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
[21]: https://www.usenix.org/conference/osdi20/session/osdi-20-accepted-papers?utm_source=chatgpt.com "OSDI '20 Accepted Papers | USENIX"
[22]: https://conferences.sigcomm.org/sigcomm/2019/program_thursday.html?utm_source=chatgpt.com "ACM SIGCOMM 2019 Thursday Program - ACM SIGCOMM 2019"
[23]: https://proceedings.mlr.press/v99/auer19b.html?utm_source=chatgpt.com "Achieving Optimal Dynamic Regret for Non-stationary Bandits without Prior Information"
[24]: https://doi.org/10.1287/mnsc.2021.4024?utm_source=chatgpt.com "Hedging the Drift: Learning to Optimize Under Nonstationarity | Management Science"
[25]: https://www.microsoft.com/en-us/research/publication/modserve-scalable-and-resource-efficient-large-multimodal-model-serving/?lang=ja&utm_source=chatgpt.com "ModServe: Scalable and Resource-Efficient Large Multimodal Model Serving - Microsoft Research"
[26]: https://arxiv.org/abs/2509.21301?utm_source=chatgpt.com "Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization"
[27]: https://arxiv.org/abs/2501.13652?utm_source=chatgpt.com "LVPruning: An Effective yet Simple Language-Guided Vision Token Pruning Approach for Multi-modal Large Language Models"
[28]: https://arxiv.org/abs/2502.11501?utm_source=chatgpt.com "Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?"

