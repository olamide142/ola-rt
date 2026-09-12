# OlaRT Papers Roadmap

> **A living publication roadmap for the OlaRT PhD research program.**

This document maps the evolution of OlaRT to possible research papers across the PhD.

It is not a commitment to publish every idea listed here.

The purpose is to ensure that implementation work is continuously connected to research questions, experiments, and publishable contributions.

The core principle is:

> **Papers should emerge from observed limitations in OlaRT, not from trying to manufacture publication topics in isolation.**

---

# 1. Publication Philosophy

OlaRT should produce a coherent research story.

The ideal progression is:

```text
characterize the problem
        ↓
build strong baselines
        ↓
identify a systems limitation
        ↓
propose a mechanism
        ↓
evaluate rigorously
        ↓
generalize the mechanism
        ↓
scale it
```

The PhD should not become a collection of unrelated papers.

Each paper should ideally depend on infrastructure, observations, or ideas from previous work.

---

# 2. Seasonal Research Cycle

Each research season should last roughly 3–4 months.

A typical cycle:

```text
Month 1
Read + reproduce + establish baselines

Month 2
Measure + profile + identify bottleneck

Month 3
Implement hypothesis + run experiments

Month 4
Write + submit + clean implementation
```

Not every season needs to produce a publication.

Possible season outputs include:

- paper submission
- technical report
- reproduction study
- benchmark suite
- negative result
- open-source release
- dataset/workload suite
- architecture revision

---

# 3. Pre-PhD / Foundation Season

## September–December 2026

### Objective

Build the minimum OlaRT infrastructure required for credible inference experiments.

### Main engineering

- Burn backend integration
- request lifecycle
- scheduler interface
- tiny LLM path
- prefill/decode separation
- KV-cache v1
- workload generator
- metrics
- simple batching
- CPU/GPU benchmark exercises

### Research output

This period does **not** need a conventional publication.

Possible outputs:

- technical report
- systems note
- reproducibility report
- benchmark note
- architecture paper draft

### Candidate technical note

**Working title**

> *Building OlaRT: An Experimental Rust Runtime for AI Inference Research*

### Questions

- What abstractions are necessary for an inference research runtime?
- Which parts should be delegated to Burn/CubeCL?
- Which runtime abstractions must remain independently controllable?

### Purpose

Document the architectural foundation and create a citation-quality internal reference for future work.

---

# 4. PhD Year 1 — LLM Inference Systems

Year 1 should establish OlaRT as a credible LLM inference research platform.

The research focus should remain narrow:

```text
LLM inference
+
scheduling
+
memory
+
batching
```

Do not rush into multimodal work before the runtime is trustworthy.

---

# 5. Year 1 — Season 1

## Theme

LLM workload characterization and scheduler baselines.

### Candidate Paper 1

**Working title**

> *Characterizing Scheduling Trade-offs in LLM Inference under Heterogeneous Sequence Lengths*

Alternative:

> *When FIFO Fails: A Measurement Study of Request Scheduling in LLM Inference*

### Core question

How much do request length, arrival patterns, prefill cost, and decode cost affect scheduler behaviour?

### Hypothesis

Simple FIFO scheduling performs poorly under heterogeneous request lengths, particularly for tail latency and TTFT.

### Required OlaRT features

- request queue
- scheduler interface
- FIFO
- SJF / estimated-shortest-job-first
- priority scheduler
- workload generator
- prefill/decode metrics
- latency tracing
- basic batching

### Workloads

Vary:

- prompt length
- generation length
- arrival rate
- burstiness
- model size
- concurrency

### Metrics

- p50 latency
- p95 latency
- p99 latency
- TTFT
- TPOT
- throughput
- fairness
- starvation rate

### Baselines

- FIFO
- round robin
- shortest-job-first
- priority

### Potential contribution

A strong characterization paper if the study reveals non-obvious scheduler behaviour.

Even if not accepted as a full paper, this becomes essential baseline work for later papers.

### Target venue class

- systems workshop
- MLSys workshop
- serving/inference workshop
- technical report

---

# 6. Year 1 — Season 2

## Theme

Memory-aware scheduling.

### Candidate Paper 2

**Working title**

> *Memory-Aware Scheduling for Continuous LLM Inference*

Alternative:

> *Beyond Queue Length: Scheduling LLM Requests under KV-Cache Pressure*

### Core question

Can scheduler decisions improve latency and throughput by explicitly considering KV-cache pressure?

### Hypothesis

Schedulers that reason about projected KV-cache usage can reduce memory pressure, fragmentation, and tail latency compared with queue-only policies.

### Required OlaRT features

- explicit memory accounting
- KV-cache manager
- continuous batching prototype
- request memory estimator
- scheduler plugin API
- workload replay
- cache occupancy metrics

### Scheduler input

Potential request features:

```text
prompt length
expected output length
current KV usage
projected KV growth
deadline
priority
```

### Baselines

- FIFO
- SJF
- priority
- queue-length-aware scheduler
- memory-unaware continuous batching

### Metrics

- throughput
- p95/p99 latency
- TTFT
- KV utilization
- allocation failures
- fragmentation
- SLO violations

### Potential contribution

A practical scheduler policy plus a characterization of memory pressure in mixed-length inference workloads.

### Target venue class

- MLSys
- ASPLOS workshop
- systems workshop
- AI systems venue

---

# 7. Year 1 — Season 3

## Theme

Continuous batching and SLO-aware inference.

### Candidate Paper 3

**Working title**

> *SLO-Aware Continuous Batching for LLM Inference*

Alternative:

> *Balancing Throughput and Tail Latency in Continuous LLM Batching*

### Core question

Can batching decisions explicitly optimize SLO attainment instead of maximizing aggregate throughput?

### Hypothesis

A scheduler that dynamically adjusts batch membership using request deadlines and predicted service cost can improve SLO compliance without sacrificing too much throughput.

### Required OlaRT features

- robust continuous batching
- deadlines/SLO metadata
- runtime cost estimation
- dynamic batch membership
- request cancellation
- reproducible workload traces

### Research variables

- batch size
- queue depth
- deadlines
- prompt length
- decode length
- model size

### Baselines

- fixed batch size
- throughput-maximizing batching
- FIFO continuous batching
- deadline-aware heuristic

### Metrics

- SLO attainment
- p99 latency
- TTFT
- throughput
- GPU utilization
- starvation

### Potential contribution

This could become the first strong systems paper if the mechanism is robust and the evaluation is broad.

### Target venue class

- MLSys
- EuroSys
- USENIX ATC
- middleware/system-performance venues

---

# 8. Year 1 — Season 4

## Theme

Consolidation and first major submission.

Possible direction:

- strengthen Paper 2 or Paper 3
- reproduce an important serving paper
- compare OlaRT against a mature serving baseline
- clean the runtime for external use

### Expected output

At least one of:

- major conference submission
- strong workshop paper
- open-source alpha release
- reproducibility artifact

---

# 9. PhD Year 2 — Multimodal and Adaptive Inference

Year 2 should introduce the central thesis shift:

> LLM serving assumptions do not necessarily transfer cleanly to multimodal inference.

The goal is to make OlaRT **modality-aware**.

---

# 10. Year 2 — Season 1

## Theme

VLM workload characterization.

### Candidate Paper 4

**Working title**

> *Characterizing Vision-Language Model Inference for Systems Scheduling*

Alternative:

> *Why VLM Serving Is Not Just LLM Serving with Images*

### Core question

How do VLM requests differ from text-only LLM requests in compute, memory, latency, and batching behaviour?

### Hypothesis

Image resolution, visual token count, encoder cost, and multimodal attention produce workload characteristics that invalidate text-only scheduling assumptions.

### Required OlaRT features

- VLM runtime
- vision encoder integration
- request modality metadata
- visual token tracking
- image resolution tracking
- encoder/decode timing
- mixed LLM/VLM workloads

### Variables

- image count
- image resolution
- aspect ratio
- visual tokens
- prompt length
- output length
- model architecture

### Metrics

- encoder latency
- TTFT
- TPOT
- memory usage
- visual-token cost
- batching efficiency
- GPU utilization

### Potential contribution

A measurement/characterization paper that motivates adaptive multimodal scheduling.

### Target venue class

- MLSys
- multimodal systems workshop
- systems measurement venue

---

# 11. Year 2 — Season 2

## Theme

Multimodal cost modelling.

### Candidate Paper 5

**Working title**

> *Predicting Multimodal Inference Cost for Adaptive Runtime Scheduling*

Alternative:

> *Cost Models for Mixed LLM and VLM Inference Workloads*

### Core question

Can the runtime predict request execution cost accurately enough before execution to improve scheduling?

### Hypothesis

A compact model using modality-specific features can predict runtime and memory sufficiently well for online scheduling decisions.

### Input features

Possible features:

```text
modality
prompt tokens
expected output tokens
image count
image pixels
visual tokens
model
device
current batch composition
```

### Prediction targets

- prefill latency
- encoder latency
- decode latency
- peak memory
- total runtime

### Models to compare

- analytical formula
- linear regression
- gradient-boosted model
- lightweight neural model

### Important constraint

Prediction overhead must remain negligible compared with inference cost.

### Required OlaRT features

- workload feature extraction
- runtime tracing
- data collection
- cost-model API
- scheduler integration

### Potential contribution

A cost model that becomes infrastructure for multiple later papers.

### Target venue class

- MLSys
- systems + ML venue
- performance modelling venue

---

# 12. Year 2 — Season 3

## Theme

Adaptive multimodal scheduling.

### Candidate Paper 6

**Working title**

> *Adaptive Scheduling for Mixed LLM and VLM Inference*

Alternative:

> *Modality-Aware Scheduling for Shared AI Inference Clusters*

### Core question

How should text and visual workloads share accelerator capacity?

### Hypothesis

Schedulers that account for modality, encoder cost, memory demand, and predicted completion time outperform generic LLM schedulers under mixed workloads.

### Required OlaRT features

- LLM + VLM execution
- cost model
- per-request modality
- adaptive scheduler
- batching
- multi-model workload
- multi-GPU if possible

### Baselines

- FIFO
- least-queue
- SJF
- text-only cost model
- generic memory-aware scheduler

### Metrics

- throughput
- TTFT
- p99 latency
- fairness across modalities
- SLO compliance
- GPU utilization
- memory pressure

### Potential contribution

This could be one of the core thesis papers.

### Target venue class

- MLSys
- EuroSys
- USENIX ATC
- ASPLOS

---

# 13. Year 2 — Season 4

## Theme

OlaDetection and vision-pipeline execution.

### Candidate Paper 7

**Working title**

> *Adaptive Execution of Multi-Stage Vision and Document AI Pipelines*

Alternative:

> *Scheduling Detection, OCR, and Language Models on Shared Accelerators*

### Core question

How should heterogeneous inference stages in a document/vision pipeline share resources?

### Example workload

```text
image
  ↓
detection
  ↓
segmentation
  ↓
OCR
  ↓
language model
  ↓
structured output
```

### Research issues

- stage-specific compute cost
- backpressure
- batching incompatibility
- pipeline parallelism
- GPU placement
- intermediate caching
- resolution adaptation

### Required OlaRT features

- OlaDetection pipeline
- stage DAG
- pipeline scheduler
- backpressure
- stage metrics
- mixed model execution

### Potential contribution

A strong applied systems paper grounded in a realistic workload.

### Target venue class

- MLSys
- vision systems workshop
- document AI workshop
- applied systems venue

---

# 14. PhD Year 3 — Distributed Adaptive Inference

Year 3 should unify scheduling, cost modelling, cache state, and resource placement.

The runtime becomes a distributed control system.

---

# 15. Year 3 — Season 1

## Theme

Multi-GPU and cache-aware placement.

### Candidate Paper 8

**Working title**

> *Cache-Aware Request Placement for Distributed AI Inference*

Alternative:

> *Locality Matters: KV-Cache-Aware Routing in Multi-GPU Inference*

### Core question

Should incoming requests be routed primarily based on utilization, queue length, or cache locality?

### Hypothesis

Cache locality can dominate load balance in some workloads, making least-loaded routing suboptimal.

### Required OlaRT features

- multi-GPU runtime
- global router
- per-device queue
- per-device memory state
- model residency
- cache metadata
- placement API

### Baselines

- round robin
- least queue
- least utilized
- least memory
- predicted completion time

### Metrics

- completion time
- TTFT
- cache reuse
- data movement
- GPU utilization
- imbalance
- SLO violations

### Target venue class

- EuroSys
- NSDI
- USENIX ATC
- distributed systems venue

---

# 16. Year 3 — Season 2

## Theme

Adaptive load balancing.

### Candidate Paper 9

**Working title**

> *Adaptive Load Balancing for Heterogeneous AI Inference Clusters*

Alternative:

> *Beyond Least-Loaded: Runtime-Aware Placement for Heterogeneous Inference*

### Core question

How should requests be distributed when accelerators differ in capability, memory, model residency, and workload?

### State

Potential inputs:

```text
queue length
GPU type
memory pressure
utilization
model residency
cache locality
request modality
predicted runtime
network cost
deadline
```

### Required OlaRT features

- heterogeneous devices
- global state
- cost model
- placement policies
- workload shifts
- multi-model support

### Baselines

- round robin
- least loaded
- least memory
- static weighted routing
- predicted completion time

### Potential contribution

A strong distributed systems paper if the runtime demonstrates meaningful gains under heterogeneous workloads.

---

# 17. Year 3 — Season 3

## Theme

Online optimization.

### Candidate Paper 10

**Working title**

> *Online Optimization for Adaptive AI Inference Scheduling*

Alternative:

> *Joint Scheduling, Placement, and Resource Control for Dynamic AI Inference*

### Core question

Can inference runtime decisions be formulated as an online optimization problem and solved effectively under uncertain arrivals?

### Possible decisions

- device placement
- batching
- admission
- cache policy
- precision
- resolution
- migration

### Example objective

\[
\min_t
\alpha L_t
+
\beta C_t
+
\gamma M_t
+
\delta V_t
\]

where:

- \(L_t\): latency
- \(C_t\): cost
- \(M_t\): memory pressure
- \(V_t\): SLO violations

### Constraints

- GPU memory
- device capacity
- model residency
- deadlines
- batching compatibility

### Baselines

- heuristics
- greedy optimization
- static policy
- predictive scheduler

### Potential contribution

This is a possible thesis-defining paper because it generalizes several earlier mechanisms into one framework.

---

# 18. Year 3 — Season 4

## Theme

Learned scheduling / RL only if justified.

### Candidate Paper 11

**Working title**

> *Learning Runtime Policies for Dynamic Multimodal Inference*

This paper should only exist if learned methods clearly outperform strong non-learning baselines.

### Core question

Can a learned policy adapt to workload regimes where analytical or heuristic policies fail?

### State

```text
queue state
device state
cache state
request features
network state
model residency
```

### Actions

```text
assign
batch
delay
migrate
change precision
change resolution
```

### Baselines

Must include:

- heuristic
- greedy
- optimization-based
- bandit where appropriate

### Requirement

Do not publish:

> "We used RL for scheduling."

Publish only if:

> "RL solves a concrete limitation that strong simpler approaches fail to solve."

---

# 19. Thesis Paper Stack

A coherent final dissertation might ultimately revolve around 5–6 central papers:

```text
Paper A
LLM scheduling characterization
        ↓
Paper B
memory/SLO-aware LLM scheduling
        ↓
Paper C
VLM workload characterization
        ↓
Paper D
adaptive multimodal scheduling
        ↓
Paper E
distributed cache-aware placement
        ↓
Paper F
online adaptive inference framework
```

Possible thesis story:

> Modern AI inference workloads are heterogeneous across request size, modality, memory behaviour, and hardware demand. Static runtime policies fail to exploit this structure. OlaRT demonstrates that workload-aware and adaptive runtime control can improve latency, throughput, memory efficiency, and SLO attainment across single-node, multimodal, and distributed inference systems.

---

# 20. Papers That Should Probably Not Be Written

Avoid weak papers such as:

- "We rewrote X in Rust"
- "We implemented a tensor library"
- "We built an inference server"
- "We used RL to schedule requests"
- "We benchmarked CUDA vs CPU"

These can be useful engineering exercises.

They are not automatically research contributions.

A paper needs:

```text
problem
+
gap
+
new idea
+
rigorous evidence
```

---

# 21. Reproduction Papers

Reproduction is valuable when used strategically.

Potential reproduction targets:

- PagedAttention
- FlashAttention
- continuous batching
- speculative decoding
- prefix caching
- VLM serving mechanisms

A reproduction project should answer:

1. Can the mechanism be reproduced?
2. Under what workloads does it help?
3. Under what workloads does it fail?
4. Which assumptions matter?
5. What changes when implemented in OlaRT?

The fifth question is where original research may begin.

---

# 22. Benchmark/Artifact Papers

Not every contribution must be a scheduler.

Potential artifact papers include:

- multimodal inference benchmark
- realistic workload traces
- reproducible VLM-serving benchmark suite
- OlaRT runtime artifact
- inference cost dataset

These become valuable if they expose a real gap in available evaluation infrastructure.

---

# 23. Negative Results

Negative results should remain in the repository.

Examples:

```text
RL scheduler does not beat greedy
dynamic resolution harms quality too much
migration cost dominates benefit
cache transfer is slower than recomputation
cost model fails under workload shift
```

These observations prevent repeating dead ends and may improve later papers.

Store them under:

```text
docs/research/negative-results/
```

---

# 24. Paper Folder Structure

Recommended:

```text
papers/
├── 2027-llm-scheduling/
│   ├── README.md
│   ├── question.md
│   ├── related-work.md
│   ├── experiments.md
│   ├── figures/
│   └── draft/
│
├── 2027-memory-aware/
├── 2028-vlm-characterization/
├── 2028-adaptive-vlm/
└── ...
```

Each paper folder should contain:

```text
research question
hypothesis
related work
required OlaRT components
experiment matrix
results
figures
limitations
draft
submission history
```

---

# 25. Submission Readiness Checklist

Before writing the paper, confirm:

- [ ] the research question is narrow and clear
- [ ] the baseline is strong
- [ ] the claimed mechanism is implemented correctly
- [ ] workloads are representative
- [ ] hardware is documented
- [ ] results are reproducible
- [ ] p95/p99 are reported where relevant
- [ ] ablations exist
- [ ] failure cases are shown
- [ ] the paper explains why the result occurs
- [ ] the contribution is not merely an implementation
- [ ] OlaRT code required to reproduce results is tagged/versioned

---

# 26. Evidence Ladder

Do not jump directly from idea to "paper contribution."

The evidence ladder is:

```text
observation
  ↓
repeatable observation
  ↓
measured bottleneck
  ↓
hypothesis
  ↓
prototype
  ↓
controlled experiment
  ↓
strong baseline comparison
  ↓
ablation
  ↓
generalization
  ↓
paper claim
```

---

# 27. Venue Strategy

Do not choose the venue before understanding the contribution.

Broadly:

## Systems / architecture

Possible families:

- MLSys
- ASPLOS
- EuroSys
- USENIX ATC
- NSDI
- SoCC
- Middleware

## ML + systems

- MLSys
- NeurIPS systems/workshop tracks when relevant
- ICML systems-oriented workshops
- ICLR systems workshops

## Vision / multimodal systems

- CVPR workshops
- ICCV/ECCV workshops
- multimodal systems workshops

Venue choice should depend on what the final contribution actually becomes.

---

# 28. Annual Publication Targets

These are targets, not quotas.

## Year 1

Aim for:

- 1 strong workshop/reproduction/measurement output
- 1 major paper submission candidate

## Year 2

Aim for:

- 1 VLM characterization/cost-model paper
- 1 strong adaptive multimodal systems paper

## Year 3

Aim for:

- 1 distributed inference paper
- 1 thesis-defining adaptive/optimization paper

A smaller number of strong connected papers is better than many weak disconnected papers.

---

# 29. Long-Term Publication Map

```text
2026
FOUNDATIONS
│
└── Technical architecture/reproduction note
      │
      ▼
2027
LLM INFERENCE
│
├── Scheduling characterization
├── Memory-aware scheduling
└── SLO-aware continuous batching
      │
      ▼
2028
MULTIMODAL
│
├── VLM characterization
├── Cost modelling
├── Adaptive multimodal scheduling
└── OlaDetection pipeline systems
      │
      ▼
2029
DISTRIBUTED / OPTIMIZATION
│
├── Cache-aware placement
├── Adaptive load balancing
├── Online optimization
└── Learned policy if justified
      │
      ▼
THESIS
```

---

# 30. Final Principle

A paper should answer:

> **What did OlaRT allow us to observe, control, or optimize that was previously poorly understood or handled?**

The project should evolve as:

```text
build
  ↓
measure
  ↓
discover limitation
  ↓
formulate question
  ↓
design mechanism
  ↓
evaluate
  ↓
publish
  ↓
feed result back into OlaRT
```

That loop is the real purpose of this roadmap.
