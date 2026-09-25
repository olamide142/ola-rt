Formalization, Optimization, and Learning of Inference Execution Policies for Heterogeneous AI Systems

A strong thesis title for the direction you’ve been shaping is:

> **Adaptive Inference Execution Policies for Efficient AI Systems Under Non-Stationary Workloads**

That title is broad enough for a 3-year PhD, but specific enough to define a coherent research program.

A slightly more systems-heavy version is:

> **Design and Optimization of Adaptive AI Inference Runtimes Under Dynamic Workloads**

And a more formal/OR-flavoured version:

> **Formalization, Optimization, and Learning of Inference Execution Policies for Heterogeneous AI Systems**

Of those three, I’d use the third if you want the thesis to clearly connect **systems + optimization + policy learning**.

Your proposal after first semester should not try to prove that you have solved the problem already. Its job is to show that:

* the problem is important,
* the problem is not adequately solved,
* you understand the literature,
* you can define the problem formally,
* you have a feasible research plan,
* and you know how you will evaluate your contribution.

## A proposal title I would use

**Formalization and Adaptive Optimization of Inference Execution Policies for Heterogeneous AI Systems**

Your presentation could then be organized around the following storyline.

### 1. Background

Start with the shift from model-centric AI to inference-centric AI systems.

Modern inference runtimes must manage:

* batching,
* KV-cache allocation,
* prefill/decode scheduling,
* quantization,
* kernel selection,
* parallelism,
* model placement,
* speculative decoding,
* heterogeneous accelerators.

Existing systems such as vLLM, TensorRT-LLM, SGLang and others expose or internally make many of these decisions.

The important observation is:

> many execution decisions are based on fixed configuration, offline tuning or specialized heuristics.

But real workloads change continuously.

For example:

```text
09:00
short interactive prompts
high TTFT sensitivity

12:00
large document workloads
long prefills

15:00
agent workloads
shared prefixes
long-lived KV state

20:00
batch inference
throughput dominant
```

Yet the runtime may continue operating under essentially the same execution configuration.

That gives you your research motivation.

---

# 2. Problem statement

A clean version:

> Modern AI inference engines employ multiple execution strategies for scheduling, batching, memory management, precision selection, model placement and parallel execution. However, these decisions are often governed by static configurations or locally designed heuristics that may perform poorly under dynamic and non-stationary workloads. There is currently no general formalism for representing inference execution policies as adaptive decision processes that jointly consider workload characteristics, hardware state and service-level objectives.

Then:

> This research proposes to formalize the concept of an inference execution policy and develop adaptive optimization methods that dynamically select execution strategies in response to changing workload and hardware conditions.

That is your PhD in two paragraphs.

---

# 3. The key research idea

Define:

$$
\pi : S \rightarrow A
$$

where \(\pi\) is an **inference execution policy**.

The runtime observes state:

$$
s_t \in S
$$

and chooses execution action:

$$
a_t \in A
$$

You might define:

$$
s_t =
(
W_t,
Q_t,
M_t,
G_t,
L_t,
H_t
)
$$

where:

* \(W_t\): workload characteristics
* \(Q_t\): queue state
* \(M_t\): memory/KV state
* \(G_t\): GPU utilization
* \(L_t\): latency/SLA state
* \(H_t\): hardware characteristics

And:

$$
a_t =
(
B_t,
C_t,
K_t,
P_t,
R_t,
D_t
)
$$

where:

* \(B_t\): batching decision
* \(C_t\): chunked-prefill policy
* \(K_t\): KV-cache policy
* \(P_t\): precision/quantization choice
* \(R_t\): routing/model placement
* \(D_t\): distributed execution strategy.

Then the problem becomes:

$$
\pi^*
=
\arg\max_\pi
\mathbb{E}
\left[
\sum_t R(s_t,\pi(s_t))
\right]
$$

subject to:

$$
P99 \leq SLA
$$

$$
Memory \leq Capacity
$$

$$
Quality \geq Q_{\min}
$$

This is where your formalization becomes important.

---

# 4. Research questions

Your proposal should probably have around four primary research questions.

### RQ1

**How can inference execution policies be formally represented in a way that captures scheduling, memory, model and hardware decisions?**

This is your foundational paper.

### RQ2

**How do static and heuristic execution policies behave under heterogeneous and non-stationary workloads?**

This establishes the weakness of existing approaches.

### RQ3

**Can adaptive optimization methods improve inference performance under changing workloads while maintaining service-level constraints?**

This leads into OR and online optimization.

### RQ4

**Can learned execution policies generalize across workload distributions, models and hardware platforms?**

This brings in reinforcement learning and learning-based systems optimization.

Potential later RQ:

**How should execution policies coordinate multiple inference dimensions jointly rather than optimizing them independently?**

That one could become very important.

---

# 5. Your hypothesis

You could state the main hypothesis as:

> **An adaptive inference execution policy that jointly considers workload characteristics, runtime resource state and service-level objectives can outperform static and isolated heuristic policies under non-stationary inference workloads.**

Sub-hypotheses could include:

$$
H_1:
\text{Adaptive scheduling reduces tail latency under workload shifts.}
$$

$$
H_2:
\text{Joint scheduling + KV management improves resource utilization.}
$$

$$
H_3:
\text{Hardware-conditioned policies transfer better than hardware-specific policies.}
$$

You don't need ten hypotheses.

Three or four are enough.

---

# 6. Research objectives

Your objectives might be:

1. Develop a formal representation for inference execution policies.

2. Characterize execution policies used by contemporary inference runtimes.

3. Build an experimental inference runtime or research platform capable of exposing execution-policy decisions.

4. Develop optimization techniques for adaptive policy selection.

5. Investigate learning-based policies under non-stationary workloads.

6. Evaluate generalization across workloads, models and heterogeneous hardware.

This fits the ola-rt idea extremely well.

---

# 7. Literature review structure

Do not make your literature review simply:

```text
Paper A did this.
Paper B did this.
Paper C did this.
```

Organize it around the problem.

For example:

```text
Inference serving
    │
    ├── Scheduling
    │      ├── continuous batching
    │      ├── prefill/decode scheduling
    │      └── request prioritization
    │
    ├── Memory
    │      ├── KV cache
    │      ├── PagedAttention
    │      ├── prefix caching
    │      └── offloading/compression
    │
    ├── Execution
    │      ├── kernels
    │      ├── quantization
    │      └── speculative decoding
    │
    ├── Distributed inference
    │      ├── TP
    │      ├── PP
    │      └── heterogeneous placement
    │
    └── Adaptive systems
           ├── autotuning
           ├── online optimization
           ├── RL systems
           └── resource scheduling
```

Then identify the gap:

> Existing work optimizes many individual components, but there is limited work providing a unified formal representation of runtime execution policy and using it for adaptive, cross-layer optimization.

That sentence could become central to your proposal.

---

# 8. Proposed research framework

Your initial architecture could look like:

```text
                       Workload
                          │
                          ▼
                 Runtime Observation
                          │
                    state vector
                          │
                          ▼
              ┌─────────────────────┐
              │ Execution Policy π  │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Scheduler      KV Manager      Model Runner
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                        GPU
                         │
                         ▼
                 Runtime Metrics
                         │
                         └───────────────┐
                                         │
                                  policy feedback
```

This diagram should probably appear fairly early in your presentation.

---

# 9. Your methodological phases

I would divide the PhD into approximately four stages.

### Phase I — Formalization and characterization

Build the vocabulary and mathematical representation.

Study runtimes such as:

```text
vLLM
SGLang
TensorRT-LLM
TGI
DeepSpeed-MII
```

Extract:

```text
state variables
decision variables
heuristics
constraints
objectives
```

Then formalize their policies.

Your first major paper could come from this.

---

### Phase II — Research runtime

Build **ola-rt** as the experimental platform.

Initially it does not need to compete with vLLM.

Its purpose is:

> expose execution decisions clearly enough to experiment with them.

Architecture:

```text
ola-rt

scheduler/
kv/
model/
runtime/
policy/
metrics/
workload/
benchmark/
```

The important module is:

```rust
trait ExecutionPolicy {
    fn decide(
        &self,
        state: &RuntimeState
    ) -> ExecutionDecision;
}
```

Then:

```rust
struct ExecutionDecision {
    token_budget: usize,
    batch_size: usize,
    prefill_strategy: PrefillStrategy,
    kv_policy: KvPolicy,
    precision: Precision,
}
```

Now your formal theory maps directly into software.

That is very powerful academically.

---

# 10. Phase III — Optimization

Start with classical optimization.

Do not jump immediately to RL.

For example:

$$
\max
\quad
\alpha T -
\beta L -
\gamma C
$$

subject to:

$$
M(B,K) \leq M_{GPU}
$$

$$
P99(B,K,S) \leq L_{SLA}
$$

Possible approaches:

```text
MILP
MINLP
dynamic programming
model predictive control
queueing models
Bayesian optimization
online optimization
```

Depending on what aspects are tractable.

This provides interpretable baselines.

---

# 11. Phase IV — Learning policies

Then introduce learned policies.

Now:

$$
\pi_\theta(s_t)
\rightarrow a_t
$$

could be trained using:

```text
contextual bandits
RL
offline RL
imitation learning
learning-to-rank
```

I would be cautious about starting with deep RL.

An excellent progression is:

```text
heuristic
   ↓
mathematical optimization
   ↓
online optimization
   ↓
contextual bandit
   ↓
RL
```

That makes your experimental story much stronger.

---

# 12. Benchmark design

Your proposal presentation needs to show that you have thought seriously about evaluation.

Create workloads representing:

### Interactive

```text
short prompt
short output
high concurrency
strict TTFT
```

### Document

```text
long prompt
short output
prefill-heavy
```

### Generation

```text
short prompt
long output
decode-heavy
```

### Agent

```text
shared system prompt
shared prefix
tool waits
persistent context
```

### Bursty

```text
10 req/s
→ 500 req/s
→ 30 req/s
```

### Distribution shift

```text
Workload A
   ↓
suddenly
   ↓
Workload B
```

That last workload is fundamental to your thesis.

---

# 13. Metrics

Measure at least:

$$
TTFT
$$

$$
TPOT
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
GPU utilization
$$

$$
memory utilization
$$

$$
KV utilization
$$

$$
energy/request
$$

and perhaps:

$$
\$/1M\ tokens
$$

for deployment-oriented evaluation.

Your main results might eventually look like:

```text
                     Static      Adaptive

P99 latency          1.8s        0.9s
throughput           2,100       2,650 tok/s
GPU utilization      71%         89%
SLA violations       7.2%        1.1%
```

Those are only illustrative, but that is the kind of evaluation table you want your research to eventually produce.

---

# 14. Contribution statement

By the end of the proposal, the committee should understand that you are aiming for several contributions:

**C1 — Formal model**

A representation for inference execution policies.

**C2 — Characterization**

A systematic analysis of existing execution policies.

**C3 — Runtime**

An experimental runtime capable of executing policy-driven inference.

**C4 — Optimization**

Adaptive algorithms for policy selection.

**C5 — Evaluation**

A benchmark methodology for evaluating execution policies under non-stationarity.

Potentially:

**C6 — Learned policy**

A generalized learned inference controller.

That is already enough material for a substantial PhD.

---

# Your first-semester presentation

If I were structuring it, I would make it roughly **15–18 slides**:

1. **Title**
2. **The evolution of AI inference**
3. **Why inference systems matter**
4. **Modern inference runtime architecture**
5. **The problem: execution decisions**
6. **Static policies vs dynamic workloads**
7. **Example workload shift**
8. **Research gap**
9. **Inference Execution Policy — definition**
10. **Formal problem formulation**
11. **Research questions**
12. **Proposed architecture**
13. **Optimization approach**
14. **Experimental methodology**
15. **Benchmark workloads and metrics**
16. **Expected research contributions**
17. **PhD roadmap**
18. **Conclusion**

The conceptual climax should probably be slide 9 or 10:

$$
\boxed{
\pi:S\rightarrow A
}
$$

and you tell the audience:

> **My central argument is that an inference runtime should not simply execute a fixed collection of heuristics. Its execution strategy should be represented explicitly as a policy that can be analyzed, optimized and eventually learned.**

That is a very clear thesis proposition.

### A useful 3-year trajectory

You could make your entire PhD story fit on one timeline:

```text
YEAR 1
────────────────────────────────────────────
Formalization
Literature
Baseline runtime
vLLM/SGLang characterization
Paper 1: Inference Execution Policies

                    ↓

YEAR 2
────────────────────────────────────────────
Optimization
Adaptive scheduling
KV + scheduling co-optimization
Non-stationary benchmarks
Paper 2
Paper 3

                    ↓

YEAR 3
────────────────────────────────────────────
Learned policies
Hardware generalization
Multi-model / multimodal execution
Full ola-rt evaluation
Paper 4+
Thesis
```

And I would intentionally keep the thesis title relatively broad:

> **Formalization and Adaptive Optimization of Inference Execution Policies for Heterogeneous AI Systems**

because it leaves room for the PhD to evolve into **scheduling, KV management, multimodal inference, heterogeneous hardware, learned policies, or even agent-runtime execution** without making the original thesis obsolete.

The central intellectual contribution remains stable:

$$
\boxed{
\text{Make the inference execution policy an explicit, optimizable object.}
}
$$

Everything else can then build on top of that.



