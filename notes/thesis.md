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
















Yes. This is exactly the right question, because a mathematical optimizer does **not magically know** that increasing batch size improves throughput but may hurt TTFT, or that increasing prefill work raises compute pressure while decode raises memory-bandwidth pressure.

Before RL, you need a **performance model**.

The basic structure is:

$$
\boxed{
\text{state} + \text{decision}
\;\xrightarrow{\text{system model}}\;
\text{predicted consequence}
}
$$

Then optimization happens over that model.

There are really three separate pieces:

$$
s_t = \text{what the system currently looks like}
$$

$$
a_t = \text{what you are allowed to change}
$$

$$
f(s_t,a_t) = \text{what you predict will happen}
$$

The optimizer solves:

$$
a_t^*
=
\arg\min_{a\in A}
J(f(s_t,a))
$$

subject to constraints.

So the important object we hadn't explicitly introduced yet is:

$$
\boxed{f:\;(s,a)\rightarrow \text{predicted system behavior}}
$$

That is your **system/performance model**.

---

# 1. First: distinguish state variables from decision variables

This is very important.

Suppose you have:

$$
s_t =
[
\text{SM utilization},
\text{HBM bandwidth},
\text{queue length},
\text{KV occupancy}
]
$$

These are mostly things you **observe**.

You don't directly command:

> Set HBM utilization to 73%.

Instead, you control things like:

$$
a_t =
[
\text{token budget},
\text{batch composition},
\text{prefill chunk size}
]
$$

Those decisions cause resource metrics to change.

So:

```text
decision
   ↓
system
   ↓
resource utilization
   ↓
latency / throughput
```

For example:

$$
\text{prefill chunk size}
\uparrow
$$

might cause:

$$
SMUtil\uparrow
$$

$$
HBMBandwidth\uparrow
$$

$$
TTFT_{\text{other requests}}\uparrow
$$

while possibly also causing:

$$
PrefillThroughput\uparrow.
$$

The optimizer needs a model encoding these relationships.

---

# 2. The simplest mathematical model

Suppose we only control:

$$
x =
\text{batch size}.
$$

And we care about:

$$
T(x)=\text{throughput}
$$

and:

$$
L(x)=\text{latency}.
$$

Maybe through benchmarking you discover approximately:

$$
T(x)=120x-0.7x^2
$$

and:

$$
L(x)=20+1.8x+0.03x^2.
$$

Now the optimizer already “knows” the consequences because they are encoded in those equations.

For example:

$$
\frac{dT}{dx}
=
120-1.4x
$$

tells you how throughput changes locally as batch size changes.

And:

$$
\frac{dL}{dx}
=
1.8+0.06x
$$

tells you how latency changes.

If:

$$
x=32
$$

then:

$$
\frac{dT}{dx}=75.2
$$

while:

$$
\frac{dL}{dx}=3.72.
$$

Interpretation:

> Around batch size 32, adding another unit of batch size is predicted to increase throughput significantly, but also increase latency.

The optimizer can therefore solve something like:

$$
\max_x T(x)
$$

subject to:

$$
L(x)\leq100ms.
$$

No RL involved whatsoever.

---

# 3. Where do those equations come from?

That's the key question.

There are several possibilities.

## A. Analytical models

You derive the relationship from system mechanics.

For example, suppose batch execution time is approximately:

$$
t(B)
=
t_{launch}
+
\frac{F(B)}{C_{GPU}}
+
\frac{D(B)}{BW_{HBM}}
$$

where:

* \(F(B)\) = FLOPs required,
* \(C_{GPU}\) = effective GPU compute throughput,
* \(D(B)\) = bytes transferred,
* \(BW_{HBM}\) = memory bandwidth.

Then throughput:

$$
Throughput(B)
=
\frac{\text{tokens processed}}
{t(B)}.
$$

Now you have a mechanistic model.

This is often preferable academically because you understand **why** the behavior occurs.

---

# 4. Roofline-style reasoning

This connects directly to the resource-vector example.

Suppose a kernel requires:

$$
F
$$

floating-point operations and moves:

$$
D
$$

bytes of memory.

Its arithmetic intensity is:

$$
AI=\frac{F}{D}.
$$

Hardware has:

$$
C_{\max}
=
\text{peak compute}
$$

and:

$$
BW_{\max}
=
\text{memory bandwidth}.
$$

The maximum attainable performance is bounded approximately by:

$$
Performance
\leq
\min(
C_{\max},
AI\cdot BW_{\max}
).
$$

This tells your controller something extremely useful.

If a workload has low arithmetic intensity:

$$
AI\downarrow
$$

then it is likely bandwidth-bound.

If:

$$
HBM=96\%
$$

and:

$$
SM=45\%
$$

you can infer:

> additional work with similar memory characteristics probably won't improve throughput much.

But perhaps compute-heavy prefill work has:

$$
AI_{\text{prefill}}
>
AI_{\text{decode}}.
$$

So mixing some prefill into the batch may exploit unused compute.

That's not learned through RL.

That's coming from a **hardware/performance model**.

---

# 5. Queueing theory gives you another model

Suppose arrivals are approximately:

$$
\lambda
$$

requests/sec and the serving rate is:

$$
\mu.
$$

For a simple \(M/M/1\) model:

$$
E[W]
=
\frac{1}{\mu-\lambda}.
$$

If:

$$
\lambda=90
$$

and:

$$
\mu=100,
$$

then:

$$
E[W]=0.1s.
$$

But if arrival rate increases slightly:

$$
\lambda=98,
$$

then:

$$
E[W]=0.5s.
$$

Notice the nonlinearity.

A tiny workload increase near saturation causes a huge queueing delay increase.

So the optimizer could understand:

```text
GPU is close to saturation
+
arrival rate rising
→
latency could explode
```

because queueing equations capture the relationship.

Again:

**no reinforcement learning required.**

---

# 6. But analytical equations won't capture everything

This is where benchmarking comes in.

Imagine you have these variables:

$$
x_1 = \text{batch token budget}
$$

$$
x_2 = \text{prefill chunk size}
$$

$$
x_3 = \text{number of active sequences}
$$

$$
x_4 = \text{KV occupancy}
$$

and metrics:

$$
y_1 = TTFT
$$

$$
y_2 = TPOT
$$

$$
y_3 = throughput.
$$

You can run experiments.

For example:

| Batch tokens | Chunk | KV util | Throughput |  TTFT |
| -----------: | ----: | ------: | ---------: | ----: |
|         4096 |   512 |     40% |       1800 |  50ms |
|         8192 |   512 |     40% |       2450 |  68ms |
|        16384 |   512 |     40% |       2700 | 125ms |
|         4096 |  2048 |     40% |       2100 |  70ms |
|         8192 |  2048 |     40% |       2850 |  95ms |

Now fit a model:

$$
\hat y=f(x).
$$

This is usually called a:

> **surrogate model** or **response-surface model**.

And it is still mathematical optimization.

---

# 7. A simple response surface

You might fit:

$$
TTFT
=
\beta_0
+
\beta_1B
+
\beta_2C
+
\beta_3M
+
\beta_{12}BC
+
\beta_{13}BM
+
\beta_{11}B^2.
$$

This is important because variables **interact**.

For example, batch size might be harmless when:

$$
KV=30\%
$$

but disastrous when:

$$
KV=95\%.
$$

So you cannot assume:

$$
effect(batch)
$$

is independent of:

$$
memory pressure.
$$

You need terms such as:

$$
B\times M.
$$

This is exactly why I said the controller should understand the **resource vector**, not individual utilization numbers.

---

# 8. The effect is actually a Jacobian

With multiple decisions and multiple outputs, the thing you're really asking about mathematically is the **Jacobian**.

Suppose:

$$
a=
\begin{bmatrix}
B\\
C\\
K
\end{bmatrix}
$$

where:

* \(B\) = batch token budget,
* \(C\) = prefill chunk size,
* \(K\) = KV retention aggressiveness.

And system outputs are:

$$
y=
\begin{bmatrix}
TTFT\\
TPOT\\
Throughput\\
Memory
\end{bmatrix}.
$$

Then:

$$
y=f(a,s).
$$

Locally:

$$
\Delta y
\approx
J\Delta a
$$

where:

$$
J=
\frac{\partial f}{\partial a}.
$$

For example:

$$
J=
\begin{bmatrix}
\frac{\partial TTFT}{\partial B}
&
\frac{\partial TTFT}{\partial C}
&
\frac{\partial TTFT}{\partial K}
\\
\frac{\partial TPOT}{\partial B}
&
\frac{\partial TPOT}{\partial C}
&
\frac{\partial TPOT}{\partial K}
\\
\frac{\partial Throughput}{\partial B}
&
\frac{\partial Throughput}{\partial C}
&
\frac{\partial Throughput}{\partial K}
\\
\frac{\partial Memory}{\partial B}
&
\frac{\partial Memory}{\partial C}
&
\frac{\partial Memory}{\partial K}
\end{bmatrix}.
$$

This matrix is effectively telling the controller:

> If I move each control knob slightly, what is predicted to happen to each performance metric?

That is one of the answers to your exact question.

---

# 9. Example

Suppose experimentally you estimate:

$$
J=
\begin{bmatrix}
+4 & +7 & -1\\
+2 & -1 & +0.5\\
+80 & +30 & +5\\
+0.2 & +0.1 & +3
\end{bmatrix}.
$$

Interpretation:

Increasing batch budget by one normalized unit roughly causes:

$$
TTFT +4
$$

$$
TPOT +2
$$

$$
Throughput +80
$$

$$
Memory +0.2.
$$

Increasing prefill chunk size:

$$
TTFT +7
$$

$$
TPOT -1
$$

$$
Throughput +30.
$$

Increasing KV retention:

$$
TTFT -1
$$

perhaps because more cached context avoids recomputation, but:

$$
Memory +3.
$$

Then the optimizer has explicit marginal tradeoffs.

---

# 10. Finite differences

What if you cannot derive derivatives analytically?

Very common.

You measure them.

Suppose:

$$
B=8192.
$$

Run:

$$
B=8192-\delta
$$

and:

$$
B=8192+\delta.
$$

Then estimate:

$$
\frac{\partial Throughput}{\partial B}
\approx
\frac{
T(B+\delta)-T(B-\delta)
}{
2\delta
}.
$$

This is finite-difference sensitivity analysis.

For example:

$$
T(7168)=2300
$$

$$
T(9216)=2480.
$$

Then:

$$
\frac{\partial T}{\partial B}
\approx
\frac{2480-2300}{2048}.
$$

Now you've empirically learned the local effect of changing batch token budget.

Again, not reinforcement learning.

Just measurement.

---

# 11. Some decisions aren't continuous

Suppose your choice is:

```text
scheduler =
    FCFS
    PRIORITY
    EDF
```

There is no meaningful derivative:

$$
\frac{\partial TTFT}
{\partial Scheduler}.
$$

Instead you evaluate each discrete option:

$$
J(s,a_1)
$$

$$
J(s,a_2)
$$

$$
J(s,a_3)
$$

and choose:

$$
a^*
=
\arg\min_{a\in A}J(s,a).
$$

This is where:

* MILP,
* mixed-integer nonlinear programming,
* dynamic programming,
* enumeration,
* branch-and-bound,

can enter.

---

# 12. Consider KV retention

Suppose block \(i\) consumes:

$$
m_i
$$

memory.

It has probability:

$$
p_i
$$

of being reused soon.

Recomputing it costs:

$$
c_i.
$$

You could define expected value of retaining the block as:

$$
v_i=p_ic_i.
$$

Decision:

$$
x_i\in\{0,1\}
$$

where:

$$
x_i=1
$$

means retain.

Optimization:

$$
\max_x
\sum_i v_ix_i
$$

subject to:

$$
\sum_i m_ix_i
\leq
M_{KV}.
$$

That's basically a knapsack problem.

The optimizer knows the effect because you supplied estimates for:

$$
p_i,\quad c_i,\quad m_i.
$$

No RL.

---

# 13. But how do we know \(p_i\)?

Now you've hit the next layer.

Maybe:

$$
p_i
$$

comes from:

* historical prefix frequencies,
* tool-call patterns,
* request-class statistics,
* TTL statistics,
* known agent structure.

That's **prediction**, not necessarily RL.

You could estimate:

$$
P(\text{reuse within }5s\mid features)
$$

using ordinary statistics.

For example:

$$
P(reuse)
=
\sigma(
\beta_0
+
\beta_1 age
+
\beta_2 frequency
+
\beta_3 requestClass
)
$$

using logistic regression.

The optimizer then consumes that probability.

This distinction is very important:

```text
Prediction model
        ↓
Optimization model
        ↓
Decision
```

The prediction model says:

> What is likely to happen?

The optimizer says:

> Given those predictions, what should I do?

That's different from RL.

---

# 14. This gives us three layers

I think this will become important for your thesis architecture:

```text
                 Observations
                      │
                      ▼
              SYSTEM ESTIMATOR
                      │
              estimates/predicts
                      │
            ┌─────────┴─────────┐
            │                   │
        workload            performance
        forecast              model
            │                   │
            └──────────┬────────┘
                       ▼
                   OPTIMIZER
                       │
              solves constrained
                 optimization
                       │
                       ▼
                 execution plan
                       │
                       ▼
                     runtime
```

So your controller is not synonymous with the optimizer.

The controller contains:

$$
\boxed{
\text{estimation}
+
\text{prediction}
+
\text{optimization}
}
$$

This is classical control-system thinking.

---

# 15. This is very close to Model Predictive Control

And I think **MPC** should absolutely be on your reading list.

At time \(t\), you observe:

$$
s_t.
$$

You have a model:

$$
s_{t+1}
=
f(s_t,a_t).
$$

Then predict several steps ahead:

$$
s_{t+1},s_{t+2},\ldots,s_{t+H}.
$$

Find actions:

$$
a_t,\ldots,a_{t+H-1}
$$

that minimize:

$$
\sum_{k=0}^{H-1}
J(s_{t+k},a_{t+k})
$$

subject to constraints.

Execute only:

$$
a_t.
$$

Then observe reality again and resolve.

So:

```text
observe
   ↓
predict future
   ↓
optimize
   ↓
execute first action
   ↓
observe again
   ↓
re-optimize
```

That's called **receding horizon control**.

No RL required.

---

# 16. Example for your inference runtime

Suppose current state is:

```text
queue             = 82
arrival rate      = 120 req/s
SM util           = 48%
HBM BW            = 93%
KV                 = 76%
TTFT P95           = 180 ms
TTFT SLO           = 250 ms
```

Candidate decisions:

```text
A:
batch_tokens=16384
prefill_chunk=4096

B:
batch_tokens=8192
prefill_chunk=2048

C:
batch_tokens=4096
prefill_chunk=1024
```

Your performance model predicts:

| Action | Throughput | TTFT P95 | HBM |  SM |
| ------ | ---------: | -------: | --: | --: |
| A      |       3200 |   330 ms | 99% | 70% |
| B      |       2850 |   225 ms | 96% | 66% |
| C      |       2200 |   165 ms | 88% | 56% |

Then solve:

$$
\max Throughput
$$

subject to:

$$
TTFT_{95}\leq250ms.
$$

A is infeasible.

C is feasible.

B is feasible and has greater throughput.

Therefore:

$$
a^*=B.
$$

There was no intelligence magic here.

The sophistication was in building a good:

$$
f(s,a).
$$

---

# 17. Now imagine HBM changes

A second later:

$$
HBM=65\%
$$

instead of 93%.

Your model may now predict:

| Action | Throughput |  TTFT |
| ------ | ---------: | ----: |
| A      |       3550 | 230ms |
| B      |       2900 | 195ms |
| C      |       2220 | 160ms |

Suddenly:

$$
A
$$

becomes feasible and optimal.

So the exact same action has different effects depending on the system state.

This is why we need:

$$
\boxed{f(s,a)}
$$

instead of merely:

$$
f(a).
$$

That's the critical point.

---

# 18. This also explains the resource vector

Earlier I said the controller should understand:

$$
[
SM,\ HBM,\ queue,\ KV,\ latency,\ldots
]
$$

rather than:

$$
GPUUtil.
$$

Now we can express why formally.

The marginal effect of batch size is actually:

$$
\frac{\partial Throughput}
{\partial B}
\Bigg|_{s}
$$

meaning:

> effect of batch size **conditional on the current state**.

When HBM pressure is low:

$$
\frac{\partial Throughput}{\partial B}>0
$$

might be large.

When HBM is saturated:

$$
\frac{\partial Throughput}{\partial B}\approx0.
$$

At even higher congestion:

$$
\frac{\partial Goodput}{\partial B}<0
$$

because latency violations dominate.

So there isn't a universal rule:

> batch size ↑ = good.

The actual relationship is:

$$
\boxed{
\text{effect of an action depends on state}
}
$$

which is precisely what the performance model captures.

---

# 19. How I would approach this in Year 1

I would **not** initially try to derive one perfect universal equation for inference.

Build a hybrid model.

### Mechanistic pieces

Use things we understand mathematically:

$$
KVMemory
$$

$$
FLOPs
$$

$$
HBM traffic
$$

$$
batch dimensions
$$

$$
queueing behavior.
$$

### Empirical pieces

Benchmark things too complex to model exactly:

$$
kernel latency
$$

$$
batch execution time
$$

$$
prefill/decode interference
$$

$$
real GPU behavior.
$$

Then:

$$
\boxed{
f
=
f_{\text{analytical}}
+
f_{\text{empirical correction}}
}
$$

Conceptually.

This is often much more robust than pretending either pure theory or pure black-box prediction is sufficient.

---

# 20. Your first experiment can be surprisingly small

On a single 5090 with vLLM:

Control:

$$
B=\texttt{max\_num\_batched\_tokens}
$$

Workload variables:

$$
\lambda,\quad
L_{prompt},\quad
L_{output}.
$$

Measure:

$$
Throughput,\quad TTFT,\quad TPOT.
$$

Run a grid:

$$
B\in
\{
1024,2048,4096,8192,16384
\}
$$

$$
\lambda\in
\{
10,25,50,100,200
\}.
$$

You'd get a response surface:

$$
Throughput=f(B,\lambda).
$$

And:

$$
TTFT=g(B,\lambda).
$$

You could literally plot:

```text
              arrival rate
                  ↑
                  │
             surface
           ╱╲  ╱╲
         ╱        ╲
       ╱            ╲
      └────────────────→ batch budget
```

Now you can see where:

$$
\frac{\partial T}{\partial B}>0
$$

and where increasing \(B\) stops helping.

That's mathematical system characterization before you ever introduce a learned controller.

---

# The core idea

The optimization algorithm doesn't inherently know:

> “increase this knob.”

It needs:

$$
\boxed{
\text{a mathematical or empirical model of cause and effect}
}
$$

You can think of the whole early-stage thesis as:

$$
\underbrace{
\text{measure the runtime}
}_{\text{system identification}}
$$

$$
\downarrow
$$

$$
\underbrace{
\text{construct }f(s,a)
}_{\text{performance model}}
$$

$$
\downarrow
$$

$$
\underbrace{
\text{analyze sensitivities}
}_{\partial f/\partial a}
$$

$$
\downarrow
$$

$$
\underbrace{
\text{solve }\arg\min/\arg\max
}_{\text{optimization}}
$$

$$
\downarrow
$$

$$
\underbrace{
\text{apply action}
}_{\text{controller}}
$$

This means there is actually an entire important research stage between **formalizing the execution policy** and **optimizing it**:

$$
\boxed{\text{System identification and performance modeling}}
$$

And I think that's a piece we had been skipping over. It may end up being fundamental to your first and second papers because without a model of

$$
(s_t,a_t)\rightarrow s_{t+1}
$$

or

$$
(s_t,a_t)\rightarrow performance,
$$

the mathematical optimizer has nothing meaningful to optimize.

