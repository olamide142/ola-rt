Yes — **“Formalizing Inference Execution Policy”** could be a very good first systems/theory paper, especially if you define something broader than just request scheduling.

The interesting distinction is:

> A scheduler decides **what runs next**.
> An inference execution policy decides **how the runtime should execute the workload given the current system state**.

That policy could encompass batching, preemption, KV-cache decisions, device placement, kernel choice, parallelism, quantization, speculative decoding, admission control and routing.

I would formalize the runtime at decision step \(t\) as:

$$
s_t =
(W_t, Q_t, M_t, H_t, C_t, S_t)
$$

where \(W_t\) is the active workload, \(Q_t\) the request queues, \(M_t\) memory/KV-cache state, \(H_t\) hardware state, \(C_t\) compute/communication state and \(S_t\) application SLOs.

Then define an execution policy

$$
\pi : \mathcal S \rightarrow \mathcal A
$$

with

$$
a_t = \pi(s_t)
$$

where the action may be something like

$$
a_t =
(B_t,P_t,K_t,D_t,R_t,Q_t^{(precision)},\ldots)
$$

representing batch composition, preemption decisions, cache management, device placement, routing, precision/quantization choice, etc.

The runtime then evolves according to

$$
s_{t+1}=F(s_t,a_t,\xi_t)
$$

where \(\xi_t\) captures uncertainty such as new arrivals, unknown generation lengths and changing resource conditions.

Now you have turned an inference engine into a **sequential decision system**.

The objective can be formulated as constrained optimization:

$$
\pi^*
=
\arg\min_\pi
\mathbb E
\left[
\sum_t
\alpha L_t
-\beta T_t
+\gamma M_t
+\delta E_t
+\eta C_t
\right]
$$

subject to constraints such as

$$
\text{TTFT}_{p99}\leq \tau_f
$$

$$
\text{TBT}_{p99}\leq\tau_b
$$

$$
M_t \leq M_{\max}
$$

and possibly fairness, power or cost constraints.

Here \(L\) could be latency, \(T\) throughput/goodput, \(M\) memory pressure, \(E\) energy consumption and \(C\) financial/compute cost.

The key contribution, however, shouldn't merely be putting an MDP equation around vLLM. There is already significant 2025–2026 research mathematically modelling LLM scheduling. Apt-Serve formally defines adaptive scheduling; recent SIGMETRICS work develops throughput-optimal scheduling; INFERMAX formulates optimal scheduling as a constraint problem; and 2026 work is explicitly arguing that LLM serving needs stronger mathematical foundations instead of isolated heuristics. ([DOI][1])

So I think your stronger research gap is:

> **Can we define a common mathematical abstraction for inference execution policies spanning multiple runtime control mechanisms, rather than deriving a different scheduler for every optimization?**

That's substantially more ambitious.

You could define an **Inference Execution Policy Model — IEPM**:

$$
\boxed{
\mathcal I =
(\mathcal W,\mathcal R,\mathcal S,\mathcal A,\mathcal T,\mathcal O,\mathcal C)
}
$$

where:

* \(\mathcal W\): workload model
* \(\mathcal R\): resource model
* \(\mathcal S\): runtime state space
* \(\mathcal A\): allowed runtime actions
* \(\mathcal T\): transition dynamics
* \(\mathcal O\): optimization objectives
* \(\mathcal C\): constraints

Then FCFS isn't a whole inference architecture. It is one policy within this model.

Continuous batching is another policy component.

PagedAttention changes available memory-management actions.

A speculative-decoding policy chooses whether and how aggressively to speculate.

A heterogeneous runtime might choose:

$$
\pi(s)=
\{
\text{GPU}_i,
\text{batch}=16,
\text{FP8},
\text{TP}=2,
\text{speculate}=4,
\text{cache-prefix}=true
\}
$$

while another state produces:

$$
\pi(s')=
\{
\text{GPU}_j,
\text{batch}=4,
\text{BF16},
\text{TP}=1,
\text{speculate}=0,
\text{cache-prefix}=false
\}.
$$

You can then introduce a useful hierarchy:

$$
\text{Execution Policy}
=
\begin{cases}
\pi_{\text{request}}\\
\pi_{\text{batch}}\\
\pi_{\text{memory}}\\
\pi_{\text{compute}}\\
\pi_{\text{placement}}\\
\pi_{\text{parallelism}}\\
\pi_{\text{precision}}\\
\pi_{\text{communication}}
\end{cases}
$$

That decomposition could itself become one of the paper's conceptual contributions.

It also opens up **policy composition**, which I think is especially interesting.

Suppose:

$$
\pi_{\text{latency}}
$$

optimizes latency and

$$
\pi_{\text{memory}}
$$

optimizes KV memory.

What happens when they disagree?

For example, the latency policy wants another request admitted into the batch while the memory policy wants eviction or preemption.

You now have a research problem:

$$
\pi^*
=
\pi_{\text{latency}}
\otimes
\pi_{\text{memory}}
\otimes
\pi_{\text{fairness}}
$$

What should \(\otimes\) mean?

Priority?

Constraint satisfaction?

Lagrangian optimization?

Pareto optimization?

Hierarchical control?

Learned arbitration?

That could become a surprisingly deep direction.

There's also clear evidence the field is moving this way. H-MAS uses a hierarchical closed-loop controller that changes scheduling behavior as workload conditions change, while recent agent-serving work proposes an explicit policy-driven runtime layer exposing primitives such as observe, score, predict and act. ([ACL Anthology][2])

So I might title the first paper:

**Formalizing Inference Execution Policies for Adaptive AI Serving Systems**

or, more systems-conference-like:

**Beyond Scheduling: A Formal Model of Inference Execution Policies**

And the thesis statement could be:

> Modern inference runtimes contain numerous interacting execution heuristics but lack a unified abstraction for representing, composing, analysing and optimizing these decisions. We introduce a formal execution-policy model that represents inference serving as constrained sequential decision-making over workload, memory, compute and hardware state.

Then `ola-rt` implements the abstraction:

```text
                 Workload
                    │
                    ▼
              Runtime State
                    │
                    ▼
        ┌──────────────────────┐
        │ Execution Policy π   │
        └──────────┬───────────┘
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
   Scheduler   Memory Policy   Placement
       │           │            │
       └───────────┼────────────┘
                   ▼
             Execution Plan
                   │
                   ▼
              CPU / GPU
                   │
                   ▼
               Metrics
                   │
                   └──────────────► state feedback
```

And then subsequent papers can build **on top of the formalism**:

**Paper 1 — Formalization**
*Beyond Scheduling: Formalizing Inference Execution Policies*

**Paper 2 — Optimization**
*Adaptive Execution Policies for Dynamic LLM Workloads*

**Paper 3 — Learning**
*Learning Inference Execution Policies under Non-Stationary Workloads*

**Paper 4 — Multimodal**
*Workload-Aware Execution Policies for VLM Inference*

**Paper 5 — Distributed systems**
*Policy-Driven Load Balancing for Heterogeneous AI Inference Clusters*

That's the point where `ola-rt` becomes something much more interesting than “Victor's Rust version of vLLM.” It becomes the **reference runtime in which your execution-policy model is implemented and experimentally tested**.

And there's a genuine opening here: current work already recognizes that output lengths, KV-cache growth, prefill/decode asymmetry, SLOs and non-stationary workloads make classical scheduling abstractions inadequate. Recent queueing and formal-schedulability work is attacking pieces of this problem. ([PubsOnLine][3])

The important PhD question becomes:

> **What is the correct abstraction for an AI inference runtime's decision-making process?**

That is a far deeper research question than “what scheduler should we use?”

[1]: https://doi.org/10.1145/3725394?utm_source=chatgpt.com "Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving | Proceedings of the ACM on Management of Data"
[2]: https://aclanthology.org/2026.findings-acl.1946/?utm_source=chatgpt.com "H-MAS: Hierarchical Multi-Agent Scheduling for Multi-Tenant LLM Serving - ACL Anthology"
[3]: https://pubsonline.informs.org/doi/10.1287/stsy.2025.0106?utm_source=chatgpt.com "Queueing, Predictions, and Large Language Models: Challenges and Open Problems | Stochastic Systems"




Yes. Your first paper can be the **foundation paper** for the rest of the PhD.

The important refinement is this: the novelty should not merely be “I write equations for vLLM.” The contribution should be a **general representation of inference execution policy** that is expressive enough to encode the behavior of multiple existing engines, compare them under a common abstraction, and then expose where static policies break down.

Something like:

$$
\Pi = (\mathcal S,\mathcal A,\mathcal C,\mathcal O,\mathcal T)
$$

where:

* \(\mathcal S\): runtime state — queue, KV cache, GPU memory, request characteristics, hardware utilization.
* \(\mathcal A\): possible runtime decisions — admit, batch, preempt, evict, route, change precision, choose device, etc.
* \(\mathcal C\): constraints — memory, SLOs, hardware capacity, fairness.
* \(\mathcal O\): objectives — latency, throughput, cost, energy, goodput.
* \(\mathcal T\): how the runtime state evolves after decisions.

Then you can take something like vLLM and say:

$$
\pi_{\text{vLLM}}(s_t) \rightarrow a_t
$$

and systematically describe what policy its scheduler, cache manager, batching system, admission logic, etc. actually implement.

You do the same for SGLang, TensorRT-LLM, llama.cpp and perhaps one or two others.

That gives you something very valuable:

$$
\text{vLLM}
\quad\longrightarrow\quad
\Pi_{vLLM}
$$

$$
\text{SGLang}
\quad\longrightarrow\quad
\Pi_{SGLang}
$$

$$
\text{TensorRT-LLM}
\quad\longrightarrow\quad
\Pi_{TRT}
$$

Now instead of comparing engines only through benchmarks, you can compare **their decision structures**.

For example, perhaps you discover:

$$
\pi_{A}
=
\pi_{\text{FCFS}}
+
\pi_{\text{continuous-batching}}
+
\pi_{\text{paged-KV}}
$$

while another runtime uses:

$$
\pi_B
=
\pi_{\text{cache-aware}}
+
\pi_{\text{continuous-batching}}
+
\pi_{\text{radix-cache}}
$$

and then ask:

> Under what workload conditions does policy \(A\) dominate \(B\)?

That's where the rest of the PhD starts.

Recent work already formalizes individual problems such as routing/scheduling and adaptive batching, so you'll need to distinguish your abstraction from those narrower formulations. For example, Apt-Serve formally defines its adaptive scheduling problem, while recent SIGMETRICS work develops a theoretical routing/scheduling framework. ([DOI][1])

Your prospective claim is broader:

> **Inference runtimes should be understood as collections of interacting execution policies rather than monolithic implementations.**

And then:

> **These policies can be represented independently of a particular engine, analyzed formally, composed, compared, and ultimately optimized.**

That is a compelling basis for a PhD.

---

### What “Learning Inference Execution Policies under Non-Stationary Workloads” means

This becomes much simpler once Paper 1 exists.

Suppose Paper 1 defines:

$$
a_t=\pi(s_t)
$$

where \(s_t\) is what the runtime currently observes and \(a_t\) is the execution decision.

Initially, you might manually design \(\pi\):

```text
if queue_depth > 100:
    increase_batch_size

if kv_cache_usage > 90%:
    preempt_low_priority_request

if latency_slo_is_being_missed:
    reduce_batch_size
```

Those are handcrafted policies.

Now consider a real production workload.

At 2 AM:

```text
50 requests/sec
short prompts
short generations
mostly chat
```

At 10 AM:

```text
500 requests/sec
moderate prompts
high concurrency
```

Then suddenly somebody launches a reasoning workload:

```text
200 requests/sec
4K-token prompts
10K-token generations
```

Then an image workload appears:

```text
VLM requests
large image embeddings
different memory profile
```

The workload distribution has changed.

Mathematically:

$$
P_t(W) \neq P_{t+k}(W)
$$

That is **non-stationarity**.

The environment generating your inference requests isn't statistically constant.

And this matters because a policy that is optimal here:

```text
short chat requests
↓
large batch
high throughput
```

might be terrible here:

```text
long reasoning requests
↓
huge KV caches
↓
memory pressure
↓
long batches block smaller requests
↓
P99 latency explodes
```

Recent serving research explicitly observes this problem: production workloads exhibit bursts, request-composition drift, and longer-term workload shifts, while fixed scheduling policies struggle to maintain QoS. ([ACL Anthology][2])

So instead of:

$$
\boxed{\pi(s)}
$$

being fixed forever, you have:

$$
\boxed{\pi_{\theta_t}(s)}
$$

where \(\theta_t\) represents the parameters governing the execution policy.

And those parameters change:

$$
\theta_{t+1}
=
\theta_t + \Delta_t
$$

based on observed runtime performance.

Now imagine ola-rt observing:

```text
queue_depth          = 212
mean_prompt_length   = 3400
mean_decode_length   = 7200
KV utilization       = 91%
GPU utilization      = 97%
P99 TTFT              = 1.8 s
target P99 TTFT       = 750 ms
```

The runtime can infer:

> My current execution policy isn't appropriate anymore.

It might change:

```text
batch limit:       64 → 24
prefill priority: low → high
preemption:       disabled → enabled
routing policy:   round-robin → cache-aware
speculation:      6 tokens → 2 tokens
```

So conceptually:

$$
s_t
\rightarrow
\pi_{\theta_t}
\rightarrow
a_t
\rightarrow
\text{runtime}
\rightarrow
\text{measure performance}
$$

then:

$$
\text{performance feedback}
\rightarrow
\theta_{t+1}
$$

giving you:

```text
                 workload
                    │
                    ▼
             ┌────────────┐
             │ Runtime    │
             │ State sₜ   │
             └──────┬─────┘
                    │
                    ▼
         ┌────────────────────┐
         │ Execution Policy   │
         │      πθ(sₜ)        │
         └─────────┬──────────┘
                   │
                   ▼
            execution plan
                   │
                   ▼
              GPU / CPU
                   │
                   ▼
         latency / throughput
            memory / cost
                   │
                   │ feedback
                   ▼
              update θ
```

That is what **learning an execution policy** means.

And “learning” doesn't necessarily mean deep RL.

You could investigate several approaches:

$$
\text{Policy adaptation}
=
\begin{cases}
\text{rules}\\
\text{online optimization}\\
\text{bandits}\\
\text{Bayesian optimization}\\
\text{MPC/control theory}\\
\text{reinforcement learning}\\
\text{learned cost models}
\end{cases}
$$

In fact, I would avoid jumping immediately to RL.

You might discover that online convex optimization or control theory is considerably more stable and interpretable.

---

The phrase **non-stationary** is particularly important because the interesting research problem isn't:

> “Can ML find the best batch size?”

That's fairly small.

It's:

> **Can an inference runtime recognize that the workload regime has changed and modify its execution policy while continuing to serve traffic and satisfy constraints?**

For example:

$$
W_1=\text{chat}
$$

then:

$$
W_2=\text{code generation}
$$

then:

$$
W_3=\text{reasoning}
$$

then:

$$
W_4=\text{multimodal}
$$

and perhaps:

$$
W_5 =
0.5 W_{\text{chat}}
+
0.2 W_{\text{reasoning}}
+
0.3 W_{\text{VLM}}
$$

The optimal execution policy can be different in every regime.

Prediction-based scheduling has already been shown to become fragile under distribution shifts, bursty arrivals and memory pressure, which is exactly the sort of phenomenon your later work could formalize and attack. ([Microsoft][3])

---

And this gives your PhD a really clean progression.

**Paper 1 — Representation**

> **Formalizing Inference Execution Policies**

Define what an execution policy is.

Define the state space, decision space, constraints and objectives.

Take existing runtimes and express their policies using your framework.

---

**Paper 2 — Analysis**

> **Characterizing Inference Execution Policies under Dynamic Workloads**

Now use the framework to ask:

$$
\pi_A \text{ vs } \pi_B
$$

under different workload regimes.

Determine where policies fail.

You might produce something like:

```text
                Short      Long       Bursty      VLM
                Chat       Reasoning

FCFS             ✓          ✗           ✗          ?
SRPT             ✓          ✓           ?          ?
Cache-aware      ✓          ✓           ✓          ?
Policy X         ...
```

But with proper mathematical/experimental analysis.

---

**Paper 3 — Optimization**

> **Adaptive Inference Execution Policies**

Given:

$$
s_t
$$

find:

$$
\pi^*_t
$$

that solves:

$$
\pi_t^*
=
\arg\min_{\pi}
J(\pi,s_t)
$$

subject to:

$$
\text{memory}\leq M
$$

$$
P99_{\text{TTFT}}\leq\tau
$$

etc.

---

**Paper 4 — Learning**

Now you get:

> **Learning Inference Execution Policies under Non-Stationary Workloads**

Instead of you determining when the runtime changes policy:

$$
\text{system learns } \pi_{\theta_t}.
$$

The runtime learns:

> “When my queue begins looking like *this* and KV pressure looks like *that*, policy \(P_3\) usually performs better.”

---

Then eventually your thesis becomes something like:

# **Inference Execution Policies for Adaptive AI Runtime Systems**

with a central intellectual thread:

$$
\boxed{
\text{Represent}
\rightarrow
\text{Understand}
\rightarrow
\text{Optimize}
\rightarrow
\text{Adapt}
\rightarrow
\text{Learn}
}
$$

That is much more coherent than doing five loosely related papers on vLLM optimizations. And importantly, Paper 1 gives you the **vocabulary and formal object that every subsequent paper operates on**.

[1]: https://doi.org/10.1145/3725394?utm_source=chatgpt.com "Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving | Proceedings of the ACM on Management of Data"
[2]: https://aclanthology.org/2026.findings-acl.1946/?utm_source=chatgpt.com "H-MAS: Hierarchical Multi-Agent Scheduling for Multi-Tenant LLM Serving - ACL Anthology"
[3]: https://www.microsoft.com/en-us/research/publication/beyond-prediction-tail-aware-scheduling-for-llm-inference/?utm_source=chatgpt.com "Beyond Prediction: Tail-Aware Scheduling for LLM Inference - Microsoft Research"
