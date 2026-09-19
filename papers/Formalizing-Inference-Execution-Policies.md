Yes. If those four papers are the backbone of the PhD, I’d make the literature review cumulative: each paper inherits the previous literature, but adds a new body of work.

## Paper 1 — Formalizing Inference Execution Policies

**Working title:** *Beyond Scheduling: Formalizing Inference Execution Policies for LLM Serving Systems*

This paper needs to establish that modern inference runtimes already make many policy decisions, but those decisions are usually embedded inside individual systems rather than represented through one common abstraction.

Start with the systems that created the modern serving model. **Orca** introduced iteration-level scheduling, showing that autoregressive serving requires scheduling at finer granularity than whole requests. ([USENIX][1]) **vLLM/PagedAttention** then made KV-cache memory management a first-class systems problem, using paging-inspired memory management to reduce fragmentation and enable larger effective batches. ([DOI][2]) **SGLang** expanded the runtime decision space further through RadixAttention, prefix reuse, cache-aware scheduling and structured generation. ([NeurIPS Papers][3])

Then study systems showing that an “execution policy” already extends far beyond scheduling. **DistServe** chooses separate resources and parallelism configurations for prefill and decode. ([arXiv][4]) **Splitwise** similarly separates the phases and considers throughput, cost and power objectives. ([arXiv][5]) **Sarathi-Serve** introduces chunked prefill to manage prefill/decode interference and the throughput–latency trade-off. ([GitHub][6]) **Llumnix** adds runtime request migration and dynamic rescheduling across model instances. ([USENIX][7])

So your literature taxonomy for Paper 1 could become:

| Policy dimension            | Representative literature      |
| --------------------------- | ------------------------------ |
| Request scheduling          | Orca, FastServe, Llumnix       |
| Batching                    | Orca, vLLM, Sarathi-Serve      |
| KV-cache allocation         | vLLM, SGLang                   |
| Cache reuse                 | SGLang                         |
| Preemption                  | FastServe, vLLM-family systems |
| Phase placement             | DistServe, Splitwise           |
| Model/device placement      | AlpaServe                      |
| Migration                   | Llumnix                        |
| Parallelism                 | DistServe, AlpaServe           |
| SLO optimization            | DistServe, Apt-Serve           |
| Resource allocation         | DistServe, Splitwise           |
| Admission/batch composition | Apt-Serve                      |

**FastServe** is especially relevant because it explicitly turns token-level preemption into a scheduler design and uses an MLFQ-inspired policy. ([arXiv][8]) **AlpaServe** is useful because it shows that model placement and parallelism selection should themselves be understood as policy decisions, especially under bursty arrivals. ([USENIX][9])

### Closest threat to your novelty

The literature is already becoming more formal.

**Apt-Serve** formally defines an adaptive scheduling optimization problem involving hybrid caching and dynamic batch composition. ([DOI][10]) More importantly, 2026 SIGMETRICS work develops an explicit theoretical framework for routing and scheduling LLM requests. ([DOI][11])

Therefore Paper 1 cannot claim:

> “Nobody has mathematically modeled inference scheduling.”

That would be false.

Your gap needs to be broader:

> Existing works formalize individual inference decisions such as scheduling, routing, caching or resource allocation. What remains missing is a runtime-independent representation that captures the **joint execution policy** governing these interacting decisions.

That distinction is critical.

Your formal object might therefore describe:

$$
\Pi =
(\mathcal S,\mathcal A,\mathcal T,\mathcal C,\mathcal O,\mathcal P)
$$

where \(\mathcal P\) could itself contain:

$$
\mathcal P =
\{
\pi_{\text{sched}},
\pi_{\text{batch}},
\pi_{\text{cache}},
\pi_{\text{placement}},
\pi_{\text{parallel}},
\pi_{\text{precision}},
\dots
\}
$$

and provide semantics for composition.

That is where I think the potentially novel contribution lives.

---

# Paper 2 — Characterizing Execution Policies Under Dynamic Workloads

**Working title:** *Characterizing LLM Inference Execution Policies under Dynamic Workloads*

Paper 1 says:

$$
\text{this is how we represent a policy}.
$$

Paper 2 asks:

$$
\text{how do different policies behave as the workload changes?}
$$

The literature review therefore shifts heavily toward **workload characterization, benchmarking, queueing and empirical serving behavior**.

One of the most important recent papers here is **ServeGen**, accepted at NSDI 2026. It analyzes production serving workloads covering conventional language models, reasoning models and multimodal models and argues that realistic workload structure matters substantially for serving-system evaluation. ([USENIX][12])

That is extremely important for your work because you don't want simplistic experiments of the form:

```text
arrival rate = fixed
prompt length = fixed
output length = fixed
```

You want distributions over:

$$
W_t =
(\lambda_t,L_{\text{in}},L_{\text{out}},M,\text{modality},\text{priority},\ldots)
$$

with those distributions changing over time.

There is also useful recent evidence from an analysis of a public production serving trace covering roughly five months and 337,000 requests. It found, among other things, that queueing can dominate TTFT and that observable request information is often a weak predictor of output length. ([IT-Universitetet i København][13])

That directly affects policies based on predicted request size.

You should also read **AlpaServe** closely. Its central insight is that bursty traffic changes the value of model parallelism and statistical multiplexing. ([arXiv][14])

And **Llumnix** motivates dynamic rescheduling precisely because inference requests are heterogeneous and unpredictable in both resource and latency demands. ([USENIX][7])

A particularly useful 2026 paper is **Beyond Prediction: Tail-Aware Scheduling for LLM Inference**. It shows that prediction-based scheduling can become fragile under workload distribution shift, bursty arrivals and GPU memory pressure—even when sophisticated size-aware policies are used. ([Microsoft][15])

That almost perfectly motivates your second paper.

Your literature taxonomy becomes:

### Workload dimensions

$$
W =
\{
\text{arrival process},
\text{prompt length},
\text{generation length},
\text{modality},
\text{SLO},
\text{cache locality},
\text{priority},
\text{model},
\text{concurrency}
\}
$$

### System-response dimensions

$$
Y =
\{
TTFT,
TPOT,
TTLT,
throughput,
goodput,
memory,
utilization,
energy,
cost
\}
$$

And then your central question becomes:

$$
f(\Pi,W)\rightarrow Y
$$

Given policy \(\Pi\) and workload \(W\), what performance behavior results?

The gap could be:

> Existing evaluations generally compare complete systems or specific scheduling mechanisms. There is comparatively little systematic analysis of how individual **policy components and policy compositions** respond to controlled changes in workload regime.

That lets you do something stronger than another “vLLM vs SGLang benchmark.”

---

# Paper 3 — Adaptive Inference Execution Policies

**Working title:** *Adaptive Inference Execution Policies for Dynamic LLM Serving*

Now things become more algorithmic.

Paper 2 establishes that:

$$
\Pi_A
$$

may work well under \(W_A\), while:

$$
\Pi_B
$$

works better under \(W_B\).

Paper 3 asks:

> Why must we choose one ahead of time?

Instead define:

$$
\Pi_t = g(s_t,W_t)
$$

and allow the runtime to alter its policy.

There is substantial adjacent work already.

**Llumnix** dynamically moves requests between model instances to deal with imbalance, fragmentation and changing requirements. ([USENIX][7])

**Apt-Serve** dynamically changes batch composition and formally formulates adaptive request scheduling. ([DOI][10])

**DistServe** co-optimizes resource allocation and parallelism according to TTFT/TPOT requirements, although much of that configuration is deployment-oriented rather than continuous runtime adaptation. ([arXiv][4])

And newer work is starting to make formerly static choices dynamic. For example, **Flying Serving** targets on-the-fly changes between data and tensor parallelism because static parallelism configurations perform poorly under bursts, priority changes and long-context traffic. ([GitHub][16])

There is also now enough literature that a 2026 survey explicitly frames LLM load balancing as a **closed-loop control and learning problem**, driven by compute, memory, communication, KV residency, heavy-tailed traffic and changing SLOs. ([ScienceDirect][17])

That paper is highly relevant to this stage of your PhD.

Your conceptual structure could be:

$$
s_t
=
[
q_t,
m_t,
u_t,
\lambda_t,
l_t,
c_t,
\dots
]
$$

runtime controller:

$$
g(s_t)\rightarrow\Pi_t
$$

then:

$$
\Pi_t(s_t)\rightarrow a_t.
$$

That is subtly different from ordinary scheduling.

There are **two decision layers**:

$$
\boxed{
\text{policy-selection}
\rightarrow
\text{execution decision}
}
$$

For example:

```text
Current regime:
high queue depth
low KV pressure
short requests
```

Controller chooses:

$$
\Pi_{\text{throughput}}
$$

Later:

```text
high KV pressure
long reasoning traffic
strict TTFT SLO
```

controller switches to:

$$
\Pi_{\text{latency/memory}}
$$

Potential techniques here include control theory, online optimization, constrained optimization, contextual bandits and model-predictive control.

I would deliberately study these **before** deep RL.

The scientific question becomes:

> How quickly can the runtime identify a regime change and transition to an execution policy that is suitable for the new regime without instability, oscillation or unacceptable switching overhead?

That's considerably stronger than “dynamically adjust batch size.”

---

# Paper 4 — Learning Inference Execution Policies under Non-Stationary Workloads

Now you cross into learning/control.

**Working title:** *Learning Inference Execution Policies under Non-Stationary Workloads*

The literature here should come from **three fields**, not only LLM systems.

You need:

**LLM systems literature**

plus

**online learning / reinforcement learning**

plus

**non-stationary control and optimization**.

On the LLM side, recent work already makes the motivation explicit: serving occurs under **non-stationary demand**, where workload composition, arrival rate, KV residency and resource pressure change over time. ([ScienceDirect][17])

ServeGen gives you empirical justification that real production workloads have complex distributions and that language, reasoning and multimodal workloads behave differently. ([USENIX][12])

Beyond Prediction gives further evidence that algorithms tuned around a presumed workload distribution can degrade when that distribution shifts. ([Microsoft][15])

You can model this as:

$$
P_t(W)\neq P_{t+k}(W).
$$

Your learner therefore cannot simply learn:

$$
\pi^*=
\arg\min_\pi
\mathbb E_{W\sim P}[J(\pi,W)]
$$

because there is no single fixed \(P\).

Instead:

$$
P=P_t
$$

and your goal becomes minimizing something like dynamic regret:

$$
R_T^{dyn}
=
\sum_{t=1}^{T}
J(\pi_t,W_t)
-
\sum_{t=1}^{T}
J(\pi_t^*,W_t)
$$

where:

$$
\pi_t^*
=
\arg\min_\pi J(\pi,W_t).
$$

That connects your systems problem directly to **online optimization under changing environments**.

You should therefore eventually review:

* non-stationary multi-armed bandits;
* contextual bandits;
* online convex optimization;
* dynamic regret;
* concept drift;
* continual learning;
* reinforcement learning in non-stationary MDPs;
* adaptive control;
* model-predictive control.

And those bodies of literature will probably influence what algorithm you ultimately choose.

There is already early direct RL work in inference systems—for example, recent work has applied deep RL to adaptive prefill chunking—but I'd treat this as evidence that learning-based control is emerging rather than as proof that RL is necessarily your best method. ([Sage Journals][18])

Your gap could instead be:

> Existing adaptive serving mechanisms largely optimize specific control knobs. We investigate whether the execution-policy representation introduced in Paper 1 can serve as a **learnable action space**, allowing a runtime to adapt coordinated groups of execution decisions under workload drift.

That is a much more interesting problem.

---

# How the literature connects

Your dissertation's literature map becomes:

```text
                 ML Serving
                     │
          Clipper / Clockwork / etc.
                     │
                     ▼
             Generative Serving
                     │
          ┌──────────┴──────────┐
          │                     │
       Scheduling             Memory
          │                     │
     Orca/FastServe           vLLM
          │                     │
          ├─────────┬───────────┤
                    │
              Modern Runtime
                    │
       ┌────────────┼─────────────┐
       │            │             │
    SGLang       DistServe     Llumnix
    caching      placement     migration
       │            │             │
       └────────────┼─────────────┘
                    │
                    ▼
            YOUR PAPER 1
       Execution-policy model
                    │
           Π = formal object
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   Workload literature       Formal scheduling
     ServeGen etc.           Apt-Serve etc.
        │                       │
        └───────────┬───────────┘
                    ▼
              YOUR PAPER 2
          policy characterization
                    │
                    ▼
              YOUR PAPER 3
             adaptive policies
                    │
       control / optimization
                    │
                    ▼
              YOUR PAPER 4
             learned policies
                    │
      online learning / RL /
        non-stationarity
```

And this gives the PhD a very clean intellectual evolution:

$$
\boxed{
\text{Representation}
\rightarrow
\text{Characterization}
\rightarrow
\text{Adaptation}
\rightarrow
\text{Learning}
}
$$

The important thing is that **Paper 1 has to be strong enough that Papers 2–4 genuinely need it**. If Paper 2 could simply be written without your formalism, reviewers may ask what Paper 1 actually contributed.

So I would make Paper 1 capable of expressing things like:

$$
\Pi_{\text{vLLM}}
$$

$$
\Pi_{\text{SGLang}}
$$

$$
\Pi_{\text{DistServe}}
$$

and then support operations such as:

$$
\text{compare}(\Pi_1,\Pi_2)
$$

$$
\text{compose}(\Pi_a,\Pi_b)
$$

$$
\text{constrain}(\Pi,C)
$$

$$
\text{optimize}(\Pi,W)
$$

and eventually:

$$
\text{learn}(W_t,s_t)\rightarrow\Pi_t.
$$

If you can create a formalism with those properties, then **“Formalizing Inference Execution Policies” really can become the foundational paper for the dissertation**, rather than just a taxonomy paper.

[1]: https://www.usenix.org/conference/osdi22/presentation/yu?utm_source=chatgpt.com "Orca: A Distributed Serving System for Transformer-Based Generative Models | USENIX"
[2]: https://doi.org/10.1145/3600006.3613165?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention | Proceedings of the 29th Symposium on Operating Systems Principles"
[3]: https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html?utm_source=chatgpt.com "SGLang: Efficient Execution of Structured Language Model Programs"
[4]: https://arxiv.org/abs/2401.09670?utm_source=chatgpt.com "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
[5]: https://arxiv.org/abs/2311.18677?utm_source=chatgpt.com "Splitwise: Efficient generative LLM inference using phase splitting"
[6]: https://github.com/microsoft/sarathi-serve?utm_source=chatgpt.com "GitHub - microsoft/sarathi-serve: A low-latency & high-throughput serving engine for LLMs · GitHub"
[7]: https://www.usenix.org/conference/osdi24/presentation/sun-biao?utm_source=chatgpt.com "Llumnix: Dynamic Scheduling for Large Language Model Serving | USENIX"
[8]: https://arxiv.org/abs/2305.05920?utm_source=chatgpt.com "Fast Distributed Inference Serving for Large Language Models"
[9]: https://www.usenix.org/conference/osdi23/presentation/li-zhouhan?utm_source=chatgpt.com "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving | USENIX"
[10]: https://doi.org/10.1145/3725394?utm_source=chatgpt.com "Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving | Proceedings of the ACM on Management of Data"
[11]: https://doi.org/10.1145/3820171.3806878?utm_source=chatgpt.com "Optimal Scheduling Algorithms for LLM Inference: Theory and Practice | ACM SIGMETRICS Performance Evaluation Review"
[12]: https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen?utm_source=chatgpt.com "ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production | USENIX"
[13]: https://researcher.itu.dk/en/publications/where-the-time-goes-analysis-of-a-public-llm-serving-system/?utm_source=chatgpt.com "Where the Time Goes: Analysis of a Public LLM Serving System - IT-University of Copenhagen"
[14]: https://arxiv.org/abs/2302.11665?utm_source=chatgpt.com "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving"
[15]: https://www.microsoft.com/en-us/research/publication/beyond-prediction-tail-aware-scheduling-for-llm-inference/?utm_source=chatgpt.com "Beyond Prediction: Tail-Aware Scheduling for LLM Inference - Microsoft Research"
[16]: https://github.com/shwgao/Flying-Serving?utm_source=chatgpt.com "GitHub - shwgao/Flying-Serving: This is the code repository for ICS'25 Flying-Serving. · GitHub"
[17]: https://www.sciencedirect.com/science/article/pii/S0925231226023714?utm_source=chatgpt.com "Load balancing for large language model inference serving: A control and learning oriented survey - ScienceDirect"
[18]: https://journals.sagepub.com/doi/10.3233/FAIA250867?utm_source=chatgpt.com "DRLServe: Adaptive Prefill Chunking with Deep Reinforcement Learning for LLM Inference, 2026"
