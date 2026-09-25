Yes — this is very relevant to your PhD direction, especially the parts around **inference execution policies, runtime optimization, scheduling, and hardware-aware execution**.

Cursor’s **Mixture-of-Kittens (MoK)** is basically a case study in what happens when you stop treating a deep-learning system as a sequence of isolated operators and instead optimize the *whole execution policy around the hardware*. Cursor built it because the MoE layer had become a major bottleneck, sometimes consuming more than half of training time. Their solution was a deterministic **MoE megakernel** that fuses communication and computation, overlaps them aggressively, minimizes CPU involvement, and exposes hardware-sensitive tuning knobs. Cursor reports up to **2.37× higher MXFP8 forward throughput** than its strongest public baseline in the configurations tested, and **1.41× end-to-end training throughput** in its production stack. ([Cursor][1])

The interesting part for your PhD is not specifically “build Mixture-of-Kittens.” It is that **MoK is almost a concrete example of the optimization problem you have been describing as an inference execution policy.**

Think about a normal runtime as having decisions like:

$$
\pi(x,h,s)\rightarrow
\{
\text{kernel},
\text{placement},
\text{batching},
\text{precision},
\text{communication},
\text{scheduling},
\text{memory policy}
\}
$$

where:

* \(x\) = workload characteristics
* \(h\) = hardware state
* \(s\) = runtime state
* \(\pi\) = execution policy

MoK contains several such decisions.

For example, Cursor explicitly tunes how many GPU SMs are assigned to **communication versus computation**, and notes that the best ratio depends on model shape, local token count, compute bandwidth, communication bandwidth, and whether execution is in the forward or backward pass. ([Cursor][1])

That is essentially:

$$
n_{\text{comm}},n_{\text{compute}}
=
\pi(
T,
H,
I,
E,
k,
B_{\text{network}},
F_{\text{GPU}}
)
$$

Their implementation exposes these as tunable parameters.

Your research question could go one step further:

> Why should those parameters remain manually tuned?

Instead:

$$
\pi^* =
\arg\min_{\pi}
\left[
\alpha L +
\beta C +
\gamma M +
\delta E
\right]
$$

subject to things such as:

$$
M \le M_{GPU}
$$

$$
L_{p99}\le SLA
$$

$$
\text{throughput}\ge T_{\min}
$$

where \(L\) might be latency, \(C\) communication overhead, \(M\) memory consumption, and \(E\) energy/cost.

That is where your **operations research / optimization component** becomes much more interesting.

---

### The bigger lesson from MoK

Traditionally, a framework might execute an MoE layer roughly like:

```text
router
 ↓
dispatch
 ↓
communication
 ↓
expert GEMM
 ↓
activation
 ↓
expert GEMM
 ↓
communication
 ↓
combine
```

Each piece might be a separate operation/kernel.

Conceptually:

```text
CPU
 ├── launch kernel A
 ├── wait
 ├── communication
 ├── launch kernel B
 ├── wait
 ├── launch kernel C
 └── ...
```

MoK instead moves toward:

```text
┌─────────────────────────────────────────┐
│              MEGAKERNEL                 │
│                                         │
│ dispatch ─┐                             │
│           ├── communication             │
│ GEMM ─────┤      ╲                      │
│           │       overlap               │
│ SwiGLU ───┤      ╱                      │
│           ├── communication             │
│ GEMM ─────┘                             │
│                                         │
│ GPU-resident scheduling                 │
└─────────────────────────────────────────┘
```

This is very similar to the architectural direction I think makes sense for **ola-rt**.

Your runtime should eventually stop thinking only in terms of:

```rust
tensor.matmul(...)
tensor.softmax(...)
tensor.matmul(...)
```

and start thinking in terms of:

```rust
ExecutionPlan {
    kernels,
    placement,
    memory,
    communication,
    overlap,
    precision,
    batching,
    scheduling,
}
```

MoK demonstrates why that distinction matters.

---

## One especially interesting idea: eliminate the CPU from the critical path

Cursor specifically discusses **ring token buffers** as a mechanism for eliminating CPU-GPU synchronization. ([Cursor][1])

This is a deep systems idea.

A naïve execution path often becomes:

```text
GPU → CPU → decision → GPU
```

which introduces synchronization barriers.

Better:

```text
GPU
 │
 ├── read work queue
 ├── execute
 ├── update state
 ├── schedule next work
 └── continue
```

Your runtime research could ask:

> How much inference scheduling can remain GPU-resident?

For example, suppose ola-rt has:

```text
TokenQueue
KVCacheManager
BatchScheduler
KernelScheduler
ExpertRouter
```

You could investigate moving progressively more runtime control onto the accelerator.

This becomes particularly interesting for:

* continuous batching
* speculative decoding
* MoE inference
* multimodal inference
* KV-cache management
* prefill/decode disaggregation

---

# Where I see a direct PhD opportunity

You could define an **Execution Policy IR**.

Something like:

```rust
ExecutionPolicy {
    compute: ComputePolicy,
    memory: MemoryPolicy,
    scheduling: SchedulingPolicy,
    communication: CommunicationPolicy,
    precision: PrecisionPolicy,
}
```

A MoE policy might become:

```rust
MoEPolicy {
    expert_parallelism: 8,
    top_k: 4,

    communication: PullBased,

    overlap: OverlapPolicy {
        communication_sms: 24,
        compute_sms: 108,
    },

    precision: MXFP8,

    fusion: FusionPolicy::MegaKernel,
}
```

A transformer decoding policy could look like:

```rust
DecodePolicy {
    attention: PagedAttention,

    batching: Continuous {
        max_tokens: 8192,
    },

    kv_cache: Paged {
        block_size: 16,
    },

    quantization: FP8,

    parallelism: TensorParallel(4),
}
```

Now you have something profound:

**MoK becomes one instance of your more general execution-policy representation.**

So instead of your thesis saying:

> “I implemented some faster kernels.”

It can say something closer to:

> **We introduce a formal representation for AI inference execution policies and formulate runtime configuration as an optimization problem over computation, communication, memory, and scheduling decisions.**

That is a much stronger research programme.

---

# Then your OR work enters naturally

Suppose you define policy variables:

$$
x =
[
b,
p,
k,
q,
c,
m,
s
]
$$

where:

* \(b\): batch size
* \(p\): parallelism
* \(k\): kernel choice
* \(q\): quantization
* \(c\): communication allocation
* \(m\): memory strategy
* \(s\): scheduling strategy

You can formulate:

$$
\max_x
\quad
\text{Throughput}(x)
$$

subject to:

$$
P99Latency(x) \le 150ms
$$

$$
Memory(x) \le 80GB
$$

$$
Quality(x) \ge Q_{min}
$$

$$
Power(x) \le P_{max}
$$

Already that could be:

* MILP
* constrained nonlinear optimization
* Bayesian optimization
* contextual bandits
* reinforcement learning

depending on how you model the system.

But real inference workloads are not static.

Suppose traffic changes:

```text
09:00 → 95% text
10:00 → large prompts
11:00 → VLM traffic spike
12:00 → short chat requests
13:00 → batch jobs
```

Then your optimization becomes:

$$
\pi_t =
f(
W_t,
H_t,
S_t
)
$$

and your runtime adapts.

That connects directly to your second paper idea:

> **Learning Inference Execution Policies under Non-Stationary Workloads**

And MoK gives you a real industrial example showing that runtime parameters such as the compute/communication allocation genuinely depend on the workload. ([Cursor][1])

---

# MoK also gives you another very important lesson: fusion is a policy

Kernel fusion often gets presented as a compiler optimization:

```text
A
B
C
```

becomes:

```text
ABC
```

But you can treat fusion itself as a runtime decision.

For example:

```text
Policy A
attention
softmax
projection
```

versus:

```text
Policy B
fused_attention
projection
```

versus:

```text
Policy C
mega_attention_decode_kernel
```

The optimal fusion may depend on:

$$
f(
\text{sequence length},
\text{batch size},
\text{GPU},
\text{memory pressure},
\text{request distribution}
)
$$

So your policy search space could include:

```rust
FusionStrategy::None
FusionStrategy::Operator
FusionStrategy::Subgraph
FusionStrategy::MegaKernel
```

That makes your PhD touch both:

**compiler optimization**

and

**operations research.**

---

# One thing I would explicitly borrow conceptually from MoK

Not their code.

Their **methodology**:

```text
Observe system
    ↓
Find dominant bottleneck
    ↓
Understand hardware
    ↓
Model execution
    ↓
Change scheduling/data movement
    ↓
Fuse work
    ↓
Benchmark
    ↓
Tune policy
```

Cursor observed that the MoE layer dominated substantial portions of training time, then redesigned execution around the hardware instead of merely optimizing individual matrix multiplications. ([Cursor][1])

That is exactly the mindset you want for ola-rt.

---

# And this changes how I would structure ola-rt

I would eventually give it roughly these layers:

```text
                 ola-rt

          MODEL / GRAPH
                │
                ▼
        ┌────────────────┐
        │ Execution IR   │
        └───────┬────────┘
                │
                ▼
       ┌───────────────────┐
       │ Policy Optimizer  │
       │                   │
       │ batching          │
       │ placement         │
       │ precision         │
       │ parallelism       │
       │ fusion            │
       │ communication     │
       │ cache strategy    │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │ Runtime Scheduler │
       └─────────┬─────────┘
                 │
         ┌───────┼────────┐
         ▼       ▼        ▼

      Kernels   Memory   Communication

         │       │        │
         └───────┼────────┘
                 ▼

               GPU
```

Burn/CubeCL can initially give you pieces toward the bottom:

```text
operators
tensor abstraction
autodiff
kernels
GPU compilation
```

Your research contribution can progressively move upward:

```text
Execution IR
      +
Policy representation
      +
Policy optimization
      +
Adaptive scheduler
```

That is much harder for an existing ML framework to hand you automatically, and therefore much more interesting academically.

---

## There is even a potential MoE-specific paper in your PhD

Something like:

**Adaptive Expert Execution Policies for Mixture-of-Experts Inference**

Instead of static:

```text
communication SMs = 24
compute SMs = 108
```

you learn:

$$
(c_t,g_t)
=
\pi(
\text{token distribution},
\text{expert load},
\text{network congestion},
\text{GPU utilization}
)
$$

where:

* \(c_t\) = communication resources
* \(g_t\) = compute resources.

Objective:

$$
\min
\quad
\alpha L_{P99}
-
\beta Throughput
+
\gamma CommunicationCost
+
\delta Imbalance
$$

You could additionally control:

```text
expert placement
expert replication
token routing
SM allocation
microbatch size
communication direction
kernel selection
```

Now you are squarely in **AI systems + optimization + scheduling**.

---

The reason I think MoK is particularly worth studying for you is that it validates one of the central assumptions behind your PhD: **modern AI performance increasingly comes from deciding how computation executes, not merely from the neural-network architecture itself.**

Cursor's MoK essentially says:

```text
Model
  +
hardware
  +
communication topology
  +
scheduling
  +
memory movement
  +
kernel structure
  =
actual performance
```

Your research can try to turn those last five terms from a collection of handwritten engineering decisions into a **formal, optimizable execution-policy abstraction**.

That's potentially the conceptual bridge between **ola-rt, your inference-policy formalization, OR/optimization, and systems work** that we've been looking for. ([Cursor][1])

[1]: https://cursor.com/pt-BR/blog/mixture-of-kittens?utm_source=chatgpt.com "Mixture-of-Kittens: our open-source MoE megakernel for NVL72s · Cursor"

