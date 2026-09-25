Yes. This is directly relevant to **ola-rt optimization**, and it gives you a useful distinction between two optimization layers:

* **kernel-level optimization**: “Which implementation of this operation should I run?”
* **runtime/policy-level optimization**: “Which sequence of kernels, batching strategy, memory layout, cache policy, device placement, and scheduling decision should I use for this workload?”

Burn/CubeCL is already giving you machinery for the first problem. **ola-rt can generalize the idea to the whole inference execution policy.**

A simple way to think about Burn's current kernel selection is:

$$
k^* = \arg\min_{k \in K(x,h)} T(k,x,h)
$$

where:

* \(k\) = candidate kernel implementation
* \(x\) = tensor/problem shape
* \(h\) = hardware
* \(T\) = measured execution time

For example, a matrix multiplication might have several implementations:

```text
MatMul
 ├── Kernel A: tiled 16×16
 ├── Kernel B: tiled 32×32
 ├── Kernel C: vectorized
 ├── Kernel D: tensor-core optimized
 └── Kernel E: special kernel for awkward shapes
```

CubeCL benchmarks them and effectively asks:

> For `[batch, M, N, K]` on this GPU, which one is fastest?

That is already a small **decision system under uncertainty**.

For ola-rt, you could lift this one level higher. Instead of only selecting:

```text
operation -> kernel
```

your runtime could select:

```text
request/workload
    ↓
execution policy
    ├── batching strategy
    ├── batch size
    ├── prefill/decode scheduling
    ├── KV-cache layout
    ├── KV-cache placement
    ├── attention implementation
    ├── quantization mode
    ├── device placement
    ├── model partitioning
    ├── speculative decoding policy
    ├── kernel family
    └── concrete kernel
```

So your optimization hierarchy might look like:

```text
                    ola-rt optimizer
                           │
              ┌────────────┴────────────┐
              │                         │
        Runtime Policy             Kernel Policy
              │                         │
      continuous batching          CubeCL autotune
      request scheduling           matmul kernel
      cache allocation             attention kernel
      model placement              reduction kernel
      precision choice             fusion strategy
```

This is especially interesting because the **shape advice in the Burn documentation becomes a runtime decision variable**.

Suppose requests naturally give you:

```text
seq_len = 1000
```

but the GPU strongly prefers:

```text
seq_len = 1024
```

A naive runtime just accepts the shape.

An intelligent ola-rt policy could evaluate:

$$
\text{pad}(1000 \rightarrow 1024)
$$

versus

$$
\text{execute}(1000)
$$

Padding introduces extra FLOPs:

$$
24/1000 = 2.4\%
$$

more sequence elements.

But if 1024 enables substantially better vectorization, fewer bounds checks, better tensor-core utilization, or a faster attention kernel, the *larger* computation may actually finish faster.

That becomes an optimization problem:

$$
\min_p
\quad
\alpha L(p)
+
\beta C(p)
+
\gamma M(p)
+
\delta E(p)
$$

where \(p\) is an execution policy and:

* \(L\): latency
* \(C\): compute cost
* \(M\): memory usage
* \(E\): energy or resource cost

subject to constraints such as:

$$
M(p) \le M_{\text{GPU}}
$$

$$
L_{p99}(p) \le L_{\text{SLO}}
$$

$$
\text{accuracy}(p) \ge A_{\min}
$$

That starts looking much more like the OR side of your PhD.

Consider a concrete LLM example. Assume ola-rt receives:

```text
Request A: prompt = 487 tokens
Request B: prompt = 513 tokens
Request C: prompt = 1002 tokens
Request D: prompt = 1017 tokens
```

The runtime could create buckets:

```text
512
1024
```

giving:

```text
487  -> 512
513  -> 1024
1002 -> 1024
1017 -> 1024
```

But that might waste compute for the 513-token request.

So the runtime could instead learn or optimize bucket boundaries:

```text
512
640
768
1024
```

Now you have a genuine optimization problem:

$$
\min_B
\sum_i
[
T(\operatorname{bucket}(s_i,B))
+
\lambda W(s_i,B)
]
$$

where \(B\) is your chosen set of bucket sizes and \(W\) is wasted padded computation.

And the optimal buckets can depend on:

```text
GPU model
model architecture
attention implementation
current request distribution
memory pressure
latency SLO
```

This is where ola-rt can go beyond Burn's autotuner.

Burn is roughly learning:

```text
(shape, hardware, operation)
        ↓
best kernel
```

Your research runtime could learn:

```text
model
hardware
current load
queue state
sequence lengths
KV-cache occupancy
SLO
available memory
historical measurements
        ↓
execution policy
```

Conceptually:

$$
\pi(s) \rightarrow a
$$

where the state \(s\) might be:

$$
s =
(
q,
m,
g,
c,
\ell,
b,
r
)
$$

with:

* \(q\): queue state
* \(m\): memory state
* \(g\): GPU characteristics
* \(c\): KV-cache state
* \(\ell\): sequence-length distribution
* \(b\): current batches
* \(r\): request/SLO characteristics

and the action \(a\) could be:

$$
a =
(
\text{batch},
\text{kernel},
\text{precision},
\text{padding},
\text{cache layout},
\text{schedule}
)
$$

That is essentially an **inference execution policy**.

And this sentence from Burn is particularly important for your research:

> “It’s better to have a slow neural network layer followed by fast ones than to propagate unevenness…”

There is a deeper systems principle hiding there: **local optimality does not imply global optimality.**

Imagine:

```text
Layer 1
1000-wide tensor
   ↓
Layer 2
1000
   ↓
Layer 3
1000
   ↓
Layer 4
1000
```

Every subsequent kernel pays the awkward-shape penalty.

Instead:

```text
Layer 1
1000
   ↓
transform once
   ↓
1024
   ↓
Layer 2 fast
   ↓
Layer 3 fast
   ↓
Layer 4 fast
```

You incur:

$$
C_\text{transform}
$$

once, but potentially save:

$$
\sum_{i=2}^{n}
(C_i^{1000}-C_i^{1024})
$$

The transformation is worthwhile when:

$$
C_\text{transform}
<
\sum_{i=2}^{n}
(C_i^{1000}-C_i^{1024})
$$

This is exactly why optimizing kernels independently can be insufficient. ola-rt could optimize over an **execution graph**, not merely one operation at a time.

You could even represent this in your future IR:

```text
Tensor {
    shape: [B, S, H],
    dtype: BF16,
    layout: RowMajor,
    alignment: 32,
}
```

and allow the optimizer to produce transformations:

```text
original:
[B, 1000, 4096]

optimized:
PadSequence(1024)
    ↓
Attention::Flash
    ↓
FusedMLP
    ↓
TrimSequence(1000)
```

The runtime planner could estimate:

```text
Plan A
-------
no padding
generic attention
generic matmul
estimated: 7.8 ms


Plan B
-------
pad 1000 -> 1024
FlashAttention kernel
aligned GEMM
trim
estimated: 5.1 ms
```

and select Plan B.

This gives you a very natural architectural relationship:

```text
Burn
  │
  ├── Tensor abstraction
  ├── Operators
  ├── CubeCL
  │    └── kernel autotuning
  │
  └────────────┐
               ↓
            ola-rt
               │
       Execution Planner
               │
       Policy Optimizer
               │
     ┌─────────┼───────────┐
     ↓         ↓           ↓
 batching   memory      graph
 policy     policy      transforms
     │         │           │
     └─────────┴─────┬─────┘
                     ↓
               Kernel selection
                     ↓
                  CubeCL
                     ↓
                    GPU
```

And your research question becomes significantly broader than:

> “Which CUDA kernel is fastest?”

It becomes something closer to:

> **Given a model, workload, hardware state, and service objectives, what execution policy minimizes inference cost while satisfying latency, throughput, memory, and accuracy constraints?**

That connects directly to the formalization work you've been considering.

You could define:

$$
P = (S,A,T,C)
$$

where:

* \(S\): runtime state space
* \(A\): allowable execution actions
* \(T\): execution transitions
* \(C\): cost model

Then an execution policy is:

$$
\pi:S\rightarrow A
$$

Burn's kernel autotuner becomes one subpolicy:

$$
\pi_\text{kernel}:
(\text{op},\text{shape},\text{device})
\rightarrow
\text{kernel}
$$

while ola-rt implements the much larger policy:

$$
\pi_\text{runtime}:
(\text{requests},
\text{hardware},
\text{memory},
\text{model},
\text{SLO})
\rightarrow
\text{execution plan}
$$

That distinction is potentially one of the strongest conceptual foundations for ola-rt: **CubeCL optimizes how an individual computation executes; ola-rt researches how the whole inference workload should execute.**

