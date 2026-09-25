For AI systems/inference engineering, **benchmarking is almost the experimental language of the field**. If ola-rt becomes your PhD research platform, you want benchmarking built into it from the beginning—not added after you have a new scheduler or memory-management idea.

The important shift is this:

> You are usually not benchmarking “how intelligent is the model?”
> You are benchmarking **how efficiently a runtime executes a fixed model/workload under specified constraints.**

A useful mental model is:

$$
\text{Model} + \text{Hardware} + \text{Workload} + \text{Runtime Policy}
\rightarrow
\text{Latency, Throughput, Memory, Cost, Energy, SLO}
$$

That last component—**runtime policy**—is especially important for your research.

## 1. What exactly are we benchmarking?

Suppose you run Llama/Qwen through ola-rt.

You might compare:

```text
                 same model
                    │
        ┌───────────┼────────────┐
        ↓           ↓            ↓
      vLLM       SGLang        ola-rt
        │           │            │
        └───────────┼────────────┘
                    ↓
             same workload
                    │
                    ↓
        ┌───────────────────────┐
        │ TTFT                  │
        │ TPOT                  │
        │ throughput            │
        │ p50/p95/p99 latency   │
        │ GPU utilization       │
        │ memory utilization    │
        │ SLO attainment        │
        │ cost/token            │
        └───────────────────────┘
```

The fixed variables matter enormously. Comparing Runtime A running FP16 on an H100 against Runtime B running INT4 on a 4090 tells you very little about the runtimes themselves.

A defensible benchmark controls things such as model, weights, precision, GPU, input distribution, output distribution, concurrency, request arrival process, sampling configuration, warm-up, and measurement methodology.

---

# 2. The fundamental inference metrics

There are several you should know almost instinctively.

### Time to First Token — TTFT

A request arrives at \(t_a\), and the first generated token appears at \(t_1\):

$$
TTFT = t_1 - t_a
$$

Example:

```text
request ──────────────────────► first token
   │                              │
 0 ms                           85 ms

TTFT = 85 ms
```

TTFT is strongly affected by queueing, scheduling and **prefill**.

For interactive applications it matters enormously because it determines how quickly the model appears to respond.

---

### Time Per Output Token — TPOT

Once generation starts:

```text
token1 ── token2 ── token3 ── token4
       20ms      19ms      21ms
```

Approximately:

$$
TPOT =
\frac{t_{\text{last}} - t_{\text{first}}}
{N_{\text{output}}-1}
$$

This captures decode performance much better than TTFT.

And:

$$
\text{tokens/sec/user} \approx \frac{1}{TPOT}
$$

So 20 ms TPOT ≈ 50 tokens/s.

---

### Inter-token latency — ITL

Rather than averaging all token gaps:

$$
ITL_i = t_i - t_{i-1}
$$

Now you get a distribution:

```text
p50 ITL = 18ms
p95 ITL = 25ms
p99 ITL = 61ms
```

This exposes jitter that an average TPOT can hide.

---

### End-to-end latency

Simply:

$$
L = t_{\text{completion}} - t_{\text{arrival}}
$$

If a request arrives at 12:00:00.000 and finishes at 12:00:02.400:

$$
L = 2.4s
$$

Useful, but it mixes prefill, queueing and decode together.

---

# 3. Throughput

There are actually several throughput numbers worth reporting.

### Request throughput

$$
\lambda = \frac{\text{completed requests}}{\text{second}}
$$

For example:

```text
ola-rt
1000 completed requests
10 seconds

= 100 req/s
```

### Output-token throughput

Often much more informative for LLM serving:

$$
T_{\text{output}}
=
\frac{\text{generated tokens}}{\text{second}}
$$

Example:

```text
vLLM       8,200 tok/s
SGLang     8,700 tok/s
ola-rt     9,100 tok/s
```

But even that number alone can be misleading.

A runtime could achieve huge throughput by batching aggressively while making individual users wait forever.

Which leads to one of the central ideas of inference systems benchmarking:

## throughput and latency must be examined together.

---

# 4. The throughput-latency curve

This is one of the most important graphs you will produce during your PhD.

Imagine increasing offered request rate:

```text
Latency
  ^
  |
1s|                         *
  |                       *
  |                    *
  |                 *
  |             *
  |       *  *
  | *  *
  +--------------------------------> throughput
      10  20  30  40  50  60
```

Initially, throughput increases without destroying latency.

Eventually you approach system capacity.

Queues form.

Latency explodes.

This region is **saturation**.

So saying:

> “ola-rt achieves 11,000 tokens/sec.”

is incomplete.

A much stronger result is:

> “ola-rt sustains 10,800 output tokens/s while maintaining p99 TTFT below 200 ms.”

That is much closer to a meaningful systems result.

---

# 5. Percentiles matter more than averages

Suppose:

```text
99 requests = 100 ms
1 request   = 10 seconds
```

Average latency is roughly:

$$
199ms
$$

That hides something important.

Instead report:

```text
p50
p90
p95
p99
p99.9
```

For example:

| Metric     | ola-rt |
| ---------- | -----: |
| p50 TTFT   |  43 ms |
| p95 TTFT   |  91 ms |
| p99 TTFT   | 178 ms |
| p99.9 TTFT | 540 ms |

Tail latency is particularly important for your scheduling research.

---

# 6. GPU metrics

Now we move underneath the serving layer.

You should measure things like:

```text
GPU utilization
SM utilization
Tensor Core utilization
HBM bandwidth
VRAM usage
memory allocation
KV-cache utilization
cache hit/miss rate
kernel execution time
kernel launch overhead
PCIe transfer
CPU↔GPU synchronization
```

Suppose:

```text
Runtime       Tok/s     GPU util     VRAM
------------------------------------------
vLLM           8,100       81%       19GB
ola-rt         8,600       93%       17GB
```

Now you have evidence about **why** your runtime may be faster rather than merely observing that it is faster.

Eventually you'll use profilers such as NVIDIA Nsight Systems and Nsight Compute to examine execution timelines and individual kernels.

---

# 7. Memory benchmarking is huge for inference

For LLM serving, memory is frequently just as interesting as compute.

Think:

$$
M =
M_{\text{weights}}
+
M_{\text{KV}}
+
M_{\text{activations}}
+
M_{\text{workspace}}
+
M_{\text{fragmentation}}
$$

KV cache becomes particularly important.

You might benchmark:

```text
Maximum concurrent sequences
Maximum context length
KV cache occupancy
KV cache fragmentation
KV allocation/deallocation overhead
prefix-cache hit rate
eviction rate
bytes/token
```

This is exactly why something like PagedAttention is interesting.

A paper doesn't merely say:

> We invented a better memory mechanism.

It constructs experiments showing how the mechanism changes memory waste, concurrency, throughput and latency.

---

# 8. Offline versus online benchmarks

This distinction is critical.

### Offline

You already have 10,000 requests and essentially ask:

> Process these as quickly as possible.

```text
requests
████████████████████
        ↓
      runtime
        ↓
maximum throughput
```

This is good for measuring maximum system capacity.

### Online

Requests arrive over time:

```text
time ───────────────────────────>

R1 ──►
      R2 ──►
 R3 ──►
                 R4 ──►
```

Perhaps:

$$
\text{arrival} \sim Poisson(\lambda)
$$

Now scheduling and queueing become enormously important.

You measure:

```text
TTFT
TPOT
queueing delay
tail latency
SLO violations
goodput
```

For **your PhD direction, online benchmarking is especially important** because execution-policy research becomes much more interesting under dynamic workloads.

---

# 9. Goodput

This is a beautiful metric for your work.

Suppose your SLO says:

```text
TTFT < 200 ms
TPOT < 30 ms
```

Runtime A processes:

```text
120 requests/sec
```

but only 80 satisfy the SLO.

Then:

$$
\text{goodput}=80\text{ req/s}
$$

Runtime B processes only:

```text
110 req/s
```

but 105 satisfy the SLO.

Its goodput is:

$$
105\text{ req/s}
$$

Therefore raw throughput says A is better.

SLO-aware goodput shows something more useful about the serving system.

---

# 10. Benchmark workloads

You shouldn't benchmark only:

```text
prompt = "Hello"
generate 100 tokens
```

You need workload classes.

For example:

| Workload        | Input | Output |
| --------------- | ----: | -----: |
| Chat            |   512 |    128 |
| Long chat       |    4K |    256 |
| Summarization   |   16K |    512 |
| Code generation |    2K |     2K |
| Long-context QA |   32K |    128 |
| Reasoning       |    2K |     8K |

Because the bottleneck changes.

Large input:

```text
prefill dominated
```

Large output:

```text
decode dominated
```

And mixed workloads stress scheduling.

---

# 11. Then introduce dynamic workloads

This is where I think your benchmark suite becomes part of the research itself.

Instead of constant traffic:

```text
100 req/s
100 req/s
100 req/s
100 req/s
```

create:

```text
requests/s

300 |                 ████
250 |                 ████
200 |       ███       ████
150 |       ███ ███   ████
100 | ███   ███ ███   ████
 50 | ███ █ ███ ███ █ ████
    +------------------------> time
```

You could simulate:

**Bursty workloads**

$$
\lambda(t)=
\begin{cases}
30 & t < 60\\
300 & 60 \le t < 90\\
50 & t\ge90
\end{cases}
$$

**Changing sequence distributions**

```text
Phase 1: mostly short chat
Phase 2: long-context requests
Phase 3: reasoning-heavy generation
Phase 4: mixed traffic
```

**Changing SLO classes**

```text
interactive:
TTFT < 100ms

batch:
TTFT < 5s

premium:
TTFT < 50ms
```

Now the runtime must make actual decisions.

---

# 12. This connects directly to your inference execution policy research

Imagine defining a policy:

$$
\pi(s_t) \rightarrow a_t
$$

State:

$$
s_t =
(
Q_t,
M_t,
U_t,
K_t,
R_t,
S_t
)
$$

where perhaps:

```text
Q = queue state
M = memory state
U = GPU utilization
K = KV-cache state
R = request characteristics
S = SLO information
```

Actions might include:

```text
select next request
batch requests
split prefill/decode
preempt sequence
evict KV blocks
change batch size
route request
choose kernel
speculate
migrate workload
```

Then benchmarking provides the objective:

$$
J(\pi)
=
\alpha T
-\beta L_{p99}
-\gamma V_{\text{SLO}}
-\delta M
-\epsilon C
$$

where:

* \(T\): throughput/goodput
* \(L\): latency
* \(V_{\text{SLO}}\): SLO violations
* \(M\): memory cost
* \(C\): compute/cost/energy

This is where your **formalization → optimization → systems evaluation** pipeline starts becoming coherent.

---

# 13. Benchmarking kernels is another level

At the CubeCL/CUDA layer you'll run microbenchmarks.

Suppose you implement attention.

You might compare:

```text
Burn attention
your CubeCL attention
FlashAttention implementation
```

across:

```text
batch size
sequence length
head dimension
number of heads
dtype
GPU
```

Measure:

$$
\text{latency}
$$

and possibly:

$$
\text{TFLOP/s}
=
\frac{\text{estimated FLOPs}}{\text{execution time}}
$$

and memory bandwidth.

This is **microbenchmarking**.

Whereas running 10,000 simulated users through ola-rt is a **system/end-to-end benchmark**.

Both matter, but they answer different questions.

---

# 14. Your benchmark architecture should therefore be a first-class part of ola-rt

I'd structure the project conceptually like:

```text
ola-rt
│
├── model/
├── tensor/
├── kernels/
├── memory/
│   └── kv_cache/
├── scheduler/
├── executor/
├── policy/
│
└── benchmark/
    ├── micro/
    │   ├── matmul
    │   ├── attention
    │   ├── kv_cache
    │   └── sampling
    │
    ├── model/
    │   ├── prefill
    │   └── decode
    │
    ├── serving/
    │   ├── throughput
    │   ├── latency
    │   ├── goodput
    │   └── concurrency
    │
    ├── workloads/
    │   ├── chat
    │   ├── long_context
    │   ├── reasoning
    │   ├── burst
    │   └── mixed
    │
    ├── baselines/
    │   ├── vllm
    │   ├── sglang
    │   └── llama_cpp
    │
    └── metrics/
```

I would go further and make benchmark reproducibility part of the runtime's design:

```text
$ ola bench serve \
    --model Qwen/... \
    --policy fcfs \
    --requests 10000 \
    --arrival poisson \
    --rate 50 \
    --input-dist trace.json \
    --slo-ttft 200ms
```

producing something like:

```text
OLA-RUNTIME BENCHMARK

Model                 Qwen...
GPU                   RTX 5090
Policy                FCFS
Requests              10,000
Arrival rate          50 req/s

Throughput             48.9 req/s
Output throughput      7,421 tok/s

TTFT
  p50                  62 ms
  p95                 121 ms
  p99                 189 ms

TPOT
  p50                18.2 ms
  p99                27.4 ms

Peak VRAM             28.1 GB
KV utilization        87.4 %
GPU utilization       91.2 %

SLO attainment        97.8 %
Goodput               47.8 req/s
```

That infrastructure could become extremely valuable during the PhD because **every subsequent experiment plugs into the same measurement framework**.

Your workflow becomes:

```text
                     ola-rt
                        │
              ┌─────────┴─────────┐
              │                   │
           workload             policy
              │                   │
              └─────────┬─────────┘
                        ↓
                    execution
                        ↓
                  instrumentation
                        ↓
              benchmark results
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
      latency        throughput       memory
        ↓               ↓               ↓
      SLO            goodput       efficiency
        └───────────────┬───────────────┘
                        ↓
                 optimization
                        ↓
                  new policy π
                        │
                        └──────► repeat
```

And this gives us an important bridge we haven't discussed deeply yet: **benchmarking isn't merely how you validate the six papers; it gives you the objective functions, workload distributions and empirical constraints that the OR/optimization part of the PhD operates on.**

The next useful thing to study is **how vLLM/SGLang/MLPerf-style inference benchmarks are actually designed, how to build statistically defensible experiments, and how to go from those measurements to an optimization problem for your inference execution policy.**

