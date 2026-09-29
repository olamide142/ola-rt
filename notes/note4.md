Yes. **This is an excellent anchor paper for your PhD direction.** It is *A Survey of LLM Inference Systems* by James Pan and Guoliang Li, first posted in June 2025. Its real value for you is not any single technique; it gives us a map of the **decision space inside an LLM inference runtime**. :chatgpt-content-reference{index="1"}

The paper can essentially be reduced to one question:

> **An LLM request has an uncertain future. Given the state of the request, GPU, memory, queue, caches and workload right now, what should the inference system do next?**

That question is remarkably close to the problem you are trying to formalize.

---

# 1. The whole paper in one picture

The authors divide inference into roughly:

\[
\boxed{
\text{Request Processing}
\rightarrow
\text{Model Execution}
\rightarrow
\text{Memory Management}
}
\]

surrounded by scheduling, batching, load balancing, hardware and the application frontend. Figure 1 is especially important because it treats inference as an **entire system**, not merely `model.forward()`. :chatgpt-content-reference{index="2"}

A simplified version is:

```text
                    Incoming Requests
                           ↓
                    ┌────────────┐
                    │   Queue    │
                    └─────┬──────┘
                          ↓
              ┌──────────────────────┐
              │ Scheduler / Batcher  │
              └──────────┬───────────┘
                         ↓
             ┌─────────────────────────┐
             │    Inference Runtime    │
             │                         │
             │ Attention               │
             │ FFN                     │
             │ Sampling                │
             │ Kernels                 │
             └──────────┬──────────────┘
                        ↓
                 GPU / accelerator

Meanwhile:

KV cache ←→ GPU memory ←→ CPU memory ←→ storage

and above everything:

load balancing
resource allocation
preemption
cache management
quantization
```

And in a distributed system:

```text
          Global Load Balancer
             /      |      \
           GPU     GPU     GPU
            ↓       ↓       ↓
        Scheduler Scheduler Scheduler
```

The inference runtime therefore acts much more like an **operating system/database scheduler** than people initially realize.

That's the first major lesson from the paper.

---

# 2. Why does LLM inference require such a complicated system?

This is the single most important part of the paper.

Suppose the prompt contains \(p\) tokens:

\[
x_1,x_2,\ldots,x_p
\]

To generate the next token:

\[
x_{p+1}
=
LLM(x_1,\ldots,x_p)
\]

Then:

\[
x_{p+2}
=
LLM(x_1,\ldots,x_p,x_{p+1})
\]

then:

\[
x_{p+3}
=
LLM(x_1,\ldots,x_p,x_{p+1},x_{p+2})
\]

and so on.

This is **autoregressive generation**. :chatgpt-content-reference{index="3"}

The critical problem is:

\[
\boxed{\text{We do not know beforehand when generation will stop.}}
\]

Imagine request A:

```text
"What is 2+2?"
```

Maybe:

\[
20\text{ output tokens}
\]

Request B:

```text
"Explain general relativity."
```

Perhaps:

\[
2500\text{ tokens}
\]

But even given the prompt, the runtime generally cannot know the exact output length before actually generating it. The paper emphasizes that this makes **processing time and eventual memory consumption uncertain**. :chatgpt-content-reference{index="4"}

And suddenly we have a stochastic systems problem.

---

# 3. This is the fundamental uncertainty

Suppose request \(i\) arrives.

We know:

\[
P_i = \text{prompt length}
\]

but the eventual generated length

\[
G_i
\]

is unknown.

Consequently:

\[
\text{ExecutionTime}_i
=
f(P_i,G_i,\text{hardware},\text{batch},\ldots)
\]

is unknown.

And:

\[
\text{Memory}_i(t)
\]

changes as tokens are generated.

Therefore:

\[
\boxed{
G_i,\quad
T_i,\quad
M_i(t)
}
\]

are all uncertain.

That uncertainty propagates throughout the system.

You don't know exactly:

```text
how long a request runs
how large its KV cache becomes
when GPU memory becomes exhausted
which request becomes a straggler
what the next batch should contain
how much capacity a GPU actually has
which GPU a new request should use
whether a preemption will soon become necessary
```

The authors identify unpredictable request lifetime and resulting memory/scheduling problems as fundamental characteristics that distinguish LLM serving from conventional processing systems. :chatgpt-content-reference{index="5"}

This is where the paper starts becoming directly useful to your research.

---

# 4. Prefill and decode

You should have these two phases almost permanently in your head.

Consider:

```text
User prompt:
"Explain Bellman's equation to me"
```

Suppose this tokenizes into 10 tokens.

## Prefill

The model processes:

\[
x_1,\ldots,x_{10}
\]

together.

Because these tokens are already known, the GPU can exploit large parallel matrix operations.

Conceptually:

```text
10 prompt tokens

x1
x2
x3
...
x10
 ↓
Transformer
 ↓
KV cache
```

The system creates the K and V representations that will subsequently be reused.

Prefill tends to expose considerable parallel computation.

---

# 5. Decode is completely different

Now the model generates:

```text
x11
```

Then it has to wait for \(x_{11}\) before it can determine \(x_{12}\).

So:

```text
x11
 ↓
x12
 ↓
x13
 ↓
x14
```

The sequence dependency prevents us from simply generating all future tokens in parallel.

Thus:

\[
x_{t+1}=f(x_{\leq t})
\]

means:

\[
x_{t+2}
\]

depends on the result of the previous inference step.

This creates repeated **decision epochs**.

That word is important for what comes later.

---

# 6. Why the KV cache exists

Inside attention, we have:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

and approximately:

\[
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt d}\right)V
\]

The survey presents the same basic attention relationship when explaining its request-processing model. :chatgpt-content-reference{index="6"}

Imagine we generated token 100.

To generate token 101, attention needs information derived from tokens:

\[
1,\ldots,100.
\]

Without caching, we would continually recompute their key and value projections.

Instead we store:

```text
K1 V1
K2 V2
K3 V3
...
K100 V100
```

So token 101 only adds:

```text
K101 V101
```

The stored matrices are the **KV cache**.

---

# 7. And here comes the systems problem

Each newly generated token increases the cache:

\[
KV_{t+1}
=
KV_t+\Delta KV.
\]

Therefore memory usage grows approximately with sequence length:

\[
M(t)
\propto
\text{number of cached tokens}.
\]

So a request isn't something like:

```text
Request arrives
reserve 600 MB
execute
free 600 MB
```

Instead:

```text
Request arrives

100 MB
 ↓
110 MB
 ↓
120 MB
 ↓
130 MB
 ↓
...
??? MB
 ↓
terminates at unknown time
```

This is why memory management is a first-class scheduling variable.

The paper explicitly connects KV growth with unknown request cost and with batching, scheduling and preemption decisions. :chatgpt-content-reference{index="7"}

---

# 8. The beautiful abstraction hidden inside this paper

We can describe the runtime at time \(t\) by a state:

\[
S_t.
\]

For example:

\[
S_t =
(
Q_t,
R_t,
M_t,
K_t,
U_t,
H_t,
L_t
)
\]

where perhaps:

\[
Q_t = \text{queued requests}
\]

\[
R_t = \text{running requests}
\]

\[
M_t = \text{free/used memory}
\]

\[
K_t = \text{KV-cache state}
\]

\[
U_t = \text{GPU utilization}
\]

\[
H_t = \text{cache-hit information}
\]

\[
L_t = \text{latency/SLO state}.
\]

The runtime then makes decisions.

Call those:

\[
A_t.
\]

For example:

\[
A_t =
(
B_t,
P_t,
D_t,
E_t,
O_t,
G_t,
C_t
)
\]

where decisions could include:

\[
B_t=\text{which requests to batch}
\]

\[
P_t=\text{priority/scheduling decision}
\]

\[
D_t=\text{device assignment}
\]

\[
E_t=\text{eviction decision}
\]

\[
O_t=\text{offloading decision}
\]

\[
G_t=\text{GPU/resource allocation}
\]

\[
C_t=\text{configuration decisions}.
\]

Then execution happens.

New requests arrive.

Some requests terminate.

KV caches grow.

GPU conditions change.

So:

\[
S_{t+1}
=
F(S_t,A_t,W_{t+1})
\]

where

\[
W_{t+1}
\]

represents exogenous information.

This can include:

```text
new request arrivals
request output length
cache growth
hardware interference
failures
workload shifts
```

Now look at what we have.

\[
\boxed{
S_t \rightarrow A_t \rightarrow S_{t+1}
}
\]

That is a **sequential decision problem**.

And this is where the survey naturally meets Powell.

---

# 9. The paper doesn't say that explicitly

This distinction is important.

The survey mostly discusses systems techniques individually:

```text
continuous batching
paged memory
SJF scheduling
cache eviction
load balancing
quantization
speculative decoding
disaggregated serving
...
```

But your research can move one abstraction level upward:

> What are these things actually?

They are **actions taken by an inference controller**.

So instead of studying:

\[
\text{better batching}
\]

alone, your question becomes:

\[
\boxed{
\text{What policy should choose batching/configuration/resource actions as system state changes?}
}
\]

The paper gets tantalizingly close to this when it argues that modern systems fundamentally depend on **load prediction, adaptive mechanisms and cost reduction**, and later says more adaptive systems, dynamic batch sizing and elastic resource provisioning are needed for workload shifts. :chatgpt-content-reference{index="8"}

That is arguably the most important passage for your thesis.

---

# 10. Now Section 2 makes much more sense: Request Processing

Section 2 asks:

> **What computation must the model perform?**

The fundamental operators are basically:

```text
attention
FFN
token sampling
```

The authors then discuss alternatives that alter the cost/quality tradeoff. :chatgpt-content-reference{index="9"}

For attention:

```text
MHA
GQA
MQA
Sparse Attention
Shared Attention
```

These are not just architectural curiosities.

Each changes the runtime's cost structure.

For example, MQA and GQA reduce how many key/value projections need to exist compared with ordinary multi-head attention. Sparse attention calculates attention over fewer locations. Shared attention exploits common attention work. :chatgpt-content-reference{index="10"}

So imagine your performance model:

\[
Latency =
f(
model,
attention\_type,
sequence\_length,
batch\_size,
hardware
).
\]

Changing attention changes \(f\).

---

# 11. The FFN

The paper writes an FFN approximately as:

\[
FFN(x)
=
g(xW_1+b_1)W_2+b_2
\]

where \(g\) is a nonlinear activation. :chatgpt-content-reference{index="11"}

Attention's cost depends strongly on context/KV size.

The FFN behaves differently: for a fixed model, its parameter dimensions do not continually increase with sequence length.

So already the workload contains computational components with different characteristics.

That matters enormously for scheduling.

---

# 12. Mixture-of-Experts adds another decision dimension

With an MoE:

```text
             token
               ↓
              gate
          /    |    \
       expert expert expert
          \    |    /
             output
```

Only a subset of experts typically execute for a token.

That reduces computation relative to activating a giant dense FFN, but creates new systems issues such as expert placement and load balancing. The survey calls attention to these additional hardware and balancing concerns. :chatgpt-content-reference{index="12"}

Notice the pattern:

Every algorithmic optimization creates **new systems decisions**.

---

# 13. Speculative decoding is another beautiful example

Normally:

```text
Large model:
token
token
token
token
token
```

Sequentially.

Speculative decoding introduces a cheap draft model:

```text
Small model:

A B C D E
```

and asks the big model to verify candidates more efficiently.

Conceptually:

```text
           draft model
                ↓
        x1 x2 x3 x4 x5
                ↓
         verification
                ↓
         accept prefix
```

Now the runtime has even more possible decisions:

\[
\text{use speculative decoding?}
\]

\[
\text{which draft model?}
\]

\[
\text{how many tokens to speculate?}
\]

The paper notes an important point: speculative decoding does **not automatically guarantee** greater throughput; its usefulness depends on characteristics such as how often the drafter succeeds. :chatgpt-content-reference{index="13"}

That's exactly a state-dependent policy problem.

---

# 14. Section 3 is probably the most important section for your PhD

Its title is:

> **Model Optimization and Execution**

This section covers:

```text
kernels
batching
scheduling
load balancing
load prediction
```

It is almost directly the territory of an **inference execution policy**. :chatgpt-content-reference{index="14"}

---

# 15. Kernel optimization

At the low level, something like attention is not one magical GPU operation.

Naively you might perform:

```text
QKᵀ
 ↓
write to GPU memory
 ↓
read
 ↓
softmax
 ↓
write
 ↓
read
 ↓
multiply V
```

Data movement is expensive.

Kernel fusion tries to perform several operations together:

```text
QKᵀ → softmax → ×V
```

without repeatedly materializing huge intermediate arrays.

The paper uses FlashAttention as the classic example: tiled matrix operations plus online softmax reduce costly memory traffic. :chatgpt-content-reference{index="15"}

This distinction is important:

\[
\text{FLOPs}
\neq
\text{actual execution cost}.
\]

Actual cost includes:

\[
\text{compute}
+
\text{memory IO}
+
\text{kernel launches}
+
\text{synchronization}
+\cdots
\]

That's why your causal/performance models cannot simply use theoretical FLOP counts.

---

# 16. Batching

Imagine three requests:

```text
A: generate 100 tokens
B: generate 4 tokens
C: generate 30 tokens
```

Traditional static batching might put:

```text
[A B C]
```

together.

After four decode rounds:

```text
A: still running
B: finished
C: still running
```

If the system retains the static batch structure until everything finishes, resources can be wasted.

The survey contrasts this with **continuous batching**, where the batch is reconstituted after decoding rounds. :chatgpt-content-reference{index="16"}

So:

```text
t0:
A B C

t1:
A B C

B completes

t2:
A C D
```

Request D can immediately enter.

This enormously improves utilization.

---

# 17. But here's the research question

How big should the batch be?

Large batch:

\[
\uparrow throughput
\]

but:

\[
\uparrow memory
\]

and potentially:

\[
\uparrow latency,
\quad
\uparrow preemption risk.
\]

Small batch:

\[
\downarrow memory
\]

but:

\[
\downarrow GPU utilization.
\]

So:

\[
b^*
=
\arg\min_b
\{
\text{latency}
-
\alpha\text{throughput}
+
\beta\text{memory risk}
\}
\]

conceptually.

The survey specifically says existing systems frequently use fixed sizes or token budgets, sometimes derived by offline benchmarking, and identifies optimal dynamic batch sizing as an open problem. :chatgpt-content-reference{index="17"}

That sentence should jump off the page for you.

Because the immediate question is:

> Why should that token budget remain fixed when the workload changes?

Exactly.

---

# 18. Scheduling

Now suppose the queue contains:

```text
r1
r2
r3
r4
...
```

Which should execute first?

The paper discusses classical policies such as:

\[
FCFS
\]

\[
SJF
\]

and

\[
MLQ.
\] :chatgpt-content-reference{index="18"}


FCFS:

```text
first request in
→ first request served
```

Fair and easy.

But imagine:

```text
r1 = 5000-token response
r2 = 5-token response
```

Then \(r_2\) sits behind \(r_1\).

That's **head-of-line blocking**.

---

# 19. Shortest Job First sounds better

SJF:

\[
\text{execute request with smallest remaining work}.
\]

Classically, this can strongly improve mean response time.

But here we hit the core uncertainty again:

\[
\boxed{\text{How do we know the job length?}}
\]

We don't.

So scheduling becomes dependent on **prediction**.

The paper discusses systems that attempt to predict output length or relative length using learned models and other techniques. :chatgpt-content-reference{index="19"}

Which gives us:

\[
\hat G_i
=
f_\theta(prompt_i).
\]

Then use:

\[
\hat G_i
\]

for scheduling.

But:

\[
\hat G_i \neq G_i.
\]

And prediction error itself affects system performance.

That's another research opportunity.

---

# 20. Scheduling isn't just latency scheduling

Because a running request owns KV cache.

Suppose:

```text
GPU memory = 80 GB
```

Current state:

```text
request A KV = 16 GB
request B KV = 12 GB
request C KV = 8 GB
...
```

You may have to preempt one.

But preempting isn't free.

The paper points out that resuming may involve either recomputing the cache through prefill or retrieving it from CPU/remote storage. :chatgpt-content-reference{index="20"}

So the scheduling decision really looks more like:

\[
\text{priority}_i =
f(
waiting\ time,
predicted\ remaining\ time,
KV\ size,
recovery\ cost,
SLO,
cache\ locality
).
\]

Now this is much richer than FCFS.

---

# 21. Load balancing

Suppose you have:

```text
GPU 1
GPU 2
GPU 3
GPU 4
```

A request arrives.

Where should it go?

Naively:

\[
\arg\min_i queue_i.
\]

But queue length may be misleading.

GPU 1:

```text
3 requests
each almost finished
```

GPU 2:

```text
2 requests
each expected to generate thousands of tokens
```

Which GPU actually has less load?

Probably GPU 1.

And now add cache locality.

Perhaps GPU 2 already contains KV blocks matching the new request's system prompt.

Then assigning there avoids prefill work.

The survey describes load balancers that incorporate estimated load and cache reuse, along with dynamic request migration when worker conditions change. :chatgpt-content-reference{index="21"}

Again:

\[
device^*
=
\pi(S_t).
\]

---

# 22. Section 4: Memory management

This section is incredibly important because modern LLM serving is often as much a **memory-management problem** as a compute problem.

The paper organizes it around ideas including:

```text
paged memory
eviction
offloading
quantization
cache persistence
``` :chatgpt-content-reference{index="22"}


---

# 23. PagedAttention becomes simple if you think about operating systems

Traditional approach:

```text
Request A:
reserve memory for maximum possible sequence

[........................................]
```

Suppose it eventually uses only:

```text
[.........]
```

Huge waste.

Instead:

```text
A:
[page][page][page]

B:
[page][page]

C:
[page][page][page][page]
```

Pages are allocated as necessary.

It is strongly analogous to virtual memory/page-based allocation.

The survey contrasts static preallocation with dynamic page/block allocation precisely because the eventual KV-cache size is uncertain. :chatgpt-content-reference{index="23"}

So:

\[
M_i(t+1)
=
M_i(t)+\text{page}
\]

only when necessary.

That is one of the main innovations that made vLLM important.

---

# 24. Eviction versus offloading

When GPU memory fills up, perhaps the cache for request A has to leave.

Option 1:

\[
\text{Evict}
\]

Delete it.

If A resumes:

\[
\text{recompute KV}.
\]

Option 2:

\[
\text{Offload}
\]

Move it:

```text
GPU
 ↓
CPU RAM
```

or potentially lower storage tiers.

When resuming:

```text
CPU RAM
 ↓
GPU
```

The tradeoff becomes:

\[
\text{recompute cost}
\quad vs \quad
\text{transfer cost}.
\]

The runtime should arguably decide based on the state.

---

# 25. Quantization

Suppose your KV cache uses FP16.

Each number requires:

\[
16\text{ bits}.
\]

If you use INT8:

\[
8\text{ bits}.
\]

Approximately half the raw representation size.

But aggressive quantization can degrade model quality.

Therefore:

\[
\downarrow \text{precision}
\Rightarrow
\downarrow \text{memory}
\]

but potentially:

\[
\downarrow \text{quality}.
\]

The paper stresses that the best quantization scheme depends on model architecture and application and calls for more systematic understanding of **when, what and how to quantize**. :chatgpt-content-reference{index="24"}

Now ask the interesting question:

> Why must quantization itself be globally static?

Could request A use one precision and request B another?

Could KV-cache precision vary according to current memory pressure?

Could layer precision change?

Could the decision depend on SLO/quality constraints?

Now you're thinking like your thesis.

---

# 26. Cache persistence

Consider 10,000 requests beginning with:

```text
"You are ChatGPT, a helpful..."
```

The beginning is identical.

Instead of recomputing its KV values 10,000 times:

```text
compute once
cache
reuse
```

Systems such as SGLang exploit prefix/cache reuse, and the survey discusses prefix matching, radix trees and persistent cache mechanisms. :chatgpt-content-reference{index="25"}

Now load balancing is no longer merely:

\[
\text{least busy GPU}.
\]

It can be:

\[
\text{best combination of load + cache hit}.
\]

The control variables are becoming coupled.

---

# 27. That coupling is critical

This is perhaps the deepest thing to learn from the paper.

You cannot really optimize these components independently.

Suppose you increase:

\[
batch\ size.
\]

Then:

\[
GPU\ utilization\uparrow
\]

but also:

\[
memory\ usage\uparrow.
\]

Therefore:

\[
preemption\ probability\uparrow.
\]

Which causes:

\[
cache\ eviction\uparrow.
\]

Which can cause:

\[
recomputation\uparrow.
\]

Which affects:

\[
latency\uparrow.
\]

Which alters:

\[
SLO\ violations\uparrow.
\]

So:

\[
batch\ size
\rightarrow
memory
\rightarrow
preemption
\rightarrow
cache
\rightarrow
compute
\rightarrow
latency.
\]

This is exactly why your earlier work on **cause/effect performance models** matters.

---

# 28. Section 5: Put everything together

The paper then stops examining components individually and asks:

> What does an actual inference system look like?

The examples include Orca, vLLM, Sarathi-Serve, SGLang, FastServe, Preble, DistServe, TetriInfer, Splitwise, Mooncake and DeepFlow, with tables comparing execution and memory-management policies. :chatgpt-content-reference{index="26"}

Notice the evolution.

Conceptually:

```text
Orca
  ↓
continuous batching

vLLM
  ↓
paged KV memory

Sarathi
  ↓
chunked prefill

SGLang
  ↓
cache-aware runtime

then multi-replica systems
  ↓
load balancing
cache placement
P/D disaggregation
resource elasticity
```

This timeline is essentially the history of **turning more static assumptions into managed runtime decisions**. :chatgpt-content-reference{index="27"}

That's a very useful way to read the literature.

---

# 29. Prefill/decode disaggregation

Traditionally:

```text
GPU
 │
 ├── prefill
 └── decode
```

Same worker does both.

Disaggregated serving:

```text
Prefill workers
      ↓
 transfer KV
      ↓
Decode workers
```

The reason this is powerful is that the two phases can now be scaled and controlled independently. The paper discusses DistServe, Splitwise, Mooncake and related systems in precisely this context. :chatgpt-content-reference{index="28"}

So if workload becomes:

```text
huge prompts
short generations
```

you might need:

\[
more\ prefill\ capacity.
\]

If it becomes:

```text
short prompts
very long generation
```

you may need:

\[
more\ decode\ capacity.
\]

Therefore even:

\[
N_\text{prefill}
\]

and

\[
N_\text{decode}
\]

become control variables.

The survey notes systems that dynamically change the ratio according to workload. :chatgpt-content-reference{index="29"}

---

# 30. Now we can finally state your research problem clearly

The survey gives us many separate mechanisms:

| Mechanism | Decision variable |
|---|---|
| Batching | which requests + how many tokens |
| Scheduling | which request runs next |
| Admission | whether a request starts |
| Preemption | which request pauses |
| Placement | which GPU executes it |
| Cache | retain / evict / offload |
| Precision | FP16 / FP8 / INT8 / etc. |
| Kernel | implementation selection |
| Speculation | draft strategy |
| P/D allocation | number/type of workers |
| Scaling | allocate/release hardware |

Most systems implement some subset using individual heuristics.

For example:

```text
batch size = fixed token budget

scheduler = FCFS

eviction = LRU

placement = least-load

quantization = configured offline
```

In mathematical notation:

\[
a_t^{batch}=\pi^{batch}(s_t)
\]

\[
a_t^{schedule}=\pi^{schedule}(s_t)
\]

\[
a_t^{cache}=\pi^{cache}(s_t)
\]

\[
a_t^{placement}=\pi^{placement}(s_t)
\]

and so on.

But these are coupled.

So you can define the overall execution policy as:

\[
\boxed{
A_t=\pi(S_t)
}
\]

where:

\[
A_t =
\begin{bmatrix}
a_t^{batch}\\
a_t^{schedule}\\
a_t^{placement}\\
a_t^{cache}\\
a_t^{precision}\\
a_t^{kernel}\\
a_t^{resource}
\end{bmatrix}
\]

That is **inference execution policy** in a mathematically useful sense.

---

# 31. And now Powell enters naturally

A Powell-style sequential decision model can be written:

\[
(S_t,A_t,W_{t+1},S^M).
\]

For your problem:

### State

\[
S_t=
(\text{requests},
\text{KV caches},
\text{memory},
\text{GPU load},
\text{queues},
\text{SLOs},
\text{hardware},
\text{cache locality})
\]

### Decision

\[
A_t=
(\text{batching},
\text{scheduling},
\text{placement},
\text{eviction},
\text{precision},
\text{resource allocation})
\]

### Exogenous information

\[
W_{t+1}
=
(\text{new requests},
\text{generated-token outcomes},
\text{terminations},
\text{hardware conditions})
\]

### Transition

\[
S_{t+1}
=
S^M(S_t,A_t,W_{t+1})
\]

### Objective

Perhaps:

\[
\min_\pi
\mathbb E
\left[
\sum_t
C(S_t,A_t)
\right].
\]

And:

\[
C=
\alpha L
+
\beta E
+
\gamma \text{SLOViolations}
+
\delta \text{Cost}
+
\eta \text{QualityLoss}.
\]

Or constrained:

\[
\max_\pi
\quad Throughput
\]

subject to:

\[
TTFT_{p99}\leq\tau_1
\]

\[
TBT_{p99}\leq\tau_2
\]

\[
Memory_t\leq M_{\max}
\]

\[
QualityLoss\leq\epsilon.
\]

Now we have moved from:

> "Let's make vLLM faster."

to:

> **Given a stochastic, dynamically changing inference environment, determine the execution policy that controls the runtime to satisfy performance, quality and resource objectives.**

That is a much more general research programme.

---

# 32. The paper's three ideas become your controller architecture

The authors repeatedly reduce techniques to three broad strategies:

\[
\boxed{\text{Prediction}}
\]

\[
\boxed{\text{Adaptation}}
\]

\[
\boxed{\text{Cost reduction}}
\] :chatgpt-content-reference{index="30"}


We can reinterpret these.

### Prediction

Estimate:

\[
\hat y_t=f(S_t)
\]

such as:

```text
future output length
future memory demand
future latency
future request arrivals
future cache pressure
```

### Optimization / decision

Choose:

\[
A_t=
\arg\min_A \hat C(S_t,A).
\]

### Adaptation

Observe:

\[
S_{t+1}
\]

and repeat.

So your runtime becomes:

```text
               ┌──────────────────┐
               │ Observe runtime  │
               └────────┬─────────┘
                        ↓
               ┌──────────────────┐
               │ Performance /    │
               │ workload model   │
               └────────┬─────────┘
                        ↓
               ┌──────────────────┐
               │ Optimizer /      │
               │ policy           │
               └────────┬─────────┘
                        ↓
              ┌────────────────────┐
              │ Execution actions  │
              └─────────┬──────────┘
                        ↓
                     ola-rt
                        │
                        └──────↺
```

Now we are at the heart of your PhD.

---

# 33. What I think is the most important sentence in the whole paper for you

Conceptually, it is the discussion near the end saying general-purpose LLM inference systems need **more accurate load prediction and adaptive mechanisms such as dynamic batch sizing and elastic resource provisioning to respond to workload shifts**. :chatgpt-content-reference{index="31"}

Because the obvious next research question is:

\[
\boxed{
\text{Adaptive according to what formal policy?}
}
\]

And then:

\[
\boxed{
\text{How do we represent that policy?}
}
\]

Then:

\[
\boxed{
\text{How do we evaluate one policy against another?}
}
\]

Then:

\[
\boxed{
\text{How do we optimize that policy?}
}
\]

Then eventually:

\[
\boxed{
\text{Can we learn the policy online under non-stationary workloads?}
}
\]

That sequence can practically become your PhD.

---

# 34. So keep this hierarchy in your head

This is the one list I want you to memorize:

1. **Model** — transformer, attention, FFN, MoE, etc.
2. **Operators** — matmul, attention, normalization, sampling.
3. **Kernels** — concrete GPU implementation of operators.
4. **Runtime mechanisms** — KV paging, continuous batching, cache persistence, preemption.
5. **Runtime decisions** — batch size, priority, placement, precision, cache action, worker allocation.
6. **Execution policy** — rules mapping runtime state to those decisions.
7. **Controller/optimizer** — determines or updates the execution policy.
8. **Objective** — latency, goodput, throughput, cost, energy, quality, fairness, etc.

Your research primarily lives around **5–8** while requiring you to deeply understand **1–4**.

That distinction will save you a lot of confusion.

---

# 35. And it explains why ola-rt matters

Your runtime isn't merely:

```rust
model.forward(x)
```

It eventually becomes your experimental plant:

```text
                 ola-rt

request manager
      │
      ├── execution graph
      │
      ├── scheduler
      │
      ├── continuous batcher
      │
      ├── KV manager
      │
      ├── kernel selector
      │
      ├── memory manager
      │
      ├── preemption manager
      │
      └── metrics
             │
             ↓
      Runtime State Sₜ
             │
             ↓
      Policy Controller
             │
             ↓
         Decision Aₜ
             │
             └─────────────↺
```

The paper is giving you the vocabulary and mechanisms required to build that experimental environment.

Your contribution then doesn't have to be "I invented batching" or "I invented KV caching."

Instead:

> **I provide a formal abstraction for expressing inference execution policies, build a runtime capable of executing these policies, construct models describing their effects, and develop optimization/adaptation methods for selecting policies under changing workloads.**

That is substantially more interesting.

---

# 36. The gap I would write in the margin of this paper

Write this beside Section 6:

> **Existing work proposes many specialized policies and heuristics for individual inference decisions. What is missing is a general formalism in which these coupled execution decisions can be represented, composed, reasoned about, optimized and adapted under workload uncertainty.**

That is not a claim the survey itself makes verbatim; it is the research question that emerges when you read the survey through the lens of your direction.

And notice how beautifully it joins the two halves of what we've been studying:

```text
COMPUTER SCIENCE
─────────────────────────
Transformers
Kernels
GPU architecture
KV caches
Batching
Scheduling
Distributed systems
Compilers
Runtime systems

              ↓

      INFERENCE POLICY
      ────────────────

              ↑

APPLIED MATHEMATICS / OR
─────────────────────────
State
Actions
Exogenous information
Transition functions
Cost models
Optimization
ADP
Policy search
Stochastic optimization
Learning
```

**That middle box is your PhD.**

The computer-science side tells you **what decisions exist and how the machine behaves**.

The OR side tells you **how to formulate and choose those decisions**.

And this survey is one of the better papers for building the bridge between them. :chatgpt-content-reference{index="32"}

The next thing I would do with this paper is go through it **properly, section by section**, beginning with **Section 2.1 and Figure 2: prefill → attention → KV cache → decode**, deriving the computation, memory complexity, arithmetic intensity, TTFT/TBT implications, and exactly what ola-rt needs to implement. Then Section 3 becomes much easier, because batching and scheduling will no longer feel like arbitrary systems tricks—they will fall naturally out of the underlying computation.
