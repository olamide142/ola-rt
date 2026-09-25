Yes. This is the part I would spend serious time formalizing before touching RL or trying to build an “adaptive scheduler.” The mathematical object itself needs to be defensible.

The first important refinement is that I would **not** ultimately describe the problem as a simple MDP.

A more realistic formulation is closer to a:

$$
\boxed{\text{Constrained Partially Observable Semi-Markov Decision Process}}
$$

because:

* you don't know future arrivals,
* you don't know exact output lengths,
* tool-call return times may be unknown,
* hardware measurements can be delayed/noisy,
* decisions have different durations,
* and there are hard constraints such as memory capacity and SLOs.

Let's build toward that carefully.

---

# 1. What exactly is an inference execution policy?

Start from:

$$
\pi:S\rightarrow A
$$

This says:

> Given the current state of the inference system, choose what the runtime should do.

That's deceptively simple.

Imagine vLLM has:

```text
74 waiting requests
28 decoding requests
3 large prefills
81% KV utilization
92% HBM bandwidth utilization
51% SM utilization
P99 TTFT close to SLA
```

The execution policy might say:

```text
Don't admit another large prefill yet.

Decode these 28 sequences.

Give 2,048 tokens of the remaining token budget
to these two prefills.

Retain these reusable KV blocks.

Evict those KV blocks.

Use speculative decoding for this request class.

Route another request to GPU 2.
```

So:

$$
\pi(s_t)=a_t
$$

is really:

$$
\boxed{
\text{system observation}
\rightarrow
\text{execution plan}
}
$$

The key contribution isn't necessarily that this mapping exists implicitly already.

Of course it does.

Every runtime has some mapping from runtime conditions to decisions.

Your contribution could be:

> **make that mapping an explicit first-class formal object.**

Then you can:

$$
\text{describe it}
\rightarrow
\text{compare it}
\rightarrow
\text{verify it}
\rightarrow
\text{optimize it}
\rightarrow
\text{learn it}
$$

That's much stronger.

---

# 2. The system dynamics

Instead of only:

$$
a_t=\pi(s_t)
$$

write:

$$
s_{t+1}
\sim
P(
s_{t+1}
\mid
s_t,a_t,\xi_t
)
$$

where \(\xi_t\) represents things the runtime does **not control**.

For example:

$$
\xi_t =
(
\text{new arrivals},
\text{output lengths},
\text{tool completions},
\text{failures},
\text{network conditions}
)
$$

This matters enormously.

Suppose you choose:

```text
batch size = 64
```

You don't control whether:

```text
1000 requests arrive 3 ms later.
```

Likewise, you don't know whether a request will generate:

```text
10 tokens
```

or:

```text
10,000 tokens.
```

So this isn't simply deterministic optimization.

There is uncertainty:

$$
s_t,a_t,\xi_t
\rightarrow
s_{t+1}
$$

And that's where OR begins to become really interesting.

---

# 3. The state

We previously wrote:

$$
s_t =
(
W_t,Q_t,M_t,G_t,L_t,H_t
)
$$

But these shouldn't be thought of as six numbers.

Each one is itself a potentially large structured state.

Think:

$$
S =
S_W
\times
S_Q
\times
S_M
\times
S_G
\times
S_L
\times
S_H
$$

Let's examine every one.

---

# \(W_t\): Workload state

This is much more than:

```text
requests per second
```

The workload determines **what type of work is entering the runtime**.

You might define:

$$
W_t =
(
\lambda_t,
P_t^{in},
P_t^{out},
C_t,
X_t,
R_t,
Z_t
)
$$

where approximately:

| Variable      | Meaning                    |
| ------------- | -------------------------- |
| \(\lambda_t\) | arrival intensity          |
| \(P^{in}\)    | input-length distribution  |
| \(P^{out}\)   | output-length distribution |
| \(C_t\)       | request-class mix          |
| \(X_t\)       | prefix locality            |
| \(R_t\)       | model/modality mix         |
| \(Z_t\)       | burstiness/correlation     |

## Arrival rate

Suppose:

$$
\lambda_t=40\text{ requests/s}
$$

Now five seconds later:

$$
\lambda_{t+5}=700\text{ requests/s}
$$

Your scheduling strategy probably shouldn't remain identical.

But average arrival rate isn't enough.

Consider two workloads.

### Workload A

```text
10 requests every second
10 requests
10 requests
10 requests
```

### Workload B

```text
0
0
0
40 requests at once
```

Both have:

$$
\bar\lambda=10/s
$$

but B is far more bursty.

So you also care about interarrival variance.

For example:

$$
CV =
\frac{\sigma_{\text{interarrival}}}
{\mu_{\text{interarrival}}}
$$

The policy should perhaps react differently when:

$$
CV \gg 1
$$

because queue spikes become likely.

---

# Prompt-length distribution

Imagine:

### workload A

$$
P(L_{prompt})
\approx 500 \text{ tokens}
$$

versus:

### workload B

$$
P(L_{prompt})
\approx 50,000 \text{ tokens}
$$

Those are fundamentally different workloads.

Long prompts cause expensive prefills.

So instead of simply recording:

$$
\text{mean prompt length}
$$

you might maintain:

$$
W_t^{prompt} =
(
P_{50},
P_{90},
P_{99},
\mu,
\sigma
)
$$

or a histogram.

---

# Output-length distribution

This one is harder.

When a request arrives, you know:

$$
L_{prompt}
$$

but generally don't know:

$$
L_{output}.
$$

Yet output length directly determines:

$$
\text{decode compute}
$$

and:

$$
\text{future KV occupancy}.
$$

Therefore:

$$
\hat L_{out}
=
f(x_i,\text{model},\text{request class})
$$

becomes a prediction problem.

But perhaps your runtime should carry uncertainty:

$$
L_{out}
\sim
P(L\mid x_i)
$$

instead of one estimate.

This becomes important for robust optimization.

Very recent work is already treating unknown output length and distribution shift explicitly in KV management, including distributionally robust optimization formulations. ([arXiv][1])

---

# Workload class

Requests might be:

```text
chat
coding
RAG
document summarization
reasoning
agent
vision
batch
```

They behave differently.

For example:

```text
RAG

long-ish prefill
short-ish output
```

while:

```text
reasoning

short/medium prefill
potentially huge decode
```

and:

```text
agent

repeated turns
shared context
tool pauses
KV reuse opportunity
```

That last category is increasingly important. Recent work such as Continuum specifically addresses the fact that tool-using agents create pauses where retaining KV state may be beneficial rather than treating finished inference calls independently. ([DOI][2])

So:

$$
W_t
$$

should probably represent not just token lengths but **program/workflow characteristics**.

---

# Prefix locality

This one is particularly interesting.

Suppose 1,000 requests contain:

```text
SYSTEM_PROMPT
+
SAME_100K_DOCUMENT
+
different question
```

versus 1,000 completely unrelated prompts.

Same:

$$
\lambda
$$

same mean input length.

But radically different execution opportunities.

The first workload may have:

$$
\text{prefix reuse} \approx 95\%
$$

while the second has:

$$
\approx 0\%.
$$

vLLM currently implements hash-based automatic prefix caching to reuse already-computed KV blocks for matching prefixes. ([vLLM][3])

So something such as:

$$
\rho_t^{prefix}
=
\frac{\text{reusable input tokens}}
{\text{total input tokens}}
$$

could be a meaningful workload feature.

---

# This produces our first deep research question

What information from the workload is actually necessary?

Because putting everything into \(W_t\) creates an enormous state space.

You need a sufficient representation:

$$
\phi(W_t)
$$

such that:

$$
\pi(\phi(W_t))
$$

makes good decisions without needing the complete workload history.

That's already potentially a research problem:

> **What is the minimal sufficient workload representation for inference execution control?**

---

# \(Q_t\): Queue state

Now we move from incoming workload characteristics to **current outstanding work**.

A queue isn't just:

$$
Q_t=83
$$

requests.

Suppose two systems both have 83 queued requests.

System A:

```text
83 × 100-token prompts
```

System B:

```text
83 × 100,000-token prompts
```

Obviously:

$$
Q_A \neq Q_B
$$

from the scheduler's perspective.

---

For each request \(i\), define something like:

$$
q_i =
(
a_i,
p_i,
d_i,
c_i,
r_i,
k_i,
u_i,
z_i
)
$$

where:

$$
a_i = \text{arrival time}
$$

$$
p_i = \text{prompt tokens remaining}
$$

$$
d_i = \text{decode tokens generated}
$$

$$
c_i = \text{request class}
$$

$$
r_i = \text{priority/SLO}
$$

$$
k_i = \text{KV occupancy}
$$

$$
u_i = \text{prefix reuse}
$$

$$
z_i = \text{execution status}.
$$

Statuses might include:

```text
WAITING
PREFILL
DECODING
PREEMPTED
TOOL_WAIT
KV_OFFLOADED
FINISHED
```

Then:

$$
Q_t=\{q_1,q_2,\ldots,q_n\}.
$$

Notice something mathematically awkward:

$$
|Q_t|
$$

changes constantly.

So your policy input has variable size.

That's important if you eventually use learning.

You can't simply do:

```text
neural_network([q1, q2, ... qN])
```

with fixed \(N\).

Possible representations include:

$$
\text{histograms}
$$

or:

$$
\text{DeepSets}
$$

or:

$$
\text{attention over requests}
$$

or aggregate queue statistics.

That's another potential design question.

---

# Request urgency

An especially useful variable is **slack**.

Suppose a request has:

$$
SLA_i=2.0s.
$$

It arrived:

$$
1.1s
$$

ago.

And estimated remaining execution is:

$$
0.6s.
$$

Then:

$$
slack_i
=
2.0-1.1-0.6
=
0.3s.
$$

That means you can only delay it another:

$$
300ms
$$

before you're likely to miss the SLO.

Compare another request with:

$$
slack=4.5s.
$$

Clearly the first one has greater urgency.

This idea has become an active research direction. Cascade, for example, continuously estimates a request's remaining latency budget using request properties, KV state and current system load, then uses that information for both scheduling and KV decisions. ([arXiv][4])

This reinforces an important point for your thesis:

> State variables shouldn't merely describe resources. They should describe **decision-relevant consequences**.

---

# \(M_t\): Memory and KV-cache state

This could become one of the richest parts of the thesis.

A naive version would be:

$$
M_t =
\text{free GPU memory}.
$$

That's nowhere near enough.

For an inference system:

$$
M_t =
(
M_{weights},
M_{KV},
M_{workspace},
M_{prefix},
M_{offload},
M_{free}
)
$$

And even \(M_{KV}\) is complex.

---

Suppose you have:

```text
32 GB HBM
```

Maybe:

```text
16 GB weights
2 GB runtime workspace
14 GB KV
```

Then the runtime has a finite number of KV blocks.

With paged allocation:

```text
Block 0
Block 1
Block 2
...
Block N
```

Current vLLM's KV manager explicitly tracks block allocation, computed/cached blocks, block reference counts, free blocks and eviction operations. ([vLLM][3])

Your memory state could therefore contain:

$$
M_t =
(
b_{free},
b_{used},
b_{cached},
b_{shared},
b_{offloaded},
h_{prefix},
e_t
)
$$

where:

$$
b_{free} = \text{free KV blocks}
$$

$$
h_{prefix} = \text{prefix-cache hit statistics}
$$

$$
e_t = \text{expected eviction/recompute cost}.
$$

---

# Why occupancy alone isn't enough

Imagine two caches each at:

$$
90\%
$$

utilization.

Cache A contains mostly:

```text
unique dead prefixes
```

Cache B contains:

```text
widely reused system prompts
```

Evicting 10% from A might barely matter.

Evicting 10% from B might trigger massive future recomputation.

Therefore the state needs something related to:

$$
\boxed{\text{value of cached state}}
$$

not only its size.

Perhaps a KV block \(b\) has a value:

$$
V(b)
=
P(\text{reuse}_b)
\times
C_{\text{recompute},b}
-
C_{\text{retention},b}.
$$

Then eviction becomes:

$$
b^*
=
\arg\min_b V(b)
$$

subject to freeing enough space.

Now KV management looks like caching theory.

---

# Multi-tier memory

Eventually:

```text
HBM
 ↓
CPU RAM
 ↓
NVMe
 ↓
remote KV store
```

could all participate.

Then block \(b\) has residency:

$$
r_b \in
\{
HBM,
DRAM,
NVMe,
remote
\}.
$$

Now a memory decision might be:

$$
HBM\rightarrow DRAM
$$

rather than simply:

$$
retain / evict.
$$

And accessing it has different costs:

$$
C_{HBM}
<
C_{DRAM}
<
C_{NVMe}.
$$

At that point you're very close to classical memory hierarchy research.

---

# \(G_t\): GPU execution state

This should not simply be:

$$
GPUUtil=90\%.
$$

Why?

Because 90% GPU utilization doesn't tell you what bottleneck you're hitting.

A workload might be:

$$
\text{compute-bound}
$$

or:

$$
\text{memory-bandwidth-bound}.
$$

vLLM's own chunked-prefill documentation explicitly describes this interaction: prefill tends to be compute-intensive while decoding tends to be memory-bandwidth intensive, and mixing them can improve utilization. ([vLLM][5])

So define:

$$
G_t =
(
U_{SM},
U_{HBM},
M_{HBM},
P_{GPU},
T_{GPU},
B_t,
K_t
)
$$

where:

$$
U_{SM}
=
\text{compute utilization}
$$

$$
U_{HBM}
=
\text{memory bandwidth utilization}
$$

$$
P_{GPU}
=
\text{power}
$$

$$
T_{GPU}
=
\text{temperature}
$$

$$
B_t
=
\text{current batch shape}
$$

$$
K_t
=
\text{kernel/execution characteristics}.
$$

---

This gives an interesting scenario.

Suppose:

$$
SM=45\%
$$

but:

$$
HBM=97\%.
$$

Adding more decode work might accomplish little.

But adding suitable compute-heavy prefill work could potentially make better use of otherwise idle compute.

Another state:

$$
SM=98\%,\quad HBM=58\%.
$$

Now adding heavy prefills could be terrible.

So your controller should ideally understand the **resource vector**, not one utilization metric.

---

# \(L_t\): Latency and SLO state

This is subtly different from queue state.

\(Q_t\) tells you:

> what work exists.

\(L_t\) tells you:

> how close the system is to failing its service objectives.

You might have:

$$
L_t =
(
TTFT_{50},
TTFT_{95},
TTFT_{99},
TPOT_{50},
TPOT_{99},
V_t,
G_t^{goodput}
)
$$

where:

$$
V_t
=
\text{SLO violation rate}.
$$

But aggregate metrics aren't sufficient either.

You probably also need per-request slack:

$$
\ell_i =
D_i -
(t-a_i)
-
\hat C_i
$$

where:

$$
D_i = \text{deadline/SLO}
$$

$$
a_i = \text{arrival time}
$$

$$
\hat C_i = \text{estimated remaining execution time}.
$$

Then:

```text
large positive slack
→ safe

near-zero slack
→ urgent

negative slack
→ likely already doomed
```

This is much more decision-useful than simply saying:

```text
P99 = 1.9 sec.
```

---

# \(H_t\): Hardware state

This one needs a distinction.

Some hardware properties change very slowly:

$$
H^{static}
$$

such as:

```text
GPU model
VRAM capacity
HBM bandwidth
tensor-core capability
interconnect topology
PCIe generation
NVLink topology
CPU architecture
```

While others change dynamically and arguably belong in \(G_t\).

For example:

```text
GPU utilization
temperature
memory pressure
```

I'd therefore probably eventually write:

$$
H =
(
H^{compute},
H^{memory},
H^{network},
H^{topology}
)
$$

as a **platform descriptor**.

Why include it in the policy?

Because you eventually want:

$$
\pi(s,h)
$$

rather than:

$$
\pi_{H100}(s)
$$

and separately:

$$
\pi_{5090}(s).
$$

In other words:

> Can one policy adapt its execution decisions to different hardware?

Suppose:

```text
GPU A
huge compute
limited memory

GPU B
less compute
huge memory

GPU C
high bandwidth
```

Optimal scheduling may differ.

That gives you a very interesting generalization problem.

---

# Now combine the state

Your actual state might eventually become:

$$
s_t =
(
W_t,
\{q_i\}_{i\in I_t},
M_t,
G_t,
L_t,
H
)
$$

with:

$$
I_t
=
\text{active requests at time }t.
$$

This is substantially richer than:

$$
[\text{queue length},\text{GPU util},\text{memory util}].
$$

---

# Now we get to the action space

We originally wrote:

$$
a_t =
(
B_t,C_t,K_t,P_t,R_t,D_t
)
$$

This is arguably even more interesting than the state.

---

# \(B_t\): Batching and admission

This isn't just:

```text
batch_size = 32
```

LLM continuous batching doesn't really behave like classical fixed batching.

A better action might include:

$$
B_t =
(
I_t^{scheduled},
T_t^{budget},
A_t^{admit},
O_t
)
$$

where:

$$
I_t^{scheduled}
=
\text{which sequences execute}
$$

$$
T_t^{budget}
=
\text{number of tokens allocated}
$$

$$
A_t^{admit}
=
\text{which waiting sequences become active}
$$

$$
O_t
=
\text{execution order}.
$$

Current vLLM exposes parameters such as maximum batched tokens and active sequences, and supports FCFS or priority scheduling; its scheduler also makes admission decisions around available KV capacity. ([vLLM][6])

Your policy could instead dynamically decide:

$$
T_t^{budget}
=
f(s_t).
$$

For example:

```text
light load:
4096 token budget

moderate:
8192

high throughput batch:
16384
```

not necessarily those values, but conceptually.

---

# \(C_t\): Prefill/decode policy

This deserves its own control dimension.

Suppose you have:

```text
20 decodes
1 × 100k-token prefill
```

Options include:

```text
run all 100k prefill
```

or:

```text
chunk into 2048-token pieces
```

or:

```text
4096
8192
16384
```

Your action might therefore be:

$$
C_t =
(
c_t,
r_t^{PD}
)
$$

where:

$$
c_t=\text{prefill chunk size}
$$

and:

$$
r_t^{PD}
=
\text{prefill/decode resource allocation}.
$$

Current vLLM already performs dynamic chunking based on the remaining batched-token budget and prioritizes decode requests under chunked prefill. ([vLLM][5])

So your research question becomes more sophisticated:

> Can a controller dynamically determine the **optimal prefill/decode composition**, rather than rely on one predefined heuristic?

---

# \(K_t\): KV-cache policy

Potential decisions:

$$
K_t=
(
retain,
evict,
offload,
prefetch,
compress,
share
)
$$

for potentially each KV object.

This becomes huge.

Suppose an agent calls a tool and will potentially return in:

$$
T_{tool}\approx500ms.
$$

Do you:

```text
retain its KV in HBM
```

or:

```text
evict it
```

or:

```text
move it to CPU
```

?

If you retain:

$$
\text{less memory available}
$$

but avoid:

$$
\text{reload/recompute}.
$$

Recent agent-serving research is explicitly exploring this exact tradeoff through KV TTL policies. ([DOI][2])

So:

$$
K_t
$$

could itself be a major dissertation chapter.

---

# \(P_t\): Precision and representation

Originally I called this quantization.

I'd broaden it to:

$$
P_t =
\text{numerical representation decisions}.
$$

Potentially:

```text
BF16
FP16
FP8
INT8
INT4
```

for:

```text
weights
KV cache
activations
```

But there's an important caveat.

Unlike:

```text
which request do I schedule?
```

changing model precision at millisecond granularity might be expensive or impossible depending on implementation.

This introduces something crucial:

# different decisions happen at different time scales.

---

# \(R_t\): Routing and placement

If you have:

```text
GPU0: 90% loaded
GPU1: 20% loaded
```

where should request \(i\) go?

But load isn't the only consideration.

GPU0 might already contain:

```text
request's prefix KV
```

while GPU1 doesn't.

So:

$$
routing
\neq
\text{least-loaded GPU}.
$$

You may need:

$$
R(i,j)
=
\alpha Load_j
+
\beta Queue_j
-
\gamma PrefixLocality_{ij}
+
\delta TransferCost_{ij}
$$

and choose:

$$
j^*
=
\arg\min_j R(i,j).
$$

Now scheduling and caching interact.

That's exactly the kind of **cross-layer coupling** that interests me most in this thesis.

---

# \(D_t\): Distributed execution

This could include:

$$
D_t =
(
TP,
PP,
DP,
EP,
CP
)
$$

for:

```text
tensor parallelism
pipeline parallelism
data parallelism
expert parallelism
context parallelism
```

The difficult part is that these aren't all fast decisions.

You generally can't cheaply say every 10 milliseconds:

```text
TP=2
```

then:

```text
TP=8
```

then:

```text
TP=4.
```

Reconfiguration can be expensive.

Therefore:

$$
D_t
$$

belongs to a slower control loop.

And that leads to what I think will become one of the most important structural pieces of your thesis.

---

# 4. A single policy probably isn't enough

Different decisions occur at radically different frequencies.

For example:

```text
Every inference iteration (~milliseconds)
-----------------------------------------
request scheduling
token budget
prefill/decode mix


Every seconds
-------------
KV retention/offloading
routing
load balancing


Every minutes/hours
-------------------
precision selection
model replicas
parallelism configuration
model placement
```

Therefore instead of:

$$
\pi(s)
$$

I would seriously consider a **hierarchical execution policy**:

$$
\Pi =
\{
\pi_{fast},
\pi_{medium},
\pi_{slow}
\}.
$$

For example:

$$
\pi_{fast}
:
S
\rightarrow
A_{schedule}
$$

$$
\pi_{medium}
:
S
\rightarrow
A_{memory,routing}
$$

$$
\pi_{slow}
:
S
\rightarrow
A_{placement,precision,parallelism}.
$$

Architecture:

```text
             Workload + system
                    │
                    ▼
        ┌─────────────────────┐
        │ Slow controller     │
        │ placement / TP / Q  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Medium controller   │
        │ KV / routing        │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Fast controller     │
        │ batching/scheduling │
        └──────────┬──────────┘
                   │
                   ▼
                  GPU
```

That is much more realistic than one giant RL agent outputting 50 decisions every millisecond.

---

# 5. The reward/objective

We originally wrote:

$$
\max_\pi
E\left[
\sum_tR(s_t,\pi(s_t))
\right].
$$

But I'd already refine that.

Why?

Because inference serving is inherently multi-objective.

You care about:

$$
Throughput
$$

$$
TTFT
$$

$$
TPOT
$$

$$
P99
$$

$$
Energy
$$

$$
Cost
$$

$$
Quality
$$

$$
Fairness.
$$

A naive reward would be:

$$
R_t
=
\alpha T_t
-\beta L_t
-\gamma C_t.
$$

But then somebody asks:

> Why \(\alpha=1.2\)?

And suddenly your scientific result depends heavily on arbitrary weights.

---

# Better: constrained optimization

I think a stronger formulation is:

$$
\max_\pi
\quad
Goodput(\pi)
$$

subject to:

$$
P99_{TTFT}\leq SLO_{TTFT}
$$

$$
P99_{TPOT}\leq SLO_{TPOT}
$$

$$
M_t\leq M_{GPU}
$$

$$
Quality(\pi)\geq Q_{min}.
$$

Then:

$$
Goodput
=
\text{requests completed while satisfying SLO}.
$$

This has a much clearer operational interpretation.

Alternatively:

$$
\min_\pi
Cost(\pi)
$$

subject to:

$$
P(SLO\ violation)\leq\epsilon.
$$

Now you have a **chance-constrained** problem.

That's very OR-friendly.

---

# 6. Hard constraints versus soft objectives

This distinction matters.

GPU memory should probably be:

$$
M_t\leq Capacity
$$

as a **hard constraint**.

You can't say:

> We violated VRAM capacity slightly but received -10 reward.

The program crashes.

Likewise certain model-quality requirements might be hard:

$$
Q \geq Q_{min}.
$$

Then throughput can be optimized within the feasible space.

So define:

$$
A_{feasible}(s)
=
\{
a:
g_j(s,a)\leq0
\}.
$$

Then:

$$
\pi(s)
\in
A_{feasible}(s).
$$

This is important if you later introduce RL.

Don't let the RL agent “learn” that OOM is bad.

Make OOM actions impossible.

This is called **action masking** or constrained action selection conceptually.

---

# 7. Why this isn't really a standard MDP

Now we can return to that point.

An MDP assumes:

$$
P(s_{t+1}|s_t,a_t)
$$

contains all relevant history.

But do we actually know everything?

No.

You don't know:

$$
\text{future requests}
$$

$$
\text{true output length}
$$

$$
\text{future prefix reuse}
$$

$$
\text{tool duration}
$$

$$
\text{future hardware contention}.
$$

So there is latent state:

$$
x_t
$$

and observation:

$$
o_t
=
O(x_t).
$$

Your policy really sees:

$$
o_t
$$

not the entire world.

Therefore:

$$
\pi(o_t)
$$

or perhaps:

$$
\pi(o_{0:t})
$$

is more accurate.

That's the **partially observable** part.

---

# Why Semi-Markov?

Because actions take different amounts of time.

A decode iteration might take:

$$
20ms.
$$

A large prefill chunk:

$$
150ms.
$$

Moving KV from CPU:

$$
40ms.
$$

Changing a distributed configuration:

$$
seconds.
$$

So transition intervals are not constant:

$$
\Delta t_t
\neq
constant.
$$

That's naturally represented by an SMDP.

So eventually you could write:

$$
\boxed{
\mathcal{M}
=
(S,A,P,R,C,\Omega,O,\tau)
}
$$

where:

$$
S=\text{latent runtime state}
$$

$$
A=\text{execution actions}
$$

$$
P=\text{state transitions}
$$

$$
R=\text{performance objective}
$$

$$
C=\text{constraints}
$$

$$
\Omega=\text{observations}
$$

$$
O=\text{observation model}
$$

$$
\tau=\text{action duration}.
$$

Now this is starting to look like a serious formal research contribution.

---

# 8. Non-stationarity

Then add one more difficulty.

Suppose:

$$
P(W_{t+1}|W_t)
$$

changes over time.

9 AM workload:

```text
interactive chatbot
```

2 PM:

```text
document analysis
```

6 PM:

```text
agents
```

Midnight:

```text
batch processing
```

So even the environment distribution isn't fixed:

$$
P_t \neq P_{t+k}.
$$

That means you aren't simply finding:

$$
\pi^*
$$

once.

You potentially need:

$$
\pi_t^*
$$

that tracks changing conditions.

This is where:

```text
online optimization
contextual bandits
adaptive control
MPC
reinforcement learning
distributionally robust optimization
```

all become candidates.

Importantly, current literature already investigates individual adaptive pieces: AugServe dynamically adjusts scheduling and token batching; Apt-Serve combines scheduling with hybrid caching; Cascade coordinates scheduling and KV hierarchy; other recent work formalizes online scheduling with KV constraints. ([arXiv][7])

So your novelty should **not** be:

> Nobody has adaptive inference scheduling.

That would already be false.

The stronger question is:

$$
\boxed{
\text{Can inference execution itself be represented as a unified,
cross-layer policy optimization problem?}
}
$$

---

# 9. This changes the thesis considerably

I would now describe the central object like this:

$$
\boxed{
\Pi_t:
(O_t,H)
\rightarrow
A_t^{feasible}
}
$$

where the runtime observes:

$$
O_t=
(
W_t,Q_t,M_t,G_t,L_t
)
$$

under hardware:

$$
H
$$

and chooses a hierarchical action:

$$
A_t=
(
B_t,C_t,K_t,P_t,R_t,D_t
)
$$

while satisfying:

$$
g_j(s_t,a_t)\leq0
$$

and optimizing long-term system utility:

$$
\max_{\Pi}
E
\left[
\sum_t
\gamma^t
U(s_t,a_t)
\right].
$$

Under:

$$
\text{uncertain}
+
\text{non-stationary}
+
\text{partially observed}
$$

workloads.

That is substantially more rigorous than merely saying:

> I'll use RL to optimize vLLM parameters.

---

# 10. And I think the thesis should have a layered research progression

This is where I would take it academically:

```text
FORMALIZATION
      │
      ▼
What is an inference execution policy?
What belongs in S?
What belongs in A?
What are the constraints?
What are the timescales?
      │
      ▼
CHARACTERIZATION
      │
      ▼
Represent vLLM policies
Represent SGLang policies
Represent other systems
Compare their decisions
      │
      ▼
BASELINE OPTIMIZATION
      │
      ▼
MILP / queueing / heuristics /
online optimization / MPC
      │
      ▼
ADAPTIVE CONTROL
      │
      ▼
React to workload changes
      │
      ▼
LEARNED POLICY
      │
      ▼
Bandit / RL / policy learning
      │
      ▼
GENERALIZATION
      │
      ▼
different workload
different GPU
different model
different modality
```

And I would resist jumping to the bottom.

**Paper 1 should probably live mostly in the first two boxes.**

If the formalization is good enough, you should be able to take a real runtime and say:

$$
\text{vLLM scheduler}
\Rightarrow
\pi_{vLLM}
$$

and explicitly describe:

$$
S_{vLLM},
A_{vLLM},
constraints_{vLLM},
objective_{implicit}.
$$

Then compare that with another runtime.

That itself could expose something extremely useful:

> today's runtime “policy” isn't one policy at all — it is a collection of independently designed decision rules at different layers and timescales.

And *that* gives you a defensible reason to investigate a unified inference execution-policy abstraction.

The next thing I would drill into is specifically **\(W_t\), the workload state**, because everything else depends on it: we can define the exact variables, which are observable vs latent, how to represent agent workloads, how to model arrival processes and output-length uncertainty, and what a Rust `WorkloadState` in ola-rt would actually contain.

[1]: https://arxiv.org/abs/2607.16892?utm_source=chatgpt.com "Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty"
[2]: https://doi.org/10.48550/arxiv.2511.02230?utm_source=chatgpt.com "[2511.02230] Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live"
[3]: https://docs.vllm.ai/en/latest/design/prefix_caching/?utm_source=chatgpt.com "Automatic Prefix Caching - vLLM"
[4]: https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com "Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving"
[5]: https://docs.vllm.ai/en/latest/configuration/optimization/?utm_source=chatgpt.com "Optimization and Tuning - vLLM"
[6]: https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/?utm_source=chatgpt.com "scheduler - vLLM"
[7]: https://arxiv.org/abs/2512.04013?utm_source=chatgpt.com "AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving"

