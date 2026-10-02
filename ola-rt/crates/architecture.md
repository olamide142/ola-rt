## 1. The high-level architecture

```text
                         ola-rt
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Model / Request Interface                                      │
│  ─────────────────────────                                      │
│  HF models     Burn models     ONNX*      Native ola models     │
│       │             │            │              │               │
│       └─────────────┴────────────┴──────────────┘               │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Model Frontend                         │  │
│  │  model loading │ config │ tokenizer │ weights │ metadata │  │
│  └─────────────────────────────┬─────────────────────────────┘  │
│                                ▼                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Graph / Runtime IR                       │  │
│  │                                                           │  │
│  │ Tensor → Linear → Attention → RMSNorm → MLP → ...        │  │
│  │                                                           │  │
│  │ shapes │ dtypes │ dependencies │ memory │ device hints   │  │
│  └─────────────────────────────┬─────────────────────────────┘  │
│                                │                                │
│                    ┌───────────▼───────────┐                    │
│                    │   Execution Planner   │                    │
│                    │                       │                    │
│                    │ What should execute?  │                    │
│                    │ Where?                │                    │
│                    │ When?                 │                    │
│                    │ How?                  │                    │
│                    └───────────┬───────────┘                    │
│                                │                                │
│          ┌─────────────────────┼──────────────────────┐         │
│          ▼                     ▼                      ▼         │
│   Request Scheduler      Memory Manager         Kernel Planner  │
│          │                     │                      │         │
│   batching               KV cache                kernel choice  │
│   priorities             paging                  fusion         │
│   prefill/decode         eviction                quantization   │
│   admission              allocation              device        │
│          │                     │                      │         │
│          └─────────────────────┼──────────────────────┘         │
│                                ▼                                │
│                    ┌─────────────────────┐                      │
│                    │ Execution Engine    │                      │
│                    └──────────┬──────────┘                      │
│                               │                                 │
│           ┌───────────────────┼────────────────────┐            │
│           ▼                   ▼                    ▼            │
│      Burn/CubeCL            CUDA                 CPU           │
│      WGPU/ROCm              kernels             SIMD           │
│           │                   │                    │            │
│           └───────────────────┼────────────────────┘            │
│                               ▼                                 │
│                       Physical hardware                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The **Execution Planner / Policy Engine** is what makes ola-rt particularly interesting.

---

# 2. Model frontend

The frontend answers:

> What computation does this model represent?

Initially, don't write everything yourself. Let Burn do a lot of this.

For example:

```rust
pub struct Llama {
    embedding: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Linear,
}
```

ola-rt could eventually translate this into its own representation:

```text
Embedding
   ↓
TransformerBlock[0]
   ├─ RMSNorm
   ├─ QKV Projection
   ├─ RoPE
   ├─ Attention
   ├─ Output Projection
   ├─ RMSNorm
   └─ SwiGLU
   ↓
TransformerBlock[1]
   ↓
...
   ↓
LM Head
```

The frontend should eventually be independent from Burn.

You might support:

```rust
trait ModelLoader {
    fn load(&self, source: ModelSource) -> Result<Model>;
}
```

with implementations such as:

```text
BurnLoader
SafetensorsLoader
HuggingFaceLoader
OnnxLoader
OlaLoader
```

---

# 3. ola-IR

This is one of the most important architectural components.

Don't let ola-rt simply execute a collection of Burn function calls.

Represent the inference computation.

For example:

```rust
pub enum Op {
    MatMul,
    Linear,
    RMSNorm,
    RoPE,
    Attention,
    SwiGLU,
    Softmax,
    Embedding,
}
```

and:

```rust
pub struct Node {
    id: NodeId,
    op: Op,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    metadata: OpMetadata,
}
```

The IR can carry information such as:

```text
dtype
shape
layout
device
memory requirements
dependencies
estimated FLOPs
estimated memory traffic
kernel candidates
quantization format
```

So instead of:

```rust
attention.forward(x)
```

ola-rt understands something more like:

```text
%q = linear(%x, %Wq)
%k = linear(%x, %Wk)
%v = linear(%x, %Wv)

%q = rope(%q)
%k = rope(%k)

%o = attention(
    q=%q,
    k=%k,
    v=%v,
    kv_cache=%cache
)
```

Now the runtime has something it can **reason about and optimize**.

---

# 4. Execution Policy

This is where I would put the intellectual centre of your PhD.

Most inference engines contain many decisions.

For a request arriving at time \(t\), define runtime state:

$$
S_t =
(W_t, H_t, M_t, C_t, Q_t)
$$

where perhaps:

* \(W_t\): workload state
* \(H_t\): hardware state
* \(M_t\): memory/KV-cache state
* \(C_t\): compute state
* \(Q_t\): request/SLA state

The runtime chooses an action:

$$
a_t = \pi(S_t)
$$

An action could mean:

```text
batch these requests
use this attention implementation
allocate these KV blocks
run prefill on GPU 0
run decode on GPU 1
quantize this model
evict this sequence
delay this request
speculatively decode 4 tokens
use tensor parallelism
move KV state
```

Architecturally:

```rust
pub trait ExecutionPolicy {
    fn decide(
        &mut self,
        state: &RuntimeState,
    ) -> ExecutionPlan;
}
```

That abstraction is extremely important.

It lets you implement:

```text
FCFSPolicy
StaticBatchPolicy
ContinuousBatchPolicy
MemoryAwarePolicy
LatencyAwarePolicy
ThroughputPolicy
CostAwarePolicy
RLPolicy
LearnedPolicy
```

without rewriting the inference engine.

---

# 5. Request scheduler

This handles incoming inference requests.

Imagine:

```text
t=0     Request A:  200 prompt tokens
t=2ms   Request B: 2000 prompt tokens
t=4ms   Request C:   50 prompt tokens
t=8ms   Request D:  800 prompt tokens
```

The scheduler maintains something like:

```text
             Request Queue

 A    B    C    D    E
 │    │    │    │    │
 └────┴────┴────┴────┘
           │
           ▼
      Scheduler
           │
    ┌──────┴───────┐
    ▼              ▼
 Prefill         Decode
 Queue            Queue
```

Eventually this should support:

```text
static batching
continuous batching
chunked prefill
priority scheduling
deadline-aware scheduling
SLA-aware scheduling
prefill/decode separation
speculative decoding
multi-tenant fairness
```

And critically:

**the scheduler doesn't decide everything itself.**

It consults the execution policy.

---

# 6. KV-cache manager

This should be a first-class subsystem rather than an implementation detail.

```text
GPU memory

┌────────────────────────────────────┐
│ Model weights                      │
├────────────────────────────────────┤
│ Runtime workspace                  │
├────────────────────────────────────┤
│ KV BLOCK POOL                      │
│                                    │
│ [01][02][03][04][05][06][07][08] │
│ [09][10][11][12][13][14][15][16] │
│                                    │
└────────────────────────────────────┘
```

Sequences don't necessarily own contiguous cache regions.

Instead:

```text
Sequence A
logical: 0  1  2  3
          │  │  │  │
          ▼  ▼  ▼  ▼
physical: 7  2  14 5
```

You could start with:

```rust
pub trait KvCache {
    fn allocate(&mut self, seq: SequenceId, tokens: usize);
    fn append(&mut self, seq: SequenceId, kv: KvTensor);
    fn free(&mut self, seq: SequenceId);
}
```

and evolve toward:

```text
ContiguousKvCache
PagedKvCache
QuantizedKvCache
TieredKvCache
DistributedKvCache
PrefixCachedKvCache
```

This gives you a natural place to reproduce PagedAttention and experiment beyond it.

---

# 7. Attention subsystem

Don't hardcode "attention".

Give ola-rt multiple implementations.

```rust
trait AttentionBackend {
    fn forward(
        &self,
        q: Tensor,
        cache: &KvCache,
        context: &ExecutionContext,
    ) -> Tensor;
}
```

Then:

```text
Attention
   │
   ├── NaiveAttention
   │
   ├── FlashAttention
   │
   ├── PagedAttention
   │
   ├── FlashDecoding
   │
   └── future experimental implementations
```

The interesting question eventually becomes:

> Which attention implementation should ola-rt use **right now**?

Rather than configuring it permanently.

---

# 8. Kernel layer

Initially:

```text
ola-rt
   ↓
Burn
   ↓
CubeCL
   ↓
CUDA / WGPU / ROCm
```

This gets you moving quickly.

You don't need to spend year one implementing GEMM.

Later:

```text
                       Kernel API
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       CubeCL          Custom CUDA       CPU SIMD
          │
      ┌───┴────┐
      ▼        ▼
    CUDA      ROCm
```

You might define:

```rust
pub trait Kernel {
    fn execute(
        &self,
        inputs: &[Tensor],
        outputs: &mut [Tensor],
        ctx: &DeviceContext,
    );
}
```

Then the planner can have multiple implementations for one operation:

```text
AttentionOp
   │
   ├── Burn reference implementation
   ├── CubeCL kernel
   ├── CUDA FlashAttention
   └── experimental kernel
```

That's excellent for research because you always retain a **correctness/reference implementation**.

---

# 9. Memory manager

Separate this from the KV-cache manager.

The general memory manager handles:

```text
weights
activations
temporary buffers
KV cache
CUDA workspaces
pinned CPU memory
device transfers
```

Something like:

```rust
pub struct MemoryManager {
    device_allocator: DeviceAllocator,
    host_allocator: HostAllocator,
    kv_manager: KvCacheManager,
    workspace_pool: WorkspacePool,
}
```

Eventually it can reason about:

$$
\text{MemoryBudget}
=
M_{\text{weights}}
+
M_{\text{KV}}
+
M_{\text{activations}}
+
M_{\text{workspace}}
$$

This becomes important for admission control.

If a new request requires 4 GB of future KV capacity and you only have 2 GB available, the runtime needs to make a decision.

---

# 10. Execution engine

The execution engine should be relatively boring.

That's desirable.

The **planner decides**.

The **engine executes**.

```text
Runtime state
      │
      ▼
Policy
      │
      ▼
ExecutionPlan
      │
      ▼
ExecutionEngine
      │
      ├── allocate
      ├── launch kernel
      ├── synchronize
      ├── transfer
      ├── update KV
      └── collect metrics
```

Conceptually:

```rust
pub struct ExecutionPlan {
    steps: Vec<ExecutionStep>,
}

pub enum ExecutionStep {
    Allocate(...),
    Transfer(...),
    Launch(...),
    Synchronize(...),
    Free(...),
}
```

That's useful because the **plan itself becomes observable**.

You can compare what different policies decided.

---

# 11. Telemetry is a core component

Don't bolt Prometheus onto it at the end.

Instrumentation should exist from the beginning.

Every request should produce something like:

```text
request_id = 7821

queue_time       3.2 ms
prefill_time    17.1 ms
TTFT            20.7 ms
decode_time    512.4 ms
ITL              8.3 ms

prompt_tokens    512
output_tokens     64

KV allocated     128 MB
peak VRAM       14.7 GB

batch sizes
8 → 12 → 14 → 11 → 9 → ...

GPU utilization
78 → 91 → 97 → 88 → ...

policy decisions
...
```

Then your research loop becomes:

```text
             workload
                 │
                 ▼
              ola-rt
                 │
         ┌───────┴────────┐
         ▼                ▼
      inference       telemetry
         │                │
         └───────┬────────┘
                 ▼
             dataset
                 │
                 ▼
          policy analysis
                 │
                 ▼
          better policy
                 │
                 └──────────► ola-rt
```

That's eventually how you get from **hand-designed policies → optimization → learned policies**.

---

# 12. Model serving layer

Then put a server around the runtime.

```text
                     Clients
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        Python       OpenAI API     Rust
           │            │            │
           └────────────┼────────────┘
                        ▼
                ┌──────────────┐
                │ ola-server   │
                └───────┬──────┘
                        │
                Request Manager
                        │
                     ola-rt
                        │
                       GPU
```

Ideally eventually:

```text
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
POST /v1/models
```

making ola-rt usable as an actual inference server rather than just a research simulator.

---

# 13. Multi-GPU comes later

Once single-device execution works:

```text
                    Global Planner
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          GPU 0        GPU 1       GPU 2
         Prefill       Decode      Decode
             │           │           │
             └───────────┼───────────┘
                         │
                   KV movement
```

Now you can investigate:

```text
tensor parallelism
pipeline parallelism
expert parallelism
prefill/decode disaggregation
KV-cache transfer
load balancing
replica selection
heterogeneous GPUs
```

This is where DistServe-style research starts fitting naturally.

---

# 14. I would structure the Rust workspace like this

```text
ola-rt/
│
├── crates/
│   ├── ola-core/
│   │   ├── tensor
│   │   ├── dtype
│   │   ├── shape
│   │   ├── device
│   │   └── error
│   │
│   ├── ola-model/
│   │   ├── loader
│   │   ├── safetensors
│   │   ├── tokenizer
│   │   └── architectures/
│   │
│   ├── ola-ir/
│   │   ├── graph
│   │   ├── op
│   │   ├── value
│   │   └── passes
│   │
│   ├── ola-kernel/
│   │   ├── attention
│   │   ├── matmul
│   │   ├── norm
│   │   ├── rope
│   │   └── quantization
│   │
│   ├── ola-memory/
│   │   ├── allocator
│   │   ├── kv_cache
│   │   ├── paging
│   │   └── workspace
│   │
│   ├── ola-scheduler/
│   │   ├── request
│   │   ├── batch
│   │   ├── admission
│   │   └── queue
│   │
│   ├── ola-policy/
│   │   ├── state
│   │   ├── action
│   │   ├── policy
│   │   └── policies/
│   │
│   ├── ola-executor/
│   │   ├── plan
│   │   ├── engine
│   │   └── context
│   │
│   ├── ola-backend-burn/
│   ├── ola-backend-cubecl/
│   ├── ola-backend-cuda/
│   ├── ola-backend-cpu/
│   │
│   ├── ola-telemetry/
│   │   ├── metrics
│   │   ├── trace
│   │   └── profiler
│   │
│   └── ola-server/
│
├── models/
├── benchmarks/
├── experiments/
├── examples/
└── papers/
```

But **do not implement all of that immediately**.

## The first version should be tiny

I'd make **ola-rt v0.1** approximately:

```text
Safetensors
     │
     ▼
Small Transformer
     │
     ▼
Burn
     │
     ▼
┌──────────────────────────┐
│ ola-rt                   │
│                          │
│ Request                  │
│    ↓                     │
│ Scheduler                │
│    ↓                     │
│ ExecutionPolicy          │
│    ↓                     │
│ KV Cache                 │
│    ↓                     │
│ Executor                 │
│    ↓                     │
│ Telemetry                │
└────────────┬─────────────┘
             ▼
           Burn
             ▼
          CubeCL
             ▼
            GPU
```

Get a small Llama/Qwen-style decoder generating tokens.

Then implement, in roughly this order:

```text
v0.1   basic autoregressive inference
  ↓
v0.2   KV cache
  ↓
v0.3   multiple concurrent requests
  ↓
v0.4   continuous batching
  ↓
v0.5   paged KV cache
  ↓
v0.6   PagedAttention
  ↓
v0.7   execution-plan IR
  ↓
v0.8   pluggable execution policies
  ↓
v0.9   profiling + workload replay
  ↓
v1.0   research-grade experimental runtime
```

And **then** your PhD gets especially interesting:

```text
                    ola-rt
                       │
                       ▼
               Runtime State Sₜ
                       │
                       ▼
                Policy π(Sₜ)
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          batching   memory    kernels
             │         │         │
             └─────────┼─────────┘
                       ▼
                 Execution
                       │
                       ▼
                  Telemetry
                       │
                       ▼
              objective function

        latency + throughput + cost
          + memory + SLA violations
                       │
                       ▼
               Optimizer / Learner
                       │
                       └────────────► π
```

