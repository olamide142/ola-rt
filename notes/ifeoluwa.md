Yes. There is a particularly strong overlap between **your inference-runtime / execution-policy / optimization direction** and your brother's **federated learning (FL)** work.

The interesting intersection is not simply "use federated learning with ola-rt." It is:

> **Treat federated learning as a distributed execution system whose execution policy can be explicitly represented, optimized, and adapted under heterogeneous/non-stationary resources.**

That fits very naturally with the direction you're developing around inference execution policies.

### Where your research overlaps

A conventional FL system looks roughly like:

$$
\text{Server}
\rightarrow
\text{select clients}
\rightarrow
\text{distribute model}
\rightarrow
\text{local training}
\rightarrow
\text{collect updates}
\rightarrow
\text{aggregate}
\rightarrow
\text{repeat}
$$

But underneath that simple loop are a lot of **systems decisions**:

$$
\pi_t =
(
\text{client selection},
\text{resource allocation},
\text{precision},
\text{local epochs},
\text{batch size},
\text{compression},
\text{aggregation timing},
\text{communication},
\text{deadline}
)
$$

Those decisions have objectives such as:

$$
\min_\pi
\quad
\alpha T(\pi)
+\beta E(\pi)
+\gamma C(\pi)
+\delta L(\pi)
$$

where \(T\) might be latency, \(E\) energy, \(C\) communication cost, and \(L\) model loss, subject to constraints on memory, bandwidth, fairness, privacy, etc.

That is extremely close conceptually to what you're already thinking about with execution policies.

## Paper topics I'd seriously consider

| Topic                                                                   | Your contribution                 | His contribution         |
| ----------------------------------------------------------------------- | --------------------------------- | ------------------------ |
| **1. Adaptive Execution Policies for Heterogeneous Federated Learning** | runtime, scheduling, optimization | FL algorithms/evaluation |
| **2. Formalizing Federated Learning Execution Policies**                | policy representation/IR          | FL semantics             |
| **3. Learning FL Execution Policies Under Non-Stationary Resources**    | online optimization/RL            | FL training              |
| **4. Multi-Objective Scheduling for Federated Learning**                | OR formulation                    | convergence/FL           |
| **5. Resource-Aware Client Selection**                                  | scheduler/runtime                 | client selection/FL      |
| **6. Straggler-Aware Federated Learning Runtime**                       | systems implementation            | FL experiments           |
| **7. Adaptive Precision and Quantization in FL**                        | kernels/runtime/quantization      | effect on convergence    |
| **8. Communication-Aware FL Scheduling**                                | network/runtime optimization      | aggregation/training     |
| **9. Federated Learning on Heterogeneous Edge Accelerators**            | CPU/GPU execution engine          | distributed FL           |
| **10. Deadline-Constrained Federated Learning**                         | scheduling/optimization           | FL algorithm             |

Of these, **#1–#4 fit your PhD direction best**.

### 1. Adaptive Execution Policies for Heterogeneous Federated Learning

This would probably be my first joint paper.

Imagine 100 FL clients:

```text
Client 1   RTX GPU     500 Mbps     32 GB
Client 2   CPU          20 Mbps      8 GB
Client 3   mobile        5 Mbps      4 GB
Client 4   GPU          80 Mbps     16 GB
...
Client 100 ...
```

Their resources also change over time.

Most importantly, instead of viewing these properties as annoying implementation details, your paper asks:

> **What execution policy should an FL system choose given heterogeneous and dynamically changing compute, memory, bandwidth, availability and data characteristics?**

You could define:

$$
s_t =
[
GPU_i,
CPU_i,
MEM_i,
BW_i,
RTT_i,
battery_i,
availability_i,
data_i,
...
]
$$

and an action:

$$
a_t =
[
clients,
epochs,
batch,
precision,
compression,
deadline,
aggregation
]
$$

Then:

$$
\pi(s_t) \rightarrow a_t
$$

Now you've converted FL orchestration into an **execution-policy optimization problem**.

You can compare static policies against heuristics, mathematical optimization, online optimization, RL, contextual bandits, etc.

That is a very clean collaboration because neither of you has to abandon your main research area.

---

### 2. Formalizing Federated Learning Execution Policies

This could actually connect directly to your **"Formalizing Inference Execution Policies"** work.

The research question becomes:

> Can federated learning execution strategies be represented using a common policy abstraction independent of the underlying FL algorithm/runtime?

For example:

```text
policy FLPolicy {
    select_clients:
        availability > 0.8
        && bandwidth > 10Mbps

    local_execution:
        epochs = adaptive
        batch_size = resource_aware
        precision = fp16

    communication:
        compression = top_k(0.1)

    aggregation:
        strategy = fedavg
        deadline = 30s

    objective:
        minimize(time_to_accuracy)
        minimize(network_bytes)
}
```

Obviously the actual formalism would be much more rigorous.

You could then express things such as FedAvg, FedProx-style execution, synchronous FL, asynchronous FL, deadline-based aggregation, resource-aware client selection, etc., through a common representation.

This gets interesting because the representation enables the next paper:

$$
\text{FL algorithm}
\neq
\text{FL execution policy}
$$

You can hold the learning algorithm constant while changing the **system execution strategy**.

That's a distinction worth exploring carefully.

---

### 3. Learning Federated Execution Policies Under Non-Stationary Workloads

This is probably the most directly aligned with your planned PhD work.

Suppose bandwidth suddenly changes:

```text
t0

clients: 100
average bandwidth: 100 Mbps
GPU availability: 70%

        ↓

t1

clients: 100
average bandwidth: 20 Mbps
GPU availability: 35%
```

A static FL configuration might become poor.

Instead:

$$
\pi_t = f(s_t)
$$

and the system continuously changes client selection, local epochs, compression, precision, aggregation deadlines, etc.

The research question becomes:

> **Can an adaptive execution-policy controller outperform static and heuristic FL execution policies under non-stationary resource conditions?**

You could deliberately create resource shifts:

```text
Bandwidth degradation
        ↓
Client dropout
        ↓
GPU contention
        ↓
CPU throttling
        ↓
Network recovery
        ↓
New clients joining
```

and measure how quickly the controller adapts.

This would let you experiment with:

* contextual bandits
* reinforcement learning
* Bayesian optimization
* online optimization
* model-predictive control

That makes it much more than "another FL algorithm."

---

### 4. Multi-Objective Optimization of Federated Learning Execution

This is where your **OR/optimization** interest becomes particularly useful.

Instead of optimizing only accuracy:

$$
\min_\pi
\begin{bmatrix}
T_{train}\\
Energy\\
NetworkBytes\\
Cost\\
1-Accuracy
\end{bmatrix}
$$

subject to:

$$
Memory_i \le M_i
$$

$$
Bandwidth_i \le B_i
$$

$$
Energy_i \le E_i
$$

$$
T_i \le deadline
$$

$$
Fairness(\pi) \ge F_{min}
$$

Then derive the **Pareto frontier**.

For example:

```text
communication
cost
 ^
 |  •
 |    •
 |       •
 |          •
 |              •
 +--------------------> training time
```

No single policy dominates everything.

One deployment might prefer:

```text
minimum latency
```

while another prefers:

```text
minimum bandwidth
```

or:

```text
minimum energy
```

This also gives you a natural route into **NSGA-II, MILP, constrained optimization, evolutionary optimization or online multi-objective optimization**.

## A particularly interesting systems paper

There's another one I would keep in reserve:

**"A Policy-Driven Runtime for Heterogeneous Federated Learning."**

Instead of primarily proposing a new FL algorithm, build a runtime:

```text
                FL Application
                     │
              ┌──────▼──────┐
              │ FL Algorithm │
              │ FedAvg/etc.  │
              └──────┬──────┘
                     │
              ┌──────▼───────┐
              │ Policy Engine │
              └──────┬───────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
 client select   scheduler    communication
       │             │             │
       └─────────────┼─────────────┘
                     │
               Runtime/Executor
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        CPU         GPU        Edge
```

The policy engine decides **how the FL workload executes**.

That could eventually become an experimental subsystem of ola-rt rather than a completely unrelated project.

## One deeper connection I think is worth pursuing

Your PhD could eventually establish a general abstraction:

$$
\boxed{
Execution\ Policy =
State \rightarrow Decisions
}
$$

Then different AI systems become instances of it.

For LLM inference:

$$
s_t
\rightarrow
\{
batching,
KV\ placement,
routing,
precision,
parallelism,
kernel
\}
$$

For VLM inference:

$$
s_t
\rightarrow
\{
vision\ resolution,
tokens,
batching,
KV,
routing,
device
\}
$$

For federated learning:

$$
s_t
\rightarrow
\{
clients,
epochs,
batch,
precision,
compression,
aggregation
\}
$$

For distributed training:

$$
s_t
\rightarrow
\{
placement,
parallelism,
checkpointing,
communication,
batch
\}
$$

That starts pointing toward something much larger than an FL side project:

> **A general theory and runtime representation of execution policies for AI systems.**

The FL collaboration could therefore provide an important test of **generality**. If your policy representation only works for LLM serving, it may be an inference-serving abstraction. If essentially the same underlying formalism can represent inference, distributed training, and FL execution decisions, you have evidence that you've found a broader AI-systems abstraction.

For a first joint paper, I'd narrow it to **adaptive execution policies for heterogeneous federated learning**: formalize the state/action/objective space, build a policy-driven prototype, compare static FedAvg-style execution against resource-aware heuristics and an adaptive optimizer, and benchmark under controlled heterogeneity and resource shifts. That gives your brother a genuine FL contribution while giving you a genuine systems/optimization contribution—and it can feed directly back into the execution-policy theory you're building for the PhD.

