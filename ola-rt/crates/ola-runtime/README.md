# ola-runtime

`ola-runtime` owns the inference work lifecycle. It is the authoritative writer
of queue membership, active membership, request status, and timing state.

Its `ModelRegistry` contains models that are loaded and available for request
admission. This differs from `ola_model::ModelCatalog`, which describes models
that could be selected and loaded. Keeping availability separate from residency
will later allow explicit load, unload, placement, and memory policies.

`Runtime` makes no ordering decisions. A `Scheduler` is a component of an
execution policy: it ranks queued requests from an immutable
`PlanningSnapshot`, which carries each request's prompt length, output limit,
priority, deadline, and arrival time. The policy validates the ranking and the
executor validates the resulting plan before any request transitions. This is
a deliberate research boundary: schedulers can be compared or replaced
without allowing experimental code to mutate runtime bookkeeping.
`Runtime::start_queue_head` remains as a policy-free lifecycle convenience for
probes that run without an executor.

The current scheduler is FIFO. The snapshot already carries what priority,
deadline, estimated-size, and memory-aware baselines need. The executor now exposes
an iteration-level boundary over several live sessions; richer admission and
batch-composition policy can build on that boundary without owning lifecycle
state.

Lifecycle timestamps are supplied by callers. That allows the same runtime to
operate with real monotonic time, simulation time, or recorded trace time.

## KV-cache block allocation

`KvBlockManager` is the first backend-neutral cache-memory mechanism. It divides
a byte capacity into fixed-size physical blocks and maintains a logical block
table for every live cache sequence. Its operations are deliberately small:

- `allocate` creates an empty logical sequence;
- `append` reserves enough blocks for additional token positions atomically;
- `free` returns every sequence block to the pool;
- `block_table` exposes the logical-to-physical mapping to a storage backend;
- `metrics` reports occupancy, unusable capacity, and internal fragmentation.

Allocation chooses the lowest available block ID so identical traces produce
identical mappings. `bytes_per_token` represents keys and values across all
layers covered by the pool; it is supplied by the model/backend boundary rather
than inferred by the runtime.

The manager owns metadata only. It does not allocate Burn tensors, move device
memory, evict sequences, or yet update `RuntimeState::memory` automatically.
Keeping those concerns separate lets CPU, CUDA, and simulated storage use the
same allocation and scheduling experiments.

## Synchronous multi-request executor

`Executor` connects lifecycle state to executable `ola-model` artifacts.
`step_iteration` is its only way forward and the continuous-batching
control-plane boundary: it advances every request that was active at
iteration entry once, then fills free slots in the order the policy's
scheduler ranks them. Each
token event identifies whether prefill or decode produced it.

Every iteration is first described by an `ExecutionPlan` (see
`EXECUTION_PLAN.md`), validated against a read-only snapshot, and only then
committed; `ExecutorIteration::execution_plan` exposes it. Because the plan is
built before any model runs, a slot is reused in the same iteration only when
its request must finish by reaching `max_new_tokens`. Slots freed by EOS or
failure are refilled on the next iteration, and the waiting queue head is
recorded as a `PlannedDeferral::Slots`. `ExecutorIteration::divergences` lists every way
execution differed from the plan: declared tensor hooks that fell back to
individual calls, admitted requests whose session creation failed, and planned
rows that ended their request without being predicted (EOS, failure, or a
one-token admission).

The iteration path groups selected prefill and decode operations by loaded
model. A model declares `CausalModel::batch_capabilities` and implements
`prefill_batch`, `prefill_chunk_batch`, and `decode_batch` to pack compatible
groups into tensor invocations; models without declared hooks, or hooks that
return `None`, retain the independent-session correctness path. Prefill batches may contain
different prompt lengths. Lifecycle events are restored to deterministic
request order even when backend invocation order groups models differently.

`ExecutorIteration::execution_batches` records decode invocations and
successful prefill invocations as `Tensor` or `Individual`, including request
IDs in tensor-row order, meaningful input positions, and artificial padding
positions. This prevents policy composition from being mistaken for actual
tensor batching and makes padding overhead measurable in benchmarks. Session
state, output tokens, KV accounting, failure, completion, and cancellation
remain request-local on both paths.

The baseline executor:

- attaches executable models to registered runtime metadata;
- accepts one text or token input per request;
- owns a configurable, positive number of active `GenerationSession` values;
- advances live requests in stable round-robin order;
- tensor-batches compatible prefill and decode through optional model hooks;
- falls back to isolated execution for models without batch support;
- admits queued work through the configured scheduler when slots are free;
- records first-token and completion lifecycle transitions;
- aggregates all live session KV bytes into `RuntimeState::memory`;
- drops session/cache state on completion, failure, and cancellation;
- cancels one live request without disturbing the remaining sessions;
- retains final token IDs, decoded text, and session metrics;
- rejects regressive iteration timestamps before advancing any live session.

## Admission policy

Admission is separate from queue selection. The scheduler first identifies a
queued candidate; an `AdmissionPolicy` then receives its tokenized prompt
length, generation limit, model KV profile, device accounting, and outstanding
reservations. A deferred candidate remains queued without a lifecycle
transition.

`MemoryAwareAdmission` is the strict baseline. It reserves worst-case KV demand
for `prompt_tokens + max_new_tokens - 1`, rounded to complete physical pages. It
enforces both the model's shared KV-pool capacity and an optional device-wide
budget in `MemoryState::capacity_bytes`. A zero device capacity means no
device-wide budget has been configured; the real model-pool bound is still
enforced. Temporary pressure produces a typed `Deferred` event. A request that
cannot fit even without competition produces `Rejected` and moves directly
from queued to failed, avoiding permanent FIFO head-of-line blocking.

## Batch composition and token budgets

Batch composition is separate from both queue selection and memory admission.
At each `step_iteration` boundary, the executor presents eligible operations as
`BatchCandidate` values. A candidate identifies its request, prefill/decode
phase, and input-token cost. Decode costs one token position; prefill costs the
tokenized prompt length.

The configured `BatchPolicy` chooses work without mutating lifecycle state.
The executor validates that every returned request is an eligible, unique
candidate before doing model work. Selected work becomes rows of
`ExecutorIteration::execution_plan`, and policy-deferred work becomes
`PlannedDeferral::Budget` entries, both in deterministic event order, while
`BatchDeferred` events explain individual deferrals to observers.

`UnboundedBatchPolicy` is the compatibility default and preserves the earlier
behavior of advancing every eligible operation. `FifoTokenBudgetPolicy` is the
bounded baseline: it selects the FIFO prefix that fits a positive per-iteration
input-token budget. If the head prefill is larger than the complete budget, it
runs alone so a valid request cannot starve forever. The plan records its real
cost even though that cost exceeds the nominal limit.

Prefill tensor grouping has a separate `PrefillBatcher` boundary. The
compatibility `UnbucketedPrefillBatcher` puts all same-model prompts together.
`PowerOfTwoPrefillBatcher` groups lengths by their next power of two, bounding
padding within a group while retaining batches for nearby prompt lengths.
`CostAwarePrefillBatcher` examines the complete same-model prefill set and
chooses between those alternatives. Its explicit invocation-overhead parameter
expresses fixed launch cost in token-equivalent units; it selects the lower sum
of dense tensor positions and invocation overhead, preferring one invocation on
ties. `decide` exposes both estimates for experiments and audit logs.
Grouping may change backend invocation order, but lifecycle events are restored
to scheduler order.

```rust,no_run
use ola_runtime::{
    ComposedPolicy, CostAwarePrefillBatcher, Executor, FifoTokenBudgetPolicy, Runtime,
};

let budget = FifoTokenBudgetPolicy::new(256).expect("the budget is positive");
let policy = ComposedPolicy::default()
    .with_batch(budget)
    .with_grouping(CostAwarePrefillBatcher::new(32));
let executor = Executor::with_max_active_requests(Runtime::default(), 8)?
    .with_policy(policy)
    .with_prefill_chunk_size(64)?;

# Ok::<(), ola_runtime::ExecutorError>(())
```

Sub-policies are components of the executor's `ExecutionPolicy`; the default
`ComposedPolicy` combines FIFO scheduling, unconstrained admission, unbounded
batching, and unbucketed grouping, and each `with_*` builder replaces one
component (see `EXECUTION_POLICY.md`). `step_iteration` is the only way to
advance the executor, so every state change goes through a validated plan.
It may tensor-batch selected prefill and decode work when the attached model
supports it; selection and execution mechanism remain separately observable.

`with_prefill_chunk_size` adds interruption points to architectures that
advertise resumable prefill. A long prompt retains its session, token cursor,
worst-case admission reservation, and KV pages between iterations. Each
segment is a normal `BatchCandidate`, so decode and continued prefill share the
same token budget. `PrefillProgress` exposes accepted versus total prompt
tokens without pretending that a first output token exists. Cancellation drops
the partial session and its pages. Architectures without support execute the
complete prompt unchanged.

Compatible resumable chunks share the optional model tensor hook even when
their request cursors and remaining lengths differ. The execution record keeps
real and padded chunk positions visible. A single selected request, or a model
without the hook, retains the `Individual` fallback.

The executor is synchronous by design. Full request preemption, result
eviction, and streaming transports remain later layers.

## Minimal lifecycle

```rust
use ola_core::{ModelDescriptor, RequestSpec, Timestamp};
use ola_runtime::Runtime;

let mut runtime = Runtime::default();
runtime.register_model(ModelDescriptor::language("tiny", "tiny-decoder"))?;

let request = runtime.submit(
    RequestSpec::text("tiny", "Explain continuous batching."),
    Timestamp::from_micros(10),
)?;
let session = runtime
    .start_queue_head(Timestamp::from_micros(20))?
    .expect("the request is queued");

assert_eq!(session.request_id, request);
runtime.mark_first_token(request, Timestamp::from_micros(30))?;
runtime.complete(request, Timestamp::from_micros(100))?;

# Ok::<(), ola_runtime::RuntimeError>(())
```

The complete runnable version is in `examples/lifecycle.rs`.
