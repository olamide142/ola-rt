# ola-runtime

`ola-runtime` owns the inference work lifecycle. It is the authoritative writer
of queue membership, active membership, request status, and timing state.

Its `ModelRegistry` contains models that are loaded and available for request
admission. This differs from `ola_model::ModelCatalog`, which describes models
that could be selected and loaded. Keeping availability separate from residency
will later allow explicit load, unload, placement, and memory policies.

A `Scheduler` receives an immutable view of `RuntimeState` and selects a queued
request ID. The runtime validates that decision before transitioning the
request. This is a deliberate research boundary: schedulers can be compared or
replaced without allowing experimental code to mutate runtime bookkeeping.

The current scheduler is FIFO. The trait is already sufficient for priority,
deadline, estimated-size, and memory-aware baselines. Batching and execution
policy will be introduced only after a model executor makes their decision
requirements measurable.

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
    .schedule_next(Timestamp::from_micros(20))?
    .expect("the request is queued");

assert_eq!(session.request_id, request);
runtime.mark_first_token(request, Timestamp::from_micros(30))?;
runtime.complete(request, Timestamp::from_micros(100))?;

# Ok::<(), ola_runtime::RuntimeError>(())
```

The complete runnable version is in `examples/lifecycle.rs`.
