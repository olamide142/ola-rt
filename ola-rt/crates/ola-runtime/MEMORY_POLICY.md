# Memory and cache policy design

Status: **accepted** (2026-10-17) with all recommended decisions.
Covers `todo.md` P7. Step 1 is implemented.

## Goal

Under a replayed cache-pressure trace, the runtime reclaims, evicts, or
preempts with recorded reasons instead of deferring indefinitely or
rejecting; prefix reuse produces identical outputs with measured savings;
and the pressure, preemption, cancellation, and recovery tests are
deterministic. That is P7's completion criterion.

## What existed

- **Reservations are worst-case and fixed.** `MemoryAwareAdmission`
  reserves prompt plus `max_new_tokens` (page-rounded) for a request's whole
  life, so the physical pool can never run out under it. The unused part of
  a reservation is exactly what the remaining tokens need, so it cannot
  shrink while the request runs; reclaiming memory requires reserving less
  up front and preempting when that bet fails.
- **Llama's pool is paged.** `KvBlockManager` hands out fixed blocks per
  session and frees them when the session ends. Nothing outlives a request,
  so there is nothing to evict until a prefix cache exists.
- **Running out of KV fails the request.** Without memory-aware admission,
  an allocation failure ended the request with an untyped error.
- **The synthetic model had no pool.** It computed byte counts, so
  simulations could not exercise allocation failure or sharing.
- **Resumable prefill exists.** Llama and the synthetic model process a
  prompt from any cursor, which a cached prefix can reuse.

## Decisions

1. **Reclaim through optimistic admission and preemption.**
   `WatermarkAdmission` reserves the prompt plus headroom. The planner
   predicts each iteration's block demand and preempts before a shortfall;
   an allocation that still fails is recovered by preempting that request
   rather than failing it. Conservative worst-case admission stays the
   default until a study shows otherwise.
2. **Preempt by recompute.** A victim's session and KV are released and the
   request is requeued with its generated tokens; on resume it prefills
   prompt plus generated tokens and continues. A `PreemptionPolicy`
   component chooses victims (none by default, newest-first, or lowest
   priority); plans carry a validated `Preempt` action. Swapping to host
   memory is evaluated once these semantics are stable.
3. **Clients do not see preemption.** Streams pause; no token repeats and
   TTFT is unchanged. Status and traces record each preemption and resume
   with its reason. Re-prefilling generated tokens must reproduce greedy
   continuations; floating-point differences could flip a near-tie, so this
   is tested on tiny Llama, not assumed.
4. **Block-granular prefix cache.** Only full blocks are shared, keyed by a
   chained hash of their token IDs and reference-counted, so shared blocks
   are never written. A new request attaches its longest cached prefix and
   computes the rest. Blocks whose last user ends move to an LRU list and are
   evicted, with a recorded reason, only when allocation needs them.
   Admission counts shared blocks once and evictable cached blocks as free.
   Off by default until equivalence tests pass.
5. **The synthetic model allocates from a real pool**, so pressure,
   preemption, prefix, and eviction tests run deterministically without a
   model or GPU.
6. **Observability.** Trace events `preempted`, `resumed`, and `prefix_hit`;
   pool metrics (cached blocks, hit rate, tokens saved, evictions,
   preemptions) in health and reports; all decisions keyed to iteration
   counters, never wall time.

## Steps

1. **Pool visibility.** **Done.** Models report `KvPoolStatus` (total, free,
   allocated, and cached blocks; block size; live sequences) through
   `CausalModel::kv_pool_status`, carried in `ModelSnapshot::kv_pool` so
   policies see physical occupancy at each iteration boundary. A pool that
   cannot grow returns `ModelExecutionError::kv_capacity_exhausted`, which
   the executor records as `PlanDivergence::KvCapacityExhausted`
   (`PlanReport` schema 4). Llama and the synthetic model both report their
   pools and type their failures; the synthetic model now allocates from a
   `KvBlockManager`. The scheduler study's samples are identical before and
   after, because its policies use memory-aware admission.
2. Recompute preemption: the `Preempt` action, the Running→Queued transition
   that keeps generated tokens, resume, `PreemptionPolicy`, events and
   traces, and output-equivalence tests on the synthetic model and tiny
   Llama.
3. Optimistic admission: `WatermarkAdmission`, predicted-shortfall
   preemption, preempt-instead-of-fail recovery, and a virtual-time
   cache-pressure study against conservative admission.
4. Prefix cache: hashed shared blocks, prefix attachment, LRU eviction,
   admission accounting, metrics, identical-output tests, and a shared-prefix
   study of prefill tokens saved.
5. A swap evaluation, a page-size and fragmentation experiment,
   replay-determinism tests, docs, and closing P7.

## Out of scope

Swapping implementation (evaluated only), cross-model or cross-device cache
sharing, partial-block sharing with copy-on-write, and KV quantization.
