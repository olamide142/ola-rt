# Memory and cache policy design

Status: **accepted** (2026-10-17) with all recommended decisions.
Covers `todo.md` P7. Steps 1–3 are implemented.

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
2. **Recompute preemption.** **Done.** `LifecycleAction::Preempt` (validated:
   only live requests, nothing else for them that iteration, slot and
   reservation released before admissions); `Runtime::preempt` moves a
   request to the front of the queue, keeping its first start and counting
   `preemptions`. The executor keeps a preempted request's prompt, output,
   and token times; its re-admission prefills prompt and output together,
   and the resulting token continues the output (no repeat, no new first
   token). `PreemptionPolicy` (`NoPreemption` default, `NewestFirst`,
   `LowestPriorityFirst`) orders victims; `ComposedPolicy::with_preemption`
   computes each pool's block demand from live work (a decode needs a block
   when its position starts one; a continued prefill needs its chunk's
   blocks), preempts until the shortfall is covered, plans the rest without
   the victims, and records a `PreemptionRecord`. `ExecutorEvent::Preempted`,
   trace event `preempted`, observer counts, `PlanReport` schema 5, and
   `preemptions` in request status. Pulled forward from step 3: with
   preemption enabled, an admission whose first prefill does not fit the
   blocks left after live growth is deferred (`AdmissionReason::KvPoolFull`);
   without it a resumed request immediately failed its own re-prefill.

   The synthetic model now continues a resumed request exactly, as a real
   model does: output tokens are at least 1000 and prompt tokens below it,
   so a resumed prompt reveals how much output it carries. Evidence: on the
   synthetic pool, three requests that cannot fit together finish with at
   least one preemption, no failures, and output identical to running
   alone; without preemption the same workload fails with
   `KvCapacityExhausted`; cancelling a preempted request releases it. On a
   random-weight Llama and on the tiny-Llama fixture, preempted requests
   produce exactly their solo greedy output, so re-prefilling generated
   tokens did not flip any near-tie in these runs.
3. **Optimistic admission and recovery.** **Done.** `WatermarkAdmission
   { headroom_tokens }` reserves prompt plus headroom (capped at the worst
   case) and still rejects requests whose worst case exceeds the pool.
   `ExecutionPolicy::recovers_kv_exhaustion` (on exactly when
   `ComposedPolicy` preempts) makes a failed allocation requeue the request
   as `PreemptionReason::AllocationFailed`, unless the request could never
   fit the whole pool. A first rule, "fail when alone in the pool", was
   wrong: when one batched decode failed for every row, requeuing the first
   rows dropped their sessions, so the last row looked alone and failed
   although the freed blocks would have let it run. The rule now depends
   only on the request. Simulation summaries count preemptions.

   **Study** (`kv-pressure`, five seeds × 120 requests, 2,048-token pool,
   24 slots, headroom 32). On `early_stopping`, watermark admission with
   preemption cut completion p95 from 473 ms to 250 ms, p50 from 231 to
   107 ms, and TTFT p95 from 419 to 183 ms, and raised throughput from 12,568
   to 19,821 tokens/s (+58%), with 32.6 preemptions per run and no failures;
   every interval separates. Worst-case reservation left a third of the
   pool unused (peak 85 KB of 131 KB against 127 KB). On `overload` and
   `mixed_short_long` the default pool never binds and both policies are
   identical; with a 256-token pool, where requests use most of their limit,
   watermark admission triggered no preemptions and stayed within 1% of
   worst-case latency and throughput with overlapping intervals. On these
   workloads it never lost; the default stays conservative until real-model
   studies agree.
4. Prefix cache: hashed shared blocks, prefix attachment, LRU eviction,
   admission accounting, metrics, identical-output tests, and a shared-prefix
   study of prefill tokens saved.
5. A swap evaluation, a page-size and fragmentation experiment,
   replay-determinism tests, docs, and closing P7.

## Out of scope

Swapping implementation (evaluated only), cross-model or cross-device cache
sharing, partial-block sharing with copy-on-write, and KV quantization.
