# Memory and cache policy design

Status: **accepted** (2026-10-17) with all recommended decisions.
Covers `todo.md` P7. All five steps are implemented; P7 is complete.

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
4. **Prefix cache.** **Done.** `KvBlockManager::with_prefix_cache`: blocks
   are reference-counted; sessions report written content through
   `record_tokens`, and each completed block is indexed by a chained FNV
   hash of its tokens and its parent, verified token by token on lookup so
   a collision cannot share the wrong KV. Freed blocks with cached content
   go to an LRU list (`cached_blocks`) and are evicted, oldest first and
   counted, only when an allocation needs them. `attach_prefix` gives an
   empty sequence the longest cached run of full blocks; a shared block is
   never written again because attachment always ends on a full block.
   Models expose `cached_prefix` and sessions `attach_prefix` (Llama via
   `LlamaKvCacheConfig::with_prefix_cache`, the synthetic model via
   `synthetic_artifact_with_prefix_cache`). The planning snapshot carries
   each queued request's cache hit (leaving at least one prompt token to
   compute, and only with resumable prefill); the planner starts the first
   prefill after it and counts the blocks it computes plus the evictable
   blocks attaching takes. The executor attaches exactly the planned prefix
   or fails the request. Worst-case reservations still count the whole
   prompt per request, so shared blocks are counted once only in the
   physical check; this is conservative, not exact. A preempted request's
   own blocks become cached when it is released, so its resume can reuse
   them. Traces show `cached_prefix_tokens` on admission.

   Correctness: on the synthetic model, a random-weight Llama, and the
   tiny-Llama fixture with its real tokenizer, outputs with the cache match
   recomputation exactly. Writing the fixture test exposed a latent Llama
   bug unrelated to caching: a prefill chunk spanning more than one block
   read past its populated prefix and panicked; any chunk size above the
   block size would have hit it. A regression test covers it.

   **Study** (`prefix-cache`, five seeds × 120 requests, 2 ms mean arrival
   gap, 8 slots). With a 256-token shared prefix (about 93% of each prompt)
   the cache computed 85% fewer input tokens (35.4k → 5.1k), cut TTFT p95
   from 190 to 4 ms and completion p95 from 208 to 21 ms, and raised
   throughput 76%: without it the server saturated. With a 32-token prefix
   it computed 45% fewer input tokens and cut completion p95 from 23.2 to
   20.6 ms (separated), while TTFT and throughput were not distinguishable
   because load did not saturate.
5. **Swap evaluation, page size, replay.** **Done.**

   *Swapping: evaluated, not implemented.* Recompute's cost was measured:
   in `kv-pressure` (payload 2, which adds input tokens), watermark
   admission computed 15.7% more input tokens than worst-case reservation
   (15,320 vs 13,236), about 64 re-prefilled tokens per preemption, and
   still delivered 58% more throughput. Swapping would replace that
   recompute with copying the request's KV out and back. Per token,
   recompute costs about 2 × parameters FLOPs; a swap round trip moves
   2 × (KV bytes per token). For a Llama-2-7B-class model in FP16
   (512 KiB of KV per token, 14 GFLOP per token), 64 tokens is 64 MiB over
   PCIe 4.0 (~25 GB/s, about 2.7 ms) against 0.9 TFLOP of prefill (about
   6 ms at 150 TFLOP/s): swapping could be roughly twice as fast per
   preemption on such hardware, and more so with grouped-query attention's
   smaller KV. But recompute batches with other prefill work, while swaps
   need a host-memory tier, asynchronous copies, and resume ordering, and
   none of this can be measured here (no usable GPU). Decision: keep
   recompute; revisit swapping as a policy choice per preemption, using
   this cost model, once a GPU can measure both. Recorded under later
   work.

   *Page size* (`kv-page-size`): a shared-prefix trace (100-token prefix)
   at 4, 8, 16, 32, and 64 tokens per page. Internal fragmentation without
   the cache grew 1.1%, 2.5%, 5.4%, 10.4%, 19.9%. Prefix reuse with the
   cache was 11,886 tokens at 4, 11,411 at 8–32 (the prefix rounds down to
   96), and 7,616 at 64, where completion p95 rose from 20.8 to 25.2 ms
   (separated intervals). On this workload 8–16-token pages balance waste
   and reuse; 64 loses both. Placement-aware admission does not apply:
   pages are fixed size, so any free page serves any request and there is
   no external fragmentation, only the internal kind measured here.

   *Replay*: identical trials produce identical iterations, events, and
   outcomes under worst-case-exceeding pressure with preemption, with the
   prefix cache (including evictions and reuse during preemption), and
   with mid-run cancellations. Writing the test confirmed that
   reservations, which count a shared prefix per request, keep a shared
   workload from over-committing the pool at all.

## Outcome

The P7 completion criterion is met. Under replayed cache-pressure traces
the runtime preempts (and recovers failed allocations) with recorded
reasons instead of deferring indefinitely or failing; prefix reuse produces
identical outputs with measured savings; and the pressure, preemption,
cancellation, and recovery tests are deterministic. Defaults stay
conservative (worst-case reservation, no preemption, no prefix cache)
until real-model studies agree with the virtual-time results.

## Out of scope

Swapping implementation (evaluated only), cross-model or cross-device cache
sharing, partial-block sharing with copy-on-write, and KV quantization.
