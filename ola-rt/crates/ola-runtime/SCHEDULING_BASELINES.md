# Scheduling baselines design proposal

Status: **accepted** (2026-10-16) with all four recommended decisions.
Covers `todo.md` P3 and builds on `EXECUTION_POLICY.md`. All seven steps are
implemented.

## Goal

Add priority, deadline-aware, shortest-job-first, and memory-aware baselines
beside FIFO. All of them are compared on the same traces by one reproducible
protocol that reports latency, throughput, memory, deferrals, fairness, and
starvation.

Done when FIFO, priority, estimated-cost, and deadline-aware policies run
through that protocol and produce byte-identical results when re-run.

## What exists

- **`Scheduler::rank(&self, &PlanningSnapshot, limit)`** is the only scheduling
  hook. The snapshot already carries each queued request's prompt length,
  output limit, `Priority` (an `i32` where larger means more important),
  optional deadline, and arrival time, plus the iteration timestamp.
- **`ComposedPolicy` stops at the first admission deferral** (FIFO
  head-of-line blocking). Decodes are always offered to the batch policy
  before prompt work.
- **`BatchContext` carries only candidates and already-scheduled work.** A
  budget policy cannot see queue depth, so it cannot adapt.
- **The benchmarks use host wall-clock time with iteration counters as
  timestamps.** Requests never *arrive* over time, so queueing never happens
  the way it would under load, and results vary from run to run.

## Proposal

### 1. Schedulers

All schedulers break ties by `(queued_at, request_id)`, so rankings are total
and deterministic.

| Scheduler | Ranks by | Descriptor |
| --- | --- | --- |
| `FifoScheduler` | arrival | `fifo` (unchanged; correctness baseline) |
| `PriorityScheduler` | priority, highest first | `priority{aging_micros=…}` |
| `EarliestDeadlineFirst` | deadline, earliest first; no deadline last | `earliest-deadline-first` |
| `ShortestJobFirst` | estimated cost, smallest first | `shortest-job-first{cost=…,aging_micros=…}` |

- **Optional aging for priority and SJF.** Every `aging_micros` of waiting
  raises a request's effective priority by one, or lowers its effective SJF
  cost by one token. Without aging, a steady stream of urgent or
  short work starves the rest. The protocol measures that rather than hiding
  it, so aging is off by default and reported as a separate configuration.
- **SJF cost** is `prompt` or `prompt_plus_output_limit`. The output limit is
  an upper bound on a request's size, which is known before it runs; EOS is
  not.

### 2. Memory-aware baseline: admission backfill

`ComposedPolicy` gains `HeadOfLine::{Block, Backfill { max_skipped }}`. With
`Backfill`, a request deferred for memory is skipped (and recorded) and
admission tries the next ranked request, up to `max_skipped` times. Combined
with `MemoryAwareAdmission`, this is the memory-aware baseline: small
requests use pool space a large head request cannot. It is also the classic
starvation risk for large requests, which the protocol measures.
`DecisionRecord` gains the skipped requests.

### 3. Prefill/decode mixing and budgets

- **`BatchContext` gains `pressure: BatchPressure { queued_requests,
  live_requests, max_active_requests }`**, filled from the snapshot.
- **`MixedTokenBudgetPolicy { max_tokens, max_prefill_tokens }`** keeps decodes
  first and caps prompt work per iteration, the piggyback form of mixing
  where long prompts cannot crowd out decodes. `max_prefill_tokens = max_tokens`
  reproduces `FifoTokenBudgetPolicy`.
- **`QueueAdaptiveTokenBudget { min_tokens, max_tokens, tokens_per_queued }`**
  grows the budget with queue depth: low latency when idle, throughput under
  load. It is stateless, computed from `pressure`, so it stays
  deterministic. Fixed against adaptive budgets is one of the comparisons.

A prefill-first ordering would need `ComposedPolicy` to change its stage
order. That is deferred unless the comparison shows the decode-first cap is
insufficient.

### 4. Comparison protocol (in `ola-bench`)

**Virtual time.** Requests arrive at trace timestamps. A cost model advances
a virtual clock by

```text
iteration_micros = fixed + per_dense_token × (input + padding) + per_invocation × invocations
```

using the iteration's `execution_batches`. When idle, the clock jumps to the
next arrival. A deterministic `SyntheticModel` (tokens in, tokens out, all
batch hooks, a KV profile, no EOS unless the trace requests one) stands in
for weights. Every metric is then exactly reproducible, independent of host
noise, and comparable across machines. Default cost parameters are
documented constants; P4 calibration can supply measured ones. Host timing
on a real model stays a separate, labelled measurement.

**Traces.** Generated from a seed with an inline SplitMix64, so there is no
new dependency. Each request carries an arrival time, prompt length, output
limit, priority, optional deadline, and optional early-EOS step.

| Trace | Exercises |
| --- | --- |
| `mixed_short_long` | SJF against FIFO; head-of-line blocking by long prompts |
| `prefill_decode_competition` | long prompts arriving while many decodes are live; mixing and budgets |
| `priority_classes` | a minority of high-priority requests in a normal stream |
| `deadlines` | slack of 1–4× an estimated solo time (caller supplies the per-iteration cost), 10% infeasible |
| `overload` | arrival rate above capacity; queue growth and starvation |
| `impossible` | requests exceeding the KV pool mixed with normal ones |

**Run.** `run_scheduling_trial(trace, policy, config) -> SchedulingRun` keeps
per-request lifecycle timestamps, outcome, and output tokens. It also keeps
per-iteration plan and decision counters: deferrals by kind, backfill
skips, capability fallbacks, and peak reserved and used KV. Every policy runs
the identical trace, model, and cost model.

**Metrics** (`SchedulingSummary`, overall and per priority or size class):

- queue latency, TTFT, and completion latency at p50, p95, p99, and max;
- throughput in output tokens per virtual second, and makespan;
- peak reserved and used KV bytes;
- deferral counts by kind (admission, budget, slot), backfill skips, and
  rejections;
- **slowdown** = end-to-end time ÷ the request's *isolated* end-to-end time
  (the same request run alone under the same cost model);
- **fairness** = Jain's index over slowdowns, `(Σx)² / (n·Σx²)`; 1.0 is
  perfectly even;
- **starvation**: a request is starved if its queue wait exceeds
  `starvation_factor ×` its isolated end-to-end time (default 10). The
  summary also reports the worst wait and the requests still queued when the
  trace horizon ends;
- deadline miss rate, for requests that carry deadlines.

**Comparison.** `compare_schedulers(trace, &[policies])` returns one summary
per policy, keyed by its stable `PolicyDescriptor`, together with the trace's
seed and parameters. Running it twice must give equal results; a test checks
this.

### 5. Isolation and correctness

- **Output equivalence:** scheduling changes *when* a request runs, never
  *what* it produces. Each policy's per-request outputs must equal FIFO's,
  checked on the synthetic model in unit tests and on the pinned tiny-Llama
  fixture in one ignored test.
- **Isolation matrix:** for every scheduler, cancelling a request, failing
  one (bad input), and adding an impossible one must leave every other
  request's output unchanged and must never leave a slot or reservation
  leaked.

## Decisions (all recommendations accepted)

1. **Virtual-time simulation as the primary measurement** *(recommended)*, or
   host wall-clock on a real model. Virtual time is exactly reproducible and
   measures scheduling rather than kernels. Real-model host timing remains
   available as a labelled secondary run.
2. **Aging off by default and compared explicitly** *(recommended)*, or on by
   default. Off keeps the baselines textbook and makes starvation visible.
3. **Admission backfill as the memory-aware baseline** *(recommended)*, or a
   ranking that orders by "fits now". Backfill composes with any scheduler
   and keeps memory logic in admission, where it already lives.
4. **Expose the comparison as one UI experiment** *(recommended)*, or keep it
   to tests and the library API for now.

## Implementation steps

Each step keeps `make verify` and `make test-fixture` green.

1. **Schedulers.** Add `PriorityScheduler`, `EarliestDeadlineFirst`, and
   `ShortestJobFirst` with optional aging; test ranking, tie-breaks, and aging
   on snapshots. **Done.**
2. **Backfill.** Add `HeadOfLine::Backfill` to `ComposedPolicy`, with skips in
   `DecisionRecord`; add conflict tests beside the P2 ones. **Done.**
   The scheduler is asked for `slots + max_skipped + 1` requests, because a
   skip consumes no slot. The `ComposedPolicy` descriptor now always states
   `head_of_line` (`block` or `backfill-N`). `DecisionRecord::backfill_skips`
   was added to `PlanReport` schema 2 in place, since that version had not
   been published.
3. **Budgets.** Add `BatchPressure`, `MixedTokenBudgetPolicy`, and
   `QueueAdaptiveTokenBudget`, with tests. **Done.**
   `MixedTokenBudgetPolicy` lets the first prompt of an iteration exceed the
   prompt cap when it fits the total budget, so an unchunked prompt longer
   than the cap is not starved by live decodes; with the cap equal to the
   budget it matches `FifoTokenBudgetPolicy` exactly (tested). Selection
   records in `PlanReport` carry queue and live counts.
4. **Simulation harness.** Add `SyntheticModel`, the virtual-clock cost
   model, seeded trace generators, and `run_scheduling_trial`. **Done**
   in `ola-bench` (`synthetic.rs`, `trace.rs`, `simulation.rs`). Default
   cost: 200 µs per iteration, 300 µs per invocation, 4 µs per dense token;
   default KV pool: 1024 tokens in 16-token pages. Under light load ranking
   rarely matters because queues stay short, so ordering comparisons use the
   `overload` trace.
5. **Metrics and comparison.** Add `SchedulingSummary` (latencies,
   throughput, memory, deferrals, slowdown, Jain fairness, starvation,
   deadline misses), `compare_schedulers`, and the reproducibility test. **Done** in
   `ola-bench/src/scheduling_summary.rs`. A probe sweep before writing
   tests changed two things and pinned the regimes the tests assert:
   - the `deadlines` trace originally set slack from the arrival gap, which
     made every deadline infeasible (120/120 missed under every policy); it
     now scales slack to an estimated solo time;
   - with the default 1024-token pool memory never binds, so backfill never
     triggers; the backfill comparison uses a 512-token pool.

   Measured effects (seed 3, 120 requests): priority cuts urgent p95
   completion more than 4× with no urgent starvation; SJF speeds short
   requests and slows long ones, lowering fairness; backfill lowers p50
   completion and starvation under a binding pool; EDF misses fewer
   deadlines only under moderate contention (2 slots, 6 ms mean gap) and
   misses more than FIFO under overload, the known EDF domino effect.
   Empty classes are omitted from summaries.
6. **Isolation and equivalence.** Add the isolation matrix, synthetic output
   equivalence, and the fixture equivalence test. **Done.**
   `run_perturbed_trial` applies `Perturbation::{Cancel, SubmitInvalid}`
   mid-run, and runs now report end-of-trial live sessions and KV. Under
   FIFO, priority, EDF, SJF, and backfill, cancelling a running and a
   queued request, injecting an invalid request, and mixing in impossible
   requests leave every other request's output unchanged and leak nothing.
   The fixture test `scheduling_policies_preserve_real_model_outputs` checks
   the same equivalence on tiny Llama, and asserts the policies produced at
   least three distinct admission orders.
7. **Surface.** Add the UI experiment and README sections, and update
   `progress_report.md`. **Done.**
   The fixed-versus-adaptive budget comparison landed here. A load and cost
   sweep found that budgets never lower latency under the default
   overhead-dominated cost model (larger iterations always win, and a
   32-token budget halves heavy-load throughput); the adaptive budget
   recovers more than 1.5× that throughput. With a token-dominated cost
   model (40 µs per token, 100 µs overhead), capping prompt work at 32
   tokens per iteration lowers p50 completion on the competition and heavy
   traces. Both regimes are pinned by tests.

## Out of scope for P3

- Persisted, versioned scheduling reports and cross-revision comparison: P4
  generalizes report persistence.
- Calibrating the cost model from measured kernels: P4.
- Preemption and swapping as responses to pressure: P7.
- Learned or predictive policies: later work.
