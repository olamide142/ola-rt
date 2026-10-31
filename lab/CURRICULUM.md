# Research curriculum

Graded experiments for learning inference runtimes, scheduling, memory,
adaptive optimization, cluster inference, and Rust, by running and changing
this codebase. Work through each track's levels in order; tracks can be
interleaved. Every item gives:

- **Run**: a `make lab-new` command that creates a notebook entry.
- **Look at**: what to read in the results.
- **Read**: the code that produced them.
- **Learn**: the mathematics or systems idea behind the result.
- **Extend**: a Rust change that makes you understand the code by altering
  it. Most extensions end with a new variant in an existing experiment, so
  you can run it straight away.
- **Open questions**: seeds for research entries, mapped to
  `notes/thesis.md` (RQ1 formalization, RQ2 static policies under
  non-stationary load, RQ3 adaptive optimization, RQ4 learned policies).

Levels: **L1** run and explain · **L2** predict, then test · **L3** change the
code · **L4** open research.

Commands assume the repository root. `make experiments` lists every
experiment's parameters; anything shown here as `SET=` can be changed.

---

## Track A: the runtime loop

Everything else builds on one loop: snapshot → policy → validated plan →
execution → events → observation.

**A1 (L1) — One iteration, end to end.**
- Run: `make lab-new NAME=fifo-trace ID=scheduler-comparison SET="trace=mixed_short_long seeds=3 requests=20"`
- Look at: `traces/run-001.jsonl`. Follow one request from `arrived` to
  `completed`: `ranked`, `admission`, `selection`, `invocation`,
  `prefill_progress`, `token`.
- Read: `ola-rt/crates/ola-runtime/src/executor.rs` (`step_iteration`),
  `composed.rs` (`plan`), `plan.rs` (`ExecutionPlan::validate`).
- Learn: why `plan` takes `&self` and `observe` `&mut self`; why
  validation happens before any state changes.
- Extend: write a short script (Python or Rust) over the trace that
  reconstructs each request's time to first token and checks it against
  the report's `ttft_p95`.

**A2 (L2) — Predict throughput from the cost model.**
- Run: `make lab-new NAME=fifo-throughput ID=fifo-workload SET="requests=200"`
- Before running, estimate iterations and makespan from `CostModel`
  (`ola-bench/src/simulation.rs`: 200 µs per iteration, 300 µs per call,
  4 µs per token). Then compare.
- Learn: Little's law, L = λW, applied to live requests, arrival rate, and
  latency.

**A3 (L3) — Add a trace event.**
- Extend: add a field (for example, the batch size of each invocation) to a
  `TraceEvent` in `ola-observe/src/trace.rs`, populate it, and use it in
  A1's script. Run `cargo test -p ola-observe`.
- Rust: `#[serde(tag = "event", rename_all = "snake_case")]`, exhaustive
  `match`, and why adding a variant forces you to handle it everywhere.

---

## Track B: batching and prefill

**B1 (L1) — Padding versus invocations.**
- Run: `make lab-new NAME=bucketing ID=prefill-bucketing` and
  `make lab-new NAME=prefill-cost ID=adaptive-prefill-cost`
- Look at: padding tokens and invocation counts per grouping.
- Read: `ola-runtime/src/batch.rs` (`PrefillBatcher`,
  `CostAwarePrefillBatcher`).
- Learn: the trade-off as an optimization problem: minimize
  `dense_tokens · a + invocations · b`.

**B2 (L2) — Calibrate a real cost model** (needs a model).
- Run: `make lab-new NAME=calibrate-cpu ID=cost-calibration MODEL=/path/to/artifact`
- Look at: the fitted `fixed_micros`, `per_dense_token_micros`,
  `per_invocation_micros`, their intervals, and `residual_rms_micros`.
- Read: `ola-bench/src/calibration.rs`.
- Learn: ordinary least squares, residuals, bootstrap confidence intervals.
- Then: run B1 and C1 with this model's ratio in mind: does batching matter
  more or less on your hardware than the default model says?

**B3 (L3) — A new prefill grouper.**
- Extend: implement `PrefillBatcher` with a different rule (for example,
  one-dimensional k-means over prompt lengths, or a greedy merge that stops
  when padding exceeds the saved invocation cost) and add it to
  `adaptive-prefill-cost`'s comparison in
  `ola-bench/src/experiment/builtin.rs`.
- Rust: trait objects (`Box<dyn PrefillBatcher>`), and why `group` returns
  one bucket per input.

**B4 (L4) — Open:** with a calibrated cost model, when does chunked prefill
(`llama-chunked-prefill`) improve tail TTFT enough to pay for its extra
invocations? Which workloads flip the answer? (RQ2)

---

## Track C: scheduling

**C1 (L1) — Baselines across workloads.**
- Run one entry per trace:
  `make lab-new NAME=sched-overload ID=scheduler-comparison SET="trace=overload seeds=10"`,
  then `mixed_short_long`, `prefill_decode_competition`, `priority_classes`,
  `deadlines`.
- Look at: completion and TTFT p95, starvation, fairness (Jain's index)
  for `fifo`, `priority`, `edf`, `sjf`, `backfill`.
- Read: `ola-runtime/src/scheduler.rs`, `ola-bench/src/scheduling_summary.rs`
  (how starvation and fairness are defined).
- Learn: SRPT optimality for mean response time, why it starves long jobs,
  and why p95 behaves differently from the mean.

**C2 (L2) — Predict the SJF trade-off.** Before running
`trace=mixed_short_long seeds=10`, predict: how much SJF lowers p95
completion against FIFO, and how many long requests it starves. Then vary
`requests` and `mean_interarrival_micros` (one entry each) to find the load
where SJF stops helping.
- Learn: load ρ = λ·E[S]; queues explode as ρ → 1. Estimate E[S] from the
  cost model.

**C3 (L3) — A new scheduler.**
- Extend: implement `Scheduler` (for example, least attained service, or
  SJF with ageing so waiting raises priority) in `scheduler.rs`, and add it
  to `SchedulerComparison::policies()` in `builtin.rs`.
- Rust: implementing a trait, `PolicyDescriptor` for stable names, and
  writing a unit test like the ones at the bottom of `scheduler.rs`.

**C4 (L4) — Open:** a scheduler that bounds starvation while keeping most
of SJF's p95 gain; formalize the trade-off as a constrained optimization
(RQ1, RQ2).

---

## Track D: memory and the KV cache

**D1 (L2) — Predict fragmentation.**
- Before running, compute the expected internal fragmentation: for lengths
  spread evenly within pages, a sequence wastes about (page − 1)/2 tokens.
- Run: `make lab-new NAME=page-size ID=kv-page-size SET="seeds=10"`
- Look at: whether `kv_fragmentation` matches your prediction for each
  page size, and how prefix reuse changes with page size.
- Read: `ola-runtime/src/kv_cache.rs` (`KvBlockManager`, `lookup_prefix`).

**D2 (L1) — Reservation versus watermark admission.**
- Run: `make lab-new NAME=kv-pressure ID=kv-pressure SET="trace=early_stopping seeds=10"`
- Look at: completion p95 and throughput for worst-case reservation
  against watermark admission with preemption.
- Read: `ola-runtime/src/admission.rs`, `preemption.rs`,
  `ola-runtime/MEMORY_POLICY.md`.
- Learn: overcommit and its risk; recompute preemption is correct because
  greedy decoding is deterministic.

**D3 (L3) — A preemption order.**
- Extend: implement `PreemptionPolicy` (for example, evict the request
  holding the most blocks, or the one with the least generated work to
  lose) and compare it in `kv-pressure`.

**D4 (L4) — Open:** joint scheduling and KV decisions (thesis H2): does
choosing admission and preemption together beat optimizing each alone?

---

## Track E: adaptive optimization

Read `ola-rt/crates/ola-runtime/ADAPTIVE_OPTIMIZATION_PLAN.md` first,
especially "Results (2026-10-31)": it records two mistakes worth not
repeating.

**E1 (L1) — Offline selection.**
- Run: `make lab-new NAME=offline-budget ID=adaptive-policy-study SET="seeds=10"`
- Look at: the selected budget and its tie with FIFO.
- Read: `ola-bench/src/scheduling_summary.rs`
  (`optimize_virtual_time_candidate`), `ola-runtime/src/adaptive.rs`
  (`FiniteCandidateOptimizer`).
- Learn: train/held-out separation, and why the grid must include the
  baseline's behaviour.

**E2 (L2) — Online replay.**
- Run: `make lab-new NAME=online-replay ID=online-adaptive-policy-study SET="seeds=10"`
- Predict first: the gap between `online` (cold start) and `fixed`.
- Then vary `window` and `min_improvement_percent`, one entry each, and
  report on seeds 6–10 only if you chose values on seeds 1–5.
- Read: `ola-bench/src/online.rs`, `AdaptivePolicy::observe` in
  `ola-runtime/src/adaptive.rs`.
- Learn: hysteresis, exploration cost, and why replay needs an accurate
  cost model.

**E3 (L3) — Make the optimum move** (the open half of H1).
- Extend: write a trace generator in `ola-bench/src/trace.rs` whose best
  token budget changes over time, and prove it changes before testing
  adaptation: run `adaptive-policy-study`-style selection separately on
  each phase and show the phases pick different budgets. Candidates:
  alternating prompt-length regimes, or a cost model where per-token cost
  dominates (`CostModel` is a parameter of `SimulationConfig`).
- Then add a variant to `online-adaptive-policy-study` and test whether
  online beats the best fixed budget.

**E4 (L3) — A different optimizer.**
- Extend: implement `AdaptiveOptimizer` as a bandit (ε-greedy or UCB over
  the same 28 candidates, scored by live metrics instead of replays) and
  compare it with `ReplayOptimizer` on E3's trace.
- Learn: regret, exploration versus exploitation, and why a bandit pays
  for exploring on live traffic while replay pays in CPU.

**E5 (L4) — Open:** how wrong can the replay's cost model be before online
adaptation hurts? Perturb the cost model the optimizer sees, not the one
that runs (RQ3). Do learned policies transfer across cost models (RQ4)?

**E6 (L2, needs a model) — Adaptive serving on hardware.** Calibrate (B2),
then serve with and without `--adaptive` under the same load generator,
and compare latency distributions. This is the open measurement in
`todo.md`.

---

## Track F: cluster inference

Read `ola-rt/crates/DISTRIBUTED.md`, especially "Implementation status".

**F1 (L1) — Routing baselines.**
- Run: `make lab-new NAME=routing-prefix ID=cluster-routing SET="trace=prefix_groups seeds=10"`
  and the same with `trace=overload`.
- Look at: completion p95 and prefix hit tokens per routing policy.
- Read: `ola-cluster/src/routing.rs`, `ola-bench/src/cluster_simulation.rs`.
- Learn: the power of two choices (maximum load falls from about
  log n / log log n to log log n), and the tension between affinity and
  balance.

**F2 (L2) — The network matters, or not.** Predict, then vary
`latency_micros` (0, 50, 500, 5000) and `bytes_per_micro` in separate
entries. At what latency does routing policy stop mattering relative to
the network?

**F3 (L2) — Heterogeneous workers.** `SET="workers=8,4,2,2"`: which policies
handle unequal capacity, and why does round-robin fail?

**F4 (L3) — A new routing policy.**
- Extend: implement `ClusterPolicy` (for example, power-of-d choices with d
  as a parameter, or affinity with a load cap based on queued tokens) in
  `routing.rs`, add it to `cluster-routing`'s policy list in
  `ola-bench/src/experiment/cluster.rs`, and keep `ClusterPlan::validate`
  passing.

**F5 (L3) — Online cluster weights.** `AdaptiveClusterPolicy` has fixed
weights. Port the replay idea: replay recent arrivals through
`run_cluster_trial` per candidate weight vector and switch on improvement.

**F6 (L1, needs a model) — A live loopback cluster.** Follow "Distributed
worker and gateway serving" in `../TESTING_GUIDE.md`; kill a worker
mid-request and watch the gateway recover it (`/ready`, and the client's
stream continuing without repeated tokens).

**F7 (L4) — Open:** when does KV transfer beat recompute for migration, on
measured links (`HandoffCostModel` in `ola-cluster/src/transfer.rs`)?
Prefill/decode disaggregation: which role split is best per workload?

---

## Track G: kernels and hardware

**G1 (L1) — Portable versus CubeCL attention.**
- Run: `make lab-new NAME=attention-cpu ID=paged-attention-kernel FEATURES=cubecl-cpu`
  (CubeCL on the CPU, no GPU needed), and on a GPU
  `make lab-new NAME=attention-wgpu ID=paged-attention-kernel FEATURES=wgpu`.
- Read: `ola-kernel/src/cubecl_extension.rs` (the `paged_attention` kernel).
- Learn: one thread per (request, head, output dimension); online softmax.

**G2 (L2) — Roofline.** Compute the arithmetic intensity of the kernel
(FLOPs per byte of K/V read) for a page size and head dimension. Predict
whether it is memory-bound on your GPU, then measure with
`paged-attention-occupancy` and Nsight Compute (`ola-bench/README.md`,
"Accelerator occupancy").

**G3 (L4) — Open:** the kernel recomputes every score for each output
dimension. Share scores across a head's dimensions and measure the
speed-up and occupancy change.

---

## Track R: Rust, through this codebase

Each topic points at real code. Read it, then do the exercise.

| Topic | Read | Exercise |
| --- | --- | --- |
| Ownership and borrowing | `ola-runtime/src/executor.rs`: `step_iteration` mutates state only after `plan.validate` | Explain in your notes why the executor can lend `&PlanningSnapshot` to the policy but not `&mut self`. |
| Enums and `match` | `ola-runtime/src/plan.rs`, `ExecutorEvent` in `executor.rs` | Add a variant to a private enum; follow every compiler error. |
| Traits, generics, `dyn` | `ola-runtime/src/policy.rs`: `ExecutionPolicy` and the blanket `impl … for Box<P>` | Write a `Logging<P: ExecutionPolicy>` wrapper that counts plans. |
| Closures and `Fn` bounds | `AdaptivePolicyFactory`'s blanket impl for closures (`adaptive.rs`) | Pass a closure capturing a value as a factory. |
| `Send` and `Sync` | `ReplayOptimizer`'s `F: Send + Sync` bound (`ola-bench/src/online.rs`) | Remove `Sync` and read why it fails to compile. |
| Lifetimes and `&'static str` | θ keys `(&'static str, String)` throughout `adaptive.rs` | Why are the keys `'static` and values owned? Try a borrowed value. |
| Errors | `CoordinatorError` (`ola-cluster/src/coordinator.rs`): enum + `Display` + `Error` + `From` | Add a variant and use `?` through two layers. |
| Serde | `WireBody` in `ola-net/src/lib.rs`: tagged enums, `skip_serializing_if` | Print a message's JSON and predict it first. |
| async and cancellation safety | `ola-cluster/src/client.rs` (one reader task per link) and `ola-net/README.md` "Reading safely" | Write a test that drops a `read_message` future mid-frame and shows the corruption. |
| `tokio::select!` | `WorkerSession::run` (`ola-cluster/src/session.rs`), the gateway driver (`ola-serve/src/cluster.rs`) | Explain why each branch's future must be cancel-safe. |
| Sans-IO design | `ola-cluster/src/coordinator.rs`: no sockets or clocks | Drive a `Coordinator` by hand in a test, playing both workers. |
| Feature flags and `cfg` | `ola-bench/src/experiment/occupancy.rs` | Build with and without `--features cubecl-cpu`; explain each `#[cfg]`. |
| Iterators | `AdaptiveObjectiveMetrics::from_observations` | Rewrite one loop as an iterator chain, and back. |
| Testing | Determinism tests (`identical_snapshots_produce_identical_decisions` in `composed.rs`) | Write a determinism test for your C3 scheduler. |
| Lints and style | `make clippy`, `[workspace.lints]` in `ola-rt/Cargo.toml` | Fix a warning without `#[allow]`; when is `#[allow(…, reason = …)]` right? |
| Performance | release builds, `cargo bench`-style timing in `paged-attention-kernel` | Profile `scheduler-comparison` with `perf record` and find the hottest function. |

`unsafe` is forbidden in this workspace (`unsafe_code = "forbid"`): learn
why each safe abstraction is enough before reaching for it elsewhere.

---

## Writing your own experiment

When no existing experiment answers your question, add one. It takes
about a hundred lines, and once registered it runs from the CLI, the lab,
and the terminal workbench with no further wiring.

1. In `ola-rt/crates/ola-bench/src/experiment/builtin.rs` (or a new module
   next to it), add a unit struct and `impl Experiment` with:
   - `descriptor()`: a stable `id`, `category`, `kind`
     (`Probe` for operational checks, `Study` for anything you will cite),
     `domain` (`Virtual`, `Host`, or `Untimed`), `payload_version`, and
     `parameters` (`ParameterSpec::integer` / `ParameterSpec::text`).
   - `run(context, parameters)`: read parameters with
     `parameters.size("name")?` or `.text("name")?`, run the workload, and
     return `finish(descriptor, context, parameters, workload_name, seeds,
     metrics, variants, summary)`. Each metric is
     `metric(name, unit, domain, direction)`; each variant is a label with
     one sample per seed. `KvBlockPressure` is the smallest example;
     `AdaptivePolicyStudy` shows a full study with held-out seeds.
2. Register it in `ExperimentRegistry::builtin()` (top of `builtin.rs`),
   export it in `experiment/mod.rs` and `lib.rs`, and add its id to the
   expected list in `experiment/tests.rs`
   (`every_builtin_experiment_runs_with_its_defaults`).
3. A study must meet the protocol minimums (3 repetitions, or 5 plus a
   warm-up in host time); the registry rejects reports that do not.
4. Run `cargo test -p ola-bench`, then `make lab-new … ID=<your-id>`.

Bump `payload_version` whenever you change what a metric means or which
parameters exist, so old and new reports are never compared as the same
experiment.
