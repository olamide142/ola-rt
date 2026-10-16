# Experiment framework design proposal

Status: **accepted** (2026-10-16) with all five recommended decisions.
Covers `todo.md` P4. Steps 1–5 are implemented.

## Goal

Any experiment, whether a policy comparison, a kernel benchmark, or a
calibration, is described, run, persisted, compared, and displayed through
one framework. A new comparison then needs neither bespoke UI code nor a
one-off report schema, which is P4's completion criterion.

## What exists

- **Eight experiments, each hand-wired.** Each one in the terminal workbench
  (`ola-ui/src/local.rs`) has its own ID constant, catalogue entry, dispatch
  arm, formatting function, and test. Most return only a formatted string.
- **One bespoke report schema.** `PrefillBenchmarkReport` (schema v2) is
  rigorous: it keeps raw paired samples, protocol, and metadata, writes
  non-overwriting files, and validates on read. But it is specific to
  prefill strategies. `PlanReport` serializes one iteration, not an
  experiment.
- **No command-line runner.** Experiments run only inside the UI or tests.
  `ola-bench` cannot load a real model, because model loading lives in
  `ola-burn`, which the UI depends on.
- **Two kinds of time are not distinguished.** P3's virtual-time numbers are
  exactly reproducible; host wall-clock numbers are noisy and machine
  specific. Nothing in a result says which kind it is.
- **Calibration is a two-point fit.** `calibrate_prefill_invocation_overhead`
  solves `time = a·dense + b·invocations` from two p50 values, with no
  residual and no uncertainty. Its result never reaches `CostModel` or
  `PolicyMeasurements`.
- **Metadata is caller-supplied.** Model, device, and revision are strings
  the caller fills in. The policy appears only implicitly, and build
  identity and kernel are absent.

## Proposal

### 1. One metric model

Every measurement is a named metric with explicit semantics:

```rust
pub struct Metric {
    pub name: MetricName,          // stable, e.g. "completion_p95"
    pub unit: Unit,                // Micros, Tokens, Bytes, Ratio, TokensPerSecond, Count
    pub domain: TimeDomain,        // Virtual | Host | Untimed
    pub better: Direction,         // Lower | Higher | Neither
}
```

- `TimeDomain` keeps simulation time and host time apart. Comparisons and
  displays never mix domains, and a report states which it contains.
- `Direction` lets comparison decide what a regression is without
  per-experiment code.

### 2. One versioned report envelope

```rust
pub struct ExperimentReport {
    pub schema: String,                  // "ola-experiment/1"
    pub experiment: ExperimentIdentity,  // id, payload version, kind, domain
    pub metadata: RunMetadata,           // time, model, device, build, host
    pub workload: WorkloadIdentity,      // name, seeds, parameters, content digest
    pub protocol: Protocol,              // warm-ups, repetitions, ordering
    pub variants: Vec<Variant>,          // label + policy descriptor + kernel identity
    pub samples: Vec<Sample>,            // raw: (repetition, variant, metric values)
    pub summaries: Vec<VariantSummary>,  // derived: per variant, per metric Estimate
    pub details: Option<serde_json::Value>, // experiment-specific extras, never needed to compare
}
```

- **Raw and derived together.** Summaries are a pure function of the samples.
  On read, validation *recomputes* them and rejects the file on mismatch,
  together with schema and payload versions and sample consistency
  (contiguous repetitions, known variants and metrics).
- **Complete identity.** `RunMetadata.build` is captured at compile time
  (crate version, git revision when available, profile, enabled features,
  target) and can be overridden by `OLA_REVISION`. `Variant` records the
  policy's `PolicyDescriptor` and, when relevant, the kernel's stable
  operation, implementation, and target metadata from `ola-kernel`.
- **Persistence.** `write_new(dir, &report)` refuses to overwrite and picks a
  unique `<experiment>-<utc>-<n>.json`. Files from the existing prefill
  schema v2 are read and converted, never written again.

### 3. Uncertainty

`Estimate { n, mean, p50, p95, min, max, ci95 }` per variant and metric. The
95% interval is a **seeded percentile bootstrap** over repetitions, so it
needs no distributional assumption and is itself reproducible.

What a repetition is depends on the domain:

- **Virtual time:** each repetition is a different trace seed. Re-running one
  seed is identical by construction, so variance must come from workload
  variation.
- **Host time:** each repetition is a re-run of the same workload, after
  warm-ups, in rotated variant order (the existing prefill protocol,
  generalized).

### 4. Comparison and regressions

`compare(baseline, candidate) -> ReportComparison`:

- matches by experiment, variant, and metric, and refuses differing
  workloads or time domains unless explicitly allowed;
- lists every identity difference (model, device, build, policy, kernel),
  so a comparison states what changed;
- reports each metric's delta and relative change, and classifies it as
  `Improved`, `Regressed`, or `Inconclusive`. A change counts only when the
  95% intervals do not overlap *and* it exceeds a relative threshold
  (default 5%). One run never yields a verdict.

### 5. Experiments as data

```rust
pub trait Experiment: Send + Sync {
    fn descriptor(&self) -> ExperimentDescriptor;   // id, name, description, kind, domain,
                                                    // requirements, parameters with defaults
    fn run(&self, context: &ExperimentContext, parameters: &Parameters)
        -> Result<ExperimentReport, ExperimentError>;
}
```

- **The registry.** `ExperimentRegistry::builtin()` lists every experiment
  in one place.
- **The context.** `ExperimentContext` carries an optional loaded model and
  its device label, supplied by the frontend, which can load models through
  `ola-burn`. It also carries build metadata and a cancellation flag.
- **Two kinds.** `ExperimentKind::Probe` is an operational check, always
  labelled *not for publication*. `ExperimentKind::Study` enforces protocol
  minimums (for example at least 5 repetitions and at least 1 warm-up for
  host time) and always reports intervals.
- **Porting.** All eight current experiments move to the trait. The
  scheduler comparison becomes a virtual-time study across seeds.

### 6. Calibration

Multi-shape calibration replaces the two-point fit:

- a grid of batch widths and prompt lengths runs as a host-time study;
- `time = c + a·dense_tokens + b·invocations` is fitted by least squares,
  with residuals and bootstrap intervals on each coefficient;
- the fit converts into a `CostModel`, so P3 comparisons run on measured
  costs, and into `PolicyMeasurements::prefill_invocation_overhead_tokens`
  for cost-aware grouping.

### 7. Frontends

- **UI:** the experiment and benchmark lists are built from the registry and
  filtered by requirements (for example, needs a loaded model). One generic
  runner shows the summary line and headline metrics, and saves the report
  when a results directory is configured. No experiment-specific code
  remains in `ola-ui`.
- **CLI:** a new `ola-cli` crate provides an `ola` binary:
  - `ola experiment list`;
  - `ola experiment run <id> [--model DIR] [--backend …] [--set key=value] [--results-dir DIR]`;
  - `ola experiment compare <baseline.json> <candidate.json>`.

  It uses the same registry, so the same named experiment runs identically
  from both frontends. A test checks this for a virtual-time experiment: the
  reports are identical apart from timestamps.

## Decisions (all recommendations accepted)

1. **A uniform metric map, plus optional `details`** *(recommended)*, or a
   typed payload per experiment inside the envelope. The uniform map is what
   makes comparison and the UI generic.
2. **Read and convert prefill schema v2 reports** *(recommended)*, or stop
   supporting existing files.
3. **Seeded bootstrap intervals** *(recommended)*, or t-intervals. Bootstrap
   makes no normality assumption and stays reproducible.
4. **A new `ola-cli` crate with an `ola` binary** *(recommended)*, or a
   `--run-experiment` flag on `ola-ui`. A separate crate keeps the UI a pure
   frontend and gives scripting a stable entry point.
5. **Capture the git revision in a build script, with an `OLA_REVISION`
   override and `unknown` fallback** *(recommended)*, or keep revision
   caller-supplied.

## Implementation steps

Each step keeps `make verify` and `make test-fixture` green.

1. **Report core.** Add `Metric`, identity, metadata, build info, and
   `ExperimentReport`, with recompute-on-read validation and non-overwriting
   persistence. **Done**
   (`src/experiment/`). `serde_json`'s `float_roundtrip` feature is enabled
   workspace-wide so recompute-on-read validation is exact. The build script
   records the commit but not whether the working tree had uncommitted
   changes.
2. **Statistics and comparison.** Add `Estimate` with seeded bootstrap,
   `compare` with identity diffs and regression classification. **Done.**
   Intervals use 2,000 resamples seeded from the values' bit patterns, so
   recompute-on-read reproduces them; a single repetition has no interval
   and therefore never yields a verdict. `SplitMix64` moved to a shared
   `random` module.
3. **Experiment trait and registry.** Add both, and port the five
   model-independent experiments, including the scheduler comparison as a
   multi-seed study. **Done.**
   `ExperimentRegistry::builtin()` holds KV block pressure, FIFO workload,
   prefill bucketing, adaptive prefill cost (probes), and the scheduler
   comparison, now a virtual-time study over five seeds by default with all
   five scheduling baselines on one chosen trace. Descriptors carry a
   category (benchmark or experiment) so the UI can keep its two lists. The
   registry checks requirements before a run and identity plus study
   minimums after it. The UI still uses its own copies until step 6.
4. **Model experiments.** Port Llama generation, prefill strategies (with
   the v2 import), and chunked prefill. **Done.**
   Llama generation and chunked prefill are probes; prefill strategies
   becomes a host-time study (calibration, then 5 rotated trials after a
   warm-up, up from 3). Its metrics are produced from `PrefillReportSample`
   for both live runs and `ExperimentReport::from_prefill_v2`, so imported
   v2 files compare directly with new runs; `read_compatible` reads either
   format. The chunked-prefill fairness sweep is kept in `details`.
5. **Calibration.** Add the multi-shape fit with intervals, feeding
   `CostModel` and `PolicyMeasurements`. **Done.**
   `cost-calibration` is a host-time study: each repetition sweeps a
   width × length grid, every shape both batched and split into one call
   per request (so dense tokens and calls vary independently), and fits
   `c + a·dense + b·calls`; intervals come from the repetitions, and the
   pooled fit with observation-bootstrap intervals is kept in `details`.
   `CostFit::from_report` turns a saved calibration into a `CostModel` for
   virtual-time comparisons and into `PolicyMeasurements`, which
   `Executor::with_policy_measurements` now delivers in every snapshot. On
   tiny Llama (debug build) one call costs about one token of work.
6. **Data-driven UI.** Build the catalogue from the registry and remove the
   bespoke code.
7. **CLI and parity.** Add `ola-cli` and the UI/CLI parity test, then update
   documentation and `progress_report.md`.

## Out of scope for P4

- Device telemetry inside reports (memory, utilization): P6.
- Serving-path experiments (HTTP load generation): P5.
- Dashboards or plotting; reports are JSON for external tools.
