# ola-cli

The `ola` command serves models over HTTP and runs, lists, and compares
experiments from
`ola_bench::ExperimentRegistry`, the same registry the terminal workbench
uses, so a named experiment produces the same report from either frontend.

```text
ola experiment list
ola experiment run <id> [--model DIR] [--backend cpu|wgpu|cuda] [--set KEY=VALUE]...
                       [--results-dir DIR] [--json] [--trace FILE]
ola experiment compare <baseline.json> <candidate.json>
                       [--threshold FRACTION] [--allow-workload-mismatch]
ola serve --model DIR [--backend cpu|wgpu|cuda] [--name ID] [--addr HOST:PORT]
          [--max-active N] [--max-in-flight N] [--drain-secs S] [--trace FILE]
```

- `run` prints the experiment's summary, labelled as a probe (not for
  publication) or a study, with its clock. `--results-dir` saves an
  `ola-experiment/1` report and never overwrites; `--json` prints the report.
- `compare` reads experiment reports, or legacy prefill v2 reports, lists
  every identity difference, and prints a verdict per variant and metric. A
  metric is improved or regressed only when the 95% intervals separate and
  the change exceeds the threshold (5% by default).

- `--trace FILE` writes request traces as JSON Lines to a new file (an
  existing file is never overwritten). `serve` traces every request and
  always keeps recent traces for `GET /v1/requests/{id}/trace`. Experiments
  marked `traceable` in `list` (those that drive an executor) label each
  executor run, such as `fifo/seed=3`; others refuse `--trace`. The
  simulator has no clock, so scheduler-comparison traces carry iteration
  start times, not the corrected virtual times in its summary.
- `serve` loads a Llama artifact, registers it under `--name` (default: the
  directory name) with memory-aware admission and up to `--max-active`
  concurrent sessions (default 4), and serves the `ola-serve` HTTP API on
  `--addr` (default `127.0.0.1:8080`). Ctrl-C refuses new work and lets
  accepted requests finish for up to `--drain-secs` (default 30).

Exit codes: `0` success, `1` `compare` found a regression (usable as a CI
gate), `2` usage or run error.

From the repository root, `make experiments` lists experiments and
`make experiment ID=<id> [MODEL=…] [SET="key=value …"] [RESULTS_DIR=…]` runs
one; `make serve MODEL=… [ADDR=…]` serves a model. WGPU and CUDA need `--features wgpu` or `--features cuda`.
