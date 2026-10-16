# ola-cli

The `ola` command runs, lists, and compares experiments from
`ola_bench::ExperimentRegistry`, the same registry the terminal workbench
uses, so a named experiment produces the same report from either frontend.

```text
ola experiment list
ola experiment run <id> [--model DIR] [--backend cpu|wgpu|cuda] [--set KEY=VALUE]...
                       [--results-dir DIR] [--json]
ola experiment compare <baseline.json> <candidate.json>
                       [--threshold FRACTION] [--allow-workload-mismatch]
```

- `run` prints the experiment's summary, labelled as a probe (not for
  publication) or a study, with its clock. `--results-dir` saves an
  `ola-experiment/1` report and never overwrites; `--json` prints the report.
- `compare` reads experiment reports, or legacy prefill v2 reports, lists
  every identity difference, and prints a verdict per variant and metric. A
  metric is improved or regressed only when the 95% intervals separate and
  the change exceeds the threshold (5% by default).

Exit codes: `0` success, `1` `compare` found a regression (usable as a CI
gate), `2` usage or run error.

From the repository root, `make experiments` lists experiments and
`make experiment ID=<id> [MODEL=…] [SET="key=value …"] [RESULTS_DIR=…]` runs
one. WGPU and CUDA need `--features wgpu` or `--features cuda`.
