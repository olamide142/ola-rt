# OlaRT testing and experiment guide

How to validate the runtime, run every experiment, serve models alone or as
a cluster, and record what you find. For the research workflow (notebook
entries, predictions, findings) see [`lab/README.md`](lab/README.md); for a
graded list of experiments to learn from, see
[`lab/CURRICULUM.md`](lab/CURRICULUM.md).

Run everything from the repository root. `make help` lists every target and
variable.

## 1. Validate the code

```bash
make verify        # fmt-check, check, test, clippy (warnings denied), doc
```

`make verify` is the gate to pass before committing code or running
experiments you intend to cite. Its parts:

| Target | What it does |
| --- | --- |
| `make fmt` / `make fmt-check` | Format, or check formatting of, every crate. |
| `make check` | Type-check all targets, including tests and examples. |
| `make test` | Run the default (CPU) test suite. |
| `make clippy` | Lint with warnings as errors. |
| `make doc` | Build API docs; fails on broken doc links. |

One crate at a time is faster while you work:

```bash
cargo test --manifest-path ola-rt/Cargo.toml -p ola-runtime
cargo test --manifest-path ola-rt/Cargo.toml -p ola-bench --lib online   # tests matching "online"
```

Crate map: `ola-core` (identities, time), `ola-model` (model traits),
`ola-kernel` (attention kernels), `ola-burn` (Llama and Mistral on Burn),
`ola-runtime` (executor, policies, KV cache), `ola-observe` (traces),
`ola-bench` (experiments, simulator), `ola-serve` (HTTP serving),
`ola-net` (wire format), `ola-cluster` (coordinator, workers), `ola-cli`
(the `ola` command), `ola-ui` (terminal workbench).

### Tests that need more than a CPU

| Command | Needs |
| --- | --- |
| `make test-fixture MODEL=/path/to/tiny-llama` | A Hugging Face-style Llama directory; runs the ignored end-to-end serving and conformance tests. |
| `make test-gpu-fixture MODEL=…` | The same, on WGPU. |
| `make check-wgpu`, `make check-cuda` | Compile-only checks of the accelerator builds. |
| `cargo test --manifest-path ola-rt/Cargo.toml -p ola-bench --features cubecl-cpu` | Nothing extra: CubeCL kernels on the CPU. |
| `make gpu-info` | A working NVIDIA driver. |

## 2. Run experiments

```bash
make experiments                       # every experiment, its kind, clock, and parameters
make experiment ID=scheduler-comparison SET="trace=overload seeds=10"
make lab-new NAME=my-question ID=scheduler-comparison SET="trace=overload seeds=10"
```

`make experiment` prints a summary. `make lab-new` does the same inside a
notebook entry that keeps the configuration, versioned reports, traces, and
your notes, which is what you want for anything you may cite (see
`lab/README.md`). Directly:

```bash
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
    experiment run <id> [--config FILE] [--set KEY=VALUE]... [--model DIR] \
    [--backend cpu|wgpu|cuda] [--results-dir DIR] [--json] [--trace FILE]
```

`--config FILE` reads `{"id": "<id>", "parameters": {"key": "value"}}`;
`--set` overrides it. Use `--release` for host-time experiments: debug
builds measure the compiler, not the runtime.

### Kinds and clocks

Every experiment is a **probe** (an operational check, not for publication)
or a **study** (a protocol with repetitions and 95% intervals that you can
cite). Its metrics come from one clock:

- **virtual time**: the deterministic simulator and its `CostModel`;
  results reproduce exactly anywhere, and say what the cost model implies;
- **host time**: wall-clock on this machine; real but noisy;
- **device time**: a hardware profiler's kernel timings;
- **untimed**: counts and ratios.

`compare` never mixes clocks, and the registry rejects studies with fewer
than 3 repetitions (5 plus a warm-up in host time).

### The experiments

Model-backed, host time (`--model DIR`):

| ID | Kind | What it measures |
| --- | --- | --- |
| `llama-generation` | probe | Greedy generation: TTFT, inter-token latency, throughput. |
| `llama-prefill-strategies` | study | Unbucketed, power-of-two, and cost-aware prefill grouping, calibrated first. |
| `llama-chunked-prefill` | probe | Whole prompts against resumable prompt chunks, and fairness across chunk sizes. |
| `cost-calibration` | study | Fits per-iteration, per-call, and per-token cost; feeds `ola serve --adaptive`. |

No model needed:

| ID | Kind, clock | What it measures |
| --- | --- | --- |
| `kv-block-pressure` | probe, host | KV block allocation, growth, freeing. |
| `fifo-workload` | probe, host | A deterministic FIFO runtime replay. |
| `prefill-bucketing` | probe, untimed | Padding against invocations. |
| `adaptive-prefill-cost` | probe, untimed | Cost-aware grouping at different invocation costs. |
| `scheduler-comparison` | study, virtual | FIFO, priority, EDF, SJF, and backfill on one trace (`trace=` `mixed_short_long`, `prefill_decode_competition`, `priority_classes`, `deadlines`, `overload`, `impossible`). |
| `adaptive-policy-study` | study, virtual | A token budget chosen offline on seed 0, against FIFO on held-out seeds. |
| `online-adaptive-policy-study` | study, virtual | Token budgets re-optimized online by replaying recent work, against the best offline budget and FIFO. |
| `kv-pressure` | study, virtual | Worst-case KV reservation against watermark admission with preemption (`trace=` `early_stopping`, `mixed_short_long`, `overload`). |
| `prefix-cache` | study, virtual | Prefill reuse for prompts sharing a system prompt. |
| `kv-page-size` | study, virtual | Internal fragmentation and prefix reuse across page sizes. |
| `cluster-routing` | study, virtual | Routing baselines across simulated workers and links (`trace=` `prefix_groups`, `mixed_short_long`, `overload`, `prefill_decode_competition`, `priority_classes`). |
| `adaptive-cluster-routing-study` | study, virtual | Cluster routing weights chosen on seed 0, on held-out seeds. |

Kernel studies, only in builds with a CubeCL feature
(`--features cubecl-cpu`, `wgpu`, or `cuda` on `ola-cli`; `FEATURES=` for
`make lab-new`):

| ID | Kind, clock | What it measures |
| --- | --- | --- |
| `paged-attention-kernel` | study, host | Portable against CubeCL paged attention: latency, correctness, page and batch sizes. |
| `paged-attention-occupancy` | study, host and device | The same shapes with Nsight Compute occupancy and utilization; see `ola-rt/crates/ola-bench/README.md`. |

Examples:

```bash
make experiment ID=scheduler-comparison SET="trace=mixed_short_long seeds=10 requests=200"
make experiment ID=cluster-routing SET="workers=4,4,4,4 latency_micros=50 bytes_per_micro=1250 seeds=10"
make experiment ID=online-adaptive-policy-study SET="seeds=10 window=64"
make lab-new NAME=attention-cpu ID=paged-attention-kernel FEATURES=cubecl-cpu
```

### Reports

`--results-dir DIR` writes an `ola-experiment/1` JSON report and never
overwrites one. It holds the experiment and payload version, build
(revision, profile, features), workload digest, protocol, every raw sample,
and per-variant summaries with means and 95% bootstrap intervals.

```bash
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment compare a.json b.json
```

`compare` lists identity differences and classifies each metric as improved
or regressed only when the intervals separate and the change exceeds 5%
(`--threshold`); it exits 1 on a regression. Summaries with `jq`:

```bash
jq '.summaries[] | {variant, p95: .estimates.completion_p95.mean, ci: .estimates.completion_p95.ci95}' report.json
```

### Reading traces

Experiments marked `traceable` in `make experiments` accept
`--trace FILE`, written as JSON Lines, one record per request event, each
labelled with its executor run (`"run": "fifo/seed=3"`):

```json
{"run":"fifo/seed=1","request":1,"iteration":1,"at_us":0,"event":"admission","outcome":"admitted","reserved_kv_bytes":1024,"budget_selected":true}
```

Event kinds: `arrived`, `ranked`, `admission`, `backfill_skipped`,
`selection`, `invocation`, `prefill_progress`, `token`, `completed`,
`failed`, `rejected`, `cancelled`, `preempted`.

```bash
jq -c 'select(.event == "admission" and .outcome != "admitted")' trace.jsonl   # deferrals
jq -c 'select(.run == "sjf/seed=1" and .request == 7)' trace.jsonl            # one request's life
jq -s 'group_by(.run) | map({run: .[0].run, tokens: map(select(.event == "token")) | length})' trace.jsonl
```

Simulator traces carry iteration start times, not the cost-model-corrected
times in the summaries.

## 3. Run models directly

Get a model first; [`lab/MODELS.md`](lab/MODELS.md) lists what loads and
what fits in memory:

```bash
lab/lab fetch HuggingFaceTB/SmolLM2-135M-Instruct     # Llama architecture, 0.5 GiB as f32
lab/lab fetch Locutusque/TinyMistral-248M             # Mistral architecture, 0.9 GiB
```

```bash
make llama MODEL=/path/to/llama PROMPT="Hello" TOKENS=16
make llama-wgpu MODEL=/path/to/llama PROMPT="Hello" TOKENS=16
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-burn --example run_mistral -- /path/to/mistral "Hello" 16
```

`ola serve` and the model-backed experiments load Llama artifacts; Mistral
runs through its example and the library API.

## 4. Serve one model

```bash
make serve MODEL=/path/to/llama ADDR=127.0.0.1:8080
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
    serve --model /path/to/llama --name llama --addr 127.0.0.1:8080 --trace server.jsonl
```

Endpoints: `GET /health`, `GET /ready`, `GET /v1/models`,
`POST /v1/generate`, `POST /v1/completions`, `GET /v1/requests/{id}`,
`DELETE /v1/requests/{id}`, `GET /v1/requests/{id}/trace`.

```bash
H='Content-Type: application/json'     # required: requests without it are refused
curl -s  -H "$H" localhost:8080/v1/generate -d '{"model":"llama","prompt":"Hello","max_new_tokens":16}'
curl -sN -H "$H" localhost:8080/v1/generate -d '{"model":"llama","prompt":"Hello","max_new_tokens":16,"stream":true}'
curl -sN -H "$H" localhost:8080/v1/completions -d '{"model":"llama","prompt":"Hello","max_tokens":16,"stream":true}'
```

Streaming (`"stream": true`) sends server-sent events. `/v1/generate`
emits `start`, one `token` event per token (`token_id`, `text`, `phase`,
`generated_tokens`), and a final `done` with the full text, timings, and
inter-token latencies. `/v1/completions` streams OpenAI-style chunks, a
final chunk with `finish_reason` and `usage`, then `data: [DONE]`, so
OpenAI client libraries can read it. To watch only the text arrive:

```bash
curl -sN -H "$H" localhost:8080/v1/generate \
    -d '{"model":"llama","prompt":"Once upon a time","max_new_tokens":64,"stream":true}' |
    sed -un 's/^data: //p' | jq --unbuffered -rj 'select(.token_id) | .text'; echo
```

Requests name the model by its `--name` (default: the directory name; the
examples use `--name llama`). Ctrl-C
drains: new work is refused and accepted requests get `--drain-secs`
(default 30) to finish. Full contract: `ola-rt/crates/ola-serve/README.md`.

### Adaptive token budgets

```bash
make lab-new NAME=calibrate ID=cost-calibration MODEL=/path/to/llama
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- \
    serve --model /path/to/llama --adaptive lab/notebook/<date>-calibrate/reports/<report>.json
```

`--adaptive` replays recent work in the simulator, with the calibrated cost
model, to re-tune how many tokens each iteration may process; it starts
from the default behaviour. Calibrate on the same model and backend you
serve. How it works and what it has (not yet) shown:
`ola-rt/crates/ola-runtime/ADAPTIVE_OPTIMIZATION_PLAN.md`.

## 5. Serve a cluster

Start the workers first, then the gateway, which connects to each:

```bash
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --role worker --model /path/to/llama --name worker-a --addr 127.0.0.1:7001
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --role worker --model /path/to/llama --name worker-b --addr 127.0.0.1:7002
cargo run --release --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --role gateway --addr 127.0.0.1:8080 --worker 127.0.0.1:7001 --worker 127.0.0.1:7002
```

The gateway serves `/health`, `/ready`, `/v1/generate`, `/v1/completions`,
and `DELETE /v1/requests/{id}`, routing with the adaptive cluster policy.
Things to try:

- `curl -s localhost:8080/health` shows connected and ready workers.
- Kill a worker mid-request: the gateway sees the connection close (or,
  for a worker that hangs, its 2 s lease lapse), marks it lost, and restarts
  its requests on the other worker; a streaming client sees no repeated
  tokens.
- Disconnect a client mid-stream: its request is cancelled on the worker.

Migration, draining, role changes, and policy pushes are library
operations (`TcpCoordinator::migrate`, `drain_worker`, `set_worker_role`,
`update_worker_policy`) that the gateway does not trigger itself; the
integration tests exercise them:

```bash
cargo test --manifest-path ola-rt/Cargo.toml -p ola-net
cargo test --manifest-path ola-rt/Cargo.toml -p ola-cluster --test network            # loss, recovery, large snapshots
cargo test --manifest-path ola-rt/Cargo.toml -p ola-cluster --test network_migration  # two-phase migration
cargo test --manifest-path ola-rt/Cargo.toml -p ola-serve --test cluster_gateway      # gateway behaviour
```

Design, status, and limits: `ola-rt/crates/DISTRIBUTED.md`.

## 6. Terminal workbench

```bash
make ui
make ui MODEL=/path/to/llama RESULTS_DIR=results
make ui-wgpu MODEL=/path/to/llama
```

The workbench runs the same registered experiments as the CLI, with chat,
benchmarks, and saved reports.

## 7. Hardware

- CPU is the default backend and needs nothing extra.
- WGPU needs `--features wgpu` (`make check-wgpu` to compile-check).
- CUDA needs `--features cuda`, a working driver, and the CUDA toolkit with
  NVRTC; check with `make gpu-info`.
- Meaningful generated text needs a trained Hugging Face-style Llama
  artifact. The tiny conformance fixture checks correctness, not quality.

## 8. A first session

1. `make verify`.
2. Read `lab/README.md`, then do curriculum items A1, C1, and D1 as lab
   entries, writing each prediction first.
3. With a model: B2 (calibrate), then serve it (section 4) and try the
   `--adaptive` flag.
4. Two workers and a gateway (section 5); kill one.
5. Pick an L4 question from `lab/CURRICULUM.md` and start a series of
   entries.

## References

- [`lab/README.md`](lab/README.md), [`lab/CURRICULUM.md`](lab/CURRICULUM.md)
- [`ola-bench/README.md`](ola-rt/crates/ola-bench/README.md) and
  [`EXPERIMENTS.md`](ola-rt/crates/ola-bench/EXPERIMENTS.md)
- [`ola-cli/README.md`](ola-rt/crates/ola-cli/README.md)
- [`ola-serve/README.md`](ola-rt/crates/ola-serve/README.md)
- [`ola-cluster/README.md`](ola-rt/crates/ola-cluster/README.md),
  [`DISTRIBUTED.md`](ola-rt/crates/DISTRIBUTED.md)
- [`OBSERVABILITY.md`](ola-rt/crates/ola-observe/OBSERVABILITY.md)
- [`ADAPTIVE_OPTIMIZATION_PLAN.md`](ola-rt/crates/ola-runtime/ADAPTIVE_OPTIMIZATION_PLAN.md)
