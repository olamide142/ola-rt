# OlaRT Testing Guide

This guide lists the runnable functionality, experiments, serving modes, and validation commands in the current repository.

Run from the repository root:

~~~bash
cd /home/olamide/dev/ola-rt
~~~

## Baseline validation

~~~bash
make check
make test
make clippy
make doc
make verify
~~~

Targeted commands:

~~~bash
cargo check --manifest-path ola-rt/Cargo.toml --workspace
cargo test --manifest-path ola-rt/Cargo.toml --workspace
cargo test --manifest-path ola-rt/Cargo.toml -p ola-runtime
cargo test --manifest-path ola-rt/Cargo.toml -p ola-burn
cargo test --manifest-path ola-rt/Cargo.toml -p ola-cluster
cargo test --manifest-path ola-rt/Cargo.toml -p ola-net
cargo test --manifest-path ola-rt/Cargo.toml -p ola-serve
~~~

## Experiment CLI

List every registered experiment and its parameters:

~~~bash
make experiments
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment list
~~~

Run or compare experiments:

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run <id> [--model DIR] [--backend cpu|wgpu|cuda] [--set KEY=VALUE]... [--results-dir DIR] [--json] [--trace FILE]
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment compare baseline.json candidate.json
~~~

### Available experiments

Model-backed host-time experiments requiring a local Llama artifact:

- `llama-generation` — greedy generation, TTFT, inter-token latency, and throughput.
- `llama-prefill-strategies` — unbucketed, power-of-two, and adaptive prefill grouping.
- `llama-chunked-prefill` — complete prompt versus resumable prompt segments.
- `cost-calibration` — fixed, per-token, and per-call execution cost fitting.

Runtime and KV experiments that do not require a model:

- `kv-block-pressure` — KV block allocation, growth, reuse, and freeing.
- `fifo-workload` — deterministic FIFO runtime replay.
- `prefill-bucketing` — padding and invocation tradeoffs.
- `adaptive-prefill-cost` — cost-aware grouping under different invocation costs.

Virtual-time scheduling experiments:

- `scheduler-comparison` — scheduling baselines on seeded traces.
- `adaptive-policy-study` — training-trace selection and held-out evaluation.
- `kv-pressure` — KV reservation versus watermark admission and preemption.
- `prefix-cache` — shared-prefix reuse and cache affinity.
- `kv-page-size` — fragmentation and prefix reuse across page sizes.

Distributed simulation experiments:

- `cluster-routing` — routing baselines across simulated workers and links, including latency, bandwidth, capacity, KV headroom, and prefix affinity.
- `adaptive-cluster-routing-study` — training-selected routing weights on held-out traces.

Kernel and accelerator experiments:

- `paged-attention-kernel` — portable versus CubeCL paged attention, correctness, latency, page sizes, and batch sizes.
- `paged-attention-occupancy` — paged-attention latency and Nsight Compute occupancy metrics.

Common scheduler trace values include `mixed_short_long`, `prefill_decode_competition`, `priority_classes`, `deadlines`, `overload`, `impossible`, and `early_stopping`.

Examples:

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run kv-block-pressure
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run scheduler-comparison --set trace=overload --set seeds=10 --set requests=200
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run cluster-routing --set workers="4,4,4,4" --set latency_micros=50 --set bytes_per_micro=1250 --set prefix_cache=1 --set seeds=10
~~~

## Reports and traces

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run scheduler-comparison --results-dir ./results
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run scheduler-comparison --json
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- experiment run scheduler-comparison --trace scheduler.jsonl
jq . scheduler.jsonl
~~~

Reports retain raw samples, metrics, confidence intervals, policy variants, build information, workload identity, and decision data. Report files are never overwritten.

Filter planning events:

~~~bash
jq 'select(.event.kind == "ranked" or .event.kind == "admission" or .event.kind == "selection")' scheduler.jsonl
~~~

## Direct model execution

Llama on CPU:

~~~bash
make llama MODEL=/path/to/llama-artifact PROMPT="Hello" TOKENS=16
~~~

Llama through WGPU:

~~~bash
make llama-wgpu MODEL=/path/to/llama-artifact PROMPT="Hello" TOKENS=16
~~~

Mistral through the Burn example:

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-burn --example run_mistral -- /path/to/mistral-artifact "Hello" 16
~~~

The CLI serving path currently loads Llama artifacts. Mistral is directly runnable through its Burn example and library APIs.

## Standalone HTTP serving

~~~bash
make serve MODEL=/path/to/llama-artifact ADDR=127.0.0.1:8080
~~~

With tracing:

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --model /path/to/llama-artifact --addr 127.0.0.1:8080 --trace server.jsonl
~~~

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/generate`
- `POST /v1/completions`
- `GET /v1/requests/{id}`
- `DELETE /v1/requests/{id}`
- `GET /v1/requests/{id}/trace`

This exercises JSON generation, SSE streaming, OpenAI-compatible completions, request status, cancellation, tracing, bounded in-flight work, memory-aware admission, and graceful draining.

## Distributed worker and gateway serving

Start two workers:

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --role worker --model /path/to/llama-artifact --name worker-a --addr 127.0.0.1:7001
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --role worker --model /path/to/llama-artifact --name worker-b --addr 127.0.0.1:7002
~~~

Start the gateway:

~~~bash
cargo run --manifest-path ola-rt/Cargo.toml -p ola-cli -- serve --role gateway --addr 127.0.0.1:8080 --worker 127.0.0.1:7001 --worker 127.0.0.1:7002
~~~

This exercises adaptive routing, heartbeats, leases, readiness, disconnect recovery, migration, recompute fallback, binary KV/session snapshot transfer, prefill/decode roles, streamed text and usage, stale-token suppression, cancellation, and graceful shutdown.

Targeted integration tests:

~~~bash
cargo test --manifest-path ola-rt/Cargo.toml -p ola-net
cargo test --manifest-path ola-rt/Cargo.toml -p ola-cluster --test network
cargo test --manifest-path ola-rt/Cargo.toml -p ola-cluster --test network_migration
cargo test --manifest-path ola-rt/Cargo.toml -p ola-serve --test cluster_gateway
~~~

## Terminal UI

~~~bash
make ui
make ui MODEL=/path/to/llama-artifact
make ui-wgpu MODEL=/path/to/llama-artifact
~~~

The UI includes runtime overview, local chat, generation benchmarks, prefill benchmarks, chunked prefill, KV pressure, scheduler experiments, report persistence, and experiment progress.

## Hardware and model constraints

- CPU is the default and safest backend.
- WGPU requires `--features wgpu`.
- CUDA requires `--features cuda`, a working NVIDIA driver, and CUDA/NVRTC.
- A trained Hugging Face-style artifact is needed for meaningful generated text.
- Check the local NVIDIA setup with `make gpu-info`.

## Recommended order

1. Run `make check` and `make test`.
2. Run `kv-block-pressure`, `prefill-bucketing`, and `scheduler-comparison`.
3. Run `cluster-routing` and `adaptive-cluster-routing-study`.
4. Run direct Llama CPU generation and host-time experiments.
5. Start and test standalone HTTP serving.
6. Start two workers and a gateway.
7. Run the distributed integration tests.
8. Try WGPU after `make check-wgpu`.
9. Attempt CUDA only after `make gpu-info` succeeds.

## Further references

- [`ola-bench/README.md`](ola-rt/crates/ola-bench/README.md)
- [`ola-bench/EXPERIMENTS.md`](ola-rt/crates/ola-bench/EXPERIMENTS.md)
- [`ola-cli/README.md`](ola-rt/crates/ola-cli/README.md)
- [`ola-serve/README.md`](ola-rt/crates/ola-serve/README.md)
- [`ola-cluster/README.md`](ola-rt/crates/ola-cluster/README.md)
- [`DISTRIBUTED.md`](ola-rt/crates/DISTRIBUTED.md)
- [`OBSERVABILITY.md`](ola-rt/crates/ola-observe/OBSERVABILITY.md)
