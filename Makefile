# Common OlaRT developer workflows. Run `make help` for the public command list.

CARGO ?= cargo
MANIFEST := ola-rt/Cargo.toml
MODEL ?=
PROMPT ?= Hello
TOKENS ?= 8
RESULTS_DIR ?=
ID ?=
SET ?=
ADDR ?= 127.0.0.1:8080

.DEFAULT_GOAL := help

.PHONY: help fmt fmt-check check test clippy doc verify \
	check-wgpu check-cuda gpu-info ui ui-wgpu llama llama-wgpu \
	test-fixture test-gpu-fixture require-model experiments experiment require-id serve

help: ## Show available commands and configurable variables.
	@awk 'BEGIN {FS = ":.*## "; print "OlaRT developer commands:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2} END {print "\nVariables: MODEL=/path/to/artifact PROMPT=Hello TOKENS=8 RESULTS_DIR=/path/to/results ID=experiment-id SET=\"key=value ...\" ADDR=127.0.0.1:8080"}' Makefile

fmt: ## Format every Rust crate.
	$(CARGO) fmt --manifest-path $(MANIFEST) --all

fmt-check: ## Check Rust formatting without modifying files.
	$(CARGO) fmt --manifest-path $(MANIFEST) --all -- --check

check: ## Type-check all workspace targets.
	$(CARGO) check --manifest-path $(MANIFEST) --workspace --all-targets

test: ## Run the default workspace test suite.
	$(CARGO) test --manifest-path $(MANIFEST) --workspace

clippy: ## Run Clippy for all workspace targets with warnings denied.
	$(CARGO) clippy --manifest-path $(MANIFEST) --workspace --all-targets -- -D warnings

doc: ## Build workspace API documentation without dependency docs.
	$(CARGO) doc --manifest-path $(MANIFEST) --workspace --no-deps

verify: fmt-check check test clippy doc ## Run the complete CPU/default quality gate.

check-wgpu: ## Compile WGPU-enabled model and UI targets.
	$(CARGO) check --manifest-path $(MANIFEST) -p ola-burn -p ola-ui --all-targets --features wgpu

check-cuda: ## Compile native-CUDA model and UI targets (execution needs NVRTC).
	$(CARGO) check --manifest-path $(MANIFEST) -p ola-burn -p ola-ui --all-targets --features cuda

gpu-info: ## Print the NVIDIA adapter, memory, driver, and compute capability.
	nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv

ui: ## Open the CPU workbench; MODEL and RESULTS_DIR are optional.
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-ui -- $(if $(MODEL),--model "$(MODEL)",) $(if $(RESULTS_DIR),--results-dir "$(RESULTS_DIR)",)

ui-wgpu: require-model ## Open WGPU workbench; MODEL required, RESULTS_DIR optional.
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-ui --features wgpu -- --backend wgpu --model "$(MODEL)" $(if $(RESULTS_DIR),--results-dir "$(RESULTS_DIR)",)

llama: require-model ## Generate on CPU (MODEL required; PROMPT and TOKENS optional).
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-burn --example run_llama -- "$(MODEL)" "$(PROMPT)" "$(TOKENS)"

llama-wgpu: require-model ## Generate on WGPU (MODEL required; PROMPT and TOKENS optional).
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-burn --features wgpu --example run_llama -- --backend wgpu "$(MODEL)" "$(PROMPT)" "$(TOKENS)"

test-fixture: require-model ## Run ignored CPU tests against the pinned external fixture.
	OLA_LLAMA_CONFORMANCE_DIR="$(MODEL)" $(CARGO) test --manifest-path $(MANIFEST) --workspace -- --ignored

test-gpu-fixture: require-model ## Run tensor-batch and UI integrations on WGPU.
	OLA_LLAMA_CONFORMANCE_DIR="$(MODEL)" $(CARGO) test --manifest-path $(MANIFEST) -p ola-burn --features wgpu llama::factory::tests::wgpu_executor_tensor_batches_prefill_and_decode -- --ignored --exact
	OLA_LLAMA_CONFORMANCE_DIR="$(MODEL)" $(CARGO) test --manifest-path $(MANIFEST) -p ola-ui --features wgpu local::tests::local_wgpu_backend_loads_and_generates_through_the_executor -- --ignored --exact

experiments: ## List registered experiments, their kinds, and parameters.
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-cli -- experiment list

experiment: require-id ## Run experiment ID (MODEL, SET="key=value ...", RESULTS_DIR optional).
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-cli -- experiment run "$(ID)" $(if $(MODEL),--model "$(MODEL)",) $(foreach pair,$(SET),--set "$(pair)") $(if $(RESULTS_DIR),--results-dir "$(RESULTS_DIR)",)

serve: require-model ## Serve MODEL over HTTP at ADDR; Ctrl-C drains in-flight requests.
	$(CARGO) run --manifest-path $(MANIFEST) -p ola-cli -- serve --model "$(MODEL)" --addr "$(ADDR)"

require-id:
	@test -n "$(ID)" || { echo "ID=<experiment-id> is required; see make experiments" >&2; exit 2; }

require-model:
	@test -n "$(MODEL)" || { echo "MODEL=/path/to/llama-artifact is required" >&2; exit 2; }
