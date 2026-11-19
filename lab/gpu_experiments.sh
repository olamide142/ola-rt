#!/usr/bin/env bash
# GPU experiments E1–E3 for the paper, from a fresh clone on one NVIDIA GPU.
#
#   E2  padding-aware cost calibration (cost-calibration, mixed_lengths=1)
#   E1  sim-versus-real validation: replay traces against `ola serve` and
#       the simulator, then compare TTFT and completion times
#   E3  llama-prefill-strategies on seeded prompt-length sets
#
# E2 runs first because E1 simulates with its fitted cost.
#
# Usage, from the repository root:
#   bash lab/gpu_experiments.sh                 # every phase, in order
#   bash lab/gpu_experiments.sh e1 e3 pack      # only these phases
#
# Environment overrides:
#   OUT        results directory          (default: ./gpu-experiments-results)
#   MODELS     model directory            (default: $HOME/models)
#   MAIN       main model Hugging Face id (default: HuggingFaceTB/SmolLM2-1.7B-Instruct)
#   SEEDS      E1 seeds                   (default: "1 2 3")
#   REQUESTS   requests per E1 run        (default: 60)
#   MAX_ACTIVE server and simulator slots (default: 8)
#   BACKEND    model backend              (default: cuda; cpu for a dry run)
#
# Every phase refuses to run from a dirty tree and records `git rev-parse
# HEAD`, so results always come from a committed revision.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
WS=$ROOT/ola-rt
OUT=${OUT:-$ROOT/gpu-experiments-results}
MODELS=${MODELS:-$HOME/models}
MAIN=${MAIN:-HuggingFaceTB/SmolLM2-1.7B-Instruct}
SEEDS=${SEEDS:-1 2 3}
REQUESTS=${REQUESTS:-60}
MAX_ACTIVE=${MAX_ACTIVE:-8}
BACKEND=${BACKEND:-cuda}
PORT=18080
OLA=$WS/target/release/ola
EVAL=$WS/target/release/examples/reviewer_eval
REPLAY=$WS/target/release/examples/replay_trace

model_dir() { echo "$MODELS/$(basename "$1" | tr '[:upper:]' '[:lower:]')"; }
log() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
mkdir -p "$OUT"

clean_tree() {
  # Results directories are ignored by the check; nothing else may differ.
  local dirty
  dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no)
  [ -z "$dirty" ] || die "working tree has uncommitted changes; commit before measuring:
$dirty"
  git -C "$ROOT" rev-parse HEAD > "$OUT/revision.txt"
  echo "revision $(cat "$OUT/revision.txt")"
}

phase_setup() {
  log "setup: Rust, GPU, CUDA toolkit"
  if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
  fi
  cargo --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || die "no GPU visible"
  command -v nvcc >/dev/null || die "nvcc not found: use a CUDA devel image (NVRTC is required)"
  python3 - <<'EOF' || die "the CUDA driver API cannot initialize (cuInit failed); try another host"
import ctypes, sys
sys.exit(ctypes.CDLL("libcuda.so.1").cuInit(0))
EOF
  echo "cuInit: ok"
}

phase_build() {
  log "build (CUDA)"
  clean_tree
  cd "$WS"
  cargo build --release -p ola-cli --features cuda
  cargo build --release -p ola-bench --example reviewer_eval --example replay_trace
}

phase_models() {
  log "models"
  "$ROOT/lab/lab" fetch "$MAIN" "$(model_dir "$MAIN")"
}

phase_e2() {
  log "E2: padding-aware cost calibration"
  clean_tree
  local dir=$OUT/e2
  rm -rf "$dir" && mkdir -p "$dir"
  "$OLA" experiment run cost-calibration --model "$(model_dir "$MAIN")" --backend "$BACKEND" \
    --set widths="1,2,4,8,16" --set lengths="16,32,64,128,192" \
    --set warmups=2 --set repetitions=10 --set mixed_lengths=1 \
    --results-dir "$dir" | tee "$dir/summary.txt"
  python3 - "$dir" <<'EOF'
import glob, json, os, sys
dir = sys.argv[1]
report = json.load(open(sorted(glob.glob(os.path.join(dir, "*.json")))[-1]))
cost = report["details"]["pooled_fit"]["cost_model"]
line = ",".join(str(cost[k]) for k in ("fixed_micros", "per_invocation_micros", "per_dense_token_micros"))
open(os.path.join(dir, "OLA_COST"), "w").write(line + "\n")
print(f"OLA_COST={line}")
EOF
}

phase_e1() {
  log "E1: sim-versus-real replay"
  clean_tree
  [ -f "$OUT/e2/OLA_COST" ] || die "run e2 first: E1 simulates with its fitted cost"
  export OLA_COST
  OLA_COST=$(cat "$OUT/e2/OLA_COST")
  local dir=$OUT/e1
  rm -rf "$dir" && mkdir -p "$dir"
  "$OLA" serve --model "$(model_dir "$MAIN")" --backend "$BACKEND" --name m \
    --addr "127.0.0.1:$PORT" --max-active "$MAX_ACTIVE" > "$dir/serve.log" 2>&1 &
  local server=$!
  trap 'kill $server 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > "$dir/models.json" && break
    kill -0 $server 2>/dev/null || die "ola serve exited; see $dir/serve.log"
    sleep 2
  done
  local kv
  kv=$(python3 -c "import json,sys; k=json.load(open(sys.argv[1]))['models'][0]['kv_cache']; \
print(f\"{k['bytes_per_token']},{k['tokens_per_block']},{k['capacity_bytes']}\")" "$dir/models.json")
  echo "server KV geometry: $kv"
  local runs=()
  for seed in $SEEDS; do
    # The overload trace's arrivals ignore the load setting, so it runs once.
    for spec in "priority_classes light" "priority_classes moderate" \
                "priority_classes overload" "overload overload"; do
      set -- $spec
      local run=$dir/$1-$2-$seed
      mkdir -p "$run"
      log "E1 run $1 / $2 / seed $seed"
      "$REPLAY" export "$1" "$seed" "$REQUESTS" "$2" "$run/trace.jsonl"
      python3 "$ROOT/lab/replay/client.py" --url "http://127.0.0.1:$PORT" --model m \
        --trace "$run/trace.jsonl" --out "$run/observed.jsonl"
      "$REPLAY" simulate "$run/observed.jsonl" "$MAX_ACTIVE" "$kv" "$run/simulated.jsonl"
      runs+=("$run")
    done
  done
  kill $server 2>/dev/null || true
  trap - EXIT
  python3 "$ROOT/lab/replay/compare.py" --out "$dir/summary.csv" "${runs[@]}" | tee "$dir/summary.md"
}

phase_e3() {
  log "E3: prefill grouping on seeded prompt-length sets"
  clean_tree
  local dir=$OUT/e3
  rm -rf "$dir" && mkdir -p "$dir"
  "$EVAL" prefill_sets "$dir/sets.csv"
  # Simulated win rates under the calibrated cost, when E2 has run.
  if [ -f "$OUT/e2/OLA_COST" ]; then
    OLA_COST=$(cat "$OUT/e2/OLA_COST") "$EVAL" bucketing "$dir/bucketing.csv"
  fi
  local index=0
  tail -n +2 "$dir/sets.csv" | while IFS=, read -r distribution size rest; do
    local lengths=${rest//\"/}
    index=$((index + 1))
    log "E3 set $index: $distribution, $size prompts"
    "$OLA" experiment run llama-prefill-strategies --model "$(model_dir "$MAIN")" --backend "$BACKEND" \
      --set prompt_lengths="$lengths" --set max_new_tokens=2 \
      --set calibration_warmups=2 --set calibration_trials=8 --set warmups=2 --set trials=10 \
      --set output_check=lengths \
      --results-dir "$dir/$distribution-$size" | tee "$dir/$distribution-$size.txt"
  done
  local simulated=()
  [ -f "$dir/bucketing.csv" ] && simulated=(--simulated "$dir/bucketing.csv")
  python3 "$ROOT/lab/prefill_summary.py" --sets "$dir/sets.csv" --out "$dir/summary.csv" \
    "${simulated[@]}" "$dir"/*/*.json | tee "$dir/summary.md"
}

phase_pack() {
  log "pack"
  {
    echo "date: $(date -u +%FT%TZ)"
    echo "revision: $(git -C "$ROOT" rev-parse HEAD)"
    cargo --version; rustc --version
    nvcc --version | tail -1
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
    lscpu | grep -E 'Model name|^CPU\(s\)'
    free -g | head -2
  } > "$OUT/machine.txt" 2>&1 || true
  nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1 || true
  tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  echo "wrote $OUT.tar.gz"
}

PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(setup build models e2 e1 e3 pack)
for phase in "${PHASES[@]}"; do
  declare -F "phase_$phase" >/dev/null || die "unknown phase '$phase'"
  "phase_$phase"
done
log "done: ${PHASES[*]}"
