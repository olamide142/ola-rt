#!/usr/bin/env bash
# Second GPU round for the paper, from a fresh clone on one NVIDIA GPU.
#
#   r1  soak: one `ola serve`, the same overload trace replayed REPS times
#       back to back, with a per-iteration server trace and an nvidia-smi
#       log, to find out whether serving slows down over its lifetime
#   r2  E1 again: a fresh `ola serve` per run, loads scaled and simulated
#       with the full calibrated cost
#   r2old  re-simulates the first round's E1 observations (OLD_E1) with
#       the same cost, for a side-by-side table
#   r3  E4: `ola serve` with and without `--adaptive ADAPTIVE` at moderate
#       load and overload, fresh server per run; only when ADAPTIVE is set
#
# Usage, from the repository root:
#   bash lab/gpu_round2.sh                      # setup build models r1 r2 r2old pack
#   bash lab/gpu_round2.sh r1 pack              # only these phases
#
# Environment overrides:
#   OUT        results directory          (default: ./gpu-experiments-2-results)
#   MODELS     model directory            (default: $HOME/models)
#   MAIN       main model Hugging Face id (default: HuggingFaceTB/SmolLM2-1.7B-Instruct)
#   OLA_COST   full calibrated cost       (default: the RTX 4090 e2-decode fit)
#   SEEDS      seeds for r2 and r3        (default: "1 2 3")
#   REQUESTS   requests per run           (default: 60)
#   REPS       r1 repetitions             (default: 10)
#   MAX_ACTIVE server and simulator slots (default: 8)
#   OLD_E1     first-round E1 directory   (r2old; skipped when unset)
#   ADAPTIVE   cost-calibration report    (r3; skipped when unset)
#   BACKEND    model backend              (default: cuda; cpu for a dry run)
#
# Every phase refuses to run from a dirty tree and records `git rev-parse
# HEAD`, so results always come from a committed revision.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
WS=$ROOT/ola-rt
OUT=${OUT:-$ROOT/gpu-experiments-2-results}
MODELS=${MODELS:-$HOME/models}
MAIN=${MAIN:-HuggingFaceTB/SmolLM2-1.7B-Instruct}
# fixed, per call, per real token, per padded position (padding-aware fit);
# decode per invocation (decode intercept minus fixed); decode per row
# (per row plus per context token times a mean decode context of 46.7).
export OLA_COST=${OLA_COST:-10745,30227,482,97,13047,24568}
SEEDS=${SEEDS:-1 2 3}
REQUESTS=${REQUESTS:-60}
REPS=${REPS:-10}
MAX_ACTIVE=${MAX_ACTIVE:-8}
BACKEND=${BACKEND:-cuda}
PORT=18080
OLA=$WS/target/release/ola
REPLAY=$WS/target/release/examples/replay_trace
SERVER=
SMI=

model_dir() { echo "$MODELS/$(basename "$1" | tr '[:upper:]' '[:lower:]')"; }
log() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
mkdir -p "$OUT"

cleanup() {
  [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null || true
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null || true
}
trap cleanup EXIT

clean_tree() {
  # Results directories are ignored by the check; nothing else may differ.
  local dirty
  dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no)
  [ -z "$dirty" ] || die "working tree has uncommitted changes; commit before measuring:
$dirty"
  git -C "$ROOT" rev-parse HEAD > "$1/revision.txt"
  echo "revision $(cat "$1/revision.txt")"
}

# Logs GPU clocks, temperature, power, and memory once a second.
smi_start() {
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu \
    --format=csv -l 1 > "$1" 2>&1 &
  SMI=$!
}
smi_stop() { [ -n "$SMI" ] && kill "$SMI" 2>/dev/null || true; SMI=; }

# start_server DIR [ARGS...]: starts `ola serve` with its log in DIR and
# waits until it answers; extra arguments go to `ola serve`.
start_server() {
  local dir=$1; shift
  "$OLA" serve --model "$(model_dir "$MAIN")" --backend "$BACKEND" --name m \
    --addr "127.0.0.1:$PORT" --max-active "$MAX_ACTIVE" "$@" > "$dir/serve.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > "$dir/models.json" && return
    kill -0 "$SERVER" 2>/dev/null || die "ola serve exited; see $dir/serve.log"
    sleep 2
  done
  die "ola serve did not answer in 240 s; see $dir/serve.log"
}

# Stops the server with Ctrl-C, so it drains and flushes its trace, and
# waits for the port to free.
stop_server() {
  kill -INT "$SERVER" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=
}

kv_geometry() {
  python3 -c "import json,sys; k=json.load(open(sys.argv[1]))['models'][0]['kv_cache']; \
print(f\"{k['bytes_per_token']},{k['tokens_per_block']},{k['capacity_bytes']}\")" "$1"
}

# replay RUN_DIR KV [WARMUPS]: plays RUN_DIR/trace.jsonl against the server
# and simulates the observed arrivals.
replay() {
  python3 "$ROOT/lab/replay/client.py" --url "http://127.0.0.1:$PORT" --model m \
    --trace "$1/trace.jsonl" --out "$1/observed.jsonl" --warmups "${3:-8}"
  "$REPLAY" simulate "$1/observed.jsonl" "$MAX_ACTIVE" "$2" "$1/simulated.jsonl"
}

phase_setup() {
  log "setup: packages, Rust, GPU, CUDA toolkit"
  if ! command -v clang >/dev/null && command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq clang pkg-config libssl-dev curl tmux
  fi
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
  clean_tree "$OUT"
  cd "$WS"
  cargo build --release -p ola-cli --features cuda
  cargo build --release -p ola-bench --example reviewer_eval --example replay_trace
}

phase_models() {
  log "models"
  "$ROOT/lab/lab" fetch "$MAIN" "$(model_dir "$MAIN")"
}

phase_r1() {
  log "R1: soak, $REPS back-to-back replays on one server"
  local dir=$OUT/r1
  rm -rf "$dir" && mkdir -p "$dir"
  clean_tree "$dir"
  smi_start "$dir/nvidia-smi.csv"
  start_server "$dir" --trace "$dir/server-trace.jsonl"
  local kv
  kv=$(kv_geometry "$dir/models.json")
  "$REPLAY" export overload 1 "$REQUESTS" overload "$dir/trace.jsonl"
  local reps=()
  for rep in $(seq 1 "$REPS"); do
    local run
    run=$dir/rep-$(printf '%02d' "$rep")
    mkdir -p "$run" && cp "$dir/trace.jsonl" "$run/"
    log "R1 repetition $rep"
    local start
    start=$(date +%s.%N)
    # Only the first repetition warms up; later ones measure the server as
    # the earlier repetitions left it.
    replay "$run" "$kv" "$([ "$rep" = 1 ] && echo 8 || echo 0)"
    echo "$(basename "$run") $start $(date +%s.%N)" >> "$dir/reps.txt"
    reps+=("$run")
  done
  stop_server
  smi_stop
  python3 "$ROOT/lab/replay/soak_summary.py" --trace "$dir/server-trace.jsonl" \
    --smi "$dir/nvidia-smi.csv" --reps "$dir/reps.txt" --out "$dir/soak.csv" "${reps[@]}" \
    | tee "$dir/soak.md"
  python3 "$ROOT/lab/replay/compare.py" --out "$dir/summary.csv" "${reps[@]}" | tee "$dir/summary.md"
}

phase_r2() {
  log "R2: E1 with a fresh server per run, OLA_COST=$OLA_COST"
  local dir=$OUT/r2
  rm -rf "$dir" && mkdir -p "$dir"
  clean_tree "$dir"
  echo "OLA_COST=$OLA_COST" > "$dir/OLA_COST"
  smi_start "$dir/nvidia-smi.csv"
  local runs=()
  for seed in $SEEDS; do
    # The overload trace's arrivals ignore the load setting, so it runs once.
    for spec in "priority_classes light" "priority_classes moderate" \
                "priority_classes overload" "overload overload"; do
      set -- $spec
      local run=$dir/$1-$2-$seed
      mkdir -p "$run"
      log "R2 run $1 / $2 / seed $seed"
      start_server "$run" --trace "$run/server-trace.jsonl"
      "$REPLAY" export "$1" "$seed" "$REQUESTS" "$2" "$run/trace.jsonl"
      replay "$run" "$(kv_geometry "$run/models.json")"
      stop_server
      runs+=("$run")
    done
  done
  smi_stop
  python3 "$ROOT/lab/replay/compare.py" --out "$dir/summary.csv" "${runs[@]}" | tee "$dir/summary.md"
}

phase_r2old() {
  if [ -z "${OLD_E1:-}" ]; then
    log "r2old: skipped (set OLD_E1 to the first round's e1 directory)"
    return
  fi
  log "R2: first-round E1 observations re-simulated with OLA_COST=$OLA_COST"
  local dir=$OUT/r2-old-resim
  rm -rf "$dir" && mkdir -p "$dir"
  clean_tree "$dir"
  echo "OLA_COST=$OLA_COST" > "$dir/OLA_COST"
  local kv runs=()
  kv=$(kv_geometry "$OLD_E1/models.json")
  for old in "$OLD_E1"/*/; do
    [ -f "$old/observed.jsonl" ] || continue
    local run
    run=$dir/$(basename "$old")
    mkdir -p "$run" && cp "$old/observed.jsonl" "$run/"
    "$REPLAY" simulate "$run/observed.jsonl" "$MAX_ACTIVE" "$kv" "$run/simulated.jsonl"
    runs+=("$run")
  done
  python3 "$ROOT/lab/replay/compare.py" --out "$dir/summary.csv" "${runs[@]}" | tee "$dir/summary.md"
}

phase_r3() {
  if [ -z "${ADAPTIVE:-}" ]; then
    log "r3: skipped (set ADAPTIVE to a cost-calibration report)"
    return
  fi
  log "R3: fixed versus --adaptive $ADAPTIVE"
  local dir=$OUT/r3
  rm -rf "$dir" && mkdir -p "$dir"
  clean_tree "$dir"
  cp "$ADAPTIVE" "$dir/adaptive-report.json"
  smi_start "$dir/nvidia-smi.csv"
  for seed in $SEEDS; do
    for load in moderate overload; do
      for variant in fixed adaptive; do
        local run=$dir/priority_classes-$load-$seed-$variant
        mkdir -p "$run"
        log "R3 run $load / seed $seed / $variant"
        local extra=(--trace "$run/server-trace.jsonl")
        [ "$variant" = adaptive ] && extra+=(--adaptive "$dir/adaptive-report.json")
        start_server "$run" "${extra[@]}"
        "$REPLAY" export priority_classes "$seed" "$REQUESTS" "$load" "$run/trace.jsonl"
        replay "$run" "$(kv_geometry "$run/models.json")"
        stop_server
      done
    done
  done
  smi_stop
  python3 "$ROOT/lab/replay/variants.py" --baseline fixed --candidate adaptive \
    --out "$dir/summary.csv" "$dir"/*-fixed "$dir"/*-adaptive | tee "$dir/summary.md"
  # Iteration and host-gap breakdown per run: an `observe` stall shows up
  # as a longer host gap under --adaptive.
  for run in "$dir"/*-fixed "$dir"/*-adaptive; do
    python3 "$ROOT/lab/replay/soak_summary.py" --trace "$run/server-trace.jsonl" \
      --out "$run/iterations.csv" "$run" | tail -1
  done | tee "$dir/iterations.md"
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
  [ -f "$ROOT/../GPU_EXPERIMENTS_2_SUMMARY.md" ] && cp "$ROOT/../GPU_EXPERIMENTS_2_SUMMARY.md" "$OUT/"
  [ -f "$OUT/GPU_EXPERIMENTS_2_SUMMARY.md" ] || echo "warning: GPU_EXPERIMENTS_2_SUMMARY.md is not in $OUT yet" >&2
  tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  echo "wrote $OUT.tar.gz"
}

PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(setup build models r1 r2 r2old pack)
for phase in "${PHASES[@]}"; do
  declare -F "phase_$phase" >/dev/null || die "unknown phase '$phase'"
  "phase_$phase"
done
log "done: ${PHASES[*]}"
