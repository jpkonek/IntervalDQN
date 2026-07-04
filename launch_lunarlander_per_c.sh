#!/usr/bin/env bash
# Launch interval DQN training per (c, seed) on the original LunarLander
# (no-perturbation single-environment) setup. Used to test whether the
# high-c overestimation pathology observed on collision and on
# credal-LunarLander also appears on the simpler benchmark.
#
# Env knobs:
#   C_VALUES        default "0.0 0.2 0.5 0.8 1.0"
#   SEEDS           default "0 1 2 3 4"
#   TOTAL_EPISODES  default 800 (matches the original LunarLander setup)
#   MAX_PARALLEL    default 5

set -euo pipefail

ROOT="/Users/jk13942/Documents/GitHub/IP-ML"
PYTHON="$ROOT/.venv/bin/python"
TOTAL_EPISODES="${TOTAL_EPISODES:-800}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
C_VALUES="${C_VALUES:-0.0 0.2 0.5 0.8 1.0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OUT_ROOT="$ROOT/checkpoints/per_c"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

run_one() {
  local c=$1
  local s=$2
  local out="$OUT_ROOT/c${c}_seed${s}"
  mkdir -p "$out"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON" lunarlander_interval_dqn_per_c.py \
      --seed "$s" --total-episodes "$TOTAL_EPISODES" \
      --c-train "$c" \
      --output-dir "$out" > "$out/stdout.log" 2>&1
}

jobs=()
for c in $C_VALUES; do
  for s in $SEEDS; do
    jobs+=("$c $s")
  done
done
total=${#jobs[@]}
echo "[launch] $total jobs (max $MAX_PARALLEL parallel)"

i=0
while [ $i -lt $total ]; do
  pids=()
  end=$((i + MAX_PARALLEL))
  if [ $end -gt $total ]; then end=$total; fi
  for ((j=i; j<end; j++)); do
    read -r c s <<< "${jobs[$j]}"
    echo "[launch] starting c=$c seed=$s"
    run_one "$c" "$s" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  i=$end
  echo "[launch] batch done, $i / $total complete"
done

echo "[launch] all done"
