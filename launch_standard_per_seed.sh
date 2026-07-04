#!/usr/bin/env bash
# Launch scalar (standard) DQN training × 5 seeds, calm LunarLander.
#
# Env knobs (override on command line):
#   SEEDS              default "0 1 2 3 4"
#   TOTAL_EPISODES     default 800
#   MAX_PARALLEL       default 3
#
# Writes to checkpoints/standard_per_seed/seed<s>/ .

set -euo pipefail

REPO="/Users/jk13942/Documents/GitHub/IP-ML"
PYTHON="${REPO}/.venv/bin/python"
TOTAL_EPISODES="${TOTAL_EPISODES:-800}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OUT_ROOT="${OUT_ROOT:-${REPO}/checkpoints/standard_per_seed}"

cd "$REPO"
mkdir -p "$OUT_ROOT"

run_one() {
  local s=$1
  local out="${OUT_ROOT}/seed${s}"
  mkdir -p "$out"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON" lunarlander_standard_dqn_per_seed.py \
      --seed "$s" --total-episodes "$TOTAL_EPISODES" \
      --output-dir "$out" > "$out/stdout.log" 2>&1
}

jobs=($SEEDS)
total=${#jobs[@]}
echo "[launch_std] $total seeds (max $MAX_PARALLEL parallel; episodes=$TOTAL_EPISODES)"

i=0
while [ $i -lt $total ]; do
  pids=()
  end=$((i + MAX_PARALLEL))
  if [ $end -gt $total ]; then end=$total; fi
  for ((j=i; j<end; j++)); do
    s="${jobs[$j]}"
    echo "[launch_std] starting seed=$s"
    run_one "$s" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  i=$end
  echo "[launch_std] batch done, $i / $total complete"
done

echo "[launch_std] all done"
