#!/usr/bin/env bash
# Launch credal-LunarLander training per c value × seed.
#
# Env knobs:
#   C_VALUES           default "0.0 0.2 0.5 0.8 1.0"
#   SEEDS              default "0 1 2 3 4"
#   TOTAL_EPISODES     default 800 (matches LunarLander interval DQN baseline)
#   MAX_PARALLEL       default 3 (low because the d_h=64 collision rerun is
#                                 also using 6 cores; tune as needed)
#   WIND_LO / WIND_HI  credal range (default 0.0 / 15.0)
#   RESULTS_SUBDIR     default "per_c"

set -euo pipefail

LL_DIR="/Users/jk13942/Documents/GitHub/IP-ML/Credal LunarLander"
PYTHON="/Users/jk13942/Documents/GitHub/IP-ML/.venv/bin/python"
TOTAL_EPISODES="${TOTAL_EPISODES:-800}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
C_VALUES="${C_VALUES:-0.0 0.2 0.5 0.8 1.0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
WIND_LO="${WIND_LO:-0.0}"
WIND_HI="${WIND_HI:-15.0}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-per_c}"

cd "$LL_DIR"
mkdir -p "results/${RESULTS_SUBDIR}"

run_one() {
  local c=$1
  local s=$2
  local out="results/${RESULTS_SUBDIR}/c${c}_seed${s}"
  mkdir -p "$out"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON" train_credal_lunarlander.py \
      --seed "$s" --total-episodes "$TOTAL_EPISODES" \
      --c-train "$c" \
      --wind-lo "$WIND_LO" --wind-hi "$WIND_HI" \
      --output-dir "$out" > "$out/stdout.log" 2>&1
}

jobs=()
for c in $C_VALUES; do
  for s in $SEEDS; do
    jobs+=("$c $s")
  done
done
total=${#jobs[@]}
echo "[launch_ll] $total jobs (max $MAX_PARALLEL parallel; wind ∈ [$WIND_LO, $WIND_HI])"

i=0
while [ $i -lt $total ]; do
  pids=()
  end=$((i + MAX_PARALLEL))
  if [ $end -gt $total ]; then end=$total; fi
  for ((j=i; j<end; j++)); do
    read -r c s <<< "${jobs[$j]}"
    echo "[launch_ll] starting c=$c seed=$s"
    run_one "$c" "$s" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  i=$end
  echo "[launch_ll] batch done, $i / $total complete"
done

echo "[launch_ll] all done"
