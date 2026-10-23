#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../adafe_diagnostics/_runtime.sh"
output=${1:?usage: run_pems08_full.sh reports/adafe_fixes/new-pems08-run}
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$project_dir"
diagnostic_check_runtime
mkdir -p "$output"
for gpu in 0 1; do
  if [[ -f "$output/worker${gpu}.pid" ]] && kill -0 "$(cat "$output/worker${gpu}.pid")" 2>/dev/null; then
    echo "Worker $gpu is already running" >&2
    exit 1
  fi
done
pids=()
for gpu in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    diagnostic_check_runtime
    exec nice -n 10 "$python_bin" -u scripts/adafe_fixes/pems08_full.py train \
      --output "$output" --plan scripts/adafe_fixes/pems08_full_plan_20261011.json --shard "$gpu"
  ) > "$output/worker${gpu}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$output/worker${gpu}.pid"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" == 0 ]]; then
  export CUDA_VISIBLE_DEVICES=1
  diagnostic_check_runtime
  "$python_bin" scripts/adafe_fixes/pems08_full.py collect \
    --output "$output" --plan scripts/adafe_fixes/pems08_full_plan_20261011.json \
    > "$output/collect.log" 2>&1
fi
printf 'PEMS08_FULL_EXIT=%s\n' "$failed"
exit "$failed"
