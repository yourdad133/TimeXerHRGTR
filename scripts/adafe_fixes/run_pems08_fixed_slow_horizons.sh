#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../adafe_diagnostics/_runtime.sh"
output="$1"
plan=scripts/adafe_fixes/pems08_fixed_slow_horizons_plan_20261012.json
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$project_dir"
diagnostic_check_runtime
mkdir -p -- "$output"
exec 9>"$output/controller.lock"
if ! flock -n 9; then
  echo "Another controller already owns this output directory" >&2
  exit 1
fi
"$python_bin" scripts/adafe_fixes/pems08_fixed_slow_horizons_collect.py \
  --output "$output" --plan "$plan" --validate-only
for gpu in 0 1; do
  pid_file="$output/worker$gpu.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Worker $gpu is already alive; refusing duplicate launch" >&2
    exit 1
  fi
done
git diff --exit-code -- layers models data_provider exp utils run.py scripts/adafe_diagnostics \
  scripts/adafe_fixes/compare.py scripts/adafe_fixes/pems08_full.py \
  scripts/adafe_fixes/result_index.py \
  scripts/adafe_fixes/pems08_fixed_slow_horizons_plan_20261012.json \
  scripts/adafe_fixes/pems08_fixed_slow_horizons_collect.py \
  scripts/adafe_fixes/run_pems08_fixed_slow_horizons.sh
if [[ -f "$output/plan.json" ]]; then
  cmp -- "$plan" "$output/plan.json"
else
  cp -- "$plan" "$output/plan.json"
fi
plan="$output/plan.json"
git rev-parse HEAD > "$output/source_commit.txt"
sha256sum dataset/PEMS08.npz > "$output/dataset.sha256"
mkdir -p -- "$output/code_snapshot"
git archive HEAD layers models data_provider exp utils run.py scripts/adafe_diagnostics \
  scripts/adafe_fixes/compare.py scripts/adafe_fixes/pems08_full.py \
  scripts/adafe_fixes/result_index.py \
  scripts/adafe_fixes/pems08_fixed_slow_horizons_plan_20261012.json \
  scripts/adafe_fixes/pems08_fixed_slow_horizons_collect.py \
  scripts/adafe_fixes/run_pems08_fixed_slow_horizons.sh \
  | tar -x -C "$output/code_snapshot"
(
  export CUDA_VISIBLE_DEVICES=0
  diagnostic_check_runtime
  exec nice -n 10 "$python_bin" -u scripts/adafe_fixes/pems08_full.py train \
    --output "$output" --plan "$plan" --shard 0
) > "$output/worker0.log" 2>&1 &
pid0=$!
printf '%s\n' "$pid0" > "$output/worker0.pid"
(
  export CUDA_VISIBLE_DEVICES=1
  diagnostic_check_runtime
  exec nice -n 10 "$python_bin" -u scripts/adafe_fixes/pems08_full.py train \
    --output "$output" --plan "$plan" --shard 1
) > "$output/worker1.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid1" > "$output/worker1.pid"
echo "WORKERS_STARTED gpu0=$pid0 gpu1=$pid1"
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" == 0 ]]; then
  export CUDA_VISIBLE_DEVICES=1
  diagnostic_check_runtime
  sha256sum -c "$output/dataset.sha256"
  "$python_bin" -u scripts/adafe_fixes/pems08_fixed_slow_horizons_collect.py \
    --output "$output" --plan "$plan" > "$output/collect.log" 2>&1
  find "$output/runs" -name best.pth -print0 | sort -z \
    | xargs -0 sha256sum > "$output/checkpoints.sha256"
fi
printf 'PEMS08_FIXED_SLOW_HORIZONS_EXIT=%s\n' "$failed"
exit "$failed"
