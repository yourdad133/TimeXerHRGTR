#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/_runtime.sh"
run_dir="${1:?Provide the existing diagnostic output directory}"
export CUDA_VISIBLE_DEVICES=0
diagnostic_check_runtime
run_dir="$(cd -- "$run_dir" && pwd -P)"
[[ -f "$run_dir/scorer_lr_plan.json" ]]
mkdir -p "$run_dir/training_scorer_lr"
printf '{"state":"running","pid":%s}\n' "$$" > "$run_dir/scorer_lr_controller.json"
trap 'code=$?; printf "{\"state\":\"finished\",\"exit_code\":%s,\"pid\":%s}\n" "$code" "$$" > "$run_dir/scorer_lr_controller.json"' EXIT
CUDA_VISIBLE_DEVICES=0 diagnostic_check_runtime > "$run_dir/scorer_lr_gpu0_preflight.log"
CUDA_VISIBLE_DEVICES=1 diagnostic_check_runtime > "$run_dir/scorer_lr_gpu1_preflight.log"
CUDA_VISIBLE_DEVICES=0 nice -n 10 "$python_bin" -u "$script_dir/run_suite.py" train --output "$run_dir/training_scorer_lr" --plan "$run_dir/scorer_lr_plan.json" --shard-index 0 --shard-count 2 > "$run_dir/scorer_lr_worker0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 nice -n 10 "$python_bin" -u "$script_dir/run_suite.py" train --output "$run_dir/training_scorer_lr" --plan "$run_dir/scorer_lr_plan.json" --shard-index 1 --shard-count 2 > "$run_dir/scorer_lr_worker1.log" 2>&1 &
pid1=$!
printf '{"gpu0_worker":%s,"gpu1_worker":%s}\n' "$pid0" "$pid1" > "$run_dir/scorer_lr_workers.json"
code0=0
code1=0
wait "$pid0" || code0=$?
wait "$pid1" || code1=$?
[[ "$code0" == 0 && "$code1" == 0 ]]
