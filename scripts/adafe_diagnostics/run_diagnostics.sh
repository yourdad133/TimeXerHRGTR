#!/usr/bin/env bash
# All new scripts and artifacts are isolated in adafe_diagnostics directories.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/_runtime.sh"
run_dir="${1:-reports/adafe_diagnostics/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES=1
diagnostic_check_runtime
mkdir -p "$run_dir"
run_dir="$(cd -- "$run_dir" && pwd -P)"
cp -- "$script_dir/default_plan.json" "$run_dir/plan.json"
printf '{"state":"running","pid":%s}\n' "$$" > "$run_dir/controller.json"
trap 'code=$?; printf "{\"state\":\"finished\",\"exit_code\":%s,\"pid\":%s}\n" "$code" "$$" > "$run_dir/controller.json"' EXIT
"$python_bin" -u "$script_dir/test_diagnostics.py" > "$run_dir/contracts.log" 2>&1
diagnostic_check_runtime
"$python_bin" -u "$script_dir/run_suite.py" audit --output "$run_dir/audit" > "$run_dir/audit.log" 2>&1
diagnostic_check_runtime
"$python_bin" -u "$script_dir/checkpoint_probe.py" --output "$run_dir/checkpoint_probes" > "$run_dir/checkpoint_probes.log" 2>&1
CUDA_VISIBLE_DEVICES=0 diagnostic_check_runtime > "$run_dir/preflight_gpu0.log"
CUDA_VISIBLE_DEVICES=1 diagnostic_check_runtime > "$run_dir/preflight_gpu1.log"
CUDA_VISIBLE_DEVICES=0 nice -n 10 "$python_bin" -u "$script_dir/run_suite.py" train --output "$run_dir/training" --plan "$run_dir/plan.json" --shard-index 0 --shard-count 2 > "$run_dir/worker0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 nice -n 10 "$python_bin" -u "$script_dir/run_suite.py" train --output "$run_dir/training" --plan "$run_dir/plan.json" --shard-index 1 --shard-count 2 > "$run_dir/worker1.log" 2>&1 &
pid1=$!
printf '{"gpu0_worker":%s,"gpu1_worker":%s}\n' "$pid0" "$pid1" > "$run_dir/workers.json"
code0=0
code1=0
wait "$pid0" || code0=$?
wait "$pid1" || code1=$?
diagnostic_check_runtime > "$run_dir/pre_summary.log"
"$python_bin" "$script_dir/summarize.py" --output "$run_dir"
[[ "$code0" == 0 && "$code1" == 0 ]]
