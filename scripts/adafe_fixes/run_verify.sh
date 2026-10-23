#!/usr/bin/env bash
# Run only on the configured SSH host, using tslib and physical GPUs 0 and 1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../adafe_diagnostics/_runtime.sh"
output=${1:?usage: run_verify.sh reports/adafe_fixes/unique-run}
export CUDA_VISIBLE_DEVICES=0,1
diagnostic_check_runtime
mkdir -p "$output"
if [[ -e "$output/worker0_finished.json" || -e "$output/worker1_finished.json" ]]; then
  echo 'A completed worker already exists; choose a new output directory.' >&2
  exit 1
fi
export PYTHONPATH="$project_dir"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$python_bin" -m unittest discover -s tests -p 'test_adafe*.py' -v > "$output/all_contracts.log" 2>&1
"$python_bin" scripts/adafe_diagnostics/test_diagnostics.py > "$output/diagnostic_contracts.log" 2>&1
pids=()
for gpu in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    diagnostic_check_runtime
    exec nice -n 10 "$python_bin" -u scripts/adafe_fixes/compare.py \
      --output "$output" --plan scripts/adafe_fixes/plan.json --shard "$gpu"
  ) > "$output/worker${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
printf 'VERIFICATION_EXIT=%s\n' "$failed"
exit "$failed"
