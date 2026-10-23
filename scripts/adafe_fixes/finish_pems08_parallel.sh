#!/usr/bin/env bash
set -euo pipefail
source scripts/adafe_diagnostics/_runtime.sh
output=$1
plan=scripts/adafe_fixes/pems08_parallel_finish_plan_20261011.json
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$project_dir"
diagnostic_check_runtime
read -r old_pid < "$output/worker0.pid"
read -r idle_pid < "$output/worker1.pid"
[[ "$old_pid" == 1668239 && "$idle_pid" == 1668240 ]]
old_start=$(awk '{print $22}' "/proc/$old_pid/stat")
old_command=$(tr '\0' ' ' < "/proc/$old_pid/cmdline")
[[ "$old_command" == *"pems08_full.py train"* && "$old_command" == *"$output"* && "$old_command" == *"--shard 0"* ]]
boundary="$output/runs/PEMS08_96_24_seed2027_fixed_slow"
paused=0
finish() {
  code=$?
  if [[ "$paused" == 1 ]] && kill -0 "$old_pid" 2>/dev/null; then kill -CONT "$old_pid"; fi
  printf 'PEMS08_FULL_RESUME_EXIT=%s\n' "$code"
}
trap finish EXIT
deadline=$((SECONDS + 7200))
while true; do
  if [[ -f "$boundary/status.json" ]]; then
    state=$(<"$boundary/status.json")
    if [[ "$state" == *'"state": "completed"'* ]]; then break; fi
    if [[ "$state" == *'"state": "failed"'* ]]; then echo 'Current case failed; no scheduling change'; exit 1; fi
  fi
  kill -0 "$old_pid"
  [[ "$(awk '{print $22}' "/proc/$old_pid/stat")" == "$old_start" ]]
  (( SECONDS < deadline ))
  sleep 2
done
# Interrupt only our next, untested attempt after the current case has completed.
kill -STOP "$old_pid"
paused=1
[[ -f "$boundary/result.json" ]]
[[ ! -f "$output/runs/PEMS08_96_24_seed2028_legacy_ste/result.json" ]]
[[ ! -f "$output/runs/PEMS08_96_24_seed2028_fixed_ste/result.json" ]]
[[ "$(awk '{print $22}' "/proc/$old_pid/stat")" == "$old_start" ]]
kill -TERM "$old_pid"
kill -CONT "$old_pid"
paused=0
for attempt in {1..30}; do
  if ! kill -0 "$old_pid" 2>/dev/null && ! kill -0 "$idle_pid" 2>/dev/null; then break; fi
  sleep 1
done
! kill -0 "$old_pid" 2>/dev/null
! kill -0 "$idle_pid" 2>/dev/null
mkdir -p "$output/execution_notes"
if [[ -d "$output/runs/PEMS08_96_24_seed2028_legacy_ste" ]]; then
  mv "$output/runs/PEMS08_96_24_seed2028_legacy_ste" "$output/execution_notes/untested_interrupted_legacy2028"
fi
date -Is > "$output/execution_notes/parallel_resume_time.txt"
printf 'Completed seed2027 fixed_slow before rescheduling. Original worker0=%s; worker1=%s. Untested final pair now runs in parallel.\n' "$old_pid" "$idle_pid" >> "$output/execution_notes/parallel_resume_time.txt"
pids=()
for gpu in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    diagnostic_check_runtime
    exec nice -n 10 "$python_bin" -u scripts/adafe_fixes/pems08_full.py train --output "$output" --plan "$plan" --shard "$gpu"
  ) > "$output/worker"$gpu"_resume.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$output/worker"$gpu".pid"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" != 0 ]]; then exit 1; fi
export CUDA_VISIBLE_DEVICES=1
diagnostic_check_runtime
"$python_bin" scripts/adafe_fixes/pems08_full.py collect --output "$output" --plan "$plan" > "$output/collect.log" 2>&1
