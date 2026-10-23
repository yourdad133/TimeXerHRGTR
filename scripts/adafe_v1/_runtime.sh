#!/usr/bin/env bash
# Run only through the configured project SSH host. No local Python fallback.
adafe_project_dir=/home/ouyangguanghong/projects/TEST/GTR-main
adafe_python=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python

adafe_check_runtime() {
  if [[ -z "${SSH_CONNECTION:-}" ]] || [[ "$(hostname)" != ps ]]; then
    echo 'AdaFE verification requires SSH to 10.21.25.109 (host ps).' >&2
    return 1
  fi
  if [[ "${SSH_CONNECTION}" != *' 10.21.25.109 22' ]]; then
    echo 'Unexpected SSH server: expected 10.21.25.109:22.' >&2
    return 1
  fi
  cd -- "$adafe_project_dir" || return 1
  [[ "$(pwd -P)" == "$adafe_project_dir" && -f run.py && -x "$adafe_python" ]] || return 1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export PATH="/home/ouyangguanghong/miniconda3/envs/tslib/bin:$PATH"
  echo "HOST=$(hostname) PROJECT=$(pwd -P) PYTHON=$adafe_python GPU=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || return 1
  "$adafe_python" -c 'import sys,torch; print("Python:",sys.executable,"CUDA:",torch.cuda.is_available(),"Devices:",torch.cuda.device_count()); assert sys.executable == sys.argv[1]; assert torch.cuda.is_available() and torch.cuda.device_count() > 0' "$adafe_python" || return 1
}
