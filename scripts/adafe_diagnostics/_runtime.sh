#!/usr/bin/env bash
set -euo pipefail
diagnostic_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir=/home/ouyangguanghong/projects/TEST/GTR-main
python_bin=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python
diagnostic_check_runtime() {
  [[ "$(cd -- "$diagnostic_dir/../.." && pwd -P)" == "$project_dir" ]]
  cd -- "$project_dir"
  [[ -x "$python_bin" && -f run.py ]]
  export PATH="$(dirname -- "$python_bin"):$PATH"
  echo "HOST=$(hostname) PROJECT=$(pwd -P) PYTHON=$python_bin GPU=${CUDA_VISIBLE_DEVICES:-0,1}"
  hostname -I
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  "$python_bin" -c 'import sys,torch; assert sys.executable == "/home/ouyangguanghong/miniconda3/envs/tslib/bin/python"; print("CUDA_AVAILABLE",torch.cuda.is_available()); assert torch.cuda.is_available()'
}
