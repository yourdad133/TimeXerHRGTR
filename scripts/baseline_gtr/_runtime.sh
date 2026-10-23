#!/usr/bin/env bash
# Source this helper before running the explicit experiment commands.
_timexer_project_root="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P
)"

timexer_check_runtime() {
  local project_dir=/home/ouyangguanghong/projects/TEST/GTR-main
  local python_bin=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python

  case "${SSH_CONNECTION:-}" in
    *" 10.21.25.109 22") ;;
    *)
      echo "Run these scripts through SSH on 10.21.25.109." >&2
      return 1
      ;;
  esac

  if [ "$_timexer_project_root" != "$project_dir" ] || [ ! -x "$python_bin" ]; then
    echo "Run these scripts through SSH from the TEST/GTR-main project with tslib installed." >&2
    return 1
  fi

  cd -- "$project_dir" || return 1
  if [ "$(pwd -P)" != "$project_dir" ] || [ ! -f run.py ]; then
    echo "The expected TEST/GTR-main project is unavailable." >&2
    return 1
  fi

  export PATH="$(dirname -- "$python_bin"):$PATH"
  hash -r
  if [ "$(command -v python)" != "$python_bin" ]; then
    echo "python must resolve to the configured tslib interpreter." >&2
    return 1
  fi

  echo "HOST=$(hostname)"
  echo "PROJECT=$(pwd -P)"
  echo "PYTHON=$(command -v python)"
  # 按需选卡的脚本可能尚未设置 CUDA_VISIBLE_DEVICES。
  # 在 set -u 下直接引用未绑定变量会报错。
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: all GPUs visible>}"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader || return 1
  python -c 'import sys, torch; print("PYTHON_EXEC=" + sys.executable); print("TORCH=" + torch.__version__); print("CUDA_AVAILABLE=" + str(torch.cuda.is_available())); print("CUDA_COUNT=" + str(torch.cuda.device_count())); assert sys.executable == sys.argv[1], "Unexpected Python interpreter"; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, "At least one visible CUDA device is required"' "$python_bin" || return 1
}

# Traffic 数据集自检：traffic.csv 应为 17545 行（1 表头 + 17544 小时）且不能有 NUL 填充。
# 背景：该文件曾因下载截断变成 648 行有效数据 + 131MB NUL，直接训练会得到空 loader / NaN。
# 用法：timexer_check_traffic || exit 1   （需先调用 timexer_check_runtime 切到项目根）
timexer_check_traffic() {
  python - "${1:-dataset/traffic.csv}" <<'PY' || return 1
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    sys.exit('[traffic] 缺少 %s' % path)

rows = nuls = 0
with open(path, 'rb') as fh:
    for line in fh:
        rows += 1
        if b'\x00' in line:
            nuls += 1

if rows < 17545 or nuls:
    sys.exit(
        '[traffic] 数据不完整：lines=%d（应为 17545），NUL 行=%d\n'
        '          修复：curl -L -o %s '
        '"https://hf-mirror.com/datasets/thuml/Time-Series-Library'
        '/resolve/main/traffic/traffic.csv"' % (rows, nuls, path)
    )
print('[traffic] %s OK: %d 行 = %d 小时 × 862 序列' % (path, rows - 1, rows - 1))
PY
}
