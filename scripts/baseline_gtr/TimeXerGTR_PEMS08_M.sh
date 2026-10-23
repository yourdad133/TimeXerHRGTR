#!/usr/bin/env bash
# PEMS08 multivariate forecasting: 96 input steps, 12/24/48/96 output steps.
# This script shares all common settings with the companion model script.

set -euo pipefail

# 本脚本在 scripts/baseline_gtr/ 下，仓库根是上两级（run.py 在根目录）。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../.." && pwd -P)"
cd "$project_root"
export PATH="/home/ouyangguanghong/miniconda3/envs/tslib/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1

model_name=TimeXerGTR
log_dir="experiment_logs/${model_name}_PEMS08"
mkdir -p "$log_dir"

seeds=(2026)
pred_lens=(12)

for seed in "${seeds[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    model_id="PEMS08_96_$pred_len"
    log_file="$log_dir/${model_id}_seed$seed.log"
    args=(
      --is_training 1
      --root_path ./dataset/
      --data_path PEMS08.npz
      --model_id "$model_id"
      --model "$model_name"
      --data PEMS
      --features M
      --seq_len 96
      --label_len 0
      --pred_len "$pred_len"
      --enc_in 170
      --dec_in 170
      --c_out 170
      --cycle 288
      --e_layers 2
      --factor 3
      --d_model 512
      --d_ff 2048
      --n_heads 8
      --patch_len 16
      --batch_size 32
      --itr 1
      --dropout 0
      --train_epochs 30
      --patience 5
      --learning_rate 0.001
      --lradj TST
      --use_revin 1
      --des Exp
      --random_seed "$seed"
    )
    args+=(--gtr_cycle_len 288 --gtr_gate_init 0.1)
    python -u run.py "${args[@]}" 2>&1 | tee "$log_file"
  done
done
