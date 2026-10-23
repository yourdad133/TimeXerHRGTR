#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PATH="/home/ouyangguanghong/miniconda3/envs/tslib/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer
log_dir="experiment_logs/${model_name}_PEMS04_M"
mkdir -p "$log_dir"

seeds=(2026 2027 2028)
pred_lens=(12 24 48 96)

for seed in "${seeds[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    model_id="PEMS04_96_$pred_len"
    log_file="$log_dir/${model_id}_seed$seed.log"
    args=(
      --is_training 1
      --root_path ./dataset/
      --data_path PEMS04.npz
      --model_id "$model_id"
      --model "$model_name"
      --data PEMS
      --features M
      --seq_len 96
      --label_len 0
      --pred_len "$pred_len"
      --enc_in 307
      --dec_in 307
      --c_out 307
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
    python -u run.py "${args[@]}" 2>&1 | tee "$log_file"
  done
done
