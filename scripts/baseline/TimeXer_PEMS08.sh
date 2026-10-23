#!/usr/bin/env bash
# Rollout of the validated high-lr recipe on PEMS08 TimeXer:
#   C1-like base + lr 1e-3 + OneCycle (TST).
# pl48 was already validated by probe P2: mse 0.1541 / mae 0.2463,
# beating the reference paper's 0.158 / 0.248 (and C1's 0.1943 by -20.7%).
# This script fills in the remaining lengths: 12 / 24 / 96.
set -euo pipefail

# 本脚本在 scripts/baseline/ 下，仓库根是上两级（run.py 在根目录）。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../.." && pwd -P)"
cd "$project_root"
export PATH="/home/ouyangguanghong/miniconda3/envs/tslib/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer
log_dir="experiment_logs/TimeXer_PEMS08"
mkdir -p "$log_dir"

for pred_len in 12 24 48 96; do
  echo "===== TimeXer pl${pred_len}  (lr=1e-3, lradj=TST) ====="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path PEMS08.npz \
    --model_id "PEMS08_96_${pred_len}_TimeXer_lr1e3TST" \
    --model "$model_name" \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len "$pred_len" \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --cycle 288 \
    --e_layers 2 \
    --factor 3 \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads 8 \
    --patch_len 16 \
    --batch_size 32 \
    --itr 1 \
    --dropout 0 \
    --train_epochs 30 \
    --patience 5 \
    --learning_rate 0.001 \
    --lradj TST \
    --use_revin 1 \
    --des Exp \
    --random_seed 2028 \
    2>&1 | tee "$log_dir/PEMS08_96_${pred_len}_seed2028.log"
done

echo "===== ROLLOUT DONE ====="
