#!/usr/bin/env bash
# Electricity: TimeXerGTR + AdaFE-ste, 96 -> 96/192/336/720.
# Three matched seeds and four horizons: 12 runs, preserving current settings.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=TimeXerGTR
seeds=(2026 2027 2028)
pred_lens=(96 192 336 720)

source "$(dirname "${BASH_SOURCE[0]}")/_runtime.sh"

timexer_check_runtime
log_dir="experiment_logs/TimeXerGTR_AdaFE_ste_Electricity"
mkdir -p "$log_dir"

for seed in "${seeds[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    timexer_check_runtime
    model_id="Electricity_96_${pred_len}_FlatDaily_AdaFEste"
    log_file="$log_dir/Electricity_${pred_len}_${seed}_flatDaily_AdaFEste.log"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id "$model_id" \
      --model $model_name \
      --use_adafe 1 \
      --adafe_mode ste \
      --adafe_fourier_degree 2 \
      --adafe_poly_degree 2 \
      --adafe_kernel 5 \
      --adafe_tau 1.0 \
      --adafe_gamma_init 0.05 \
      --data custom \
      --features M \
      --target OT \
      --freq h \
      --seq_len 96 \
      --label_len 0 \
      --pred_len "$pred_len" \
      --e_layers 2 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 2048 \
      --n_heads 8 \
      --patch_len 16 \
      --batch_size 32 \
      --num_workers 4 \
      --itr 1 \
      --cycle 168 \
      --gtr_cycle_len 24 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 5 \
      --learning_rate 0.001 \
      --lradj TST \
      --use_revin 1 \
      --random_seed "$seed" \
      2>&1 | tee "$log_file"
  done
done
