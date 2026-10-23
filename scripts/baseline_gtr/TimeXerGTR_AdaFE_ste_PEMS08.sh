#!/usr/bin/env bash
# PEMS08: TimeXerGTR + AdaFE-Hard, 96 -> 12/24/48/96.
# Same explicit command layout as TimeXerGTR_Electricity.sh (seed 2027).
# Data/model/training settings match TimeXerGTR_PEMS08_M.sh for seed 2027.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=TimeXerGTR
seed=2028

source "$(dirname "${BASH_SOURCE[0]}")/_runtime.sh"

# Prediction length: 12
timexer_check_runtime
mkdir -p experiment_logs/TimeXerGTR_AdaFE_ste_PEMS08
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_12_AdaFESte \
  --model $model_name \
  --use_adafe 1 \
  --adafe_mode ste \
  --adafe_fourier_degree 2 \
  --adafe_poly_degree 2 \
  --adafe_kernel 5 \
  --adafe_tau 1.0 \
  --adafe_gamma_init 0.05 \
  --data PEMS \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --patch_len 16 \
  --batch_size 32 \
  --num_workers 10 \
  --itr 1 \
  --cycle 288 \
  --gtr_cycle_len 288 \
  --gtr_gate_init 0.1 \
  --dropout 0 \
  --train_epochs 30 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_AdaFE_ste_PEMS08/PEMS08_96_12_seed${seed}_AdaFESte.log"

# Prediction length: 24
timexer_check_runtime
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_24_AdaFEste \
  --model $model_name \
  --use_adafe 1 \
  --adafe_mode ste \
  --adafe_fourier_degree 2 \
  --adafe_poly_degree 2 \
  --adafe_kernel 5 \
  --adafe_tau 1.0 \
  --adafe_gamma_init 0.05 \
  --data PEMS \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --patch_len 16 \
  --batch_size 32 \
  --num_workers 10 \
  --itr 1 \
  --cycle 288 \
  --gtr_cycle_len 288 \
  --gtr_gate_init 0.1 \
  --dropout 0 \
  --train_epochs 30 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_AdaFE_ste_PEMS08/PEMS08_96_24_seed${seed}_AdaFESte.log"

# Prediction length: 48
timexer_check_runtime
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_48_AdaFEste \
  --model $model_name \
  --use_adafe 1 \
  --adafe_mode ste \
  --adafe_fourier_degree 2 \
  --adafe_poly_degree 2 \
  --adafe_kernel 5 \
  --adafe_tau 1.0 \
  --adafe_gamma_init 0.05 \
  --data PEMS \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --patch_len 16 \
  --batch_size 32 \
  --num_workers 10 \
  --itr 1 \
  --cycle 288 \
  --gtr_cycle_len 288 \
  --gtr_gate_init 0.1 \
  --dropout 0 \
  --train_epochs 30 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_AdaFE_ste_PEMS08/PEMS08_96_48_seed${seed}_AdaFESte.log"

# Prediction length: 96
timexer_check_runtime
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_96_AdaFEste \
  --model $model_name \
  --use_adafe 1 \
  --adafe_mode ste \
  --adafe_fourier_degree 2 \
  --adafe_poly_degree 2 \
  --adafe_kernel 5 \
  --adafe_tau 1.0 \
  --adafe_gamma_init 0.05 \
  --data PEMS \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --patch_len 16 \
  --batch_size 32 \
  --num_workers 10 \
  --itr 1 \
  --cycle 288 \
  --gtr_cycle_len 288 \
  --gtr_gate_init 0.1 \
  --dropout 0 \
  --train_epochs 30 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_AdaFE_ste_PEMS08/PEMS08_96_96_seed${seed}_AdaFESte.log"
