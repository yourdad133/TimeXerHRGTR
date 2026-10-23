#!/usr/bin/env bash
# TimeXerGTR with a daily cycle table, using the shared remote runtime.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

model_name=TimeXerGTR
seed=2027

source "$(dirname "${BASH_SOURCE[0]}")/_runtime.sh"

# Prediction length: 96
timexer_check_runtime
mkdir -p experiment_logs/TimeXerGTR_manual
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_96_96_FlatDaily \
  --model $model_name \
  --data custom \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
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
  --learning_rate 0.003 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_manual/Electricity_96_${seed}_flatDaily.log"

# Prediction length: 192
timexer_check_runtime
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_96_192_FlatDaily \
  --model $model_name \
  --data custom \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
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
  --learning_rate 0.003 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_manual/Electricity_192_${seed}_flatDaily.log"

# Prediction length: 336
timexer_check_runtime
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_96_336_FlatDaily \
  --model $model_name \
  --data custom \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
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
  --learning_rate 0.003 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_manual/Electricity_336_${seed}_flatDaily.log"

# Prediction length: 720
timexer_check_runtime
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_96_720_FlatDaily \
  --model $model_name \
  --data custom \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
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
  --learning_rate 0.003 \
  --lradj TST \
  --use_revin 1 \
  --random_seed "$seed" \
  2>&1 | tee "experiment_logs/TimeXerGTR_manual/Electricity_720_${seed}_flatDaily.log"
