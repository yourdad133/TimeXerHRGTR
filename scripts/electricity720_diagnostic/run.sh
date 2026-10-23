#!/usr/bin/env bash
# Reproduce only Electricity 96 -> 720, retaining the original 30-epoch schedule.
set -euo pipefail
cd /home/ouyangguanghong/projects/TEST/GTR-main
export CUDA_VISIBLE_DEVICES=0
export DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-reports/electricity720_c168_ff1024_s2026_diag_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$DIAGNOSTIC_DIR"
/home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u scripts/electricity720_diagnostic/run_diagnostic.py \
  --is_training 1 \
  --root_path ./dataset/ --data_path electricity.csv \
  --model_id "$(basename "$DIAGNOSTIC_DIR")" --model TimeXerGTR \
  --data custom --features M --target OT --freq h \
  --seq_len 96 --label_len 0 --pred_len 720 \
  --enc_in 321 --dec_in 321 --c_out 321 \
  --cycle 168 --gtr_cycle_len 168 --use_revin 1 \
  --d_model 512 --d_ff 1024 --e_layers 2 --n_heads 8 --factor 3 --patch_len 16 \
  --dropout 0.05 --batch_size 32 --num_workers 4 --itr 1 \
  --use_adafe 1 --adafe_mode ste \
  --adafe_fourier_degree 2 --adafe_poly_degree 2 --adafe_kernel 5 \
  --adafe_tau 1.0 --adafe_gamma_init 0.05 \
  --adafe_constant_reparam 1 --adafe_proxy_normalize 1 --adafe_scorer_lr_scale 0.1 \
  --guided_adafe_use_prior 1 --guided_adafe_prior_gate_init 0.05 \
  --train_epochs 30 --patience 8 --learning_rate 0.0005 --lradj TST --pct_start 0.1 \
  --random_seed 2026 --des Exp \
  2>&1 | tee "$DIAGNOSTIC_DIR/train.log"
