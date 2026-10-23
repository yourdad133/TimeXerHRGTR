#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="/home/ouyangguanghong/miniconda3/envs/tslib/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

model_name=TimeXerGTR
log_dir="experiment_logs/${model_name}Guide_PEMS07_M_dp005"
mkdir -p "$log_dir"
seeds=(2026)
pred_lens=(48)
for seed in "${seeds[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    model_id="PEMS07_96_${pred_len}Guide_dp005"
    log_file="$log_dir/${model_id}_seed${seed}.log"
    args=(
      --is_training 1
      --root_path ./dataset/
      --data_path PEMS07.npz
      --model_id "$model_id"
      --model "$model_name"
      --use_adafe 1
      --adafe_mode ste
      --adafe_fourier_degree 2
      --adafe_poly_degree 2
      --adafe_kernel 5
      --adafe_tau 1.0
      --adafe_gamma_init 0.05
      --adafe_constant_reparam 1
      --adafe_proxy_normalize 1
      --adafe_scorer_lr_scale 0.1
      --guided_adafe_use_prior 1
      --guided_adafe_prior_gate_init 0.05
      --data PEMS
      --features M
      --seq_len 96
      --label_len 0
      --pred_len "$pred_len"
      --enc_in 883
      --dec_in 883
      --c_out 883
      --gtr_cycle_len 288
      --cycle 288
      --gtr_gate_init 0.1
      --e_layers 2
      --factor 3
      --d_model 512
      --d_ff 1024
      --n_heads 8
      --patch_len 16
      --batch_size 16
      --num_workers 10
      --itr 1
      --dropout 0.05
      --train_epochs 21
      --patience 5
      --learning_rate 0.0003
      --lradj TST
      --max_grad_norm 0
      --use_revin 0
      --gpu 0
      --des Exp
      --random_seed "$seed"
    )
    python -u scripts/baseline_gtr/run_gtr_guided_adafe.py "${args[@]}" 2>&1 | tee "$log_file"
  done
done
