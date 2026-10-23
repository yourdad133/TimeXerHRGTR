#!/usr/bin/env bash
# PEMS08: GTR-guided AdaFE-STE, three seeds and four prediction lengths.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$script_dir/_runtime.sh"
export CUDA_VISIBLE_DEVICES=1


model_name=TimeXerGTR
log_dir="experiment_logs/Guide_PEMS08_dp002"
seeds=(2027 2028)
pred_lens=(12 24 48 96)

for seed in "${seeds[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    timexer_check_runtime
    mkdir -p "$log_dir"
    model_id="PEMS08_96_${pred_len}_Guide_dp002"
    log_file="$log_dir/${model_id}_seed${seed}.log"
    args=(
      --is_training 1
      --root_path ./dataset/
      --data_path PEMS08.npz
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
      --target OT
      --freq h
      --seq_len 96
      --label_len 0
      --pred_len "$pred_len"
      --enc_in 170
      --dec_in 170
      --c_out 170
      --cycle 288
      --gtr_cycle_len 288
      --gtr_gate_init 0.1
      --e_layers 2
      --factor 3
      --d_model 512
      --d_ff 2048
      --n_heads 8
      --patch_len 16
      --batch_size 32
      --num_workers 10
      --itr 1
      --dropout 0.02
      --train_epochs 30
      --patience 5
      --learning_rate 0.001
      --lradj TST
      --max_grad_norm 0
      --use_revin 1
      --des Exp
      --random_seed "$seed"
    )
    python -u scripts/baseline_gtr/run_gtr_guided_adafe.py "${args[@]}" 2>&1 | tee "$log_file"
  done
done
