#!/usr/bin/env bash
# Execute through SSH on the configured server. One isolated PEMS07 stability run.
set -euo pipefail
case " $(hostname -I) " in
  *" 10.21.25.109 "*) ;;
  *) echo 'Run this script through SSH on 10.21.25.109.' >&2; exit 1 ;;
esac
cd "$(dirname "$0")/../.."
[[ "$PWD" == /home/ouyangguanghong/projects/TEST/GTR-main ]] || exit 1
python_bin=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
seed="${SEED:-2026}"
epochs="${TRAIN_EPOCHS:-12}"
learning_rate="${LEARNING_RATE:-3e-4}"
clip_norm="${MAX_GRAD_NORM:-1}"
run_tag="${RUN_TAG:-PEMS07_stability_revin0_pl12_seed${seed}_lr${learning_rate}_clip${clip_norm}_$(date +%Y%m%d_%H%M%S)}"
[[ "$run_tag" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'RUN_TAG must be a simple directory name.' >&2; exit 1; }
report_dir="reports/pems07_stability_runs/$run_tag"
[[ ! -e "$report_dir" ]] || { echo 'Use a new RUN_TAG to preserve the existing run.' >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
"$python_bin" -c 'import os,sys,torch; assert torch.cuda.is_available(), "Remote CUDA is required"; print("Python",sys.executable,"visible GPUs",os.environ["CUDA_VISIBLE_DEVICES"],"CUDA",torch.cuda.is_available())'
mkdir -p "$report_dir" experiment_logs/pems07_stability_runs
trap 'status=$?; printf "%s\n" "$status" > "$report_dir/exit_status.txt"' EXIT
"$python_bin" -u run.py \
  --is_training 1 --model_id "$run_tag" --model TimeXerGTR \
  --root_path ./dataset/ --data_path PEMS07.npz --data PEMS --features M \
  --seq_len 96 --label_len 0 --pred_len 12 --enc_in 883 --dec_in 883 --c_out 883 \
  --gtr_cycle_len 288 --cycle 288 --e_layers 2 --factor 3 \
  --d_model 512 --d_ff 2048 --n_heads 8 --patch_len 16 \
  --batch_size 16 --itr 1 --dropout 0 --train_epochs "$epochs" --patience 5 \
  --learning_rate "$learning_rate" --lradj constant --max_grad_norm "$clip_norm" \
  --use_revin 0 --use_adafe 1 --adafe_mode ste \
  --adafe_fourier_degree 2 --adafe_poly_degree 2 --adafe_kernel 5 \
  --adafe_tau 1 --adafe_gamma_init 0.05 --adafe_scorer_lr_scale 0.1 \
  --adafe_report_dir "$report_dir" --adafe_log_interval 25 \
  --checkpoints ./checkpoints/pems07_stability_runs/ \
  --des Stability --random_seed "$seed" \
  2>&1 | tee "experiment_logs/pems07_stability_runs/$run_tag.log"
