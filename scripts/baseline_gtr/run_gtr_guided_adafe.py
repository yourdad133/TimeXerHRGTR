"""Train the existing GTR-guided AdaFE through the standard experiment loop.

The TimeXerGTR routing key is retained for its five-input forward signature
and cycle-aware data loader. Only this entry point builds the guided model.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from exp.exp_main import Exp_Main
from models.TimeXerGTRGuidedAdaFE import Model as GuidedModel
from scripts.adafe_diagnostics.common import parser_from_run, seed_all
from utils.adafe_reporting import adafe_setting_suffix


class GuidedExperiment(Exp_Main):
    def _acquire_device(self):
        # CUDA_VISIBLE_DEVICES is selected by the shell script; use its local index.
        device = torch.device("cuda", self.args.gpu)
        print(f"Use GPU: {device} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<all>')})")
        return device

    def _build_model(self):
        return GuidedModel(self.args).float()


def parse_args(arguments=None):
    parser = parser_from_run()
    parser.description = "Full training and test of TimeXerGTR with GTR-guided AdaFE"
    parser.add_argument("--guided_adafe_use_prior", type=int, choices=[0, 1], default=1)
    parser.add_argument("--guided_adafe_prior_gate_init", type=float, default=0.05)
    parser.add_argument("--check_config", action="store_true",
                        help="validate and print the configuration without starting training")
    args = parser.parse_args(arguments)
    if args.model != "TimeXerGTR" or args.use_adafe != 1:
        parser.error("This entry requires --model TimeXerGTR --use_adafe 1")
    if args.use_multi_gpu or not args.use_gpu:
        parser.error("Use one visible remote CUDA GPU per training process")
    if args.itr < 1:
        parser.error("--itr must be positive")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm < 0:
        parser.error("--max_grad_norm must be finite and nonnegative")
    if not np.isfinite(args.adafe_scorer_lr_scale) or args.adafe_scorer_lr_scale <= 0:
        parser.error("--adafe_scorer_lr_scale must be finite and positive")
    if (not np.isfinite(args.guided_adafe_prior_gate_init)
            or abs(args.guided_adafe_prior_gate_init) >= 1):
        parser.error("--guided_adafe_prior_gate_init must lie strictly within (-1, 1)")
    return args


def check_remote_runtime(args):
    project = Path("/home/ouyangguanghong/projects/TEST/GTR-main")
    python = "/home/ouyangguanghong/miniconda3/envs/tslib/bin/python"
    if ROOT != project or Path.cwd().resolve() != project or sys.executable != python:
        raise RuntimeError("Run through SSH in the configured project with tslib Python")
    connection = os.environ.get("SSH_CONNECTION", "").split()
    if len(connection) != 4 or connection[-2:] != ["10.21.25.109", "22"]:
        raise RuntimeError("The configured SSH connection to 10.21.25.109 is required")
    print(f"HOST={subprocess.check_output(['hostname'], text=True).strip()}")
    print(f"PROJECT={ROOT} PYTHON={sys.executable}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                    "--format=csv,noheader"], check=True)
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError("The selected remote CUDA GPU is unavailable")
    print(f"CUDA_AVAILABLE=True CUDA_COUNT={torch.cuda.device_count()}", flush=True)


def experiment_setting(args, iteration=0):
    setting = (f"{args.model_id}_{args.model}_{args.data}_ft{args.features}"
               f"_sl{args.seq_len}_pl{args.pred_len}_cycle{args.cycle}_seed{args.random_seed}")
    setting += adafe_setting_suffix(args)
    setting += (f"_revin{args.use_revin}_guide{args.guided_adafe_use_prior}"
                f"_pg{args.guided_adafe_prior_gate_init:g}")
    if args.itr > 1:
        setting += f"_itr{iteration}"
    return setting


def main():
    args = parse_args()
    check_remote_runtime(args)
    print("Effective model: models.TimeXerGTRGuidedAdaFE.Model")
    print("Args in experiment:")
    print(args)
    if args.check_config:
        print(json.dumps({"setting": experiment_setting(args), "args": vars(args)}))
        return
    seed_all(args.random_seed)
    if args.is_training:
        for iteration in range(args.itr):
            setting = experiment_setting(args, iteration)
            experiment = GuidedExperiment(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>", flush=True)
            experiment.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", flush=True)
            experiment.test(setting)
            if args.do_predict:
                experiment.predict(setting, True)
            del experiment
            torch.cuda.empty_cache()
    else:
        setting = experiment_setting(args)
        experiment = GuidedExperiment(args)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", flush=True)
        experiment.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
