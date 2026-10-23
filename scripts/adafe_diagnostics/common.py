"""Shared contracts for isolated AdaFE diagnostics."""
import argparse
import ast
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_provider.data_factory import data_dict
from models.TimeXerGTR import Model

@dataclass(frozen=True)
class Recipe:
    filename: str
    loader: str
    channels: int
    cycle: int
    freq: str = "h"
    horizon: int = 96

RECIPES = {
    "ETTh1": Recipe("ETTh1.csv", "ETTh1", 7, 24),
    "ETTh2": Recipe("ETTh2.csv", "ETTh2", 7, 24),
    "ETTm1": Recipe("ETTm1.csv", "ETTm1", 7, 96, "t"),
    "ETTm2": Recipe("ETTm2.csv", "ETTm2", 7, 96, "t"),
    "Weather": Recipe("weather.csv", "custom", 21, 144, "t"),
    "Electricity": Recipe("electricity.csv", "custom", 321, 24),
    "Traffic": Recipe("traffic.csv", "custom", 862, 24),
    "PEMS03": Recipe("PEMS03.npz", "PEMS", 358, 288, "h", 24),
    "PEMS04": Recipe("PEMS04.npz", "PEMS", 307, 288, "h", 24),
    "PEMS07": Recipe("PEMS07.npz", "PEMS", 883, 288, "h", 24),
    "PEMS08": Recipe("PEMS08.npz", "PEMS", 170, 288, "h", 24),
}

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)

def runtime(output):
    if ROOT != Path("/home/ouyangguanghong/projects/TEST/GTR-main"):
        raise RuntimeError("The configured remote project is required")
    expected = "/home/ouyangguanghong/miniconda3/envs/tslib/bin/python"
    if sys.executable != expected:
        raise RuntimeError("The configured remote tslib Python is required")
    addresses = subprocess.check_output(["hostname", "-I"], text=True)
    if "10.21.25.109" not in addresses.split():
        raise RuntimeError("Unexpected remote host")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA is unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    record = dict(host=socket.gethostname(), addresses=addresses.split(), project=str(ROOT),
                  python=sys.executable, torch=torch.__version__, cuda=True,
                  visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES", ""), gpu_inventory=gpu)
    save_json(Path(output) / "runtime.json", record)
    print("RUNTIME " + json.dumps(record, ensure_ascii=False), flush=True)
    return torch.device("cuda:0")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parser_from_run():
    tree = ast.parse((ROOT / "run.py").read_text())
    statements = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets):
            statements.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "parser" and func.attr == "add_argument":
                statements.append(node)
    scope = {"argparse": argparse}
    exec(compile(ast.Module(body=statements, type_ignores=[]), "run_parser", "exec"), scope)
    return scope["parser"]

def config(dataset, seed, batch=32, width=512, epochs=30, patience=10, enabled=1):
    recipe = RECIPES[dataset]
    cli = ["--is_training", "1", "--model_id", "AdaFEDiagnostics", "--model", "TimeXerGTR",
           "--data", recipe.loader, "--root_path", str(ROOT / "dataset"),
           "--data_path", recipe.filename, "--features", "M", "--freq", recipe.freq,
           "--enc_in", str(recipe.channels), "--dec_in", str(recipe.channels), "--c_out", str(recipe.channels),
           "--seq_len", "96", "--label_len", "0", "--pred_len", str(recipe.horizon),
           "--cycle", str(recipe.cycle), "--gtr_cycle_len", str(recipe.cycle),
           "--d_model", str(width), "--d_ff", str(4 * width), "--n_heads", "8", "--e_layers", "2",
           "--patch_len", "16", "--dropout", "0", "--use_revin", "1", "--batch_size", str(batch),
           "--learning_rate", "0.001", "--lradj", "TST", "--train_epochs", str(epochs),
           "--patience", str(patience), "--num_workers", "0", "--random_seed", str(seed),
           "--use_adafe", str(enabled), "--adafe_mode", "ste", "--adafe_tau", "1",
           "--adafe_gamma_init", "0.05", "--adafe_constant_reparam", "0",
           "--adafe_proxy_normalize", "0", "--adafe_scorer_lr_scale", "1"]
    return parser_from_run().parse_args(cli)

def dataset_for(args, flag):
    return data_dict[args.data](root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        target=args.target, timeenc=1, freq=args.freq, cycle=args.gtr_cycle_len)

def window_indices(length, limit):
    if length <= 0:
        raise ValueError("Empty dataset")
    if not limit or limit >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, limit, dtype=np.int64)

def loader_for(dataset, batch, indices, train, seed):
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch,
        shuffle=train, drop_last=train, num_workers=0,
        generator=torch.Generator().manual_seed(seed + 173), pin_memory=True)

def forward_batch(model, batch, device):
    x, y, xm, ym, cycle = batch[:5]
    x, y, xm, ym = (tensor.float().to(device, non_blocking=True) for tensor in (x, y, xm, ym))
    cycle = cycle.int().to(device, non_blocking=True)
    pred_len = model.pred_len
    decoder = torch.zeros_like(y[:, -pred_len:, :])
    prediction = model(x, xm, decoder, ym, cycle)
    return prediction, y[:, -pred_len:, :]

def evaluate(model, loader, device):
    model.eval()
    squared = absolute = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            prediction, target = forward_batch(model, batch, device)
            error = prediction - target
            squared += error.double().square().sum().item()
            absolute += error.double().abs().sum().item()
            count += error.numel()
    if not count:
        raise RuntimeError("Empty evaluation loader")
    return {"mse": squared / count, "mae": absolute / count, "target_values": count}

def initial_hash(model, adafe=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith("adafe_adapter.") != adafe:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
