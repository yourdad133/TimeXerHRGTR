"""Checkpoint-started, fresh-Adam LR/clip sensitivity probe; train/val only.

This does not resume the original optimizer, shuffle state or OneCycle momentum.
Outputs and weights are isolated from the formal experiment.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from models.TimeXerGTR import Model
from scripts.adafe_diagnostics.common import (
    dataset_for, evaluate, forward_batch, loader_for, runtime, save_json, seed_all,
)
from utils.adafe_reporting import diagnostic_snapshot

ARMS = (("high_lr", 8.1e-4, 0.0), ("low_lr", 3e-4, 0.0),
        ("high_lr_clip1", 8.1e-4, 1.0), ("low_lr_clip1", 3e-4, 1.0))
SOURCES = ("models/TimeXerGTR.py", "models/TimeXer.py", "models/GTR.py",
           "layers/adafe_v1.py", "data_provider/data_loader.py",
           "scripts/adafe_diagnostics/common.py",
           "scripts/gtr_guided_adafe/probe_training_instability.py")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_args(log):
    line = next(line for line in log.read_text().splitlines() if line.startswith("Namespace("))
    call = ast.parse(line, mode="eval").body
    return SimpleNamespace(**{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})


def groups_for(model, lr, scale):
    scorer = list(model.adafe_adapter.adafe.score.parameters())
    ids = {id(p) for p in scorer}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return [{"params": other, "lr": lr}, {"params": scorer, "lr": lr * scale}]


def mask_probe(model, batch, device):
    capture = {}
    def hook(module, inputs, scores):
        capture["mask"] = (scores > scores.mean()).detach().cpu()
        capture["input_rms"] = float(inputs[0].detach().float().square().mean().sqrt())
        capture["input_abs_max"] = float(inputs[0].detach().float().abs().max())
    handle = model.adafe_adapter.adafe.score.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        pred, target = forward_batch(model, batch, device)
        capture["mse"] = float(F.mse_loss(pred, target))
    handle.remove()
    return capture


def run_arm(name, lr, clip, state, args, train, val, indices, val_indices, options, device):
    seed_all(args.random_seed)
    model = Model(args).to(device)
    model.load_state_dict(state, strict=True)
    optim = torch.optim.Adam(groups_for(model, lr, args.adafe_scorer_lr_scale))
    loader = loader_for(train, options.batch_size, indices, True, 20261013)
    validation = loader_for(val, options.batch_size, val_indices, False, 20261013)
    probe_batch = next(iter(loader_for(val, options.batch_size, val_indices[:options.batch_size], False, 20261013)))
    initial = mask_probe(model, probe_batch, device)
    initial_validation = evaluate(model, validation, device)
    rows, evaluations = [], [{"step": 0, **initial_validation}]
    start = time.monotonic()
    model.train()
    for step, batch in enumerate(loader, 1):
        if step > options.steps:
            break
        sample = step == 1 or step % 16 == 0 or step == options.steps
        model.adafe_adapter.adafe.capture_diagnostics = sample
        optim.zero_grad(set_to_none=True)
        pred, target = forward_batch(model, batch, device)
        loss = F.mse_loss(pred, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss in {name}, step {step}")
        loss.backward()
        diagnostics = diagnostic_snapshot(model) if sample else {}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip or float("inf"), error_if_nonfinite=True)
        row = {"step": step, "train_mse": float(loss.detach()), "grad_norm_before_clip": float(norm),
               "clipped": bool(clip and float(norm) > clip), **diagnostics}
        optim.step()
        rows.append(row)
        if sample:
            fixed = mask_probe(model, probe_batch, device)
            row.update(fixed_probe_mse=fixed["mse"], scorer_input_rms=fixed["input_rms"],
                       scorer_input_abs_max=fixed["input_abs_max"],
                       fixed_mask_flip_from_initial=float((fixed["mask"] != initial["mask"]).float().mean()))
            metrics = evaluate(model, validation, device)
            evaluations.append({"step": step, **metrics})
            print("STEP " + json.dumps({"arm": name, **row, "val": metrics}), flush=True)
            model.train()
            save_json(options.output / name / "history.json", rows)
            save_json(options.output / name / "validation.json", evaluations)
    result = {"arm": name, "lr": lr, "scorer_lr": lr * args.adafe_scorer_lr_scale,
              "adam_betas": list(optim.param_groups[0]["betas"]), "clip_norm": clip,
              "steps": len(rows), "initial_validation": initial_validation,
              "final_validation": evaluations[-1], "max_train_mse": max(row["train_mse"] for row in rows),
              "mean_train_mse": float(np.mean([row["train_mse"] for row in rows])),
              "max_grad_norm": max(row["grad_norm_before_clip"] for row in rows),
              "clip_steps": sum(row["clipped"] for row in rows),
              "seconds": time.monotonic() - start}
    save_json(options.output / name / "result.json", result)
    print("RESULT " + json.dumps(result), flush=True)
    del model, optim, pred, target, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-windows", type=int, default=128)
    options = parser.parse_args()
    if (options.output / "summary.json").exists():
        raise FileExistsError("Use a new output directory")
    device = runtime(options.output)
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    args = load_args(options.log)
    assert args.use_revin == 0 and args.use_adafe == 1 and args.adafe_mode == "ste"
    args.root_path, args.num_workers = str(ROOT / "dataset"), 0
    state = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    train, val = dataset_for(args, "train"), dataset_for(args, "val")
    indices = np.linspace(0, len(train)-1, options.steps * options.batch_size, dtype=np.int64)
    val_indices = np.linspace(0, len(val)-1, options.val_windows, dtype=np.int64)
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    summary = {"state": "running", "checkpoint_sha256": file_hash(options.checkpoint),
               "checkpoint": str(options.checkpoint), "source_hashes": source_hashes,
               "args": vars(args), "probe_args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(options).items()},
               "train_indices": indices.tolist(), "val_indices": val_indices.tolist(),
               "loader_seed": 20261013 + 173, "split": "train/val only", "results": [],
               "limitations": "Fresh Adam with betas=(0.9,0.999); same checkpoint and batches. Not original optimizer, shuffle, per-step LR or OneCycle beta1 state; no exact replay or formal benchmark."}
    save_json(options.output / "summary.json", summary)
    for name, lr, clip in ARMS:
        summary["results"].append(run_arm(name, lr, clip, state, args, train, val, indices, val_indices, options, device))
        save_json(options.output / "summary.json", summary)
    summary["source_unchanged"] = source_hashes == {name: file_hash(ROOT/name) for name in SOURCES}
    summary["state"] = "completed"
    save_json(options.output / "summary.json", summary)


if __name__ == "__main__":
    main()
