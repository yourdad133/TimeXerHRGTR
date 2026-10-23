"""Paired train/validation screening for GTR-guided AdaFE; SSH/tslib CUDA only.

The default 1,024/256-window, six-epoch run is a pilot, not a final benchmark.
No test dataset is constructed or evaluated. Use a new output directory to rerun.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from models.TimeXerGTR import Model as BaselineModel
from models.TimeXerGTRGuidedAdaFE import Model as GuidedModel
from scripts.adafe_diagnostics.common import (
    config, dataset_for, evaluate, forward_batch, loader_for, runtime,
    save_json, seed_all, window_indices,
)

ARMS = (
    ("gtr_only_norm1", False, False, 1),
    ("fixed_adafe_norm1", True, False, 1),
    ("fixed_adafe_norm0", True, False, 0),
    ("guided_without_prior_norm1", True, True, 1),
    ("guided_with_prior_norm1", True, True, 1),
)
SOURCE_FILES = (
    "models/TimeXerGTR.py", "models/TimeXerGTRGuidedAdaFE.py",
    "models/TimeXer.py", "models/GTR.py", "layers/adafe_v1.py",
    "layers/gtr_guided_adafe.py", "data_provider/data_loader.py",
    "scripts/adafe_diagnostics/common.py", "scripts/gtr_guided_adafe/run_pilot.py",
)
BASE_LR = 3e-4
SCORER_LR_SCALE = 0.1
CLIP_NORM = 1.0
MEMORY_FRACTION = 0.8


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def tensor_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def make_args(options, arm, enabled, use_norm):
    args = config(options.dataset, options.seed, batch=options.batch_size,
                  width=options.width, epochs=options.epochs, enabled=int(enabled))
    args.model_id = arm
    args.pred_len = options.pred_len
    args.factor = 3
    args.use_revin = use_norm
    args.adafe_constant_reparam = 1
    args.adafe_proxy_normalize = 1
    args.adafe_scorer_lr_scale = SCORER_LR_SCALE
    args.learning_rate = BASE_LR
    args.lradj = "constant"
    args.guided_adafe_use_prior = arm == "guided_with_prior_norm1"
    args.guided_adafe_prior_gate_init = 0.05
    return args


def capture(model, enabled):
    adapter = getattr(model, "adafe_adapter", None)
    local = getattr(adapter, "adafe", None)
    if local is not None:
        local.capture_diagnostics = enabled


def scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().float().mean().cpu().item()
    value = float(value)
    return value if math.isfinite(value) else None


def diagnostic_snapshot(model):
    row = {"gtr_gamma": scalar(getattr(model, "gamma", None))}
    adapter = getattr(model, "adafe_adapter", None)
    local = getattr(adapter, "adafe", None)
    if local is None:
        return row
    row["adafe_gamma"] = scalar(getattr(adapter, "gamma", None))
    for name in ("last_active_ratio", "last_score_std", "last_proxy_scale",
                 "last_proxy_saturated_fraction", "last_guide_ratio",
                 "last_prior_ratio", "last_prior_to_score_ratio"):
        row[name] = scalar(getattr(local, name, None))
    prior_gate = getattr(local, "prior_gate", None)
    row["prior_gate_effective"] = scalar(prior_gate.tanh()) if prior_gate is not None else None
    return row


def gradient_group(name):
    if name.startswith("adafe_adapter.adafe.score."):
        return "scorer"
    if name == "gamma":
        return "gtr_gamma"
    if name == "adafe_adapter.gamma":
        return "adafe_gamma"
    if name.startswith("adafe_adapter.adafe.prior_"):
        return "guide"
    if name.startswith("adafe_adapter.adafe.score_norm."):
        return "score_norm"
    return "backbone_and_other"


def check_gradients(model, collect=False):
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("No gradients were produced")
    finite = torch.stack([torch.isfinite(gradient).all() for _, gradient in gradients]).all()
    if not bool(finite.item()):
        bad = [name for name, gradient in gradients if not bool(torch.isfinite(gradient).all().item())]
        raise FloatingPointError("Non-finite gradients: " + ", ".join(bad))
    if getattr(model, "use_adafe", False):
        for name, parameter in model.adafe_adapter.adafe.score.named_parameters():
            if parameter.requires_grad and parameter.grad is None:
                raise RuntimeError("Disconnected STE scorer: " + name)
    if getattr(model, "guided_adafe_use_prior", False):
        for name, parameter in model.adafe_adapter.adafe.named_parameters():
            if name.startswith("prior_") and parameter.requires_grad and parameter.grad is None:
                raise RuntimeError("Disconnected periodic guide: " + name)
    if not collect:
        return {}
    sums, counts = {}, {}
    for name, gradient in gradients:
        group = gradient_group(name)
        value = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        sums[group] = sums.get(group, 0.0) + value * value
        counts[group] = counts.get(group, 0) + 1
    return {group: {"norm": math.sqrt(sums.get(group, 0.0)),
                    "tensors_with_grad": counts.get(group, 0)}
            for group in ("scorer", "gtr_gamma", "adafe_gamma", "guide",
                          "score_norm", "backbone_and_other")}


def make_optimizer(model):
    scorer = []
    if getattr(model, "use_adafe", False):
        scorer = [parameter for parameter in model.adafe_adapter.adafe.score.parameters()
                  if parameter.requires_grad]
    scorer_ids = {id(parameter) for parameter in scorer}
    other = [parameter for parameter in model.parameters()
             if parameter.requires_grad and id(parameter) not in scorer_ids]
    groups = [{"params": other, "lr": BASE_LR, "group_name": "other"}]
    if scorer:
        groups.append({"params": scorer, "lr": BASE_LR * SCORER_LR_SCALE,
                       "group_name": "scorer"})
    return torch.optim.Adam(groups)


def train_arm(options, spec, directory, device, base_state, adapter_state,
              datasets, indices, expected_hash, expected_adapter_hash, hashes, run_record):
    arm, enabled, guided, use_norm = spec
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"state": "building", "arm": arm, "pid": os.getpid(),
              "started_unix": time.time()}
    save_json(directory / "status.json", status)
    model = optimizer = prediction = target = loss = None
    try:
        args = make_args(options, arm, enabled, use_norm)
        seed_all(options.seed)
        model = GuidedModel(args) if guided else BaselineModel(args)
        state = model.state_dict()
        for name, value in base_state.items():
            if name not in state or state[name].shape != value.shape:
                raise AssertionError("Backbone/GTR key mismatch: " + name)
            state[name] = value.clone()
        if enabled:
            for name, value in adapter_state.items():
                if name not in state or state[name].shape != value.shape:
                    raise AssertionError("Common AdaFE key mismatch: " + name)
                state[name] = value.clone()
        model.load_state_dict(state, strict=True)
        del state
        actual_base_hash = tensor_hash({name: model.state_dict()[name] for name in base_state})
        if actual_base_hash != expected_hash:
            raise AssertionError("Paired backbone/GTR initialization differs")
        adapter_hash = None
        if enabled:
            adapter_hash = tensor_hash({name: model.state_dict()[name] for name in adapter_state})
            if adapter_hash != expected_adapter_hash:
                raise AssertionError("Paired common AdaFE initialization differs")
        guide_hash = tensor_hash({name: value for name, value in model.state_dict().items()
                                 if name.startswith("adafe_adapter.adafe.prior_")
                                 or name.startswith("adafe_adapter.adafe.score_norm.")}) if guided else None
        model = model.to(device)
        optimizer = make_optimizer(model)
        loaders = {split: loader_for(datasets[split], options.batch_size, indices[split],
                                     split == "train", options.seed)
                   for split in ("train", "val")}
        if not len(loaders["train"]):
            raise ValueError("The training subset must contain at least one full batch")
        configuration = {
            "args": vars(args), "arm": arm, "model_class": type(model).__module__,
            "pilot_only": True, "full_train_validation": options.full,
            "test_used_for_selection": False, "test_evaluations": 0,
            "requested_windows": {"train": options.train_windows, "val": options.val_windows},
            "dataset_windows": {split: len(value) for split, value in datasets.items()},
            "selected_windows": {split: len(value) for split, value in indices.items()},
            "train_windows_per_epoch": len(loaders["train"]) * options.batch_size,
            "train_drop_last": True, "validation_drop_last": False,
            "validation_shuffle": False, "index_hashes": {
                split: hashlib.sha256(value.tobytes()).hexdigest() for split, value in indices.items()},
            "initial_backbone_gtr_hash": actual_base_hash,
            "initial_common_adafe_hash": adapter_hash, "initial_guide_hash": guide_hash,
            "shuffle_seed": options.seed + 173, "forward_rng_seed": options.seed + 2341,
            "optimizer": "Adam", "constant_lr": BASE_LR,
            "scorer_lr_scale": SCORER_LR_SCALE, "gradient_clip_norm": CLIP_NORM,
            "selection": "minimum validation MSE", "metric_space": "standardized",
            "source_hashes": hashes,
        }
        save_json(directory / "configuration.json", configuration)
        save_json(directory / "runtime.json", run_record)
        save_json(directory / "history.json", [])
        best, best_epoch, history = float("inf"), 0, []
        # Reset forward RNG after construction and state loading. Loader randomness
        # has an independent, identical generator in every arm.
        seed_all(options.seed + 2341)
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, options.epochs + 1):
            epoch_started = time.monotonic()
            model.train()
            squared_total, target_count = 0.0, 0
            sampled_gradients = []
            for step, batch in enumerate(loaders["train"], 1):
                sampled = step == 1 or step % options.log_interval == 0 or step == len(loaders["train"])
                capture(model, sampled)
                optimizer.zero_grad(set_to_none=True)
                prediction, target = forward_batch(model, batch, device)
                loss = F.mse_loss(prediction, target)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                norms = check_gradients(model, collect=sampled)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM,
                                                           error_if_nonfinite=True)
                if sampled:
                    sampled_gradients.append({"step": step, "pre_clip_global_norm": scalar(total_norm),
                                              "groups": norms, "diagnostics": diagnostic_snapshot(model)})
                optimizer.step()
                squared_total += float(loss.detach().item()) * target.numel()
                target_count += target.numel()
                if sampled:
                    status.update(state="training", epoch=epoch, step=step,
                                  steps_per_epoch=len(loaders["train"]), heartbeat_unix=time.time(),
                                  elapsed_seconds=time.monotonic() - started)
                    save_json(directory / "status.json", status)
                    if options.full:
                        print(f"STEP {arm} epoch={epoch} {step}/{len(loaders['train'])} "
                              f"loss={loss.item():.7f}", flush=True)
            capture(model, True)
            validation = evaluate(model, loaders["val"], device)
            if not all(math.isfinite(validation[key]) for key in ("mse", "mae")):
                raise FloatingPointError("Non-finite validation metrics")
            row = {"epoch": epoch, "train_mse": squared_total / target_count,
                   "validation": validation, "diagnostics": diagnostic_snapshot(model),
                   "sampled_train_gradients": sampled_gradients,
                   "learning_rates": [group["lr"] for group in optimizer.param_groups],
                   "epoch_seconds": time.monotonic() - epoch_started}
            history.append(row)
            if validation["mse"] < best:
                best, best_epoch = validation["mse"], epoch
                torch.save(model.state_dict(), directory / "best.pth")
            save_json(directory / "history.json", history)
            status.update(state="training", epoch=epoch, best_epoch=best_epoch,
                          best_validation_mse=best, heartbeat_unix=time.time(),
                          elapsed_seconds=time.monotonic() - started)
            save_json(directory / "status.json", status)
            print(f"EPOCH {arm} {epoch}/{options.epochs} train={row['train_mse']:.7f} "
                  f"val={validation['mse']:.7f} best={best:.7f}", flush=True)
        if source_hashes() != hashes:
            raise RuntimeError("Source changed during the paired pilot")
        # Report the recorded best validation epoch; there is no test evaluation.
        best_row = history[best_epoch - 1]
        result = {
            "arm": arm, "state": "completed", "pilot_only": True,
            "dataset": options.dataset, "seed": options.seed, "pred_len": options.pred_len,
            "best_epoch": best_epoch, "epochs_completed": len(history),
            "validation": best_row["validation"], "best_epoch_diagnostics": best_row["diagnostics"],
            "test_evaluations": 0, "full_train_validation": options.full,
            "selected_windows": configuration["selected_windows"],
            "initial_backbone_gtr_hash": actual_base_hash,
            "initial_common_adafe_hash": adapter_hash, "initial_guide_hash": guide_hash,
            "index_hashes": configuration["index_hashes"],
            "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "elapsed_seconds": time.monotonic() - started,
        }
        save_json(directory / "result.json", result)
        status.update(state="completed", best_epoch=best_epoch, best_validation_mse=best,
                      elapsed_seconds=time.monotonic() - started)
        save_json(directory / "status.json", status)
        return result
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        save_json(directory / "status.json", status)
        raise
    finally:
        del model, optimizer, prediction, target, loss
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS07"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-windows", type=int, default=1024)
    parser.add_argument("--val-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--arms", nargs="+", choices=[spec[0] for spec in ARMS],
                        help="Run only these arms, retaining the declared order")
    parser.add_argument("--full", action="store_true",
                        help="Use every train/validation window; still no test evaluation")
    options = parser.parse_args()
    for name in ("epochs", "train_windows", "val_windows", "batch_size", "width",
                 "pred_len", "log_interval"):
        if getattr(options, name) <= 0:
            parser.error(name.replace("_", "-") + " must be positive")
    if options.width % 8:
        parser.error("width must be divisible by the eight attention heads")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("this isolated pilot requires CUDA_VISIBLE_DEVICES=0")
    selected_arms = [spec for spec in ARMS if options.arms is None or spec[0] in options.arms]
    output = options.output if options.output.is_absolute() else ROOT / options.output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the configured remote project")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be new or empty; completed or partial runs are never overwritten")
    # common.runtime rejects local execution and checks remote project, host,
    # tslib Python, nvidia-smi and torch.cuda.is_available before CUDA work.
    device = runtime(output / "runtime")
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    run_record = json.loads((output / "runtime" / "runtime.json").read_text())
    run_record["per_process_memory_fraction"] = MEMORY_FRACTION
    save_json(output / "runtime" / "runtime.json", run_record)
    plan = {"options": {**vars(options), "output": str(output)},
            "arms": [spec[0] for spec in selected_arms], "pilot_only": True,
            "test_evaluations": 0, "constant_learning_rate": BASE_LR,
            "scorer_lr_scale": SCORER_LR_SCALE, "gradient_clip_norm": CLIP_NORM,
            "source_hashes": source_hashes(),
            "interpretation": "Paired validation screening; subset and six epochs are not final performance",
            "comparisons": {
                "fixed_vs_gtr": "Adds production AdaFE under the same normalization",
                "norm0_vs_norm1": "Changes only use_revin for production AdaFE",
                "guided_prior_vs_no_prior": "Changes only the periodic-prior enable switch",
                "guided_no_prior_vs_fixed": "Combined token-local threshold/scale and score-input normalization",
            }}
    save_json(output / "plan.json", plan)
    status = {"state": "building", "pid": os.getpid(), "started_unix": time.time()}
    save_json(output / "status.json", status)
    try:
        common_args = make_args(options, "fixed_adafe_norm1", True, 1)
        seed_all(options.seed)
        canonical = BaselineModel(common_args)
        canonical_state = cpu_state(canonical)
        del canonical
        base_state = {name: value for name, value in canonical_state.items()
                      if not name.startswith("adafe_adapter.")}
        adapter_state = {name: value for name, value in canonical_state.items()
                         if name.startswith("adafe_adapter.")}
        expected_hash, expected_adapter_hash = tensor_hash(base_state), tensor_hash(adapter_state)
        datasets = {split: dataset_for(common_args, split) for split in ("train", "val")}
        indices = {split: window_indices(len(datasets[split]), 0 if options.full else limit)
                   for split, limit in (("train", options.train_windows), ("val", options.val_windows))}
        np.savez(output / "selected_indices.npz", **indices)
        results = []
        for spec in selected_arms:
            if source_hashes() != plan["source_hashes"]:
                raise RuntimeError("Source changed before an arm")
            status.update(state="training", active_arm=spec[0], heartbeat_unix=time.time())
            save_json(output / "status.json", status)
            row = train_arm(options, spec, output / "runs" / spec[0], device,
                            base_state, adapter_state, datasets, indices, expected_hash,
                            expected_adapter_hash, plan["source_hashes"], run_record)
            results.append(row)
            save_json(output / "results.json", results)
        if len({row["initial_backbone_gtr_hash"] for row in results}) != 1:
            raise AssertionError("Paired backbone/GTR hashes differ")
        guided = [row for row in results if row["initial_guide_hash"] is not None]
        if guided and len({row["initial_guide_hash"] for row in guided}) != 1:
            raise AssertionError("Prior-on/off initial conditioning parameters differ")
        baseline = next((row for row in results if row["arm"] == "gtr_only_norm1"), None)
        baseline_mse = baseline["validation"]["mse"] if baseline else None
        for row in results:
            row["validation_mse_change_vs_gtr_pct"] = (
                100.0 * (row["validation"]["mse"] / baseline_mse - 1.0)
                if baseline_mse is not None and baseline_mse > 0 else None)
        summary = {"state": "completed", "pilot_only": True, "test_evaluations": 0,
                   "results": results, "paired_backbone_gtr_hash": expected_hash,
                   "selected_windows": {split: len(value) for split, value in indices.items()},
                   "full_train_validation": options.full,
                   "interpretation": plan["interpretation"]}
        save_json(output / "results.json", results)
        save_json(output / "summary.json", summary)
        status.update(state="completed", active_arm=None, heartbeat_unix=time.time())
        save_json(output / "status.json", status)
        print("COMPLETED PILOT " + json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), heartbeat_unix=time.time())
        save_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()


