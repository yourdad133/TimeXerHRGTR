"""Frozen Electricity 96->96 Guide diagnostics. Run through remote SSH/tslib."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from models.TimeXerGTRGuidedAdaFE import Model
from scripts.adafe_diagnostics.common import (
    dataset_for, forward_batch, loader_for, runtime, save_json, seed_all,
)

DEFAULT_LOG = "experiment_logs/Guide_Electricity/Electricity_96_2026Guide_dff1024.log"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_project(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Diagnostic inputs and outputs must remain inside this project.")
    return path


def ratio(numerator, denominator):
    return math.sqrt(numerator / max(denominator, 1e-30))


def parse_log(payload):
    text = payload.decode("utf-8")
    call = ast.parse(next(line for line in text.splitlines()
                          if line.startswith("Namespace(")), mode="eval").body
    args = argparse.Namespace(**{item.arg: ast.literal_eval(item.value)
                                 for item in call.keywords})
    if (args.data_path != "electricity.csv" or args.seq_len != 96
            or args.pred_len != 96 or args.features != "M"
            or not args.guided_adafe_use_prior or args.use_amp):
        raise ValueError("This probe expects the guide-enabled, FP32 Electricity M 96->96 run.")
    if "Effective model: models.TimeXerGTRGuidedAdaFE.Model" not in text:
        raise ValueError("The supplied log does not identify the Guided model.")
    setting = next(line.split("start training :", 1)[1].strip().rstrip(">").strip()
                   for line in text.splitlines() if "start training :" in line)
    epochs = [(int(m[1]), float(m[2])) for m in re.finditer(
        r"Epoch: (\d+), Steps: \d+ \| Train Loss: [\d.]+ Vali Loss: ([\d.]+)", text)]
    best_epoch, best_val = min(epochs, key=lambda row: row[1])
    val_windows = int(re.search(r"^val (\d+)$", text, re.MULTILINE)[1])
    return args, setting, best_epoch, best_val, val_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--output", default=None)
    options = parser.parse_args()
    ssh = os.environ.get("SSH_CONNECTION", "").split()
    if len(ssh) != 4 or ssh[2] != "10.21.25.109":
        raise RuntimeError("Run this diagnostic through SSH on 10.21.25.109.")
    output = inside_project(options.output or (
        "reports/electricity96_guide_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = runtime(output)
    log_path = inside_project(options.log)
    log_bytes = log_path.read_bytes()
    args, setting, best_epoch, best_val, expected_windows = parse_log(log_bytes)
    recorded_args = vars(args).copy()
    args.root_path = str(inside_project(args.root_path))
    args.num_workers = 0
    checkpoint = inside_project(Path(args.checkpoints) / setting / "checkpoint.pth")
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha = digest_bytes(checkpoint_bytes)
    dataset_path = inside_project(Path(args.root_path) / args.data_path)
    data_sha = digest_file(dataset_path)
    source_names = [
        "models/TimeXerGTRGuidedAdaFE.py", "models/TimeXerGTR.py",
        "models/TimeXer.py", "models/GTR.py", "layers/gtr_guided_adafe.py",
        "layers/adafe_v1.py", "data_provider/data_factory.py",
        "data_provider/data_loader.py", "scripts/adafe_diagnostics/common.py",
        str(Path(__file__).relative_to(ROOT)),
    ]
    source_hashes = {name: digest_file(ROOT / name) for name in source_names}
    seed_all(args.random_seed)
    data = dataset_for(args, "val")
    if len(data) != expected_windows:
        raise ValueError("Validation window count differs from the training log.")
    indices = np.arange(len(data), dtype=np.int64)
    loader = loader_for(data, args.batch_size, indices, False, args.random_seed)
    model = Model(args)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del checkpoint_bytes
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = model.adafe_adapter
    local = adapter.adafe
    original_scores = local.conditioned_scores
    original_model_prior, original_local_prior = model.guided_adafe_use_prior, local.use_prior
    original_capture = local.capture_diagnostics
    local.capture_diagnostics = False
    capture = {}

    def scored(self, x, prior=None):
        result = original_scores(x, prior)
        capture["scores"] = result.detach()
        return result

    def captured_adapter(module, inputs, result):
        capture["patch"] = inputs[0][:, :-1, :].detach()
        capture["enhanced"] = result[:, :-1, :].detach()

    local.conditioned_scores = MethodType(scored, local)
    hook = adapter.register_forward_hook(captured_adapter)

    def forward(enabled, batch):
        model.guided_adafe_use_prior = enabled
        local.use_prior = enabled
        capture.clear()
        prediction, target = forward_batch(model, batch, device)
        return prediction, target, dict(capture)

    totals = {name: 0.0 for name in [
        "on_squared_error", "off_squared_error", "on_absolute_error", "off_absolute_error",
        "prediction_delta_squared", "prediction_delta_absolute", "prediction_on_squared",
        "patch_squared", "on_residual_squared", "off_residual_squared",
        "residual_delta_squared", "local_centered_squared", "guide_centered_squared",
    ]}
    count = bits = tokens = flips = on_active = off_active = changed_tokens = 0
    prediction_max_delta = 0.0
    flip_histogram = np.zeros(args.d_model + 1, dtype=np.int64)
    channel_on = np.zeros(args.enc_in, dtype=np.float64)
    channel_off = np.zeros(args.enc_in, dtype=np.float64)
    rows = []
    restored_max_delta = None
    save_json(output / "configuration.json", {
        "args": recorded_args, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha, "log_sha256": digest_bytes(log_bytes),
        "dataset_sha256": data_sha, "source_hashes": source_hashes,
        "recorded_best_epoch": best_epoch, "recorded_best_validation_mse": best_val,
        "validation_windows": len(data), "validation_indices_sha256": digest_bytes(indices.tobytes()),
        "training_steps": 0, "test_evaluations": 0,
    })
    (output / "training_log_snapshot.log").write_bytes(log_bytes)

    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader):
                pred_on, target, on = forward(True, batch)
                pred_off, target_off, off = forward(False, batch)
                if not torch.equal(on["patch"], off["patch"]) or not torch.equal(target, target_off):
                    raise RuntimeError("Guide intervention changed the main-route input or targets.")
                if batch_number == 0:
                    restored, _, restored_capture = forward(True, batch)
                    restored_max_delta = float((restored - pred_on).abs().max())
                    if not torch.allclose(restored, pred_on, atol=1e-7, rtol=1e-6):
                        raise RuntimeError("Restoring Guide failed to restore baseline predictions.")
                    del restored, restored_capture
                    capture.clear()
                if not bool(torch.isfinite(pred_on).all() and torch.isfinite(pred_off).all()):
                    raise FloatingPointError("Non-finite predictions.")

                scores_on, scores_off = on["scores"], off["scores"]
                mask_on = scores_on > scores_on.mean(-1, keepdim=True)
                mask_off = scores_off > scores_off.mean(-1, keepdim=True)
                token_flips = (mask_on != mask_off).sum(-1)
                flips += int(token_flips.sum())
                changed_tokens += int((token_flips > 0).sum())
                bits += mask_on.numel()
                tokens += token_flips.numel()
                on_active += int(mask_on.sum())
                off_active += int(mask_off.sum())
                flip_histogram += torch.bincount(
                    token_flips, minlength=args.d_model + 1).cpu().numpy()
                patch = on["patch"].float()
                residual_on = on["enhanced"].float() - patch
                residual_off = off["enhanced"].float() - off["patch"].float()
                local_centered = scores_off - scores_off.mean(-1, keepdim=True)
                guide = scores_on - scores_off
                guide_centered = guide - guide.mean(-1, keepdim=True)
                for name, value in [
                    ("patch_squared", patch), ("on_residual_squared", residual_on),
                    ("off_residual_squared", residual_off),
                    ("residual_delta_squared", residual_on - residual_off),
                    ("local_centered_squared", local_centered),
                    ("guide_centered_squared", guide_centered),
                ]:
                    totals[name] += value.square().sum(dtype=torch.float64).item()

                error_on = pred_on.double() - target.double()
                error_off = pred_off.double() - target.double()
                delta = pred_off.double() - pred_on.double()
                for name, value in [
                    ("on_squared_error", error_on.square()),
                    ("off_squared_error", error_off.square()),
                    ("on_absolute_error", error_on.abs()),
                    ("off_absolute_error", error_off.abs()),
                    ("prediction_delta_squared", delta.square()),
                    ("prediction_delta_absolute", delta.abs()),
                    ("prediction_on_squared", pred_on.double().square()),
                ]:
                    totals[name] += value.sum().item()
                count += target.numel()
                prediction_max_delta = max(prediction_max_delta, float(delta.abs().max()))
                channel_on += error_on.square().sum((0, 1)).cpu().numpy()
                channel_off += error_off.square().sum((0, 1)).cpu().numpy()
                window_metrics = torch.stack([
                    error_on.square().mean((1, 2)), error_off.square().mean((1, 2)),
                    error_on.abs().mean((1, 2)), error_off.abs().mean((1, 2)),
                    delta.square().mean((1, 2)).sqrt(),
                ], dim=-1).cpu().tolist()
                cycles = batch[4].reshape(-1).tolist()
                for cycle, values in zip(cycles, window_metrics):
                    rows.append([len(rows), int(cycle)] + values)
                if (batch_number + 1) % 10 == 0 or batch_number + 1 == len(loader):
                    print(json.dumps({
                        "batches": batch_number + 1, "total_batches": len(loader),
                        "windows": len(rows), "mse_on": totals["on_squared_error"] / count,
                        "mse_off": totals["off_squared_error"] / count,
                        "mask_flip_fraction": flips / bits,
                    }), flush=True)
                del on, off, pred_on, pred_off, target, target_off, error_on, error_off, delta
                capture.clear()
    finally:
        hook.remove()
        local.conditioned_scores = original_scores
        model.guided_adafe_use_prior = original_model_prior
        local.use_prior = original_local_prior
        local.capture_diagnostics = original_capture

    on_metrics = {"mse": totals["on_squared_error"] / count,
                  "mae": totals["on_absolute_error"] / count}
    off_metrics = {"mse": totals["off_squared_error"] / count,
                   "mae": totals["off_absolute_error"] / count}
    weights_unchanged = all(torch.equal(value.detach().cpu(), state[name])
                            for name, value in model.state_dict().items())
    baseline_matches = math.isclose(on_metrics["mse"], best_val, rel_tol=1e-5, abs_tol=1e-6)
    sources_unchanged = all(digest_file(ROOT / name) == value
                            for name, value in source_hashes.items())
    files_unchanged = (digest_file(checkpoint) == checkpoint_sha
                       and digest_file(dataset_path) == data_sha)
    passed = baseline_matches and weights_unchanged and sources_unchanged and files_unchanged
    mse_change = off_metrics["mse"] - on_metrics["mse"]
    summary = {
        "status": "completed" if passed else "identity_or_reproduction_mismatch",
        "verification_status": "VERIFIED" if passed else "UNVERIFIED",
        "dataset": args.data_path, "seq_len": args.seq_len, "pred_len": args.pred_len,
        "seed": args.random_seed, "dropout": args.dropout,
        "validation_windows": len(rows), "target_values": count,
        "metric_space": "training-scaler standardized", "batch_size": args.batch_size,
        "recorded_best_epoch": best_epoch, "recorded_best_validation_mse": best_val,
        "baseline_matches_recorded_validation": baseline_matches,
        "baseline_match_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "weights_unchanged": weights_unchanged, "sources_unchanged": sources_unchanged,
        "checkpoint_and_data_files_unchanged": files_unchanged,
        "gtr_main_route_unchanged": True, "patch_inputs_identical": True,
        "restored_guide_max_prediction_difference": restored_max_delta,
        "training_steps": 0, "test_evaluations": 0,
        "guide_on": on_metrics, "guide_off": off_metrics,
        "guide_off_minus_on_mse": mse_change,
        "guide_off_minus_on_mse_pct": 100 * mse_change / on_metrics["mse"],
        "mask_changed_fraction": flips / bits,
        "tokens_with_any_changed_feature_fraction": changed_tokens / tokens,
        "active_fraction_on": on_active / bits, "active_fraction_off": off_active / bits,
        "mask_changed_feature_count_histogram": flip_histogram.tolist(),
        "adafe_residual_to_patch_rms_ratio_on": ratio(totals["on_residual_squared"], totals["patch_squared"]),
        "adafe_residual_to_patch_rms_ratio_off": ratio(totals["off_residual_squared"], totals["patch_squared"]),
        "guide_induced_residual_change_to_patch_rms_ratio": ratio(totals["residual_delta_squared"], totals["patch_squared"]),
        "centered_guide_to_local_score_rms_ratio": ratio(totals["guide_centered_squared"], totals["local_centered_squared"]),
        "prediction_delta_rmse": math.sqrt(totals["prediction_delta_squared"] / count),
        "prediction_delta_mae": totals["prediction_delta_absolute"] / count,
        "prediction_delta_max_abs": prediction_max_delta,
        "prediction_delta_to_on_rms_ratio": ratio(totals["prediction_delta_squared"], totals["prediction_on_squared"]),
        "learned_adafe_gamma": float(adapter.gamma),
        "learned_prior_gate_tanh": float(local.prior_gate.tanh()),
        "learned_gtr_gamma": float(model.gamma),
        "checkpoint_sha256": checkpoint_sha, "duration_seconds": time.monotonic() - started,
        "interpretation_limit": "Frozen-weight intervention, not a retrained ablation. Overlapping windows are not independent samples.",
    }
    save_json(output / "results.json", summary)
    with (output / "per_window.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["validation_window_index", "forecast_origin_cycle_index",
                         "mse_on", "mse_off", "mae_on", "mae_off", "prediction_delta_rmse"])
        writer.writerows(rows)
    with dataset_path.open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream))
    channels = [name for name in columns if name not in ("date", args.target)] + [args.target]
    with (output / "per_channel.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel_index", "channel_name", "mse_on", "mse_off", "mse_off_minus_on"])
        denominator = len(data) * args.pred_len
        for index, name in enumerate(channels):
            a, b = channel_on[index] / denominator, channel_off[index] / denominator
            writer.writerow([index, name, a, b, b - a])
    report = [
        "# Electricity 96->96 Guide 检查",
        "",
        "## Material Passport",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: " + datetime.now(timezone.utc).isoformat(),
        "- Verification Status: " + summary["verification_status"],
        "- Version Label: electricity96_frozen_probe_v1",
        "",
        "固定同一最佳权重和完整验证集；仅关闭 Guide 评分贡献，主路径 GTR 保持开启。",
        "未执行训练更新或测试集评估。逐窗口结果包含重叠窗口，不视为独立统计样本。",
        "",
        "| 检查项 | 数值 |",
        "|---|---:|",
        f"| 验证窗口数 | {len(rows)} |",
        f"| Guide 开启 MSE | {on_metrics['mse']:.9f} |",
        f"| Guide 关闭 MSE | {off_metrics['mse']:.9f} |",
        f"| 关闭后的 MSE 相对变化 | {summary['guide_off_minus_on_mse_pct']:+.6f}% |",
        f"| 掩码元素改变比例 | {summary['mask_changed_fraction']:.6%} |",
        f"| 至少一个特征改变的 token 比例 | {summary['tokens_with_any_changed_feature_fraction']:.6%} |",
        f"| AdaFE 残差 / 原 patch 的 RMS（开启 Guide） | {summary['adafe_residual_to_patch_rms_ratio_on']:.6f} |",
        f"| Guide 引起的残差变化 / 原 patch RMS | {summary['guide_induced_residual_change_to_patch_rms_ratio']:.6f} |",
        f"| 去均值周期评分 / 本地评分 RMS | {summary['centered_guide_to_local_score_rms_ratio']:.6f} |",
        f"| 预测变化 RMS | {summary['prediction_delta_rmse']:.9f} |",
        "",
        f"记录的最佳轮次：{best_epoch}；验证 MSE：{best_val:.9f}。",
        f"复现检查：{baseline_matches}；参数保持不变：{weights_unchanged}；源文件保持不变：{sources_unchanged}。",
        "",
        "关闭后的 MSE 上升说明当前已训练模型从该分支获益；下降说明当前固定权重下关闭分支更好。",
        "这不等价于从头训练的有/无 Guide 消融，也不能证明跨种子或测试集收益。",
        "若复现或身份检查失败，应先解决配置/权重/数据不一致，不能将本报告归因于指定训练轮次。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    print("OUTPUT " + str(output), flush=True)
    if not passed:
        raise SystemExit("Identity or baseline reproduction check failed; see results.json.")


if __name__ == "__main__":
    main()
