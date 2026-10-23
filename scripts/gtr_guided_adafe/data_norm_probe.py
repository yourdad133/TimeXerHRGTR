"""Read-only PEMS normalization diagnosis, with mandatory remote preflight.

All arrays are examined on the configured remote CPU. No forecasting model
or checkpoint is instantiated and no array is copied to CUDA. common.runtime
still verifies the SSH host, project, tslib interpreter, nvidia-smi and CUDA.

Exact raw zeros can be genuine zero flow or placeholders. Their presence
alone does not establish missing data, and this probe does not impute values.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.adafe_diagnostics.common import (
    config, dataset_for, runtime, save_json, window_indices,
)

THRESHOLDS = (1e-3, 1e-2, 0.05)
QUANTILES = (
    ("min", 0), ("p01", .01), ("p05", .05), ("p10", .1),
    ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9),
    ("p95", .95), ("p99", .99), ("max", 1),
)


def quantiles(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {name: None for name, _ in QUANTILES}
    result = np.quantile(finite, [fraction for _, fraction in QUANTILES])
    return {name: float(value) for (name, _), value in zip(QUANTILES, result)}


def raw_zero_summary(raw):
    finite = np.isfinite(raw)
    zero = finite & (raw == 0)
    finite_count = int(finite.sum())
    node_denominator = finite.sum(axis=0)
    node_ratio = np.divide(
        zero.sum(axis=0), node_denominator,
        out=np.zeros(raw.shape[1], dtype=np.float64), where=node_denominator > 0)
    top_nodes = np.argsort(-node_ratio, kind="stable")[:5]
    return {
        "observations": int(raw.size),
        "finite_observations": finite_count,
        "nonfinite_observations": int(raw.size - finite_count),
        "exact_zero_count": int(zero.sum()),
        "exact_zero_fraction_among_finite": (
            float(zero.sum() / finite_count) if finite_count else None),
        "per_node_zero_fraction_quantiles": quantiles(node_ratio[node_denominator > 0]),
        "top5_nodes_by_zero_fraction": [
            {"node_index": int(node), "zero_fraction": float(node_ratio[node]),
             "finite_observations": int(node_denominator[node])}
            for node in top_nodes],
    }


def concentration(std_matrix, phases, threshold, cycle):
    finite = np.isfinite(std_matrix)
    low = finite & (std_matrix < threshold)
    finite_total = int(finite.sum())
    node_count = finite.sum(axis=0)
    node_ratio = np.divide(
        low.sum(axis=0), node_count,
        out=np.zeros(std_matrix.shape[1], dtype=np.float64), where=node_count > 0)
    top_nodes = np.argsort(-node_ratio, kind="stable")[:5]
    phase_rows = []
    for phase in range(cycle):
        selected = phases == phase
        support = int(selected.sum())
        if not support:
            continue
        denominator = int(finite[selected].sum())
        count = int(low[selected].sum())
        phase_rows.append({
            "phase": phase, "sample_windows": support, "finite_node_windows": denominator,
            "near_constant_node_windows": count,
            "fraction": float(count / denominator) if denominator else None,
        })
    phase_rows.sort(key=lambda row: (
        -(row["fraction"] if row["fraction"] is not None else -1),
        -row["sample_windows"], row["phase"]))
    return {
        "threshold": threshold, "comparison": "strictly less than",
        "near_constant_node_windows": int(low.sum()),
        "finite_node_windows": finite_total,
        "fraction": float(low.sum() / finite_total) if finite_total else None,
        "nodes_with_at_least_one_near_constant_window": int(low.any(axis=0).sum()),
        "windows_with_at_least_one_near_constant_node": int(low.any(axis=1).sum()),
        "per_node_fraction_quantiles": quantiles(node_ratio[node_count > 0]),
        "top5_nodes": [
            {"node_index": int(node), "fraction": float(node_ratio[node]),
             "near_constant_windows": int(low[:, node].sum()),
             "finite_windows": int(node_count[node])}
            for node in top_nodes],
        "top5_input_start_phases": phase_rows[:5],
        "phase_support_warning": (
            "Phase rankings use only uniformly sampled windows; inspect sample_windows "
            "before interpreting a concentration as a reproducible daily pattern."),
    }


def normalized_window_summary(dataset, args, window_limit):
    indices = window_indices(len(dataset), window_limit)
    std_matrix = np.empty((len(indices), args.enc_in), dtype=np.float64)
    phases = np.empty(len(indices), dtype=np.int64)
    for row, start in enumerate(indices):
        window = np.asarray(dataset.data_x[start:start + args.seq_len], dtype=np.float64)
        std_matrix[row] = np.std(window, axis=0, ddof=0)
        phases[row] = int(dataset.cycle_index[start])
    finite = np.isfinite(std_matrix)
    return {
        "available_windows": len(dataset),
        "sampled_windows": len(indices),
        "sampling": "uniformly spaced valid window starts, including split endpoints",
        "sampled_window_start_indices_within_split": indices.tolist(),
        "index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "input_start_phase_sha256": hashlib.sha256(phases.tobytes()).hexdigest(),
        "window_length": args.seq_len,
        "future_horizon_defining_valid_windows": args.pred_len,
        "node_count": args.enc_in,
        "node_indices_are_zero_based_npz_column_indices": True,
        "space": "Dataset_PEMS.data_x; StandardScaler fitted on the training split only",
        "std_definition": "population std over each input window separately for each node",
        "finite_node_windows": int(finite.sum()),
        "nonfinite_node_windows": int(std_matrix.size - finite.sum()),
        "window_node_std_quantiles": quantiles(std_matrix),
        "revin_denominator_quantiles": quantiles(np.sqrt(std_matrix ** 2 + 1e-5)),
        "revin_epsilon": 1e-5,
        "threshold_results": {
            format(threshold, "g"): concentration(
                std_matrix, phases, threshold, args.gtr_cycle_len)
            for threshold in THRESHOLDS},
        "training_scaler_node_scale_quantiles": quantiles(dataset.scaler.scale_),
    }


def dataset_summary(name, options):
    args = config(name, options.seed, enabled=0)
    args.seq_len = options.seq_len
    args.pred_len = options.pred_len
    path = Path(args.root_path) / args.data_path
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise ValueError(f"{name}: NPZ lacks data")
        values = archive["data"]
        if values.ndim != 3 or values.shape[2] < 1:
            raise ValueError(f"{name}: expected [time, nodes, features]")
        raw = np.asarray(values[:, :, 0])
        if raw.shape[1] != args.enc_in:
            raise ValueError(f"{name}: unexpected node count {raw.shape[1]}")
        # Match Dataset_PEMS's non-overlapping raw 60/20/20 split boundaries.
        training_end = int(raw.shape[0] * .6)
        validation_end = training_end + int(raw.shape[0] * .2)
        raw_summaries = {
            "entire_npz_descriptive_only": raw_zero_summary(raw),
            "training_raw_split": raw_zero_summary(raw[:training_end]),
            "validation_raw_split": raw_zero_summary(raw[training_end:validation_end]),
        }
        raw_shape = list(values.shape)
    result = {
        "dataset": name, "npz_path": str(path), "npz_shape": raw_shape,
        "feature_selected": 0, "node_count": args.enc_in, "cycle_len": args.gtr_cycle_len,
        "raw_zero_summary": raw_summaries,
        "raw_zero_interpretation": (
            "Exact zeros are observed values; they may indicate zero traffic or placeholders. "
            "Missingness has not been established."),
        "normalized_windows": {},
    }
    for split in ("train", "val"):
        dataset = dataset_for(args, split)
        result["normalized_windows"][split] = normalized_window_summary(
            dataset, args, options.windows)
        del dataset
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("PEMS04", "PEMS07"),
                        default=["PEMS04", "PEMS07"])
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty output directory inside the remote project")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    options = parser.parse_args()
    if min(options.windows, options.seq_len, options.pred_len) <= 0:
        parser.error("windows, seq-len and pred-len must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("Use CUDA_VISIBLE_DEVICES=0 for mandatory preflight; arrays stay on the remote CPU")
    output = options.output if options.output.is_absolute() else ROOT / options.output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the configured remote project")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be new or empty; existing diagnostic reports are not overwritten")
    runtime(output / "runtime")
    report = {
        "state": "running", "read_only_data_diagnosis": True,
        "models_or_checkpoints_instantiated": False, "model_training": False,
        "array_compute_device": "configured remote CPU via NumPy",
        "options": {**vars(options), "output": str(output)},
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "data_provider/data_loader.py", "scripts/adafe_diagnostics/common.py",
                "scripts/gtr_guided_adafe/data_norm_probe.py")},
        "interpretation": (
            "Input window variability describes the scale available to RevIN. Frequent small "
            "scales would support investigating normalization, but these descriptive statistics "
            "alone do not establish forecast degradation or missing data."),
        "datasets": [],
    }
    save_json(output / "summary.json", report)
    for name in options.datasets:
        row = dataset_summary(name, options)
        report["datasets"].append(row)
        save_json(output / "summary.json", report)
        compact = {
            "dataset": name,
            "raw_zero_fraction": row["raw_zero_summary"]["entire_npz_descriptive_only"]
                                    ["exact_zero_fraction_among_finite"],
            "near_constant_fractions": {
                split: {threshold: values["fraction"]
                        for threshold, values in stats["threshold_results"].items()}
                for split, stats in row["normalized_windows"].items()},
        }
        print("DATA_NORM " + json.dumps(compact, ensure_ascii=False), flush=True)
    report["state"] = "completed"
    save_json(output / "summary.json", report)
    print("DATA_NORM_PROBE_COMPLETED " + str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()

