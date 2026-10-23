"""Streaming, post-training diagnostics for the isolated Electricity 720 run."""
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STAT_NAMES = (
    "count", "sse", "sae", "error_sum", "prediction_sum", "truth_sum",
    "prediction_sq_sum", "truth_sq_sum", "prediction_truth_sum",
)
DECOMP_NAMES = (
    "window_bias_mse", "window_amplitude_mse", "window_shape_mse",
    "window_centered_mse",
)


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv_write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def sufficient_statistics(prediction, truth, axes=None):
    """Return additive count/error/value moments, with reduction axes explicit."""
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("Prediction and truth shapes differ.")
    if axes is None:
        axes = tuple(range(p.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    axes = tuple(a % p.ndim for a in axes)
    d = p - t
    values = [
        np.sum(d * d, axis=axes, dtype=np.float64),
        np.sum(np.abs(d), axis=axes, dtype=np.float64),
        np.sum(d, axis=axes, dtype=np.float64),
        np.sum(p, axis=axes, dtype=np.float64),
        np.sum(t, axis=axes, dtype=np.float64),
        np.sum(p * p, axis=axes, dtype=np.float64),
        np.sum(t * t, axis=axes, dtype=np.float64),
        np.sum(p * t, axis=axes, dtype=np.float64),
    ]
    count = float(np.prod([p.shape[a] for a in axes], dtype=np.int64))
    n = np.full(np.shape(values[0]), count, dtype=np.float64)
    return np.stack([n] + values, axis=-1)


def metrics_from_statistics(statistics):
    s = np.asarray(statistics, dtype=np.float64)
    n = float(s[0])
    if n <= 0:
        raise ValueError("Cannot compute metrics for an empty group.")
    result = {name: (int(round(value)) if name == "count" else float(value))
              for name, value in zip(STAT_NAMES, s)}
    mp, mt = s[4] / n, s[5] / n
    vp = max(float(s[6] / n - mp * mp), 0.0)
    vt = max(float(s[7] / n - mt * mt), 0.0)
    covariance = float(s[8] / n - mp * mt)
    bias = float(s[3] / n)
    mse = max(float(s[1] / n), 0.0)
    sp, st = np.sqrt(vp), np.sqrt(vt)
    correlation = None
    if sp > 1e-15 and st > 1e-15:
        correlation = float(np.clip(covariance / (sp * st), -1.0, 1.0))
    result.update({
        "mse": mse, "mae": float(s[2] / n), "rmse": float(np.sqrt(mse)),
        "mean_signed_error": bias,
        "pooled_bias_squared": bias * bias,
        "centered_residual_mse": max(mse - bias * bias, 0.0),
        "pooled_prediction_mean": float(mp), "pooled_truth_mean": float(mt),
        "pooled_prediction_std": float(sp), "pooled_truth_std": float(st),
        "pooled_std_ratio": float(sp / st) if st > 1e-15 else None,
        "pooled_correlation": correlation,
    })
    return result


def window_decomposition(prediction, truth):
    """Exact MSE decomposition along lead time for every window and channel.

    Returns arrays [windows, channels]. MSE = bias + amplitude + shape.
    Shape is 2 * (std_prediction * std_truth - covariance); it is not a
    pooled correlation and is not a causal attribution to a model module.
    """
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.ndim != 3 or p.shape != t.shape or p.shape[1] == 0:
        raise ValueError("Expected matching [windows, leads, channels] arrays.")
    pm, tm = p.mean(axis=1), t.mean(axis=1)
    pc = p - pm[:, None, :]
    tc = t - tm[:, None, :]
    h = p.shape[1]
    vp = np.einsum("bhc,bhc->bc", pc, pc) / h
    vt = np.einsum("bhc,bhc->bc", tc, tc) / h
    covariance = np.einsum("bhc,bhc->bc", pc, tc) / h
    bias = (pm - tm) ** 2
    amplitude = (np.sqrt(np.maximum(vp, 0)) -
                 np.sqrt(np.maximum(vt, 0))) ** 2
    centered = np.maximum(vp + vt - 2 * covariance, 0.0)
    shape = np.maximum(centered - amplitude, 0.0)
    return dict(zip(DECOMP_NAMES, (bias, amplitude, shape, centered)))


class SplitAccumulator:
    """Only one prediction batch and small additive summaries are retained."""

    def __init__(self, channel_names, horizon=720):
        self.channel_names = [str(c) for c in channel_names]
        self.horizon = int(horizon)
        self.channel_stats = np.zeros((len(self.channel_names), len(STAT_NAMES)),
                                      dtype=np.float64)
        self.month_stats = {}
        self.lead_bins = [(f"{lo}-{min(hi, self.horizon)}", lo - 1,
                           min(hi, self.horizon))
                          for lo, hi in ((1, 168), (169, 336), (337, 720))
                          if lo <= self.horizon]
        self.lead_stats = {label: np.zeros(len(STAT_NAMES), dtype=np.float64)
                           for label, _, _ in self.lead_bins}
        self.decomposition_sums = {
            key: np.zeros(len(self.channel_names), dtype=np.float64)
            for key in DECOMP_NAMES
        }
        self.windows = 0

    def update(self, prediction, truth, target_months):
        p = np.asarray(prediction, dtype=np.float64)
        t = np.asarray(truth, dtype=np.float64)
        months = np.asarray(target_months)
        if p.shape != t.shape or p.ndim != 3:
            raise ValueError("Expected matching 3-D prediction and truth.")
        if p.shape[1:] != (self.horizon, len(self.channel_names)):
            raise ValueError("Unexpected prediction horizon or channel count.")
        if months.shape != p.shape[:2]:
            raise ValueError("Target-month matrix must match windows and leads.")
        self.channel_stats += sufficient_statistics(p, t, axes=(0, 1))
        decomposition = window_decomposition(p, t)
        for key, values in decomposition.items():
            self.decomposition_sums[key] += values.sum(axis=0)
        for label, lo, hi in self.lead_bins:
            self.lead_stats[label] += sufficient_statistics(p[:, lo:hi, :],
                                                           t[:, lo:hi, :])
        for month in np.unique(months):
            label = str(month)
            mask = months == month
            if label not in self.month_stats:
                self.month_stats[label] = np.zeros(len(STAT_NAMES), dtype=np.float64)
            self.month_stats[label] += sufficient_statistics(p[mask], t[mask])
        self.windows += p.shape[0]

    def records(self):
        if not self.windows:
            raise ValueError("No evaluation windows were accumulated.")
        overall = metrics_from_statistics(self.channel_stats.sum(axis=0))
        overall["windows"] = self.windows
        channels = []
        for i, channel in enumerate(self.channel_names):
            row = {"channel": channel, **metrics_from_statistics(self.channel_stats[i])}
            for key in DECOMP_NAMES:
                row[key] = float(self.decomposition_sums[key][i] / self.windows)
            row["window_decomposition_mse"] = sum(row[k] for k in DECOMP_NAMES[:3])
            row["window_decomposition_residual"] = row["mse"] - row["window_decomposition_mse"]
            channels.append(row)
        for key in DECOMP_NAMES:
            overall[key] = float(self.decomposition_sums[key].sum() /
                                 (self.windows * len(self.channel_names)))
        overall["window_decomposition_mse"] = sum(overall[k] for k in DECOMP_NAMES[:3])
        overall["window_decomposition_residual"] = overall["mse"] - overall["window_decomposition_mse"]
        months = [{"target_month": label, **metrics_from_statistics(stats)}
                  for label, stats in sorted(self.month_stats.items())]
        leads = [{"lead_bin": label, **metrics_from_statistics(self.lead_stats[label])}
                 for label, _, _ in self.lead_bins]
        return {"overall": overall, "per_channel": channels,
                "per_month": months, "per_lead_bin": leads}


def _state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_load(path):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in snapshot or "epoch" not in snapshot:
        raise ValueError(f"Not a diagnostic snapshot: {path}")
    return snapshot


def _data_metadata(args):
    path = Path(args.root_path) / args.data_path
    columns = list(pd.read_csv(path, nrows=0).columns)
    columns.remove("date")
    columns.remove(args.target)
    columns.append(args.target)
    if args.features == "MS":
        columns = [args.target]
    elif args.features == "S":
        columns = [args.target]
    dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"])
    months = dates.to_numpy(dtype="datetime64[ns]").astype("datetime64[M]").astype(str)
    n = len(months)
    train = int(n * .7)
    test = int(n * .2)
    boundaries = {"val": (train, n - test), "test": (n - test, n)}
    return columns, months, boundaries


def _evaluate_split(experiment, args, dataset, loader, channel_names, months,
                    bounds, label, split):
    lo, hi = bounds
    expected = hi - lo - args.pred_len + 1
    if len(dataset) != expected:
        raise ValueError(f"{split} window count differs from Dataset_Custom: "
                         f"{len(dataset)} != {expected}")
    accumulator = SplitAccumulator(channel_names, args.pred_len)
    offset = 0
    experiment.model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch_x, batch_y, x_mark, y_mark, cycle = batch[:5]
            x = batch_x.float().to(experiment.device)
            y = batch_y.float()
            xm = x_mark.float().to(experiment.device)
            ym = y_mark.float().to(experiment.device)
            cyc = cycle.int().to(experiment.device)
            decoder = torch.zeros_like(y[:, -args.pred_len:, :]).float()
            decoder = torch.cat([y[:, :args.label_len, :], decoder], dim=1).to(experiment.device)
            context = torch.cuda.amp.autocast() if args.use_amp else nullcontext()
            with context:
                output = experiment.model(x, xm, decoder, ym, cyc)
            fdim = -1 if args.features == "MS" else 0
            p = output[:, -args.pred_len:, fdim:].detach().cpu().numpy().astype(np.float64)
            t = y[:, -args.pred_len:, fdim:].numpy().astype(np.float64)
            batch_size = p.shape[0]
            target_indices = (lo + offset + np.arange(batch_size)[:, None] +
                              np.arange(args.pred_len)[None, :])
            if target_indices.min() < lo or target_indices.max() >= hi:
                raise ValueError("Target-time indices escaped their evaluation split.")
            accumulator.update(p, t, months[target_indices])
            offset += batch_size
            if (batch_index + 1) % 50 == 0:
                print(f"Analysis {label} {split}: {offset}/{len(dataset)} windows", flush=True)
    if offset != len(dataset):
        raise ValueError(f"Evaluation lost or repeated windows: {offset} != {len(dataset)}")
    return accumulator.records()


def _delta_rows(before, after, key, total_count):
    old = {str(row[key]): row for row in before}
    new = {str(row[key]): row for row in after}
    if set(old) != set(new):
        raise ValueError(f"Group membership changed between snapshots: {key}")
    records = []
    change_fields = ("mse", "mae", "mean_signed_error", "pooled_std_ratio",
                     "pooled_correlation") + DECOMP_NAMES
    for name in old:
        a, b = old[name], new[name]
        if a["count"] != b["count"]:
            raise ValueError("Different target counts between snapshot evaluations.")
        row = {
            key: name, "count": a["count"],
            "before_mse": a["mse"], "after_mse": b["mse"],
            "delta_mse": b["mse"] - a["mse"],
            "delta_sse": b["sse"] - a["sse"],
            "contribution_to_overall_mse_change":
                (b["sse"] - a["sse"]) / total_count,
        }
        for field in change_fields:
            if field == "mse":
                continue
            if field in a and field in b:
                row[f"before_{field}"] = a[field]
                row[f"after_{field}"] = b[field]
                row[f"delta_{field}"] = (
                    b[field] - a[field]
                    if a[field] is not None and b[field] is not None else None
                )
        records.append(row)
    return records


def _comparison(before, after, run_dir):
    comparison = {"before_epoch": 4, "after_epoch": 12, "splits": {}}
    output = Path(run_dir) / "analysis" / "epoch04_to_epoch12"
    for split in ("val", "test"):
        a, b = before[split], after[split]
        overview = {}
        for field in ("mse", "mae", "mean_signed_error") + DECOMP_NAMES:
            overview[field] = {
                "before": a["overall"][field], "after": b["overall"][field],
                "change": b["overall"][field] - a["overall"][field],
            }
        groups = {}
        for group, key in (("per_channel", "channel"),
                           ("per_month", "target_month"),
                           ("per_lead_bin", "lead_bin")):
            rows = _delta_rows(a[group], b[group], key, a["overall"]["count"])
            _csv_write(output / f"{split}_{group}_delta.csv", rows)
            groups[group] = {
                "largest_worsening": sorted(
                    [r for r in rows if r["delta_mse"] > 0],
                    key=lambda r: r["contribution_to_overall_mse_change"],
                    reverse=True)[:10],
                "largest_improvement": sorted(
                    [r for r in rows if r["delta_mse"] < 0],
                    key=lambda r: r["contribution_to_overall_mse_change"])[:10],
                "sum_of_contributions": sum(
                    r["contribution_to_overall_mse_change"] for r in rows),
                "delta_csv": str(output / f"{split}_{group}_delta.csv"),
            }
        comparison["splits"][split] = {"overall": overview, "groups": groups}
    _json_write(output / "comparison.json", comparison)
    return comparison


def evaluate_checkpoints(experiment, args, run_dir, snapshot_paths):
    """Evaluate retained snapshots; formal selection remains validation-only."""
    run_dir = Path(run_dir)
    paths = [Path(path) for path in snapshot_paths]
    channel_names, target_months, boundaries = _data_metadata(args)
    data_loaders = {split: experiment._get_data(split) for split in ("val", "test")}
    source_path = Path(__file__).resolve()
    summary = {
        "complete": False,
        "evaluator_source": str(source_path),
        "evaluator_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "metric_space": "training-set StandardScaler space",
        "prediction_error_sign": "prediction minus truth",
        "target_month_definition": "month of each future target timestamp, not forecast origin",
        "lead_bins": ["1-168", "169-336", "337-720"],
        "snapshot_order": [], "snapshots": {},
        "decomposition_definition": {
            "axis": "each window x channel across the complete forecast horizon",
            "identity": "MSE = window_bias_mse + window_amplitude_mse + window_shape_mse",
            "bias": "squared difference of per-window temporal means",
            "amplitude": "squared difference of per-window temporal standard deviations",
            "shape": "2 * (std_prediction * std_truth - covariance)",
            "scope": "overall and per-channel; pooled moments for months/leads are separate diagnostics",
        },
        "interpretation_limits": [
            "Test-set metrics are retrospective diagnostics, never the checkpoint-selection rule.",
            "Pooled standard deviations and correlations are not per-window amplitude attribution.",
            "Error-component changes do not by themselves establish a causal model-module effect.",
        ],
    }
    summary_path = run_dir / "summary.json"
    _json_write(summary_path, summary)
    seen = {}
    payloads = {}
    epoch_payloads = {}
    best_path = next((p for p in paths if p.stem == "validation_best"), None)
    try:
        for path in paths:
            snapshot = _snapshot_load(path)
            epoch = int(snapshot["epoch"])
            label = path.stem
            state = snapshot["model_state_dict"]
            fingerprint = _state_fingerprint(state)
            entry = {
                "label": label, "epoch": epoch, "snapshot_path": str(path),
                "state_sha256": fingerprint,
                "losses": snapshot.get("losses", {}),
                "gates": snapshot.get("gates", {}),
            }
            dedup_key = (epoch, fingerprint)
            if dedup_key in seen:
                original = seen[dedup_key]
                entry["alias_of"] = original
                entry["splits"] = summary["snapshots"][original]["splits"]
                payloads[label] = payloads[original]
                print(f"Analysis {label}: same epoch and exact model state as {original}; reused", flush=True)
            else:
                print(f"Analysis {label}: loading epoch {epoch}", flush=True)
                experiment.model.load_state_dict(state, strict=True)
                del state
                payload = {}
                split_entries = {}
                for split, (dataset, loader) in data_loaders.items():
                    print(f"Analysis {label}: evaluating {split}", flush=True)
                    records = _evaluate_split(
                        experiment, args, dataset, loader, channel_names,
                        target_months, boundaries[split], label, split,
                    )
                    directory = run_dir / "analysis" / label / split
                    _json_write(directory / "overall.json", records["overall"])
                    for name in ("per_channel", "per_month", "per_lead_bin"):
                        _csv_write(directory / f"{name}.csv", records[name])
                    payload[split] = records
                    split_entries[split] = {
                        "overall": records["overall"],
                        "directory": str(directory),
                    }
                    print(f"Analysis {label} {split}: MSE={records['overall']['mse']:.9f}, "
                          f"MAE={records['overall']['mae']:.9f}", flush=True)
                entry["splits"] = split_entries
                payloads[label] = payload
                seen[dedup_key] = label
            summary["snapshot_order"].append(label)
            summary["snapshots"][label] = entry
            epoch_payloads.setdefault(epoch, payloads[label])
            _json_write(summary_path, summary)
            del snapshot
        if best_path is not None:
            best_label = best_path.stem
        else:
            def validation_loss(label):
                losses = summary["snapshots"][label]["losses"]
                return float(losses.get("validation_mse_standardized", float("inf")))
            best_label = min(summary["snapshot_order"], key=validation_loss)
        selected = summary["snapshots"][best_label]
        summary["validation_best_label"] = best_label
        summary["best_validation_epoch"] = selected["epoch"]
        summary["selected_validation"] = selected["splits"]["val"]["overall"]
        summary["selected_test"] = selected["splits"]["test"]["overall"]
        if "alias_of" in selected:
            summary["validation_best_alias"] = selected["alias_of"]
        if 4 in epoch_payloads and 12 in epoch_payloads:
            summary["epoch4_to_epoch12"] = _comparison(
                epoch_payloads[4], epoch_payloads[12], run_dir,
            )
        summary["gates_by_snapshot"] = {
            label: summary["snapshots"][label]["gates"]
            for label in summary["snapshot_order"]
        }
        if 4 in epoch_payloads and 12 in epoch_payloads:
            gate4 = next(entry["gates"] for entry in summary["snapshots"].values()
                         if entry["epoch"] == 4)
            gate12 = next(entry["gates"] for entry in summary["snapshots"].values()
                          if entry["epoch"] == 12)
            summary["gate_changes_epoch4_to_epoch12"] = {
                key: float(gate12[key]) - float(gate4[key])
                for key in ("gtr_gamma", "adafe_gamma", "guide_prior_gate_effective")
                if key in gate4 and key in gate12
            }
        summary["complete"] = True
        _json_write(summary_path, summary)
        return summary
    except Exception as error:
        summary["error"] = repr(error)
        _json_write(summary_path, summary)
        raise
    finally:
        if best_path is not None:
            best = _snapshot_load(best_path)
            experiment.model.load_state_dict(best["model_state_dict"], strict=True)
            experiment.model.eval()
