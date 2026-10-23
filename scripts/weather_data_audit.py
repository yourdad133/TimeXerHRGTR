import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_LEN = 96
PRED_LENS = (96, 192, 336, 720)
SENTINEL = -9999.0


def intervals(mask, dates):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        {
            "start_row": int(a),
            "end_row_inclusive": int(b - 1),
            "start_date": str(dates.iloc[a]),
            "end_date": str(dates.iloc[b - 1]),
            "rows": int(b - a),
        }
        for a, b in zip(starts, ends)
    ]


def window_counts(n, invalid_input_rows, invalid_target_rows, pred_len):
    n_train, n_test = int(n * 0.7), int(n * 0.2)
    n_val = n - n_train - n_test
    borders = {
        "train": (0, n_train),
        "val": (n_train - SEQ_LEN, n_train + n_val),
        "test": (n - n_test - SEQ_LEN, n),
    }
    span = SEQ_LEN + pred_len
    result = {}
    for split, (begin, end) in borders.items():
        candidates = max(0, end - begin - span + 1)
        input_prefix = np.r_[0, np.cumsum(invalid_input_rows[begin:end], dtype=np.int64)]
        target_prefix = np.r_[0, np.cumsum(invalid_target_rows[begin:end], dtype=np.int64)]
        starts = np.arange(candidates, dtype=np.int64)
        bad_input = input_prefix[starts + SEQ_LEN] - input_prefix[starts]
        bad_target = target_prefix[starts + span] - target_prefix[starts + SEQ_LEN]
        valid = int(np.count_nonzero((bad_input == 0) & (bad_target == 0)))
        loader_used = valid if split == "test" else valid // 4 * 4
        result[split] = {
            "candidate_windows": int(candidates),
            "eligible_windows": valid,
            "excluded_windows": int(candidates - valid),
            "loader_used_windows": int(loader_used),
            "batch_tail_dropped": int(valid - loader_used),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, outdir = Path(args.data), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    if "date" not in df or "OT" not in df:
        raise ValueError("Expected date and OT columns")
    dates = pd.to_datetime(df["date"], errors="raise")
    names = [c for c in df.columns if c != "date"]
    values = df[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    sent = values == SENTINEL
    invalid = ~np.isfinite(values) | sent
    bad_rows = invalid.any(axis=1)
    n_train = int(len(df) * 0.7)

    stats = []
    for i, name in enumerate(names):
        x = values[:n_train, i]
        valid = x[np.isfinite(x) & (x != SENTINEL)]
        stats.append({
            "feature": name,
            "sentinel_count_all_rows": int(sent[:, i].sum()),
            "other_invalid_count_all_rows": int((~np.isfinite(values[:, i]) & ~sent[:, i]).sum()),
            "train_valid_values": int(valid.size),
            "train_mean_with_sentinel": float(np.nanmean(x)),
            "train_std_with_sentinel": float(np.nanstd(x)),
            "train_mean_valid_only": float(valid.mean()) if valid.size else None,
            "train_std_valid_only": float(valid.std()) if valid.size else None,
        })
    intervals_by_feature = {
        name: intervals(sent[:, i], dates) for i, name in enumerate(names) if sent[:, i].any()
    }
    deltas = dates.diff()
    cadence = pd.Series(deltas.dropna()).value_counts().head(10)
    cadence_summary = [{"interval": str(k), "count": int(v)} for k, v in cadence.items()]
    timestamp_anomalies = [
        {
            "previous_row": int(i - 1),
            "row": int(i),
            "previous_date": str(dates.iloc[i - 1]),
            "date": str(dates.iloc[i]),
            "delta": str(delta),
        }
        for i, delta in enumerate(deltas)
        if i > 0 and delta != pd.Timedelta(minutes=10)
    ]
    samples = {str(h): window_counts(len(df), bad_rows, invalid[:, names.index("OT")], h) for h in PRED_LENS}
    ot = stats[names.index("OT")]
    report = {
        "source_file": str(source),
        "rows": int(len(df)),
        "date_start": str(dates.iloc[0]),
        "date_end": str(dates.iloc[-1]),
        "features": names,
        "timestamp_deltas_top10": cadence_summary,
        "timestamp_anomalies_from_10min": timestamp_anomalies,
        "sentinel": SENTINEL,
        "sentinel_only_in_OT": bool(sent[:, [i for i, name in enumerate(names) if name != "OT"]].sum() == 0),
        "sentinel_counts_by_feature": {name: int(sent[:, i].sum()) for i, name in enumerate(names)},
        "other_invalid_counts_by_feature": {
            name: int((~np.isfinite(values[:, i]) & ~sent[:, i]).sum()) for i, name in enumerate(names)
        },
        "sentinel_intervals_by_feature": intervals_by_feature,
        "train_rows": int(n_train),
        "batch_size": 4,
        "feature_train_stats": stats,
        "train_ot_stats": ot,
        "sample_counts_by_prediction_length": samples,
    }
    json_path = outdir / "weather_data_audit.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    common = ", ".join("{} ({:,})".format(x["interval"], x["count"]) for x in cadence_summary[:3])
    sentinel_summary = ", ".join(
        "{}={:,}".format(name, int(sent[:, i].sum())) for i, name in enumerate(names) if sent[:, i].any()
    ) or "none"
    lines = [
        "# Weather data audit (before cleaning)",
        "",
        "- Source: {}".format(source),
        "- Rows: {:,}; features excluding date: {}".format(len(df), len(names)),
        "- Date range: {} to {}".format(dates.iloc[0], dates.iloc[-1]),
        "- Most common timestamp intervals: {}".format(common),
        "- -9999 counts by feature: {}".format(sentinel_summary),
        "- Other non-finite values: {:,}".format(int((~np.isfinite(values) & ~sent).sum())),
        "- Timestamp intervals differing from 10 minutes: {}".format(len(timestamp_anomalies)),
        "",
        "## Timestamp irregularities",
        "",
        "| Previous timestamp | Current timestamp | Delta |",
        "|---|---|---|",
    ]
    for anomaly in timestamp_anomalies:
        lines.append("| {} | {} | {} |".format(
            anomaly["previous_date"], anomaly["date"], anomaly["delta"]
        ))
    lines += [
        "",
        "## Sentinel intervals",
        "",
        "| Feature | Start | End | Rows |",
        "|---|---|---|---:|",
    ]
    for name, spans in intervals_by_feature.items():
        for span in spans:
            lines.append("| {} | {} | {} | {:,} |".format(
                name, span["start_date"], span["end_date"], span["rows"]
            ))
    lines += [
        "",
        "## OT training statistics (population standard deviation)",
        "",
        "| Statistic | Before (sentinel included) | After (valid train values only) |",
        "|---|---:|---:|",
        "| Mean | {:.10g} | {:.10g} |".format(ot["train_mean_with_sentinel"], ot["train_mean_valid_only"]),
        "| Std. dev. | {:.10g} | {:.10g} |".format(ot["train_std_with_sentinel"], ot["train_std_valid_only"]),
        "| Valid train values | {:,} | {:,} |".format(n_train, ot["train_valid_values"]),
        "",
        "## Eligible windows (seq_len=96; input has no invalid feature and forecast OT labels are valid)",
        "",
        "| pred_len | Split | Candidate | Eligible | Excluded | Loader used | Batch tail dropped |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for h, splits in samples.items():
        for split, counts in splits.items():
            lines.append("| {} | {} | {:,} | {:,} | {:,} | {:,} | {:,} |".format(
                h,
                split,
                counts["candidate_windows"],
                counts["eligible_windows"],
                counts["excluded_windows"],
                counts["loader_used_windows"],
                counts["batch_tail_dropped"],
            ))
    (outdir / "weather_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_json": str(json_path),
        "audit_markdown": str(outdir / "weather_data_audit.md"),
        "rows": len(df),
        "sentinel_counts_by_feature": report["sentinel_counts_by_feature"],
        "sentinel_intervals_by_feature": intervals_by_feature,
        "train_ot_stats": ot,
        "sample_counts_by_prediction_length": samples,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


