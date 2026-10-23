"""Collect all diagnostic phases and verify matched protocols; no test-based selection."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from common import ROOT, runtime, save_json

PHASES = [
    ("main", "plan.json", "training", "controller.json"),
    ("constant_reparam", "constant_reparam_plan.json", "training_followup", "followup_controller.json"),
    ("scorer_lr", "scorer_lr_plan.json", "training_scorer_lr", "scorer_lr_controller.json"),
]
VARIANTS = ["baseline", "ste", "gamma_fixed", "proxy_scaled", "token_threshold",
            "stable", "constant_reparam", "scorer_slow"]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def mean(values):
    return statistics.mean(values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    options = parser.parse_args()
    output = Path(options.output).resolve()
    if not output.is_relative_to(ROOT / "reports/adafe_diagnostics"):
        raise ValueError("Use an isolated diagnostic output directory")
    runtime(output / "collection_runtime")
    rows, pending, phases, seen, signatures = [], [], [], set(), {}
    for phase, plan_file, training_folder, controller_file in PHASES:
        if not (output / plan_file).exists():
            continue
        plan = read(output / plan_file)
        controller = read(output / controller_file) if (output / controller_file).exists() else {}
        phases.append({"phase": phase, "planned": len(plan["jobs"]), "controller": controller})
        for job in plan["jobs"]:
            case = f'{job["dataset"]}_seed{job["seed"]}_{job["variant"]}'
            if case in seen:
                raise AssertionError(f"Duplicate diagnostic case: {case}")
            seen.add(case)
            directory = output / training_folder / "runs" / case
            if not (directory / "result.json").exists():
                status = read(directory / "status.json") if (directory / "status.json").exists() else {"state": "missing"}
                pending.append({"phase": phase, "case": case, "status": status})
                continue
            result = read(directory / "result.json")
            configuration = read(directory / "configuration.json")
            history = read(directory / "history.json")
            if result["state"] != "completed":
                raise AssertionError(f"Non-completed result: {case}")
            if result["best_epoch"] != min(history, key=lambda r: r["validation"]["mse"])["epoch"]:
                raise AssertionError(f"Checkpoint did not minimize validation MSE: {case}")
            if len(history) != result["epochs_completed"]:
                raise AssertionError(f"Epoch history mismatch: {case}")
            if configuration["test_used_for_selection"] or configuration["validation_shuffle"] or configuration["validation_drop_last"]:
                raise AssertionError(f"Unexpected evaluation protocol: {case}")
            args = {k:v for k,v in configuration["args"].items() if k not in {"use_adafe", "adafe_mode"}}
            signature = json.dumps({
                "args": args, "windows": configuration["windows"],
                "indices": configuration["index_hashes"],
                "initial_shared_hash": configuration["initial_shared_hash"],
                "metric": configuration["metric_aggregation"],
            }, sort_keys=True)
            group = (job["dataset"], job["seed"])
            if group in signatures and signatures[group] != signature:
                raise AssertionError(f"Unmatched initialization/data/protocol: {case}")
            signatures[group] = signature
            diagnostic = result["best_diagnostics"]
            max_std = max((h["diagnostics"].get("score_std", 0) for h in history), default=0)
            max_saturation = max((h["diagnostics"].get("proxy_saturated_fraction", 0) for h in history), default=0)
            rows.append({
                "case": case, "phase": phase, "dataset": job["dataset"], "seed": job["seed"],
                "variant": job["variant"], "validation_mse": result["validation_mse"],
                "test_mse": result["test"]["mse"], "test_mae": result["test"]["mae"],
                "best_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"],
                "gamma": diagnostic.get("gamma"), "score_std": diagnostic.get("score_std"),
                "saturation": diagnostic.get("proxy_saturated_fraction"),
                "residual_ratio": diagnostic.get("residual_to_patch_norm"),
                "max_epoch_score_std": max_std, "max_epoch_saturation": max_saturation,
                "full_data": result["full_dataset_training"],
                "train_windows": configuration["windows"]["train"],
                "validation_windows": configuration["windows"]["val"],
                "test_windows": configuration["windows"]["test"],
                "batch_size": configuration["args"]["batch_size"],
                "physical_gpu": configuration["physical_gpu"],
                "initial_shared_hash": result["initial_shared_hash"],
                "index_hashes": configuration["index_hashes"],
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_memory_mb": result["peak_memory_mb"],
                "directory": str(directory.relative_to(output)),
            })
    for row in rows:
        group_rows = [r for r in rows if (r["dataset"], r["seed"]) == (row["dataset"], row["seed"])]
        baseline = next((r for r in group_rows if r["variant"] == "baseline"), None)
        ste = next((r for r in group_rows if r["variant"] == "ste"), None)
        for name, reference in [("baseline", baseline), ("ste", ste)]:
            row[f"test_change_vs_{name}_pct"] = 100*(row["test_mse"]/reference["test_mse"]-1) if reference else None
            row[f"validation_change_vs_{name}_pct"] = 100*(row["validation_mse"]/reference["validation_mse"]-1) if reference else None
    all_finished = all(p["controller"].get("state") == "finished" and p["controller"].get("exit_code") == 0 for p in phases)
    audit = read(output / "audit/audit.json")
    audit_cases = sum(len(r["variants"]) for r in audit)
    audit_updates = sum(len(v["steps"]) for r in audit for v in r["variants"])
    summary = {
        "planned": sum(p["planned"] for p in phases), "completed": len(rows),
        "pending_or_failed": pending, "phases": phases, "controllers_finished_successfully": all_finished,
        "matched_groups_checked": len(signatures), "protocol_checks_passed": True,
        "test_used_for_selection": False, "audit_datasets": [r["dataset"] for r in audit],
        "audit_cases": audit_cases, "audit_updates": audit_updates, "runs": rows,
        "interpretation": "Descriptive diagnostic trials. ETTh2 is full data; Electricity/Weather/ETTm2 are fixed-window pilots. Variant follow-ups were specified after preceding phase observations.",
    }
    save_json(output / "collected_summary.json", summary)
    if rows:
        flat = [{**r, "index_hashes": json.dumps(r["index_hashes"], sort_keys=True)} for r in rows]
        with (output / "all_runs.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    table = [
        "# GPU 排查的受控训练结果",
        "",
        f"已完成 {len(rows)}/{summary['planned']} 组；失败或待完成 {len(pending)} 组。所有配对的骨干/GTR 初始权重、窗口索引、批量大小、学习率日程及验证口径均已核对。",
        "",
        "指标为原有数据划分和训练集标准化下的测试 MSE。负变化表示误差降低。各轮最佳模型只按验证 MSE 选择，测试只在选择完成后计算一次。",
        "",
        "ETTh2 为完整时间划分；Weather/ETTm2 为固定 4096/512/1024 个训练/验证/测试窗口；Electricity 为 512/256/256 窗口、batch=8。小预算试验不代表完整数据集的正式基准成绩。",
        "",
    ]
    for dataset, seeds in [("ETTh2", [2026, 2027]), ("ETTh2", [2028]), ("Weather", [2026]), ("ETTm2", [2026]), ("Electricity", [2026])]:
        group_rows = [r for r in rows if r["dataset"] == dataset and r["seed"] in seeds]
        if not group_rows:
            continue
        table.extend([f"## {dataset}，种子 {', '.join(map(str,seeds))}", "",
            "| 变体 | 种子数 | 平均验证 MSE | 平均测试 MSE | 对基线的测试变化 | 对原始 STE 的测试变化 |",
            "|---|---:|---:|---:|---:|---:|"])
        base = [r["test_mse"] for r in group_rows if r["variant"] == "baseline"]
        ste = [r["test_mse"] for r in group_rows if r["variant"] == "ste"]
        for variant in VARIANTS:
            candidates = [r for r in group_rows if r["variant"] == variant]
            if not candidates:
                continue
            value = mean([r["test_mse"] for r in candidates])
            base_change = f"{100*(value/mean(base)-1):+.3f}%" if base else "—"
            ste_change = f"{100*(value/mean(ste)-1):+.3f}%" if ste else "—"
            table.append(f"| {variant} | {len(candidates)} | {mean([r['validation_mse'] for r in candidates]):.6f} | {value:.6f} | {base_change} | {ste_change} |")
        table.append("")
    table.extend(["## 每次运行", "",
        "| 数据集 | seed | 变体 | 验证 MSE | 测试 MSE | 测试 MAE | 最佳轮/实际轮数 | 最佳模型饱和比例* |",
        "|---|---:|---|---:|---:|---:|---|---:|"])
    for row in sorted(rows, key=lambda r:(r["dataset"],r["seed"],VARIANTS.index(r["variant"]))):
        saturation = f"{100*row['saturation']:.3f}%" if row["saturation"] is not None else "—"
        table.append(f"| {row['dataset']} | {row['seed']} | {row['variant']} | {row['validation_mse']:.6f} | {row['test_mse']:.6f} | {row['test_mae']:.6f} | {row['best_epoch']}/{row['epochs_completed']} | {saturation} |")
    table.extend(["", "*打分标准差、代理 sigmoid 饱和比例和残差范数来自固定的第一个验证批次，是表征诊断，不能当成整个验证集的统计。饱和定义为 sigmoid < 0.01 或 > 0.99。",
        "", "本表保留所有预先指定的变体，包括变差的结果；不进行显著性检验，不据测试指标选择默认实现。"])
    (output / "training_comparison.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    scripts = ROOT / "scripts/adafe_diagnostics"
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(scripts.iterdir()) if p.is_file()}
    save_json(output / "code_manifest.json", manifest)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"runs","audit_datasets","phases"}}))
    if (pending or not all_finished) and not options.allow_incomplete:
        raise RuntimeError("Not all planned diagnostic phases have finished successfully")

if __name__ == "__main__":
    main()
