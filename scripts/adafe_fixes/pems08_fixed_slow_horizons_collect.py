"""Validate and summarize the requested fixed_slow PEMS08 horizon queue."""
import argparse
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_plan(plan):
    assert plan["dataset"] == "PEMS08" and plan["seq_len"] == 96
    assert plan["pred_lens"] == [12, 48, 96]
    assert plan["seeds"] == [2026, 2027]
    assert plan["variants"] == ["fixed_slow"]
    assert plan["adafe_constant_reparam"] == plan["adafe_proxy_normalize"] == 1
    assert plan["adafe_scorer_lr_scale"] == 0.1
    assert plan["full_dataset"] is True
    assert plan["validation_shuffle"] is False
    assert plan["validation_drop_last"] is False
    assert plan["test_used_for_selection"] is False
    jobs = plan["jobs"]
    expected = {(horizon, seed) for horizon in plan["pred_lens"] for seed in plan["seeds"]}
    assert len(jobs) == 6
    assert {(job["pred_len"], job["seed"]) for job in jobs} == expected
    for job in jobs:
        assert job["dataset"] == "PEMS08" and job["variant"] == "fixed_slow"
        assert job["batch_size"] == 32 and job["epochs"] == 30 and job["patience"] == 5
        assert job["full_dataset"] is True and job["max_seconds"] == 14400
        assert job["shard"] == job["seed"] - 2026


def collect(output, plan):
    results, configurations = [], []
    for job in plan["jobs"]:
        case = f'PEMS08_96_{job["pred_len"]}_seed{job["seed"]}_fixed_slow'
        directory = output / "runs" / case
        result = read_json(directory / "result.json")
        configuration = read_json(directory / "configuration.json")
        args = configuration["args"]
        assert configuration["job"] == job
        assert result["case"] == case and result["state"] == "completed"
        assert result["seed"] == job["seed"] and result["pred_len"] == job["pred_len"]
        assert result["variant"] == "fixed_slow" and result["test_evaluations"] == 1
        assert result["full_dataset_training"] is True
        assert args["seq_len"] == 96 and args["pred_len"] == job["pred_len"]
        assert args["use_adafe"] == 1 and args["adafe_mode"] == "ste"
        assert args["adafe_constant_reparam"] == args["adafe_proxy_normalize"] == 1
        assert args["adafe_scorer_lr_scale"] == 0.1
        assert configuration["full_dataset_training"] is True
        assert configuration["validation_shuffle"] is False
        assert configuration["validation_drop_last"] is False
        assert configuration["test_used_for_selection"] is False
        assert configuration["metric_aggregation"] == "target_element_weighted_mean"
        assert configuration["metric_space"] == "standardized"
        assert result["windows"] == configuration["windows"]
        assert result["source_hashes"] == configuration["source_hashes"]
        assert result["test"]["target_values"] == (
            result["windows"]["test"] * job["pred_len"] * args["c_out"])
        assert result["best_diagnostics"]["gradients_finite"] is True
        assert result["best_diagnostics"]["scorer_missing_grad_count"] == 0
        assert all(math.isfinite(result["test"][metric]) for metric in ("mse", "mae"))
        results.append(result)
        configurations.append(configuration)
    assert all(row["source_hashes"] == configurations[0]["source_hashes"] for row in configurations)
    assert len({row["scaler_hash"] for row in configurations}) == 1
    means = {}
    for horizon in plan["pred_lens"]:
        matched = [row for row in configurations if row["job"]["pred_len"] == horizon]
        comparable = [{key: value for key, value in row["args"].items()
                       if key not in {"random_seed"}} for row in matched]
        assert comparable[0] == comparable[1]
        assert matched[0]["index_hashes"] == matched[1]["index_hashes"]
        rows = [row for row in results if row["pred_len"] == horizon]
        means[horizon] = {f"{metric}_{statistic}": operation(
            [row["test"][metric] for row in rows])
            for metric in ("mse", "mae")
            for statistic, operation in (("mean", statistics.mean), ("sd", statistics.stdev))}
    summary = dict(plan=plan, results=results, means_by_horizon=means,
                   verification="passed", scope="PEMS08 full data, fixed_slow only, two seeds per horizon")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# PEMS08 修复 AdaFE＋打分器学习率×0.1",
        "",
        "输入96；预测12、48、96；种子2026、2027；每项使用完整数据划分。",
        "最多30轮，早停耐心值5，batch=32、d_model=512；按验证MSE选模，选定后测试一次。",
        "验证集固定顺序并保留最后不足一批的窗口；指标按目标元素加权，使用标准化空间。",
        "",
        "| 预测步长 | 种子 | 测试MSE | 测试MAE | 最佳轮 | 实际训练轮 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(f'| {row["pred_len"]} | {row["seed"]} | {row["test"]["mse"]:.7f} | '
                     f'{row["test"]["mae"]:.7f} | {row["best_epoch"]} | {row["epochs_completed"]} |')
    lines.extend(["", "| 预测步长 | MSE均值±样本标准差 | MAE均值±样本标准差 |",
                  "|---:|---:|---:|"])
    for horizon, values in means.items():
        lines.append(f'| {horizon} | {values["mse_mean"]:.7f} ± {values["mse_sd"]:.7f} | '
                     f'{values["mae_mean"]:.7f} ± {values["mae_sd"]:.7f} |')
    lines.extend(["", "本轮仅运行用户指定方案；是否性能倒退需要同一步长和种子的对照结果。", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("HORIZON_SUMMARY", json.dumps(means), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    plan = read_json(Path(options.plan))
    validate_plan(plan)
    if options.validate_only:
        print("PLAN_VALIDATED: fixed_slow only; 12/48/96; seeds 2026/2027; six jobs", flush=True)
        return
    collect(Path(options.output), plan)


if __name__ == "__main__":
    main()
