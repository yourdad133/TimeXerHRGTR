"""Isolate the output head's temporal-mean projection at fixed epoch12 features."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.baseline_gtr.run_gtr_guided_adafe import GuidedExperiment, check_remote_runtime
from scripts.adafe_diagnostics.common import seed_all
from scripts.electricity720_diagnostic.analysis_metrics import _data_metadata, _evaluate_split

def main():
    run = Path(os.environ["DIAGNOSTIC_DIR"]).resolve()
    args = SimpleNamespace(**json.loads((run / "config.json").read_text())["args"])
    check_remote_runtime(args)
    seed_all(args.random_seed)
    experiment = GuidedExperiment(args)
    load = lambda name: torch.load(run / "snapshots" / name, map_location="cpu", weights_only=False)
    best = load("validation_best.pth")
    last = load("epoch12.pth")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"]
    experiment.model.load_state_dict(last["model_state_dict"], strict=True)
    head = experiment.model.head.linear
    weight = last["model_state_dict"]["head.linear.weight"]
    bias = last["model_state_dict"]["head.linear.bias"]
    old_weight = best["model_state_dict"]["head.linear.weight"]
    old_bias = best["model_state_dict"]["head.linear.bias"]
    assert head.out_features == args.pred_len == 720
    with torch.no_grad():
        head.weight.copy_(weight - weight.mean(dim=0, keepdim=True) + old_weight.mean(dim=0, keepdim=True))
        head.bias.copy_(bias - bias.mean() + old_bias.mean())
    names, months, boundaries = _data_metadata(args)
    result = {"definition": "At epoch12, replace only temporal-row mean of head.linear weight/bias by validation-best values.",
              "reference_epoch": int(best["epoch"]),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "splits": {}}
    try:
        for split in ("val", "test"):
            dataset, loader = experiment._get_data(split)
            records = _evaluate_split(experiment, args, dataset, loader, names, months, boundaries[split], "mean_head_to_best", split)
            actual = records["overall"]
            baseline = summary["snapshots"]["epoch12"]["splits"][split]["overall"]
            fields = ("mse", "window_bias_mse", "window_amplitude_mse", "window_shape_mse", "mean_signed_error")
            delta = {field: actual[field] - baseline[field] for field in fields}
            assert abs(delta["window_amplitude_mse"]) < 1e-6
            assert abs(delta["window_shape_mse"]) < 1e-6
            result["splits"][split] = {"overall": actual, "delta_from_epoch12": delta}
            print("MEAN_HEAD_PROBE " + split + " " + json.dumps(delta), flush=True)
        (run / "mean_head_rollback.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        experiment.model.load_state_dict(best["model_state_dict"], strict=True)
        experiment.model.eval()
    print("MEAN_HEAD_PROBE_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
