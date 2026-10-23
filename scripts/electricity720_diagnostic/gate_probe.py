"""One-at-a-time gate sensitivity checks after the retained-snapshot analysis."""
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


def gate_rollback_probe(experiment, snapshot4, snapshot12):
    model = experiment.model
    state4 = snapshot4["model_state_dict"]
    state12 = snapshot12["model_state_dict"]
    original = {name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()}
    was_training = model.training
    criterion = experiment._select_criterion()
    datasets = {split: experiment._get_data(split) for split in ("val", "test")}

    def score():
        return {split: float(experiment.vali(dataset, loader, criterion))
                for split, (dataset, loader) in datasets.items()}

    gates = {
        "gtr_gamma_to_epoch4": "gamma",
        "adafe_gamma_to_epoch4": "adafe_adapter.gamma",
        "guide_gate_to_epoch4": "adafe_adapter.adafe.prior_gate",
    }
    results = {}
    try:
        model.load_state_dict(state12, strict=True)
        model.eval()
        baseline = score()
        results["epoch12_baseline"] = baseline
        print("Gate probe baseline " + json.dumps(baseline), flush=True)
        for label, name in gates.items():
            model.load_state_dict(state12, strict=True)
            parameter = dict(model.named_parameters())[name]
            with torch.no_grad():
                parameter.copy_(state4[name].to(device=parameter.device, dtype=parameter.dtype))
            metrics = score()
            results[label] = {
                **metrics,
                "delta_val_from_epoch12": metrics["val"] - baseline["val"],
                "delta_test_from_epoch12": metrics["test"] - baseline["test"],
                "raw_gate_epoch4": float(state4[name].item()),
                "raw_gate_epoch12": float(state12[name].item()),
            }
            print(label + " " + json.dumps(results[label]), flush=True)
        return results
    finally:
        model.load_state_dict(original, strict=True)
        model.train(was_training)


def main():
    run_dir = Path(os.environ["DIAGNOSTIC_DIR"]).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    args = SimpleNamespace(**config["args"])
    check_remote_runtime(args)
    summary = json.loads((run_dir / "summary.json").read_text())
    if not summary["complete"]:
        raise RuntimeError("Complete retained-snapshot evaluation before the gate probe.")
    seed_all(args.random_seed)
    experiment = GuidedExperiment(args)
    load = lambda name: torch.load(run_dir / "snapshots" / name, map_location="cpu", weights_only=False)
    best = load("validation_best.pth")
    experiment.model.load_state_dict(best["model_state_dict"], strict=True)
    experiment.model.eval()
    result = gate_rollback_probe(experiment, load("epoch04.pth"), load("epoch12.pth"))
    reference = summary["snapshots"]["epoch12"]["splits"]
    for split in ("val", "test"):
        delta = result["epoch12_baseline"][split] - reference[split]["overall"]["mse"]
        if abs(delta) > 1e-6:
            raise RuntimeError(f"Gate-probe baseline mismatch for {split}: {delta}")
    source = Path(__file__).resolve()
    output = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "definition": "Freeze epoch12 state; reset the full state before each one-scalar rollback to epoch4.",
        "interpretation": "Functional sensitivity at fixed epoch12 weights, not a training-causality proof or test-based selection.",
        "results": result,
    }
    (run_dir / "gate_rollback.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("GATE_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
