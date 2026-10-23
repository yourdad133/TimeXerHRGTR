"""Frozen validation interventions for the trained GTR-guided AdaFE pilot.

Only the guide's phase is changed; the GTR main route keeps its correct
cycle index. These interventions are not retrained baselines and do not
establish an independent test-set improvement. Execute via remote tslib.
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.TimeXerGTRGuidedAdaFE import Model as GuidedModel
from scripts.adafe_diagnostics.common import (
    dataset_for, evaluate, loader_for, runtime, save_json, seed_all,
)

ARM = "guided_with_prior_norm1"
BATCH_SIZE = 16
MEMORY_FRACTION = 0.35


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def resolve_project_path(path):
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Run and output paths must stay inside the configured project")
    return resolved


def current_source_hashes(expected):
    hashes = {}
    for name in expected:
        path = resolve_project_path(Path(name))
        hashes[name] = sha256(path.read_bytes())
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Completed pilot root containing runs/ and selected_indices.npz")
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty project directory for runtime.json and results.json")
    options = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("This isolated validation probe requires CUDA_VISIBLE_DEVICES=0")
    try:
        run = resolve_project_path(options.run)
        output = resolve_project_path(options.output)
    except ValueError as error:
        parser.error(str(error))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be new or empty; existing probe results are never overwritten")

    arm_directory = run / "runs" / ARM
    configuration_path = arm_directory / "configuration.json"
    checkpoint_path = arm_directory / "best.pth"
    index_path = run / "selected_indices.npz"
    result_path = arm_directory / "result.json"
    required = (configuration_path, checkpoint_path, index_path, result_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing completed-pilot inputs: " + ", ".join(missing))

    # Read immutable input snapshots once, including the exact checkpoint bytes
    # later passed to torch.load. Concurrent replacement cannot mix hash and load.
    configuration_bytes = configuration_path.read_bytes()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = sha256(checkpoint_bytes)
    index_bytes = index_path.read_bytes()
    result_bytes = result_path.read_bytes()
    configuration = json.loads(configuration_bytes)
    recorded_result = json.loads(result_bytes)
    if configuration.get("arm") != ARM or recorded_result.get("state") != "completed":
        parser.error("The guide-enabled pilot arm must be completed")
    args = argparse.Namespace(**configuration["args"])
    if not args.guided_adafe_use_prior:
        parser.error("The checkpoint must have been trained with the periodic guide enabled")
    if args.data != "PEMS" or args.data_path not in ("PEMS04.npz", "PEMS07.npz"):
        parser.error("This probe is restricted to the paired PEMS04/PEMS07 pilot")
    expected_sources = configuration.get("source_hashes", {})
    if not expected_sources:
        parser.error("The pilot configuration must include frozen source hashes")
    source_hashes = current_source_hashes(expected_sources)
    if source_hashes != expected_sources:
        changed = [name for name in expected_sources
                   if source_hashes[name] != expected_sources[name]]
        parser.error("Pilot source changed before the frozen probe: " + ", ".join(changed))

    with np.load(io.BytesIO(index_bytes), allow_pickle=False) as selected:
        if "val" not in selected:
            parser.error("selected_indices.npz must contain val indices")
        val_indices = selected["val"].copy()
    if (val_indices.ndim != 1 or not len(val_indices)
            or not np.issubdtype(val_indices.dtype, np.integer)):
        parser.error("Validation indices must be a nonempty integer vector")
    index_hash = sha256(val_indices.tobytes())
    if index_hash != configuration["index_hashes"]["val"]:
        parser.error("Validation indices differ from the completed pilot")

    # This rejects local execution and checks host, remote project, tslib
    # Python, nvidia-smi, and CUDA before constructing CUDA model state.
    device = runtime(output)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    runtime_record = json.loads((output / "runtime.json").read_text())
    runtime_record["per_process_memory_fraction"] = MEMORY_FRACTION
    save_json(output / "runtime.json", runtime_record)
    seed_all(args.random_seed)

    # Only validation is constructed. The dataset loader fits its scaler from
    # the training partition as usual; no test dataset or test metric is used.
    validation_dataset = dataset_for(args, "val")
    if np.any(val_indices < 0) or np.any(val_indices >= len(validation_dataset)):
        raise ValueError("Recorded validation indices are outside the current dataset")
    if len(val_indices) != configuration["selected_windows"]["val"]:
        raise ValueError("Validation window count differs from the completed pilot")
    if len(validation_dataset) != configuration["dataset_windows"]["val"]:
        raise ValueError("Validation dataset size differs from the completed pilot")
    loader = loader_for(validation_dataset, BATCH_SIZE, val_indices, False,
                        args.random_seed)

    model = GuidedModel(args)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state, checkpoint_bytes
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.adafe_adapter.adafe.capture_diagnostics = False
    original_guide = model._period_guide
    original_model_prior = model.guided_adafe_use_prior
    original_adapter_prior = model.adafe_adapter.adafe.use_prior

    def shifted_guide(self, endogenous, cycle_index):
        guide_index = torch.as_tensor(cycle_index, device=endogenous.device)
        return original_guide(endogenous, guide_index + self.gtr_cycle_len // 2)

    def fixed_phase_guide(self, endogenous, cycle_index):
        guide_index = torch.as_tensor(cycle_index, device=endogenous.device)
        return original_guide(endogenous, torch.zeros_like(guide_index))

    interventions = (
        ("correctly_aligned", None, True),
        ("half_day_shift", shifted_guide, True),
        ("fixed_phase_zero", fixed_phase_guide, True),
        ("prior_disabled", None, False),
    )
    results = {}
    try:
        for name, replacement, enabled in interventions:
            model._period_guide = (original_guide if replacement is None
                                  else MethodType(replacement, model))
            model.guided_adafe_use_prior = enabled
            model.adafe_adapter.adafe.use_prior = enabled
            metrics = evaluate(model, loader, device)
            if not all(math.isfinite(metrics[key]) for key in ("mse", "mae")):
                raise FloatingPointError("Non-finite frozen-intervention metrics: " + name)
            results[name] = metrics
            print("INTERVENTION " + name + " " + json.dumps(metrics), flush=True)
    finally:
        model._period_guide = original_guide
        model.guided_adafe_use_prior = original_model_prior
        model.adafe_adapter.adafe.use_prior = original_adapter_prior

    aligned = results["correctly_aligned"]
    for metrics in results.values():
        metrics["mse_change_vs_correctly_aligned_pct"] = (
            100.0 * (metrics["mse"] / aligned["mse"] - 1.0)
            if aligned["mse"] > 0 else None)
    if current_source_hashes(expected_sources) != source_hashes:
        raise RuntimeError("Pilot source changed during the frozen validation probe")
    recorded_validation = recorded_result["validation"]
    matches_recorded = all(
        math.isclose(aligned[key], recorded_validation[key], rel_tol=1e-6, abs_tol=1e-8)
        for key in ("mse", "mae"))
    report = {
        "state": "completed",
        "run": str(run),
        "arm": ARM,
        "dataset": args.data_path,
        "seed": args.random_seed,
        "pred_len": args.pred_len,
        "recorded_best_epoch": recorded_result["best_epoch"],
        "batch_size": BATCH_SIZE,
        "validation_windows": len(val_indices),
        "metric_space": "training-scaler standardized",
        "frozen_weights": True,
        "gtr_main_route_cycle_unchanged": True,
        "guide_half_day_shift_steps": model.gtr_cycle_len // 2,
        "test_evaluations": 0,
        "training_steps": 0,
        "results": results,
        "recorded_best_validation": recorded_validation,
        "aligned_matches_recorded_validation": matches_recorded,
        "input_hashes": {
            "configuration_sha256": sha256(configuration_bytes),
            "checkpoint_sha256": checkpoint_hash,
            "selected_indices_file_sha256": sha256(index_bytes),
            "validation_indices_sha256": index_hash,
            "recorded_result_sha256": sha256(result_bytes),
        },
        "source_hashes": source_hashes,
        "probe_source_sha256": sha256(Path(__file__).read_bytes()),
        "interpretation": (
            "Frozen validation interventions on the guide-enabled checkpoint, "
            "not retrained baselines. Only _period_guide receives altered cycle "
            "indices; the GTR main route retains the correct original cycle. "
            "Sensitivity to guide phase does not by itself establish causal "
            "periodicity learning or independent test-set improvement."
        ),
    }
    save_json(output / "results.json", report)
    print("COMPLETED FROZEN GUIDANCE PROBE " + str(output / "results.json"), flush=True)


if __name__ == "__main__":
    main()