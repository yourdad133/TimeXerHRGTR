"""Freeze the current historical Electricity evidence and probe newly completed STE checkpoints."""
import argparse
import ast
import gc
import re
from datetime import datetime
from pathlib import Path
import hashlib

import torch
from torch.utils.data._utils.collate import default_collate

from common import ROOT, Model, dataset_for, forward_batch, runtime, save_json, seed_all
from checkpoint_probe import parse_log
from variants import capture, gradients, install, snapshot

FOLDERS = {
    "baseline": "TimeXerGTR_manual",
    "hard": "TimeXerGTR_AdaFE_Hard_Electricity",
    "ste": "TimeXerGTR_AdaFE_ste_Electricity",
}

def scan():
    records = []
    for mode, folder in FOLDERS.items():
        for path in sorted((ROOT/"experiment_logs"/folder).glob("Electricity_*flatDaily*.log")):
            raw = path.read_text()
            namespace = re.search(r"Namespace\([^\r\n]*\)", raw)
            if not namespace:
                continue
            node = ast.parse(namespace.group(0), mode="eval").body
            args = {kw.arg:ast.literal_eval(kw.value) for kw in node.keywords}
            final = list(re.finditer(r"^mse:([\d.eE+-]+), mae:([\d.eE+-]+)", raw, re.MULTILINE))
            epochs = list(re.finditer(r"Epoch: (\d+), Steps: \d+ \| Train Loss: ([\d.eE+-]+) Vali Loss: ([\d.eE+-]+) Test Loss: ([\d.eE+-]+)", raw))
            seed = args.get("random_seed")
            if seed is None:
                candidate = re.search(r"_(202[678])_", path.name)
                seed = int(candidate.group(1)) if candidate else None
            records.append({
                "file": str(path.relative_to(ROOT)), "mode": mode, "seed": seed,
                "horizon": args["pred_len"], "completed": bool(final),
                "mse": float(final[-1].group(1)) if final else None,
                "mae": float(final[-1].group(2)) if final else None,
                "epochs_recorded": len(epochs),
                "best_logged_validation": min(float(match.group(3)) for match in epochs) if epochs else None,
                "last_modified_local": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "sha256": hashlib.sha256(raw.encode()).hexdigest(), "args": args,
            })
    baselines = {(r["seed"],r["horizon"]):r for r in records if r["mode"]=="baseline" and r["completed"]}
    for row in records:
        base = baselines.get((row["seed"],row["horizon"]))
        if row["mode"]!="baseline" and row["completed"] and base:
            row["baseline_mse"] = base["mse"]
            row["test_mse_change_pct"] = 100*(row["mse"]/base["mse"]-1)
            row["baseline_patience"] = base["args"]["patience"]
            row["adafe_patience"] = row["args"]["patience"]
    return records

def probe(path, device):
    args, setting = parse_log(path)
    args.root_path = str(ROOT/"dataset")
    seed_all(args.random_seed)
    model = install(Model(args),"ste").to(device)
    initial = {name:p.detach().clone() for name,p in model.adafe_adapter.adafe.score.named_parameters()}
    model.load_state_dict(torch.load(ROOT/"checkpoints"/setting/"checkpoint.pth",
                                    map_location=device,weights_only=True),strict=True)
    deltas = {name:float((p.detach()-initial[name]).norm()) for name,p in model.adafe_adapter.adafe.score.named_parameters()}
    validation = dataset_for(args,"val")
    batch = default_collate([validation[i] for i in [0,1,800,1600]])
    model.eval()
    capture(model)
    prediction, target = forward_batch(model,batch,device)
    loss = torch.nn.functional.mse_loss(prediction,target)
    loss.backward()
    result = {
        "file": str(path.relative_to(ROOT)), "checkpoint": str(Path("checkpoints")/setting/"checkpoint.pth"),
        "validation_indices":[0,1,800,1600], "scorer_parameter_delta_norms":deltas,
        "scorer_changed_tensors": sum(value>0 for value in deltas.values()),
        "validation_batch_mse":float(loss), **snapshot(model), **gradients(model),
    }
    del model, initial, validation, batch, prediction, target, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    options = parser.parse_args()
    output = Path(options.output)
    device = runtime(output)
    records = scan()
    probes = []
    for row in records:
        if row["mode"]=="ste" and row["completed"]:
            try:
                probes.append({"state":"completed",**probe(ROOT/row["file"],device)})
            except Exception as error:
                probes.append({"state":"unavailable","file":row["file"],"error":f"{type(error).__name__}: {error}"})
    save_json(output/"historical_snapshot.json",{
        "snapshot_local_time":datetime.now().astimezone().isoformat(),
        "records":records,"completed_ste_probes":probes,
        "interpretation":"Historical patience differs; ongoing cases have no final comparison. Fixed-window gradient probes do not use test labels.",
    })
    print({"historical_completed":sum(r["completed"] for r in records),"historical_pending":sum(not r["completed"] for r in records),
           "completed_ste_probe_states":[p["state"] for p in probes]})

if __name__=="__main__":
    main()
