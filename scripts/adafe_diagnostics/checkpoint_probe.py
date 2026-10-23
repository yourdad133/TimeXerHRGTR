"""Validation-only interventions on existing checkpoints; test labels are not used."""
import argparse
import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from common import ROOT, Model, dataset_for, evaluate, forward_batch, loader_for, runtime, save_json, seed_all, window_indices
from variants import capture, gradients, install, snapshot

LOGS = [
    ("Electricity","TimeXerGTR_AdaFE_ste_Electricity/Electricity_96_2026_flatDaily_AdaFEste.log"),
    ("Electricity336","TimeXerGTR_AdaFE_Hard_Electricity/Electricity_336_2027_flatDaily_AdaFEHard.log"),
    ("PEMS08","TimeXerGTR_AdaFE_ste_PEMS08/PEMS08_96_24_seed2027_AdaFESte.log"),
]

def parse_log(path):
    raw = path.read_text()
    node = ast.parse(re.search(r"Namespace\([^\r\n]*\)",raw).group(0),mode="eval").body
    args = argparse.Namespace(**{kw.arg:ast.literal_eval(kw.value) for kw in node.keywords})
    matches = list(re.finditer(r">>>>>>>testing : (.+?)<{6,}",raw))
    if not matches:
        raise ValueError("Completed testing setting absent")
    return args,matches[-1].group(1).strip()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--windows",type=int,default=512)
    options=parser.parse_args()
    device=runtime(options.output)
    rows=[]
    for label,relative in LOGS:
        path=ROOT/"experiment_logs"/relative
        if not path.exists() and label=="PEMS08":
            candidates=sorted((ROOT/"experiment_logs/TimeXerGTR_AdaFE_ste_PEMS08").glob("*24*2027*.log"))
            if candidates:
                path=candidates[0]
        row={"dataset":label,"log":str(path.relative_to(ROOT))}
        started=time.time()
        try:
            args,setting=parse_log(path)
            args.root_path=str(ROOT/"dataset")
            seed_all(args.random_seed)
            model=Model(args).to(device).eval()
            model.load_state_dict(torch.load(ROOT/"checkpoints"/setting/"checkpoint.pth",map_location=device,weights_only=True),strict=True)
            validation=dataset_for(args,"val")
            indices=window_indices(len(validation),options.windows)
            loader=loader_for(validation,16,indices,False,args.random_seed)
            original_gamma=float(model.adafe_adapter.gamma)
            sweeps=[]
            for gamma in sorted(set([0.,.05,.1,.2,original_gamma])):
                model.adafe_adapter.gamma.data.fill_(gamma)
                result=evaluate(model,loader,device)
                sweeps.append({"gamma":gamma,**result})
                print(f"CHECKPOINT_GAMMA {label} {gamma:.5f} {result['mse']:.7f}",flush=True)
            model.adafe_adapter.gamma.data.fill_(original_gamma)
            modes=["ste","token_threshold"] if args.adafe_mode=="ste" else ["hard"]
            group_results=[]
            for variant in modes:
                install(model,variant)
                for order_seed in [731,947,1223]:
                    ordered=indices.copy()
                    np.random.default_rng(order_seed).shuffle(ordered)
                    result=evaluate(model,loader_for(validation,16,ordered,False,args.random_seed),device)
                    group_results.append({"variant":variant,"order_seed":order_seed,**result})
            args_index=[0,1,2,3]
            fixed_batch=default_collate([validation[index] for index in args_index])
            proxy_results=[]
            if args.adafe_mode=="ste":
                for variant in ["ste","proxy_scaled"]:
                    install(model,variant)
                    capture(model)
                    model.zero_grad(set_to_none=True)
                    prediction,target=forward_batch(model,fixed_batch,device)
                    torch.nn.functional.mse_loss(prediction,target).backward()
                    proxy_results.append({"variant":variant,**snapshot(model),**gradients(model)})
                    capture(model,False)
            row.update(state="completed",window_count=len(indices),evaluation_batch_size=16,
                original_gamma=original_gamma,gamma_sweep=sweeps,
                group_permutations=group_results,proxy_gradients=proxy_results,
                elapsed_seconds=time.time()-started,
                interpretation="Validation interventions only; gamma=0 retains coadapted backbone weights.")
            del model,validation,loader
            torch.cuda.empty_cache()
        except Exception as error:
            row.update(state="unavailable",error=f"{type(error).__name__}: {error}")
        rows.append(row)
        save_json(Path(options.output)/"checkpoint_interventions.json",rows)
    print("CHECKPOINT_PROBES_COMPLETE",flush=True)

if __name__=="__main__":
    main()
