"""Separate omitted-validation-window noise from batch regrouping on frozen models."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checkpoint_probe import parse_log
from common import ROOT, Model, dataset_for, evaluate, forward_batch, loader_for, runtime, save_json, seed_all

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    options=parser.parse_args()
    device=runtime(options.output)
    records=[]
    for label,relative in [
        ("Electricity96","TimeXerGTR_AdaFE_ste_Electricity/Electricity_96_2026_flatDaily_AdaFEste.log"),
        ("Electricity336","TimeXerGTR_AdaFE_Hard_Electricity/Electricity_336_2027_flatDaily_AdaFEHard.log")]:
        args,setting=parse_log(ROOT/"experiment_logs"/relative)
        args.root_path=str(ROOT/"dataset")
        seed_all(args.random_seed)
        model=Model(args).to(device).eval()
        model.load_state_dict(torch.load(ROOT/"checkpoints"/setting/"checkpoint.pth",
                                         map_location=device,weights_only=True),strict=True)
        data=dataset_for(args,"val")
        indices=np.arange(len(data),dtype=np.int64)
        ordered=loader_for(data,32,indices,False,args.random_seed)
        per_window=[]
        with torch.inference_mode():
            for batch in ordered:
                prediction,target=forward_batch(model,batch,device)
                per_window.extend((prediction-target).double().square().mean(dim=(1,2)).cpu().tolist())
        per_window=np.asarray(per_window)
        retained=len(data)//32*32
        dropped=len(data)-retained
        permutation_metrics=[]
        for order_seed in [731,947,1223]:
            order=indices.copy()
            np.random.default_rng(order_seed).shuffle(order)
            keep=order[:retained]
            actual=evaluate(model,loader_for(data,32,keep,False,args.random_seed),device)
            permutation_metrics.append({"seed":order_seed,"retained":retained,"dropped":dropped,
                                        "actual_mse":actual["mse"],"cached_window_mse":float(per_window[keep].mean())})
            print("VALIDATION_REPEAT "+label+" "+str(order_seed)+" "+str(actual["mse"]),flush=True)
        rng=np.random.default_rng(20261011)
        simulated=np.asarray([per_window[rng.permutation(len(data))[:retained]].mean() for _ in range(1000)])
        row={"dataset":label,"windows":len(data),"batch_size":32,"randomly_omitted":dropped,
             "ordered_all_windows_mse":float(per_window.mean()),
             "random_drop_repeats":permutation_metrics,
             "omission_only_simulations":1000,
             "omission_only_std":float(simulated.std(ddof=1)),
             "omission_only_quantiles_025_50_975":np.quantile(simulated,[.025,.5,.975]).tolist(),
             "note":"Cached-window simulation holds chronological-batch predictions fixed; actual repeats recompute predictions."}
        records.append(row)
        save_json(Path(options.output)/"validation_stability.json",records)
        del model,data,ordered
        torch.cuda.empty_cache()

if __name__=="__main__":
    main()
