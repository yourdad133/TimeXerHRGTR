"""GPU audit and matched training, isolated from run.py's logs and results."""
import argparse
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (ROOT, RECIPES, Model, config, dataset_for, evaluate, forward_batch,
                    initial_hash, loader_for, runtime, save_json, seed_all, window_indices)
from variants import VARIANTS, capture, gradients, install, snapshot

def build(args, variant, device):
    args.use_adafe = int(VARIANTS[variant]["enabled"])
    args.adafe_mode = VARIANTS[variant].get("mode", "ste")
    seed_all(args.random_seed)
    model = install(Model(args), variant)
    shared_hash = initial_hash(model)
    return model.to(device), shared_hash

def audit(options, device):
    results = []
    for name in options.datasets:
        args = config(name, 2026, batch=2)
        datasets = {flag: dataset_for(args, flag) for flag in ("train", "val", "test")}
        counts = {flag: len(data) for flag, data in datasets.items()}
        recipe = RECIPES[name]
        for flag, data in datasets.items():
            if len(data) <= 0 or data.data_x.shape[1] != recipe.channels:
                raise RuntimeError(f"{name}/{flag} has an invalid dataset shape")
            if not np.isfinite(data.data_x).all():
                raise RuntimeError(f"{name}/{flag} contains non-finite normalized data")
        if name == "Traffic" and counts["train"] < 10000:
            raise RuntimeError("Traffic data appears truncated")
        row = {"dataset": name, "recipe": vars(recipe), "split_windows": counts, "variants": []}
        for variant in ("baseline", "hard", "ste", "stable"):
            model, shared_hash = build(args, variant, device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
            loader = loader_for(datasets["train"], 2, window_indices(counts["train"], 8), True, 2026)
            torch.cuda.reset_peak_memory_stats()
            capture(model)
            history = []
            for step, batch in enumerate(loader):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pred, target = forward_batch(model, batch, device)
                loss = F.mse_loss(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{name}/{variant} non-finite loss")
                loss.backward()
                diagnostic = {**snapshot(model), **gradients(model), "loss": float(loss), "step": step+1}
                if not all(diagnostic[key] for key in diagnostic if key.endswith("_grad_finite")):
                    raise FloatingPointError(f"{name}/{variant} non-finite gradient")
                optimizer.step()
                history.append(diagnostic)
                if step == 2:
                    break
            row["variants"].append({"variant": variant, "initial_shared_hash": shared_hash,
                "steps": history, "peak_memory_mb": torch.cuda.max_memory_allocated()/1024**2})
            del model, optimizer, loader
            gc.collect()
            torch.cuda.empty_cache()
        if len({case["initial_shared_hash"] for case in row["variants"]}) != 1:
            raise AssertionError(f"{name}: backbone/GTR initialization differs")
        results.append(row)
        save_json(Path(options.output)/"audit.json", results)
        print("AUDIT_COMPLETE " + name, flush=True)
        del datasets
        gc.collect()
    return results

def monitor_batch(model, loader, device):
    model.eval()
    capture(model)
    with torch.inference_mode():
        forward_batch(model, next(iter(loader)), device)
    row = snapshot(model)
    capture(model, False)
    return row

def make_optimizer(model, variant, args, train_steps):
    scorer_scale = VARIANTS[variant].get("scorer_lr_scale", 1.0)
    maximum_lr = args.learning_rate
    if scorer_scale != 1.0:
        scorer_parameters = list(model.adafe_adapter.adafe.score.parameters())
        scorer_ids = {id(parameter) for parameter in scorer_parameters}
        other_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad and id(parameter) not in scorer_ids]
        optimizer = torch.optim.Adam([
            {"params":other_parameters,"lr":args.learning_rate},
            {"params":scorer_parameters,"lr":args.learning_rate*scorer_scale},
        ])
        maximum_lr = [args.learning_rate, args.learning_rate*scorer_scale]
    else:
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=maximum_lr,
        epochs=args.train_epochs,steps_per_epoch=train_steps,pct_start=args.pct_start)
    return optimizer, scheduler

def train_case(job, output, device, cache):
    case_id = f'{job["dataset"]}_seed{job["seed"]}_{job["variant"]}'
    directory = Path(output)/"runs"/case_id
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"result.json").exists():
        print("EXISTING_COMPLETED " + case_id, flush=True)
        return
    started = time.time()
    status = {"case": case_id, "pid": os.getpid(), "state": "running", "started_unix": started,
              "job": job, "heartbeat_unix": started}
    save_json(directory/"status.json", status)
    args = config(job["dataset"], job["seed"], batch=job.get("batch_size",32),
                  epochs=job.get("epochs",30), patience=job.get("patience",10))
    model, shared_hash = build(args, job["variant"], device)
    dataset_name = job["dataset"]
    if dataset_name not in cache:
        cache[dataset_name] = {flag: dataset_for(args, flag) for flag in ("train","val","test")}
    data = cache[dataset_name]
    limits = {"train":job.get("train_windows",0), "val":job.get("validation_windows",0),
              "test":job.get("test_windows",0)}
    indices = {flag:window_indices(len(ds), limits[flag]) for flag,ds in data.items()}
    loaders = {flag:loader_for(ds, args.batch_size, indices[flag], flag=="train", args.random_seed)
               for flag,ds in data.items()}
    actual_windows = {flag:len(index) for flag,index in indices.items()}
    index_hashes = {flag:hashlib.sha256(index.tobytes()).hexdigest() for flag,index in indices.items()}
    save_json(directory/"configuration.json", {"args":vars(args), "variant":VARIANTS[job["variant"]],
              "job":job, "windows":actual_windows, "index_hashes":index_hashes,
              "initial_shared_hash":shared_hash, "validation_shuffle":False,
              "validation_drop_last":False, "metric_aggregation":"sum_error / total_target_values",
              "physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"), "test_used_for_selection":False})
    optimizer, scheduler = make_optimizer(model, job["variant"], args, len(loaders["train"]))
    history, best, best_epoch, stale = [], float("inf"), 0, 0
    checkpoint = directory/"best.pth"
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(1,args.train_epochs+1):
            model.train()
            total, count = 0.0, 0
            last_gradient = {}
            for step,batch in enumerate(loaders["train"],1):
                collect = step == 1 or step == len(loaders["train"])
                capture(model, collect)
                optimizer.zero_grad(set_to_none=True)
                prediction,target = forward_batch(model,batch,device)
                loss = F.mse_loss(prediction,target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                if collect:
                    last_gradient = {**snapshot(model), **gradients(model)}
                    if not all(last_gradient[key] for key in last_gradient if key.endswith("_grad_finite")):
                        raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                scheduler.step()
                total += float(loss)*target.numel()
                count += target.numel()
                if step % 40 == 0:
                    status.update(epoch=epoch,step=step,heartbeat_unix=time.time(),
                                  elapsed_seconds=time.time()-started)
                    save_json(directory/"status.json",status)
                if time.time()-started > job.get("max_seconds",1800):
                    raise TimeoutError("Predeclared per-case time budget exceeded")
            capture(model,False)
            validation = evaluate(model,loaders["val"],device)
            diagnostics = monitor_batch(model,loaders["val"],device)
            row = {"epoch":epoch,"train_mse":total/count,"validation":validation,
                   "diagnostics":diagnostics,"last_train_gradient":last_gradient,
                   "lr":scheduler.get_last_lr()[0],"elapsed_seconds":time.time()-started}
            history.append(row)
            if validation["mse"] < best:
                best,best_epoch,stale = validation["mse"],epoch,0
                torch.save(model.state_dict(),checkpoint)
            else:
                stale += 1
            status.update(epoch=epoch,state="running",heartbeat_unix=time.time(),
                          best_epoch=best_epoch,best_validation_mse=best,elapsed_seconds=time.time()-started)
            save_json(directory/"history.json",history)
            save_json(directory/"status.json",status)
            print(f"EPOCH {case_id} {epoch} train={total/count:.7f} val={validation['mse']:.7f} "
                  f"gamma={diagnostics.get('gamma')} score_std={diagnostics.get('score_std')} "
                  f"sat={diagnostics.get('proxy_saturated_fraction')}",flush=True)
            if stale >= args.patience:
                break
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        test = evaluate(model,loaders["test"],device)
        diagnostics = monitor_batch(model,loaders["val"],device)
        result = {"case":case_id,"dataset":job["dataset"],"seed":job["seed"],"variant":job["variant"],
                  "best_epoch":best_epoch,"epochs_completed":len(history),"validation_mse":best,
                  "test":test,"best_diagnostics":diagnostics,"window_counts":actual_windows,
                  "full_dataset_training":job.get("train_windows",0)==0,
                  "initial_shared_hash":shared_hash,"elapsed_seconds":time.time()-started,
                  "peak_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"state":"completed"}
        save_json(directory/"result.json",result)
        status.update(state="completed",heartbeat_unix=time.time(),elapsed_seconds=time.time()-started)
        save_json(directory/"status.json",status)
        print("COMPLETED " + json.dumps(result),flush=True)
    except Exception as error:
        status.update(state="failed",error=f"{type(error).__name__}: {error}",heartbeat_unix=time.time())
        save_json(directory/"status.json",status)
        print("FAILED " + json.dumps(status),flush=True)
        raise
    finally:
        capture(model,False)
        del model,optimizer,scheduler,loaders
        gc.collect()
        torch.cuda.empty_cache()

def train(options,device):
    plan=json.loads(Path(options.plan).read_text())
    cache={}
    failures=[]
    for index,job in enumerate(plan["jobs"]):
        if index % options.shard_count != options.shard_index:
            continue
        try:
            train_case(job,options.output,device,cache)
        except Exception as error:
            failures.append({"job":job,"error":f"{type(error).__name__}: {error}"})
            case_id=f'{job["dataset"]}_seed{job["seed"]}_{job["variant"]}'
            status_path=Path(options.output)/"runs"/case_id/"status.json"
            status=json.loads(status_path.read_text()) if status_path.exists() else {"case":case_id}
            status.update(state="failed",error=f"{type(error).__name__}: {error}",heartbeat_unix=time.time())
            save_json(status_path,status)
    save_json(Path(options.output)/f"worker{options.shard_index}_finished.json",
              {"state":"completed" if not failures else "completed_with_failures","failures":failures,
               "pid":os.getpid(),"finished_unix":time.time()})
    if failures:
        raise RuntimeError(f"{len(failures)} diagnostic cases failed; details are saved")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","train"])
    parser.add_argument("--output",required=True)
    parser.add_argument("--datasets",nargs="+",default=list(RECIPES))
    parser.add_argument("--plan")
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--shard-count",type=int,default=1)
    options=parser.parse_args()
    device=runtime(Path(options.output)/f"worker{options.shard_index}_runtime" if options.stage=="train" else options.output)
    if options.stage=="audit":
        audit(options,device)
    else:
        if not options.plan:
            parser.error("train requires --plan")
        train(options,device)

if __name__=="__main__":
    main()
