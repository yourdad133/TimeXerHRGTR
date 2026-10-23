"""Frozen validation probes; no weight updates or test-set tuning."""
import argparse, ast, gc, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from models.TimeXerGTR import Model
from scripts.adafe_diagnostics.common import dataset_for, evaluate, forward_batch, loader_for, runtime, save_json
from utils.adafe_reporting import diagnostic_snapshot

def probe(dataset, horizon, device, output, windows):
    log = ROOT / f'experiment_logs/TimeXerGTR_{dataset}_M/{dataset}_96_{horizon}_seed2026.log'
    lines = log.read_text().splitlines()
    call = ast.parse(next(x for x in lines if x.startswith('Namespace(')), mode='eval').body
    args = SimpleNamespace(**{x.arg: ast.literal_eval(x.value) for x in call.keywords})
    args.root_path, args.num_workers = str(ROOT/'dataset'), 0
    setting = next(x.split('start training :',1)[1].strip().strip('>').strip() for x in lines if 'start training :' in x)
    checkpoint = ROOT/'checkpoints'/setting/'checkpoint.pth'
    model = Model(args).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.adafe_adapter.adafe.capture_diagnostics = True
    data = dataset_for(args, 'val')
    indices = np.linspace(0, len(data)-1, windows, dtype=np.int64)
    row = dict(dataset=dataset, horizon=horizon, split='val', windows=windows,
               checkpoint=str(checkpoint.relative_to(ROOT)), args=vars(args))
    loader = loader_for(data, 16, indices, False, 20261013)
    row['frozen_ablations'] = {'both_on': evaluate(model,loader,device)}
    row['batch_size_validation'] = {1: evaluate(model,loader_for(data,1,indices,False,20261013),device), 16: row['frozen_ablations']['both_on']}
    with torch.no_grad():
        ag, gg = model.adafe_adapter.gamma.clone(), model.gamma.clone()
        model.adafe_adapter.gamma.zero_()
        row['frozen_ablations']['adafe_off'] = evaluate(model,loader,device)
        model.adafe_adapter.gamma.copy_(ag)
        model.gamma.zero_()
        row['frozen_ablations']['gtr_off'] = evaluate(model,loader,device)
        model.adafe_adapter.gamma.zero_()
        row['frozen_ablations']['both_off'] = evaluate(model,loader,device)
        model.gamma.copy_(gg)
        model.adafe_adapter.gamma.copy_(ag)
    capture = {}
    hook = model.adafe_adapter.adafe.score.register_forward_hook(
        lambda module, inputs, scores: capture.update(scores=scores.detach()))
    changes, flips = [], []
    with torch.no_grad():
        for anchor in indices[:8]:
            prediction,_ = forward_batch(model, default_collate([data[int(anchor)]]),device)
            scores = capture['scores']
            mask = scores > scores.mean()
            companions = indices[windows//2:windows//2+15]
            batched,_ = forward_batch(model,default_collate([data[int(anchor)]]+[data[int(i)] for i in companions]),device)
            bs = capture['scores']
            flips.append(float((mask != (bs>bs.mean())[:scores.shape[0]]).float().mean()))
            changes.append(float((prediction[0]-batched[0]).abs().max()))
    hook.remove()
    row['batch_companion_probe'] = dict(max_abs_prediction_changes=changes,
        mask_changed_fractions=flips, max_abs_prediction_change=max(changes),
        mean_mask_changed_fraction=float(np.mean(flips)))
    batch = default_collate([data[int(i)] for i in indices[:4]])
    model.zero_grad(set_to_none=True)
    prediction,target = forward_batch(model,batch,device)
    loss = F.mse_loss(prediction,target)
    loss.backward()
    row['best_checkpoint_gradients'] = diagnostic_snapshot(model)
    row['best_checkpoint_gradients']['probe_mse'] = float(loss.detach())
    x,_,_,_,cycle = batch
    x = x.float().to(device)
    with torch.no_grad():
        normalized = (x-x.mean(1,keepdim=True))/(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt()
        pos = (cycle.to(device).long().view(-1,1)-model.seq_len+torch.arange(model.seq_len,device=device))%model.gtr_cycle_len
        prior = model.GTR.linear(model.Q[pos].transpose(1,2))
        weight = model.GTR.conv2d.weight
        shape = normalized.transpose(1,2).shape
        lp = F.conv1d(normalized.transpose(1,2).reshape(-1,1,model.seq_len),weight[:,:,0,:],
                     padding=model.GTR.period_len//2).reshape(shape)
        qp = F.conv1d(prior.reshape(-1,1,model.seq_len),weight[:,:,1,:],
                     padding=model.GTR.period_len//2).reshape(shape)
        rms = lambda t: float(t.float().square().mean().sqrt())
        row['gtr_decomposition'] = dict(gtr_gamma=float(model.gamma),adafe_gamma=float(model.adafe_adapter.gamma),
            q_std=float(model.Q.std()),normalized_input_rms=rms(normalized),
            gated_local_conv_rms=rms(model.gamma*lp),gated_period_conv_rms=rms(model.gamma*qp))
    save_json(output/f'{dataset}_{horizon}.json',row)
    print('PROBE '+json.dumps({k:row[k] for k in ('dataset','horizon','frozen_ablations',
        'batch_companion_probe','best_checkpoint_gradients','gtr_decomposition')}),flush=True)
    del model,prediction,target,loss,data,loader
    gc.collect()
    torch.cuda.empty_cache()
    return row

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['PEMS04','PEMS07'],required=True)
    parser.add_argument('--horizons',type=int,nargs='+',default=[12,96])
    parser.add_argument('--windows',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    device=runtime(options.output)
    save_json(options.output/'summary.json',[probe(options.dataset,h,device,options.output,options.windows) for h in options.horizons])
if __name__=='__main__':
    main()


