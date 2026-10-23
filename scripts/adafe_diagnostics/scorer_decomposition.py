"""Inspect constant-basis redundancy and measure its Adam update amplification."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from checkpoint_probe import parse_log
from common import ROOT, Model, config, dataset_for, forward_batch, runtime, save_json, seed_all

def embedding(model,x,cycle):
    x=x.float()
    means=x.mean(dim=1,keepdim=True)
    centered=x-means
    x=centered/torch.sqrt(centered.var(dim=1,keepdim=True,unbiased=False)+1e-5) if model.use_norm else x
    enhanced=model._retrieve(x,cycle.int())
    tokens,_=model.en_embedding(enhanced.transpose(1,2))
    return tokens[:,:-1,:].reshape(-1,tokens.shape[-1])

def decomposition(model,args,device):
    data=dataset_for(args,"val")
    batch=default_collate([data[index] for index in [0,1,800,1600]])
    with torch.inference_mode():
        tokens=embedding(model,batch[0].to(device),batch[4].to(device))
        scorer=model.adafe_adapter.adafe.score
        scores=scorer(tokens)
        const_input=scorer.cos_coeffs[:,:,0].sum(dim=0,keepdim=True)+scorer.poly_coeffs[:,:,0].sum(dim=0,keepdim=True)+scorer.bias
        constant=scorer.interaction(const_input)
        dynamic=scores-constant
        centered_constant=constant-constant.mean()
        centered_dynamic=dynamic-dynamic.mean()
        probability=(scores>scores.mean()).float().mean(dim=0)
        return {"score_std":float(scores.std(unbiased=False)),
                "constant_std":float(constant.std(unbiased=False)),
                "dynamic_std":float(dynamic.std(unbiased=False)),
                "constant_dynamic_covariance":float((centered_constant*centered_dynamic).mean()),
                "constant_variance":float(constant.var(unbiased=False)),
                "dynamic_variance":float(dynamic.var(unbiased=False)),
                "score_variance":float(scores.var(unbiased=False)),
                "feature_mask_probability_q05_q50_q95":torch.quantile(probability,torch.tensor([.05,.5,.95],device=device)).tolist(),
                "feature_always_on_or_off_fraction":float(((probability==0)|(probability==1)).float().mean()),
                "feature_nearly_static_fraction":float(((probability<.01)|(probability>.99)).float().mean())}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    options=parser.parse_args()
    device=runtime(options.output)
    args=config("ETTh2",2026,batch=4)
    seed_all(2026)
    model=Model(args).to(device)
    data=dataset_for(args,"train")
    batch=default_collate([data[index] for index in [0,1,2,3]])
    score=model.adafe_adapter.adafe.score
    before_const=(score.cos_coeffs[:,:,0].sum(0)+score.poly_coeffs[:,:,0].sum(0)).detach().clone()
    before_bias=score.bias.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    prediction,target=forward_batch(model,batch,device)
    torch.nn.functional.mse_loss(prediction,target).backward()
    cosine=score.cos_coeffs.grad[:,:,0]
    polynomial=score.poly_coeffs.grad[:,:,0]
    row={"hidden_dim":args.d_model,
         "constant_cos_grad_row_difference":float((cosine-cosine[:1]).abs().max()),
         "constant_poly_grad_row_difference":float((polynomial-polynomial[:1]).abs().max()),
         "cos_poly_grad_difference":float((cosine-polynomial).abs().max()),
         "constant_vs_bias_grad_difference":float((cosine[0]-score.bias.grad[0]).abs().max())}
    optimizer.step()
    constant_delta=score.cos_coeffs[:,:,0].sum(0)+score.poly_coeffs[:,:,0].sum(0)-before_const
    bias_delta=score.bias.detach()-before_bias
    row.update(constant_delta_norm=float(constant_delta.norm()),bias_delta_norm=float(bias_delta.norm()),
               constant_update_amplification=float(constant_delta.norm()/bias_delta.norm()),
               expected_amplification=2*args.d_model)
    del model,data,optimizer
    torch.cuda.empty_cache()
    logfile=ROOT/"experiment_logs/TimeXerGTR_AdaFE_ste_Electricity/Electricity_96_2026_flatDaily_AdaFEste.log"
    args,setting=parse_log(logfile)
    args.root_path=str(ROOT/"dataset")
    seed_all(args.random_seed)
    model=Model(args).to(device).eval()
    model.load_state_dict(torch.load(ROOT/"checkpoints"/setting/"checkpoint.pth",map_location=device,weights_only=True))
    result={"initial_adam_step":row,"electricity_checkpoint":decomposition(model,args,device),
            "interpretation":"Two repeated constant bases yield 2*D redundant contributions. These updates are distinct from direct bias learning."}
    save_json(Path(options.output)/"scorer_decomposition.json",result)
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()
