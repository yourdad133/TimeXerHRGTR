"""Meaningful GPU contracts for diagnostic variants and evaluation."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import ROOT, config, evaluate, initial_hash, Model, runtime, seed_all
from variants import install
from layers.adafe_v1 import AdaFEAdapter

class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device=runtime(ROOT/"reports/adafe_diagnostics/test_runtime")

    def wrapped(self,adapter,variant):
        model=SimpleNamespace(use_adafe=True,adafe_adapter=adapter)
        install(model,variant)
        return model.adafe_adapter

    def test_legacy_forward_and_gradients(self):
        for mode in ["hard","ste"]:
            torch.manual_seed(71)
            original=AdaFEAdapter(32,mode=mode,constant_reparam=False,proxy_normalize=False).to(self.device)
            diagnostic=self.wrapped(copy.deepcopy(original),mode)
            a=torch.randn(4,7,32,device=self.device,requires_grad=True)
            b=a.detach().clone().requires_grad_()
            first,second=original(a),diagnostic(b)
            self.assertTrue(torch.equal(first,second))
            first.square().sum().backward()
            second.square().sum().backward()
            self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-6,rtol=1e-5))
            for p,q in zip(original.parameters(),diagnostic.parameters()):
                self.assertEqual(p.grad is None,q.grad is None)
                if p.grad is not None:
                    self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-5,rtol=1e-5))

    def test_proxy_scaling_preserves_forward_and_global_token(self):
        torch.manual_seed(19)
        original=AdaFEAdapter(32,mode="ste",constant_reparam=False,proxy_normalize=False).to(self.device)
        diagnostic=self.wrapped(copy.deepcopy(original),"proxy_scaled")
        tokens=torch.randn(3,7,32,device=self.device)
        self.assertTrue(torch.allclose(original(tokens),diagnostic(tokens),atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.equal(diagnostic(tokens)[:,-1],tokens[:,-1]))

    def test_token_threshold_independent_of_other_tokens(self):
        torch.manual_seed(43)
        adapter=self.wrapped(AdaFEAdapter(32,mode="ste",constant_reparam=False,proxy_normalize=False).to(self.device),"token_threshold")
        token=torch.randn(1,32,device=self.device)
        companions=torch.randn(9,32,device=self.device)*12
        alone=adapter.adafe(token)
        grouped=adapter.adafe(torch.cat([token,companions]))[:1]
        self.assertTrue(torch.allclose(alone,grouped,atol=1e-6,rtol=1e-5))

    def test_fixed_gamma_remains_fixed_while_scorer_updates(self):
        torch.manual_seed(53)
        adapter=self.wrapped(AdaFEAdapter(32,mode="ste",constant_reparam=False,proxy_normalize=False).to(self.device),"gamma_fixed")
        initial_gamma=adapter.gamma.detach().clone()
        initial_score=adapter.adafe.score.interaction.weight.detach().clone()
        optimizer=torch.optim.Adam((p for p in adapter.parameters() if p.requires_grad),lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            adapter(torch.randn(4,7,32,device=self.device)).square().mean().backward()
            optimizer.step()
        self.assertTrue(torch.equal(initial_gamma,adapter.gamma))
        self.assertFalse(torch.equal(initial_score,adapter.adafe.score.interaction.weight))

    def test_shared_backbone_and_gtr_initialization(self):
        hashes=[]
        for variant in ["baseline","ste","gamma_fixed","proxy_scaled","token_threshold","stable"]:
            args=config("ETTh2",2026,width=64,enabled=int(variant!="baseline"))
            seed_all(2026)
            model=Model(args)
            install(model,variant)
            hashes.append(initial_hash(model))
        self.assertEqual(len(set(hashes)),1)

    def test_constant_reparam_preserves_initial_function(self):
        torch.manual_seed(73)
        original=AdaFEAdapter(32,mode="ste",constant_reparam=False,proxy_normalize=False).to(self.device)
        diagnostic=self.wrapped(copy.deepcopy(original),"constant_reparam")
        tokens=torch.randn(4,7,32,device=self.device)
        flat=tokens[:,:-1].reshape(-1,32)
        self.assertTrue(torch.allclose(original.adafe.score(flat),diagnostic.adafe.score(flat),atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(original(tokens),diagnostic(tokens),atol=1e-6,rtol=1e-6))

    def test_constant_reparam_removes_redundant_updates(self):
        torch.manual_seed(83)
        adapter=self.wrapped(AdaFEAdapter(32,mode="ste",constant_reparam=False,proxy_normalize=False).to(self.device),"constant_reparam")
        original_bias=adapter.adafe.score.bias.detach().clone()
        optimizer=torch.optim.Adam(adapter.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad()
            adapter(torch.randn(4,7,32,device=self.device)).square().mean().backward()
            optimizer.step()
        self.assertEqual(float(adapter.adafe.score.cos_coeffs[:,:,0].abs().max()),0.)
        self.assertEqual(float(adapter.adafe.score.poly_coeffs[:,:,0].abs().max()),0.)
        self.assertFalse(torch.equal(original_bias,adapter.adafe.score.bias))

    def test_scorer_lr_changes_only_scorer_updates(self):
        from run_suite import make_optimizer
        args=config("ETTh2",2026,width=64,epochs=2)
        seed_all(2026)
        raw=install(Model(args),"ste").to(self.device)
        slow=copy.deepcopy(raw)
        original={name:parameter.detach().clone() for name,parameter in raw.named_parameters()}
        raw_optimizer,_=make_optimizer(raw,"ste",args,10)
        slow_optimizer,_=make_optimizer(slow,"scorer_slow",args,10)
        for model,optimizer in [(raw,raw_optimizer),(slow,slow_optimizer)]:
            for parameter in model.parameters():
                parameter.grad=torch.ones_like(parameter)
            optimizer.step()
        raw_params=dict(raw.named_parameters())
        slow_params=dict(slow.named_parameters())
        for name in ["Q","gamma","en_embedding.value_embedding.weight"]:
            self.assertTrue(torch.equal(raw_params[name],slow_params[name]))
        name="adafe_adapter.adafe.score.bias"
        raw_delta=(raw_params[name]-original[name]).norm()
        slow_delta=(slow_params[name]-original[name]).norm()
        self.assertAlmostEqual(float(slow_delta/raw_delta),.1,places=5)

    def test_weighted_metrics_include_partial_batch(self):
        class Zero(nn.Module):
            pred_len=1
            def forward(self,x,xm,decoder,ym,cycle):
                return torch.zeros_like(x[:,-1:,:])
        samples=[]
        for value in [1.,1.,3.]:
            samples.append((torch.zeros(1,1),torch.tensor([[value]]),
                            torch.zeros(1,4),torch.zeros(1,4),torch.tensor(0)))
        metrics=evaluate(Zero().to(self.device),DataLoader(samples,batch_size=2),self.device)
        self.assertAlmostEqual(metrics["mse"],11/3)
        self.assertEqual(metrics["target_values"],3)

if __name__=="__main__":
    unittest.main(verbosity=2)
