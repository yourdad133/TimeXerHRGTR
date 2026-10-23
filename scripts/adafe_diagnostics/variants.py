"""Experimental AdaFE variants; legacy project modules remain untouched."""
from copy import deepcopy

import torch
import torch.nn.functional as F

from layers.adafe_v1 import AdaFELocal, AdaFEAdapter

VARIANTS = {
    "baseline": dict(enabled=False),
    "hard": dict(enabled=True, mode="hard", threshold="batch", proxy="raw", gamma="learned"),
    "ste": dict(enabled=True, mode="ste", threshold="batch", proxy="raw", gamma="learned"),
    "gamma_fixed": dict(enabled=True, mode="ste", threshold="batch", proxy="raw", gamma="fixed"),
    "proxy_scaled": dict(enabled=True, mode="ste", threshold="batch", proxy="scaled", gamma="learned"),
    "token_threshold": dict(enabled=True, mode="ste", threshold="token", proxy="raw", gamma="learned"),
    "scorer_slow": dict(enabled=True, mode="ste", threshold="batch", proxy="raw", gamma="learned", scorer_lr_scale=0.1),
    "constant_reparam": dict(enabled=True, mode="ste", threshold="batch", proxy="raw", gamma="learned"),
    "stable": dict(enabled=True, mode="ste", threshold="token", proxy="scaled", gamma="fixed"),
}

class DiagnosticLocal(AdaFELocal):
    def forward(self, x):
        scores = self.score(x)
        threshold = scores.mean() if self.threshold_scope == "batch" else scores.mean(dim=-1, keepdim=True)
        hard = (scores > threshold).to(x.dtype)
        scale = torch.ones((), device=x.device, dtype=x.dtype)
        if self.proxy_kind == "scaled":
            if self.threshold_scope == "batch":
                scale = scores.detach().std(unbiased=False).clamp_min(1e-4)
            else:
                scale = scores.detach().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        soft = torch.sigmoid((scores - threshold) / (self.tau * scale))
        active = hard if self.mode == "hard" else (hard - soft).detach() + soft
        output = F.conv1d((x * active).unsqueeze(1), self.weight, padding=self.padding).squeeze(1)
        self.last_active_ratio = hard.detach().mean()
        if self.capture_diagnostics:
            self.snapshot = {
                "score_std": float(scores.detach().std(unbiased=False)),
                "active_ratio": float(hard.detach().mean()),
                "proxy_saturated_fraction": float(((soft < .01) | (soft > .99)).float().mean()),
                "proxy_scale_mean": float(scale.mean()),
            }
        return output

class DiagnosticAdapter(AdaFEAdapter):
    def forward(self, en_embed):
        patches = en_embed[:, :-1, :]
        global_token = en_embed[:, -1:, :]
        shape = patches.shape
        extracted = self.adafe(patches.reshape(-1, shape[-1])).reshape(shape)
        residual = self.gamma * self.norm(extracted)
        if self.adafe.capture_diagnostics:
            self.adafe.snapshot.update(
                gamma=float(self.gamma.detach()),
                residual_to_patch_norm=float(residual.detach().norm() / patches.detach().norm().clamp_min(1e-12)),
            )
        return torch.cat([patches + residual, global_token], dim=1)

def install(model, variant):
    options = VARIANTS[variant]
    if not options["enabled"]:
        return model
    adapter = model.adafe_adapter
    adapter.__class__ = DiagnosticAdapter
    adapter.adafe.__class__ = DiagnosticLocal
    adapter.adafe.mode = options["mode"]
    adapter.adafe.threshold_scope = options["threshold"]
    adapter.adafe.proxy_kind = options["proxy"]
    adapter.adafe.capture_diagnostics = False
    adapter.adafe.snapshot = {}
    if variant == "constant_reparam":
        scorer = adapter.adafe.score
        # Two constant bases are exactly redundant with the explicit scorer bias.
        # Fold their initial contribution into that bias, preserving the function.
        with torch.no_grad():
            scorer.bias.add_(scorer.cos_coeffs[:, :, 0].sum(dim=0, keepdim=True)
                             + scorer.poly_coeffs[:, :, 0].sum(dim=0, keepdim=True))
            scorer.cos_coeffs[:, :, 0].zero_()
            scorer.poly_coeffs[:, :, 0].zero_()
        def freeze_constant_column(gradient):
            gradient = gradient.clone()
            gradient[:, :, 0] = 0
            return gradient
        adapter.constant_hook_handles = [
            scorer.cos_coeffs.register_hook(freeze_constant_column),
            scorer.poly_coeffs.register_hook(freeze_constant_column),
        ]
    if options["gamma"] == "fixed":
        adapter.gamma.requires_grad_(False)
    return model

def capture(model, enabled=True):
    if getattr(model, "use_adafe", False):
        model.adafe_adapter.adafe.capture_diagnostics = enabled

def snapshot(model):
    return deepcopy(model.adafe_adapter.adafe.snapshot) if getattr(model, "use_adafe", False) else {}

def gradients(model):
    groups = {"scorer": [], "gtr": [], "backbone": []}
    for name, parameter in model.named_parameters():
        group = "scorer" if ".adafe.score." in name else (
            "gtr" if name.startswith(("Q", "GTR.")) or name == "gamma" else "backbone")
        if parameter.requires_grad:
            groups[group].append(parameter)
    output = {}
    for group, parameters in groups.items():
        existing = [parameter.grad for parameter in parameters if parameter.grad is not None]
        output[group + "_missing_grad_tensors"] = len(parameters) - len(existing)
        output[group + "_grad_norm"] = sum(float(gradient.detach().double().square().sum()) for gradient in existing) ** .5
        output[group + "_grad_finite"] = all(bool(torch.isfinite(gradient).all()) for gradient in existing)
    return output
