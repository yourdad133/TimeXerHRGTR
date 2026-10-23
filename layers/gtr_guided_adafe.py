"""Experimental AdaFE with an explicit, patch-aligned periodic guide.

The scorer is token-local: neither its threshold nor its STE scale depends
on another sample, node, or patch. The existing hidden-feature convolution
and patch residual are retained to isolate the conditioning change.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from layers.adafe_v1 import AdaFELocal


class GTRGuidedAdaFELocal(AdaFELocal):
    """Filter local features using a separately supplied periodic prior."""

    def __init__(self, d_model, degree=2, p_degree=2, kernel_size=5,
                 mode="ste", tau=1.0, constant_reparam=True,
                 proxy_normalize=True, use_prior=True, prior_gate_init=0.05):
        if not math.isfinite(prior_gate_init) or abs(prior_gate_init) >= 1:
            raise ValueError("prior_gate_init must be finite and strictly within (-1, 1).")
        super().__init__(d_model, degree, p_degree, kernel_size, mode, tau,
                         constant_reparam, proxy_normalize)
        self.score_norm = nn.LayerNorm(d_model)
        self.prior_norm = nn.LayerNorm(d_model)
        self.prior_projection = nn.Linear(d_model, d_model, bias=False)
        self.prior_gate = nn.Parameter(torch.tensor(math.atanh(float(prior_gate_init))))
        self.use_prior = bool(use_prior)
        self.last_prior_to_score_ratio = None

    def conditioned_scores(self, x, prior=None):
        """Expose conditioned scores for controlled representation probes."""
        if x.ndim != 2 or x.shape[-1] != self.d_model or x.shape[0] == 0:
            raise ValueError("AdaFE input must be nonempty [tokens, d_model].")
        scores = self.score(self.score_norm(x))
        if self.use_prior:
            if prior is None or prior.shape != x.shape:
                raise ValueError("An enabled periodic prior must match the token matrix.")
            contribution = self.prior_gate.tanh() * self.prior_projection(
                self.prior_norm(prior))
            if self.capture_diagnostics:
                # Only centered contributions can change a token's mean-threshold mask.
                local_centered = scores.detach().float()
                local_centered = local_centered - local_centered.mean(-1, keepdim=True)
                prior_centered = contribution.detach().float()
                prior_centered = prior_centered - prior_centered.mean(-1, keepdim=True)
                self.last_prior_to_score_ratio = (
                    prior_centered.square().mean().sqrt()
                    / local_centered.square().mean().sqrt().clamp_min(1e-8))
            scores = scores + contribution
        elif self.capture_diagnostics:
            self.last_prior_to_score_ratio = scores.new_zeros(())
        return scores

    def surrogate(self, scores, threshold):
        # Normalize each token across features, preserving sample independence.
        scale = (scores.detach().float().std(dim=-1, keepdim=True, unbiased=False)
                 .clamp_min(1e-4) if self.proxy_normalize
                 else torch.ones_like(threshold, dtype=torch.float32))
        soft = torch.sigmoid((scores.float() - threshold.float()) / (self.tau * scale))
        return soft, scale

    def forward(self, x, prior=None):
        scores = self.conditioned_scores(x, prior)
        threshold = scores.mean(dim=-1, keepdim=True)
        hard = (scores > threshold).to(x.dtype)
        if self.mode == "hard":
            active = hard
        else:
            soft, scale = self.surrogate(scores, threshold)
            active = hard + (soft - soft.detach()).to(x.dtype)
        output = F.conv1d((x * active).unsqueeze(1), self.weight,
                          padding=self.padding).squeeze(1)
        self.last_active_ratio = hard.detach().mean()
        if self.capture_diagnostics:
            if self.mode == "hard":
                soft, scale = self.surrogate(scores, threshold)
            self.last_score_std = scores.detach().float().std(unbiased=False)
            self.last_proxy_scale = scale.detach().mean()
            self.last_proxy_saturated_fraction = (
                (soft.detach() < .01) | (soft.detach() > .99)).float().mean()
        return output


class GTRGuidedAdaFEAdapter(nn.Module):
    """Condition patch filtering while preserving TimeXer's global token."""

    def __init__(self, d_model, degree=2, p_degree=2, mode="ste",
                 gamma_init=0.05, kernel_size=5, tau=1.0,
                 constant_reparam=True, proxy_normalize=True,
                 use_prior=True, prior_gate_init=0.05):
        super().__init__()
        if not math.isfinite(gamma_init):
            raise ValueError("gamma_init must be finite.")
        self.adafe = GTRGuidedAdaFELocal(
            d_model, degree, p_degree, kernel_size, mode, tau,
            constant_reparam, proxy_normalize, use_prior, prior_gate_init)
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, en_embed, prior=None):
        if (en_embed.ndim != 3 or en_embed.shape[1] < 2
                or en_embed.shape[-1] != self.adafe.d_model):
            raise ValueError("Expected [batch*channels, patches+1, d_model].")
        patch_tokens = en_embed[:, :-1, :]
        if self.adafe.use_prior and (prior is None or prior.shape != patch_tokens.shape):
            raise ValueError("The periodic guide must match endogenous patch tokens.")
        hidden = patch_tokens.shape[-1]
        flat_prior = prior.reshape(-1, hidden) if self.adafe.use_prior else None
        extracted = self.adafe(patch_tokens.reshape(-1, hidden), flat_prior)
        extracted = extracted.reshape_as(patch_tokens)
        enhanced = patch_tokens + self.gamma * self.norm(extracted)
        return torch.cat([enhanced, en_embed[:, -1:, :]], dim=1)

