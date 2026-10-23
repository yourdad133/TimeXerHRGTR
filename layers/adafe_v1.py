"""AdaFE feature filtering and a patch-only residual adapter.

FourierPolyMask is adapted from lllucky111/FeTS, models/FeTS.py at
457d4477c9258e4d7be6b498dd723dce821baa74:
https://github.com/lllucky111/FeTS/blob/457d4477c9258e4d7be6b498dd723dce821baa74/models/FeTS.py
Upstream has no LICENSE file or license declaration at this revision; no
license grant is inferred. See scripts/adafe_v1/reference/upstream/ for provenance.

The repaired scorer folds duplicate constants into one bias; legacy tensor
shapes and initialization RNG are retained for checkpoint compatibility.
Hard keeps the global threshold; STE normalizes only its backward estimator.
The shared convolution slides across embedding features, not time.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class FourierPolyMask(nn.Module):
    def __init__(self, input_dim, output_dim, fourier_degree, poly_degree,
                 constant_reparam=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_degree = fourier_degree
        self.poly_degree = poly_degree
        self.cos_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, fourier_degree + 1))
        self.sin_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, fourier_degree))
        self.poly_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, poly_degree + 1))

        # Keep the original draws so paired runs have the same initial function.
        nn.init.normal_(self.cos_coeffs, mean=0.0, std=1 / (input_dim * (fourier_degree + 1)))
        nn.init.normal_(self.sin_coeffs, mean=0.0, std=1 / (input_dim * fourier_degree))
        nn.init.normal_(self.poly_coeffs, mean=0.0, std=1 / input_dim * (poly_degree + 1))
        self.register_buffer("k_cos", torch.arange(0, fourier_degree + 1))
        self.register_buffer("k_sin", torch.arange(1, fourier_degree + 1))
        self.register_buffer("k_poly", torch.arange(0, poly_degree + 1))
        self.interaction = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(1, output_dim))
        self.constant_reparam = bool(constant_reparam)
        if self.constant_reparam:
            with torch.no_grad():
                self.bias.add_(self.cos_coeffs[:, :, 0].sum(0, keepdim=True)
                               + self.poly_coeffs[:, :, 0].sum(0, keepdim=True))
                self.cos_coeffs[:, :, 0].zero_()
                self.poly_coeffs[:, :, 0].zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keys = [prefix + name for name in ('cos_coeffs', 'poly_coeffs', 'bias')]
        if self.constant_reparam and all(key in state_dict for key in keys):
            cos, poly, bias = (state_dict[key] for key in keys)
            if (cos.shape == self.cos_coeffs.shape
                    and poly.shape == self.poly_coeffs.shape
                    and bias.shape == self.bias.shape):
                # Migrate old checkpoints without changing the caller's tensors.
                # Repaired checkpoints already have zero constant columns.
                state_dict[keys[2]] = (bias + cos[:, :, 0].sum(0, keepdim=True)
                                      + poly[:, :, 0].sum(0, keepdim=True))
                cos, poly = cos.clone(), poly.clone()
                cos[:, :, 0].zero_()
                poly[:, :, 0].zero_()
                state_dict[keys[0]], state_dict[keys[1]] = cos, poly
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        start = 1 if self.constant_reparam else 0
        x_cos = torch.cos(x.unsqueeze(-1) * self.k_cos[start:] * torch.pi)
        x_sin = torch.sin(x.unsqueeze(-1) * self.k_sin * torch.pi)
        x_poly = x.unsqueeze(-1).pow(self.k_poly[start:])
        y_cos = torch.einsum("bid,iod->bo", x_cos, self.cos_coeffs[:, :, start:])
        y_sin = torch.einsum("bid,iod->bo", x_sin, self.sin_coeffs)
        y_poly = torch.einsum("bid,iod->bo", x_poly, self.poly_coeffs[:, :, start:])
        return self.interaction(y_cos + y_sin + y_poly + self.bias)


class AdaFELocal(nn.Module):
    """Apply feature filtering to a two-dimensional token matrix."""

    def __init__(self, d_model, degree=2, p_degree=2, kernel_size=5,
                 mode="hard", tau=1.0, constant_reparam=True,
                 proxy_normalize=True):
        super().__init__()
        if d_model <= 0 or degree < 1 or p_degree < 0:
            raise ValueError("AdaFE needs d_model > 0, degree >= 1, p_degree >= 0.")
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("AdaFE kernel_size must be positive and odd.")
        if mode not in {"hard", "ste"}:
            raise ValueError("AdaFE mode must be 'hard' or 'ste'.")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("AdaFE tau must be finite and positive.")
        self.d_model = d_model
        self.mode = mode
        self.tau = float(tau)
        self.proxy_normalize = bool(proxy_normalize)
        self.score = FourierPolyMask(d_model, d_model, degree, p_degree,
                                     constant_reparam=constant_reparam)
        self.weight = nn.Parameter(torch.randn(1, 1, kernel_size))
        self.padding = kernel_size // 2
        self.last_active_ratio = None
        self.capture_diagnostics = False
        self.last_score_std = None
        self.last_proxy_scale = None
        self.last_proxy_saturated_fraction = None

    def surrogate(self, scores, threshold):
        # FP32 avoids low-precision overflow in the backward estimator.
        scale = (scores.detach().float().std(unbiased=False).clamp_min(1e-4)
                 if self.proxy_normalize else scores.new_ones(()).float())
        soft = torch.sigmoid((scores.float() - threshold.float())
                             / (self.tau * scale))
        return soft, scale

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.d_model or x.shape[0] == 0:
            raise ValueError("AdaFE input must be nonempty [tokens, d_model].")
        scores = self.score(x)
        threshold = scores.mean()  # One scalar over every token and feature.
        hard = (scores > threshold).to(x.dtype)
        if self.mode == "hard":
            active = hard
        else:
            soft, scale = self.surrogate(scores, threshold)
            # Exactly hard in forward, including AMP; sigmoid supplies backward.
            active = hard + (soft - soft.detach()).to(x.dtype)
        output = F.conv1d(
            (x * active).unsqueeze(1), self.weight, padding=self.padding
        ).squeeze(1)
        self.last_active_ratio = hard.detach().mean()
        if self.capture_diagnostics:
            if self.mode == "hard":
                soft, scale = self.surrogate(scores, threshold)
            self.last_score_std = scores.detach().float().std(unbiased=False)
            self.last_proxy_scale = scale.detach()
            self.last_proxy_saturated_fraction = (
                (soft.detach() < .01) | (soft.detach() > .99)
            ).float().mean()
        return output


class AdaFEAdapter(nn.Module):
    """Enhance endogenous patch tokens and preserve the final global token."""

    def __init__(self, d_model, degree=2, p_degree=2, mode="hard",
                 gamma_init=0.05, kernel_size=5, tau=1.0,
                 constant_reparam=True, proxy_normalize=True):
        super().__init__()
        if not math.isfinite(gamma_init):
            raise ValueError("AdaFE gamma_init must be finite.")
        self.adafe = AdaFELocal(
            d_model, degree, p_degree, kernel_size, mode, tau,
            constant_reparam, proxy_normalize
        )
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, en_embed):
        if (en_embed.ndim != 3 or en_embed.shape[1] < 2
                or en_embed.shape[-1] != self.adafe.d_model):
            raise ValueError("Expected [batch*channels, patches+1, d_model].")
        patch_tokens = en_embed[:, :-1, :]
        global_token = en_embed[:, -1:, :]
        batch_channels, patches, hidden = patch_tokens.shape
        extracted = self.adafe(patch_tokens.reshape(-1, hidden)).reshape(
            batch_channels, patches, hidden
        )
        enhanced = patch_tokens + self.gamma * self.norm(extracted)
        return torch.cat([enhanced, global_token], dim=1)
