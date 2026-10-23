"""Isolated TimeXerGTR prototype: pure periodic queries guide AdaFE scores.

This model is intentionally absent from the production experiment registry.
Use ``guided_adafe_use_prior=False`` for the matched unguided ablation.
"""

import copy

import torch

from layers.gtr_guided_adafe import GTRGuidedAdaFEAdapter
from models.TimeXerGTR import Model as TimeXerGTRModel


class Model(TimeXerGTRModel):
    def __init__(self, configs):
        backbone_configs = copy.copy(configs)
        backbone_configs.use_adafe = False
        super().__init__(backbone_configs)
        self.use_adafe = True
        self.guided_adafe_use_prior = bool(getattr(configs, "guided_adafe_use_prior", True))
        # The optional adapter must not change backbone or subsequent RNG draws.
        with torch.random.fork_rng(devices=[]):
            self.adafe_adapter = GTRGuidedAdaFEAdapter(
                d_model=configs.d_model,
                degree=getattr(configs, "adafe_fourier_degree", 2),
                p_degree=getattr(configs, "adafe_poly_degree", 2),
                mode=getattr(configs, "adafe_mode", "ste"),
                gamma_init=getattr(configs, "adafe_gamma_init", 0.05),
                kernel_size=getattr(configs, "adafe_kernel", 5),
                tau=getattr(configs, "adafe_tau", 1.0),
                constant_reparam=getattr(configs, "adafe_constant_reparam", True),
                proxy_normalize=getattr(configs, "adafe_proxy_normalize", True),
                use_prior=self.guided_adafe_use_prior,
                prior_gate_init=getattr(configs, "guided_adafe_prior_gate_init", 0.05),
            )

    def _period_guide(self, endogenous, cycle_index):
        """Project aligned Linear(Q) patches without local input or position codes."""
        batch, length, channels = endogenous.shape
        if length != self.seq_len or channels != self.gtr_channels:
            raise ValueError("Unexpected endogenous input shape for the periodic guide.")
        if cycle_index is None:
            raise ValueError("The periodic guide requires cycle_index.")
        cycle_index = torch.as_tensor(cycle_index, device=endogenous.device)
        if cycle_index.numel() != batch:
            raise ValueError("cycle_index must contain one value per batch item.")
        cycle_index = cycle_index.reshape(batch, 1).long()
        offsets = torch.arange(self.seq_len, device=endogenous.device).view(1, -1)
        positions = (cycle_index - self.seq_len + offsets) % self.gtr_cycle_len
        query = self.Q[positions].transpose(1, 2)
        periodic = self.GTR.linear(query)
        patches = periodic.unfold(-1, self.patch_len, self.patch_len)
        patches = patches.reshape(batch * channels, patches.shape[-2], self.patch_len)
        return self.en_embedding.value_embedding(patches)

    def apply_adafe_to_en_embed(self, en_embed, period_guide=None):
        return self.adafe_adapter(en_embed, period_guide)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index):
        # Keep the production TimeXerGTR main route and normalization unchanged.
        if (x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len
                or x_enc.shape[2] != self.input_channels):
            raise ValueError("x_enc must have shape (batch, seq_len, enc_in).")
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True,
                                        unbiased=False) + 1e-5)
            normalized = centered / stdev
        else:
            means = stdev = None
            normalized = x_enc
        if self.features == "MS":
            endogenous = normalized[:, :, -1:]
            exogenous = normalized[:, :, :-1]
            output_mean = means[:, :, -1:] if self.use_norm else None
            output_stdev = stdev[:, :, -1:] if self.use_norm else None
        else:
            endogenous = normalized
            exogenous = normalized
            output_mean = means
            output_stdev = stdev
        if exogenous.shape[-1] == 0 and x_mark_enc is None:
            raise ValueError("Cross-attention requires exogenous or time tokens.")

        enhanced = self._retrieve(endogenous, cycle_index)
        en_embed, n_vars = self.en_embedding(enhanced.transpose(1, 2))
        period_guide = (self._period_guide(endogenous, cycle_index)
                        if self.guided_adafe_use_prior else None)
        en_embed = self.apply_adafe_to_en_embed(en_embed, period_guide)
        ex_embed = self.ex_embedding(exogenous, x_mark_enc)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2],
                                  enc_out.shape[-1]).permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * output_stdev + output_mean
        return dec_out

