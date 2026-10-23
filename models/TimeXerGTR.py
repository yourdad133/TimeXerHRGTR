"""TimeXer with gated global temporal retrieval on predicted variables."""

import torch
from torch import nn

from models import TimeXer
from models.GTR import GTR as GlobalTemporalRetriever


class Model(TimeXer.Model):
    """Fuse GTR with TimeXer's target and cross-variable token paths.

    In MS mode, GTR enriches the last (target) channel and the other channels
    remain the exogenous context. In M mode, GTR enriches every predicted
    channel while the original normalized window provides cross-variable
    context. The dataset's cycle index marks the first step after the input
    window, so global retrieval starts ``seq_len`` steps before that index.
    """

    def __init__(self, configs):
        if configs.features not in {"MS", "M"}:
            raise ValueError("TimeXerGTR supports features='MS' or 'M'.")
        super().__init__(configs)

        self.input_channels = int(configs.enc_in)
        self.gtr_hierarchical = bool(getattr(configs, "gtr_hierarchical", False))
        self.gtr_cycle_len = int(
            getattr(configs, "gtr_cycle_len", getattr(configs, "cycle", 24))
        )
        if not self.gtr_hierarchical and self.gtr_cycle_len <= 0:
            raise ValueError("gtr_cycle_len must be positive.")

        if self.gtr_hierarchical:
            short_cycle_len = getattr(configs, "gtr_short_cycle_len", None)
            long_cycle_len = getattr(configs, "gtr_long_cycle_len", None)
            if short_cycle_len is None or long_cycle_len is None:
                raise ValueError(
                    "Hierarchical GTR requires short and long cycle lengths."
                )
            self.gtr_short_cycle_len = int(short_cycle_len)
            self.gtr_long_cycle_len = int(long_cycle_len)
            if self.gtr_short_cycle_len <= 0 or self.gtr_long_cycle_len <= 0:
                raise ValueError("Hierarchical cycle lengths must be positive.")
            if (
                self.gtr_long_cycle_len <= self.gtr_short_cycle_len
                or self.gtr_long_cycle_len % self.gtr_short_cycle_len != 0
            ):
                raise ValueError(
                    "gtr_long_cycle_len must be a larger integer multiple of "
                    "gtr_short_cycle_len."
                )
            self.gtr_num_blocks = (
                self.gtr_long_cycle_len // self.gtr_short_cycle_len
            )
            self.gtr_residual_scale = float(
                getattr(configs, "gtr_residual_scale", 1.0)
            )

        period_len = int(getattr(configs, "gtr_period_len", 24))
        if period_len <= 0:
            raise ValueError("gtr_period_len must be positive.")

        self.gtr_channels = 1 if self.features == "MS" else self.input_channels
        gate_init = float(getattr(configs, "gtr_gate_init", 0.1))
        if self.gtr_hierarchical:
            self.Q_short = nn.Parameter(
                torch.randn(self.gtr_short_cycle_len, self.gtr_channels) * 0.02
            )
            self.R_long_raw = nn.Parameter(
                torch.zeros(
                    self.gtr_num_blocks,
                    self.gtr_short_cycle_len,
                    self.gtr_channels,
                )
            )
        else:
            self.Q = nn.Parameter(
                torch.randn(self.gtr_cycle_len, self.gtr_channels) * 0.02
            )
        self.gamma = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.GTR = GlobalTemporalRetriever(
            d_series=self.seq_len,
            c=self.gtr_channels,
            CI=False,
            period_len=period_len,
        )

    def get_centered_residual(self):
        """Keep the long-cycle memory zero mean across short-cycle blocks."""
        if not self.gtr_hierarchical:
            raise RuntimeError(
                "Centered residual is only available in hierarchical mode."
            )
        return self.R_long_raw - self.R_long_raw.mean(dim=0, keepdim=True)

    def _retrieve(self, endogenous, cycle_index):
        """Add the aligned global period segment to the local window."""
        if cycle_index is None:
            raise ValueError("TimeXerGTR requires cycle_index.")

        batch_size, window_len, channels = endogenous.shape
        if window_len != self.seq_len or channels != self.gtr_channels:
            raise ValueError("Unexpected endogenous input shape for GTR.")

        cycle_index = torch.as_tensor(cycle_index, device=endogenous.device)
        if cycle_index.numel() != batch_size:
            raise ValueError("cycle_index must contain one value per batch item.")
        cycle_index = cycle_index.reshape(batch_size, 1).long()
        offsets = torch.arange(self.seq_len, device=endogenous.device).view(1, -1)
        if self.gtr_hierarchical:
            # cycle_index marks the first forecast step, not the first input step.
            positions = (
                cycle_index - self.seq_len + offsets
            ) % self.gtr_long_cycle_len
            short_positions = positions % self.gtr_short_cycle_len
            block_positions = positions // self.gtr_short_cycle_len
            centered_residual = self.get_centered_residual()
            query = (
                self.Q_short[short_positions]
                + self.gtr_residual_scale
                * centered_residual[block_positions, short_positions]
            )
        else:
            positions = (cycle_index - self.seq_len + offsets) % self.gtr_cycle_len
            query = self.Q[positions]
        query = query.transpose(1, 2)

        global_information = self.GTR(endogenous.transpose(1, 2), query)
        return endogenous + self.gamma * global_information.transpose(1, 2)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index):
        if (
            x_enc.ndim != 3
            or x_enc.shape[1] != self.seq_len
            or x_enc.shape[2] != self.input_channels
        ):
            raise ValueError("x_enc must have shape (batch, seq_len, enc_in).")

        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized = centered / stdev
        else:
            means = stdev = None
            normalized = x_enc

        if self.features == "MS":
            # Dataset_Custom places the target in the last channel.
            endogenous = normalized[:, :, -1:]
            exogenous = normalized[:, :, :-1]
            output_mean = means[:, :, -1:] if self.use_norm else None
            output_stdev = stdev[:, :, -1:] if self.use_norm else None
        else:
            # Every channel is a target in M mode. As in TimeXer's M path,
            # all original channels serve as cross-variable context.
            endogenous = normalized
            exogenous = normalized
            output_mean = means
            output_stdev = stdev

        if exogenous.shape[-1] == 0 and x_mark_enc is None:
            raise ValueError("Cross-attention requires exogenous or time tokens.")

        enhanced = self._retrieve(endogenous, cycle_index)
        en_embed, n_vars = self.en_embedding(enhanced.transpose(1, 2))
        ex_embed = self.ex_embedding(exogenous, x_mark_enc)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(
            -1, n_vars, enc_out.shape[-2], enc_out.shape[-1]
        ).permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * output_stdev + output_mean
        return dec_out

    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index, mask=None
    ):
        return self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index
        )[:, -self.pred_len :, :]
