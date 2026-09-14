# No-calibration U-Net with decoder-gated refiners on the two deep skips:
# a Mamba gate (MambaUNet3LNoCal) or a Transformer / Conformer gate (TransformerUNet3LNoCal).
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import (
    Mamba, DROPOUT_P, BASE_CHANNELS,
    DoubleConv1D, SinusoidalPositionalEncoding1D,
    PlainUNet3LNoCal, WaveformLoss,
)

__all__ = ["MambaUNet3LNoCal", "TransformerUNet3LNoCal", "PlainUNet3LNoCal", "WaveformLoss"]

# Identity-safe residual initialisation for Mamba/Transformer skip refiners: zero-init the value.
SKIP_REFINER_ZERO_INIT_VALUE = True
SKIP_REFINER_INIT_ALPHA_BIAS = -3.0  # the registry sets -4.0 for unet_mamba_skip

# When False, the shallowest (highest-resolution, L=1024) U-Net skip is passed through.
SKIP_REFINER_USE_SHALLOW_SKIP = False

# Deep-skip gate: 0 = Transformer, 1 = Conformer.
USE_CONFORMER_GATE = 0


# Bi-directional Mamba Attention Gate with additive residual refinement.
class MambaAttentionGate(nn.Module):

    def __init__(self, c_skip: int, c_gate: int, c_inter: Optional[int] = None,
                 d_state: int = 8, d_conv: int = 4, expand: int = 2,
                 dropout: float = DROPOUT_P,
                 init_alpha_bias: Optional[float] = None,
                 bidirectional: bool = True, use_channel_attn: bool = True,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaAttentionGate requires mamba_ssm.Mamba. The depthwise-Conv1d "
                "fallback has been removed because it would silently turn a "
                "'Mamba U-Net' into a plain conv U-Net. Install mamba_ssm."
            )
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        if c_inter is None:
            c_inter = max(c_skip // 2, 8)
        self.c_inter = c_inter
        self.bidir = bool(bidirectional)
        self.use_channel_attn = bool(use_channel_attn)

        self.W_x = nn.Conv1d(c_skip, c_inter, kernel_size=1, bias=False)
        self.W_g = nn.Conv1d(c_gate, c_inter, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c_inter)

        self.norm = nn.LayerNorm(c_inter)
        self.mamba_fwd = Mamba(d_model=c_inter, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = (
            Mamba(d_model=c_inter, d_state=d_state, d_conv=d_conv, expand=expand)
            if self.bidir else None
        )

        self.proj_v = nn.Conv1d(c_inter, c_skip, kernel_size=1, bias=True)

        self.psi_t = nn.Conv1d(c_inter, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(c_inter // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(c_inter, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, c_skip, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Identity-safe init: zero-init the value projection so the residual delta is exactly 0 at step 0.
        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xg = F.relu(self.bn(self.W_x(skip) + self.W_g(gate)))   # [B, c_inter, L]

        seq = xg.transpose(1, 2)            # [B, L, c_inter]
        seq = self.norm(seq)
        f = self.mamba_fwd(seq)
        if self.bidir and self.mamba_bwd is not None:
            b = torch.flip(self.mamba_bwd(torch.flip(seq, dims=[1])), dims=[1])
            seq = 0.5 * (f + b)
        else:
            seq = f
        seq = self.dropout(seq)
        seq = seq.transpose(1, 2)           # [B, c_inter, L]

        v = self.proj_v(seq)                                  # [B, c_skip, L]
        a_t = torch.sigmoid(self.psi_t(seq))                  # [B, 1,      L]
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq))              # [B, c_skip, 1]
            alpha = a_t * a_c
        else:
            alpha = a_t
        return skip + alpha * v


# Bi-directional Transformer Attention Gate with additive residual.
class TransformerAttentionGate(nn.Module):

    def __init__(self, c_skip: int, c_gate: int, c_inter: Optional[int] = None,
                 n_heads: int = 4, ff_mult: int = 2,
                 dropout: float = DROPOUT_P,
                 init_alpha_bias: Optional[float] = None,
                 use_channel_attn: bool = True,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        if c_inter is None:
            c_inter = max(c_skip // 2, 8)
        # Ensure embed dim divisible by heads; halve heads until it fits.
        while n_heads > 1 and (c_inter % n_heads != 0):
            n_heads //= 2
        self.c_inter = c_inter
        self.n_heads = n_heads
        self.use_channel_attn = bool(use_channel_attn)

        self.W_x = nn.Conv1d(c_skip, c_inter, kernel_size=1, bias=False)
        self.W_g = nn.Conv1d(c_gate, c_inter, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c_inter)

        # Positional encoding added right before the first attention block.
        self.pos_enc = SinusoidalPositionalEncoding1D(c_inter)

        self.norm1 = nn.LayerNorm(c_inter)
        self.attn = nn.MultiheadAttention(
            embed_dim=c_inter, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(c_inter)
        ff_hidden = c_inter * int(ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(c_inter, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, c_inter),
        )

        self.proj_v = nn.Conv1d(c_inter, c_skip, kernel_size=1, bias=True)

        self.psi_t = nn.Conv1d(c_inter, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(c_inter // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(c_inter, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, c_skip, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xg = F.relu(self.bn(self.W_x(skip) + self.W_g(gate)))   # [B, c_inter, L]
        seq = xg.transpose(1, 2)                                # [B, L, c_inter]
        seq = self.pos_enc(seq)                                 # add 1D PE
        h = self.norm1(seq)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.dropout(attn_out)
        h = self.norm2(seq)
        seq = seq + self.dropout(self.ff(h))
        seq = seq.transpose(1, 2)                               # [B, c_inter, L]

        v = self.proj_v(seq)                                    # [B, c_skip, L]
        a_t = torch.sigmoid(self.psi_t(seq))                    # [B, 1,      L]
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq))                # [B, c_skip, 1]
            alpha = a_t * a_c
        else:
            alpha = a_t
        return skip + alpha * v


# Decoder-gated Transformer skip refiner with internal pooled attention.
class PooledTransformerAttentionGate(nn.Module):

    def __init__(self, c_skip: int, c_gate: int, c_inter: Optional[int] = None,
                 n_heads: int = 2, ff_mult: int = 2,
                 pool_factor: int = 4,
                 dropout: float = DROPOUT_P,
                 init_alpha_bias: Optional[float] = None,
                 use_channel_attn: bool = True,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        if c_inter is None:
            c_inter = max(c_skip // 2, 8)
        while n_heads > 1 and (c_inter % n_heads != 0):
            n_heads //= 2
        self.c_inter = c_inter
        self.n_heads = n_heads
        self.pool_factor = max(1, int(pool_factor))
        self.use_channel_attn = bool(use_channel_attn)

        self.W_x = nn.Conv1d(c_skip, c_inter, kernel_size=1, bias=False)
        self.W_g = nn.Conv1d(c_gate, c_inter, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c_inter)

        if self.pool_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=self.pool_factor,
                                     stride=self.pool_factor)
            self.upsample = nn.Upsample(scale_factor=self.pool_factor,
                                        mode='linear', align_corners=True)
        else:
            self.pool = nn.Identity()
            self.upsample = nn.Identity()

        # PE applied at the pooled scale, where attention actually runs.
        self.pos_enc = SinusoidalPositionalEncoding1D(c_inter)

        self.norm1 = nn.LayerNorm(c_inter)
        self.attn = nn.MultiheadAttention(
            embed_dim=c_inter, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(c_inter)
        ff_hidden = c_inter * int(ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(c_inter, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, c_inter),
        )

        self.proj_v = nn.Conv1d(c_inter, c_skip, kernel_size=1, bias=True)

        self.psi_t = nn.Conv1d(c_inter, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(c_inter // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(c_inter, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, c_skip, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xg_full = F.relu(self.bn(self.W_x(skip) + self.W_g(gate)))   # [B, c_int, L]
        L = int(xg_full.size(-1))

        if self.pool_factor > 1:
            # Right-pad to a multiple of pool_factor so the pool/upsample roundtrip lands on the original.
            pad_amount = 0
            if L % self.pool_factor != 0:
                target_L = int(math.ceil(L / self.pool_factor)
                               * self.pool_factor)
                pad_amount = target_L - L
                xg_padded = F.pad(xg_full, (0, pad_amount))
            else:
                xg_padded = xg_full
            xg_pooled = self.pool(xg_padded)                          # [B, c_int, L_pool]
        else:
            pad_amount = 0
            xg_pooled = xg_full

        seq = xg_pooled.transpose(1, 2)                               # [B, L_pool, c_int]
        seq = self.pos_enc(seq)                                       # add 1D PE
        h = self.norm1(seq)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.dropout(attn_out)
        h = self.norm2(seq)
        seq = seq + self.dropout(self.ff(h))
        seq_pooled = seq.transpose(1, 2)                              # [B, c_int, L_pool]

        if self.pool_factor > 1:
            seq_up = self.upsample(seq_pooled)
            if pad_amount:
                seq_up = seq_up[..., :L]
        else:
            seq_up = seq_pooled

        v = self.proj_v(seq_up)                                       # [B, c_skip, L]
        a_t = torch.sigmoid(self.psi_t(seq_up))                       # [B, 1, L]
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq_up))                   # [B, c_skip, 1]
            alpha = a_t * a_c
        else:
            alpha = a_t
        return skip + alpha * v


# Conformer block adapted for 1D BP-waveform skip refinement.
class ConformerBlock1D(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 4, ff_mult: int = 4,
                 conv_kernel: int = 31, dropout: float = DROPOUT_P):
        super().__init__()
        if conv_kernel % 2 == 0:
            conv_kernel += 1  # keep symmetric "same" padding

        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                      padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(a)
        h = self.norm_conv(x).transpose(1, 2)            # [B, D, L]
        x = x + self.conv(h).transpose(1, 2)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_final(x)


# Conformer-based decoder-gated skip refiner with additive residual.
class ConformerAttentionGate(nn.Module):

    def __init__(self, c_skip: int, c_gate: int, c_inter: Optional[int] = None,
                 n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4,
                 conv_kernel: int = 31,
                 dropout: float = DROPOUT_P,
                 init_alpha_bias: Optional[float] = None,
                 use_channel_attn: bool = True,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        if c_inter is None:
            c_inter = max(c_skip // 2, 8)
        while n_heads > 1 and (c_inter % n_heads != 0):
            n_heads //= 2
        self.c_inter = c_inter
        self.n_heads = n_heads
        self.use_channel_attn = bool(use_channel_attn)

        self.W_x = nn.Conv1d(c_skip, c_inter, kernel_size=1, bias=False)
        self.W_g = nn.Conv1d(c_gate, c_inter, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c_inter)

        self.pos_enc = SinusoidalPositionalEncoding1D(c_inter)

        self.blocks = nn.ModuleList([
            ConformerBlock1D(
                d_model=c_inter, n_heads=n_heads, ff_mult=ff_mult,
                conv_kernel=conv_kernel, dropout=dropout,
            )
            for _ in range(int(n_layers))
        ])

        self.proj_v = nn.Conv1d(c_inter, c_skip, kernel_size=1, bias=True)

        self.psi_t = nn.Conv1d(c_inter, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(c_inter // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(c_inter, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, c_skip, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xg = F.relu(self.bn(self.W_x(skip) + self.W_g(gate)))   # [B, c_inter, L]
        seq = xg.transpose(1, 2)                                # [B, L, c_inter]
        seq = self.pos_enc(seq)
        for blk in self.blocks:
            seq = blk(seq)
        seq = seq.transpose(1, 2)                               # [B, c_inter, L]

        v = self.proj_v(seq)                                    # [B, c_skip, L]
        a_t = torch.sigmoid(self.psi_t(seq))                    # [B, 1,      L]
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq))                # [B, c_skip, 1]
            alpha = a_t * a_c
        else:
            alpha = a_t
        return skip + alpha * v


# 3-level UNet decoder with decoder-gated Mamba skip refiners on the deeper skips only.
class MambaSkipDecoder3LNoCal(nn.Module):

    def __init__(self, base_channels: int = BASE_CHANNELS,
                 ag_d_state: int = 8, ag_d_conv: int = 4, ag_expand: int = 2,
                 use_shallow_skip: Optional[bool] = None):
        super().__init__()
        if use_shallow_skip is None:
            use_shallow_skip = bool(SKIP_REFINER_USE_SHALLOW_SKIP)
        self.use_shallow_skip = bool(use_shallow_skip)

        self.ag2 = MambaAttentionGate(
            c_skip=base_channels * 2, c_gate=base_channels * 4,
            c_inter=base_channels,
            d_state=ag_d_state, d_conv=ag_d_conv, expand=ag_expand,
        )
        self.ag1 = MambaAttentionGate(
            c_skip=base_channels, c_gate=base_channels * 2,
            c_inter=max(base_channels // 2, 8),
            d_state=ag_d_state, d_conv=ag_d_conv, expand=ag_expand,
        )
        if self.use_shallow_skip:
            self.ag0 = MambaAttentionGate(
                c_skip=base_channels, c_gate=base_channels,
                c_inter=max(base_channels // 4, 8),
                d_state=ag_d_state, d_conv=ag_d_conv, expand=ag_expand,
            )
        else:
            self.ag0 = None

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        f0, f1, f2, _ = features
        x = bottleneck_out                       # [B, 4*base, 128]

        x = self.up3(x)                          # [B, 4*base, 256] (gate)
        skip2 = self.ag2(f2, x)                  # gated skip [B, 2*base, 256]
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)                        # [B, 2*base, 256]

        x = self.up2(x)                          # [B, 2*base, 512] (gate)
        skip1 = self.ag1(f1, x)                  # gated skip [B, base, 512]
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)                        # [B, base, 512]

        x = self.up1(x)                          # [B, base, 1024] (gate)
        skip0_up = self.up1(f0)                  # [B, base, 1024]
        if self.ag0 is not None:
            skip0 = self.ag0(skip0_up, x)        # gated skip [B, base, 1024]
        else:
            skip0 = skip0_up                     # identity pass-through
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)                        # [B, base, 1024]
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        return x.squeeze(1)

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        return x, out.squeeze(1)


# 3-level UNet decoder; deep skips go through the gate USE_CONFORMER_GATE selects.
class TransformerSkipDecoder3LNoCal(nn.Module):

    def __init__(self, base_channels: int = BASE_CHANNELS,
                 n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4,
                 conv_kernel: int = 31,
                 ag0_pool_factor: int = 4,
                 use_shallow_skip: Optional[bool] = None,
                 use_conformer_gate: Optional[int] = None):
        super().__init__()
        if use_shallow_skip is None:
            use_shallow_skip = bool(SKIP_REFINER_USE_SHALLOW_SKIP)
        self.use_shallow_skip = bool(use_shallow_skip)

        if use_conformer_gate is None:
            use_conformer_gate = USE_CONFORMER_GATE
        if int(use_conformer_gate) not in (0, 1):
            raise ValueError(
                f"use_conformer_gate must be 1 (Conformer gate) or 0 "
                f"(plain Transformer gate), got {use_conformer_gate!r}."
            )
        self.use_conformer_gate = int(use_conformer_gate)

        # Deep skips (L=256, L=512).
        if self.use_conformer_gate == 1:
            self.ag2 = ConformerAttentionGate(
                c_skip=base_channels * 2, c_gate=base_channels * 4,
                c_inter=base_channels,
                n_heads=n_heads, n_layers=n_layers, ff_mult=ff_mult,
                conv_kernel=conv_kernel,
            )
            self.ag1 = ConformerAttentionGate(
                c_skip=base_channels, c_gate=base_channels * 2,
                c_inter=max(base_channels // 2, 8),
                n_heads=max(1, n_heads // 2), n_layers=n_layers, ff_mult=ff_mult,
                conv_kernel=conv_kernel,
            )
        else:
            self.ag2 = TransformerAttentionGate(
                c_skip=base_channels * 2, c_gate=base_channels * 4,
                c_inter=base_channels,
                n_heads=n_heads, ff_mult=ff_mult,
            )
            self.ag1 = TransformerAttentionGate(
                c_skip=base_channels, c_gate=base_channels * 2,
                c_inter=max(base_channels // 2, 8),
                n_heads=max(1, n_heads // 2), ff_mult=ff_mult,
            )
        if self.use_shallow_skip:
            self.ag0 = PooledTransformerAttentionGate(
                c_skip=base_channels, c_gate=base_channels,
                c_inter=max(base_channels // 4, 8),
                n_heads=max(1, n_heads // 2), ff_mult=max(1, ff_mult // 2),
                pool_factor=int(ag0_pool_factor),
            )
        else:
            self.ag0 = None

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        f0, f1, f2, _ = features
        x = bottleneck_out

        x = self.up3(x)                          # [B, 4*base, 256] (gate)
        skip2 = self.ag2(f2, x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)                        # [B, 2*base, 256]

        x = self.up2(x)                          # [B, 2*base, 512] (gate)
        skip1 = self.ag1(f1, x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)                        # [B, base, 512]

        x = self.up1(x)                          # [B, base, 1024] (gate)
        skip0_up = self.up1(f0)                  # [B, base, 1024]
        if self.ag0 is not None:
            skip0 = self.ag0(skip0_up, x)        # gated skip [B, base, 1024]
        else:
            skip0 = skip0_up                     # identity pass-through
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)                        # [B, base, 1024]
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        return x.squeeze(1)

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        return x, out.squeeze(1)


class MambaUNet3LNoCal(PlainUNet3LNoCal):
    """No-calibration UNet with decoder-gated Mamba skip refiners."""
    def __init__(self, input_channels: int = 3,
                 base_channels: int = BASE_CHANNELS,
                 use_signal_stats: bool = True,
                 ag_d_state: int = 8, ag_d_conv: int = 4, ag_expand: int = 2):
        super().__init__(input_channels=input_channels,
                         base_channels=base_channels,
                         use_signal_stats=use_signal_stats)
        self.decoder = MambaSkipDecoder3LNoCal(
            base_channels=base_channels,
            ag_d_state=ag_d_state, ag_d_conv=ag_d_conv, ag_expand=ag_expand,
        )


class TransformerUNet3LNoCal(PlainUNet3LNoCal):
    """No-calibration UNet with decoder-gated skip refiners (see USE_CONFORMER_GATE)."""
    def __init__(self, input_channels: int = 3,
                 base_channels: int = BASE_CHANNELS,
                 use_signal_stats: bool = True,
                 n_heads: int = 4, n_layers: int = 2, ff_mult: int = 4,
                 conv_kernel: int = 31,
                 use_conformer_gate: Optional[int] = None):
        super().__init__(input_channels=input_channels,
                         base_channels=base_channels,
                         use_signal_stats=use_signal_stats)
        self.decoder = TransformerSkipDecoder3LNoCal(
            base_channels=base_channels,
            n_heads=n_heads, n_layers=n_layers, ff_mult=ff_mult,
            conv_kernel=conv_kernel,
            use_conformer_gate=use_conformer_gate,
        )
