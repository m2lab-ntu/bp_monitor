import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model configuration defaults (overridden by train.py)
FS = 125
USE_MULTITASK_SBP_DBP = True
DROPOUT_P = 0.2
BASE_CHANNELS = 32

# Skip refiner / Mamba backend configuration When True.
REQUIRE_REAL_MAMBA = True

# Identity-safe residual initialisation for Mamba/Transformer skip refiners: zero-init the value.
SKIP_REFINER_ZERO_INIT_VALUE = True
SKIP_REFINER_INIT_ALPHA_BIAS = -3.0

# When False, the shallowest (highest-resolution, L=1024) U-Net skip is passed through.
SKIP_REFINER_USE_SHALLOW_SKIP = False

# Deep-skip gate: 0 = Transformer, 1 = Conformer.
USE_CONFORMER_GATE = 0

try:
    from mamba_ssm import Mamba
    logging.info("mamba_ssm.Mamba backend is available.")
except ImportError as _mamba_err:
    if REQUIRE_REAL_MAMBA:
        raise ImportError(
            "REQUIRE_REAL_MAMBA=True but mamba_ssm could not be imported. "
            "Install mamba_ssm to run the Mamba U-Net experiment, or set "
            "REQUIRE_REAL_MAMBA=False in this file to disable the strict check."
        ) from _mamba_err
    Mamba = None

# Helper functions for model internals

# Right-pad temporal axis to be divisible by mult; return (padded, orig_len).
def _pad_to_multiple(x: torch.Tensor, mult: int):
    orig_len = int(x.size(-1))
    if mult <= 1 or orig_len % int(mult) == 0:
        return x, orig_len
    target_len = int(math.ceil(orig_len / int(mult)) * int(mult))
    pad = target_len - orig_len
    return F.pad(x, (0, pad)), orig_len


def _crop_to_length(y: torch.Tensor, orig_len: Optional[int]):
    if orig_len is None:
        return y
    return y[..., :int(orig_len)]


def _mean_std_pool(x: torch.Tensor, dim: int = 2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=False)
    return torch.cat([mean, std], dim=1)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(DROPOUT_P)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        out = self.dropout(out)
        return out


# ResNet encoder with one fewer downsampling level (3-level UNet).
class ResNetEncoder3L(nn.Module):
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 1024 -> 512
        )
        self.layer1 = ResNetBlock(base_channels, base_channels)
        self.layer2 = nn.Sequential(
            ResNetBlock(base_channels, base_channels * 2, stride=2,
                        downsample=nn.Sequential(
                            nn.Conv1d(base_channels, base_channels * 2, 1, stride=2, bias=False),
                            nn.BatchNorm1d(base_channels * 2)
                        )),
            ResNetBlock(base_channels * 2, base_channels * 2)
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(base_channels * 2, base_channels * 4, stride=2,
                        downsample=nn.Sequential(
                            nn.Conv1d(base_channels * 2, base_channels * 4, 1, stride=2, bias=False),
                            nn.BatchNorm1d(base_channels * 4)
                        )),
            ResNetBlock(base_channels * 4, base_channels * 4)
        )

    def forward(self, x):
        features = []
        x = self.initial_conv(x)  # [B, base, 512]
        features.append(x)        # f0
        x = self.layer1(x)        # [B, base, 512]
        features.append(x)        # f1
        x = self.layer2(x)        # [B, 2*base, 256]
        features.append(x)        # f2
        x = self.layer3(x)        # [B, 4*base, 128]
        features.append(x)        # f3 (deepest)
        return features


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout: Optional[float] = None):
        super().__init__()
        if dropout is None:
            dropout = DROPOUT_P
        layers = [
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout1d(dropout))
        layers.extend([
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ])
        if dropout and dropout > 0:
            layers.append(nn.Dropout1d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Classic conv bottleneck for the 3-level UNet variant.
class ClassicUNetBottleneck3L(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = DoubleConv1D(channels, channels)

    def forward(self, x):
        return self.block(x)


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


# 1D sinusoidal positional encoding shaped for batch-first sequences.
class SinusoidalPositionalEncoding1D(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = int(d_model)
        pe = self._build(self.d_model, int(max_len))
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, d_model))
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = int(seq.size(1))
        if L > self.pe.size(1):
            pe = self._build(self.d_model, L).to(self.pe.device, self.pe.dtype)
            self.pe = pe
        return seq + self.pe[:, :L, :].to(seq.dtype)


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

        # 1D sinusoidal positional encoding added to the batch-first sequence right before the first.
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


class ResidualFeatureAdapter1D(nn.Module):
    """Lightweight residual adapter inserted between the calibrated decoder's last DoubleConv."""
    def __init__(self, channels: int, bottleneck: int = 16,
                 init_alpha: float = 1.0):
        super().__init__()
        self.down = nn.Conv1d(channels, bottleneck, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.up = nn.Conv1d(bottleneck, channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='linear')
        nn.init.zeros_(self.down.bias)
        # LoRA-style zero init on the up projection => identity at init.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.act(self.down(x)))


# 3-level UNet decoder with Mamba-based Attention Gates on each skip.
class CalibAGDecoder3L(nn.Module):

    def __init__(self, base_channels: int = BASE_CHANNELS,
                 ag_d_state: int = 8, ag_d_conv: int = 4, ag_expand: int = 2):
        super().__init__()

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
        # Stage 3 (shallowest, L=1024): plain identity skip on f0 (no Mamba gate).

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)

        # Hot-swappable per-master adapter; identity by default so the pre-trained checkpoint loads/runs.
        self.feature_adapter: nn.Module = nn.Identity()

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        # features: [f0(base,512), f1(base,512), f2(2*base,256), f3(4*base,128)]
        f0, f1, f2, _ = features
        x = bottleneck_out                       # [B, 4*base, 128]

        # Stage 1: 128 -> 256
        x = self.up3(x)                          # [B, 4*base, 256] (gate)
        skip2 = self.ag2(f2, x)                  # gated skip [B, 2*base, 256]
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)                        # [B, 2*base, 256]

        # Stage 2: 256 -> 512
        x = self.up2(x)                          # [B, 2*base, 512] (gate)
        skip1 = self.ag1(f1, x)                  # [B, base, 512]
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)                        # [B, base, 512]

        # Stage 3: 512 -> 1024 (shallowest, identity skip — no Mamba gate)
        x = self.up1(x)                          # [B, base, 1024]
        skip0 = self.up1(f0)                     # [B, base, 1024]
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)                        # [B, base, 1024]
        # Per-master residual feature adapter (identity by default).
        x = self.feature_adapter(x)
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        return x.squeeze(1)

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        return x, out.squeeze(1)


class CalibrationUNet(nn.Module):
    """UNet conditioned on two per-subject calibration segments."""
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS):
        super().__init__()

        self.encoder = ResNetEncoder3L(input_channels, base_channels)
        bottleneck_c = base_channels * 4

        self.bottleneck = ClassicUNetBottleneck3L(bottleneck_c)
        self.decoder = CalibAGDecoder3L(base_channels)

        self.calib_a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.calib_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Calibration context fusion
        cal_bp_embed_dim = base_channels
        self.cal_bp_encoder = nn.Sequential(
            nn.Linear(4, cal_bp_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cal_bp_embed_dim, cal_bp_embed_dim),
            nn.ReLU(inplace=True),
        )

        stats_embed_dim = base_channels
        self.stats_encoder = nn.Sequential(
            nn.Linear(4, stats_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(stats_embed_dim, stats_embed_dim),
            nn.ReLU(inplace=True),
        )

        self._film_channels = {
            'b':  bottleneck_c,
            's2': base_channels * 2,
            's1': base_channels,
            's0': base_channels,
        }
        total_film_c = sum(self._film_channels.values())

        fusion_in_dim = (bottleneck_c * 3) + cal_bp_embed_dim + stats_embed_dim
        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, total_film_c * 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_film_c * 2, total_film_c * 2),
        )
        self._bottleneck_c = bottleneck_c
        self._total_film_c = total_film_c

        if USE_MULTITASK_SBP_DBP:
            self.sbp_head = nn.Sequential(
                nn.LayerNorm(bottleneck_c),
                nn.Linear(bottleneck_c, 1),
            )
            self.dbp_head = nn.Sequential(
                nn.LayerNorm(bottleneck_c),
                nn.Linear(bottleneck_c, 1),
            )
        else:
            self.sbp_head = None
            self.dbp_head = None

    # Run shared encoder, return (features_list, pooled_deep, orig_len).
    def _encode_and_pool(self, x):
        x, orig_len = _pad_to_multiple(x, 8)
        features = self.encoder(x)
        deep = features[-1]
        pooled = torch.mean(deep, dim=2)
        return features, pooled, orig_len

    def _build_film_params(self, pooled_target, pooled_cal1, pooled_cal2,
                           cal_bp, signal_stats):
        B = pooled_target.size(0)
        dev = pooled_target.device
        dtype = pooled_target.dtype

        if cal_bp is not None:
            bp_embed = self.cal_bp_encoder(cal_bp)
        else:
            bp_embed = torch.zeros(B, self.cal_bp_encoder[0].in_features,
                                   device=dev, dtype=dtype)
            bp_embed = self.cal_bp_encoder(bp_embed)

        if signal_stats is not None:
            stats_embed = self.stats_encoder(signal_stats)
        else:
            stats_embed = torch.zeros(
                B, self.stats_encoder[0].out_features,
                device=dev, dtype=dtype)

        parts = [pooled_target, pooled_cal1, pooled_cal2, bp_embed, stats_embed]
        fused = torch.cat(parts, dim=1)
        film_params = self.context_fusion(fused)

        gamma_all, beta_all = film_params.split(self._total_film_c, dim=1)

        ch_list = [self._film_channels[k] for k in ('b', 's2', 's1', 's0')]
        g_b, g_s2, g_s1, g_s0 = gamma_all.split(ch_list, dim=1)
        b_b, b_s2, b_s1, b_s0 = beta_all.split(ch_list, dim=1)

        film = {
            'b':  ((g_b  + 1.0).unsqueeze(-1), b_b.unsqueeze(-1)),
            's2': ((g_s2 + 1.0).unsqueeze(-1), b_s2.unsqueeze(-1)),
            's1': ((g_s1 + 1.0).unsqueeze(-1), b_s1.unsqueeze(-1)),
            's0': ((g_s0 + 1.0).unsqueeze(-1), b_s0.unsqueeze(-1)),
        }
        return film

    def forward(self, x_target, x_cal1, x_cal2, cal_bp,
                signal_stats: Optional[torch.Tensor] = None):
        tgt_feats, pooled_tgt, orig_len = self._encode_and_pool(x_target)
        _, pooled_cal1, _ = self._encode_and_pool(x_cal1)
        _, pooled_cal2, _ = self._encode_and_pool(x_cal2)

        film = self._build_film_params(
            pooled_tgt, pooled_cal1, pooled_cal2, cal_bp, signal_stats)

        deep_features = tgt_feats[-1]
        gamma_b, beta_b = film['b']
        bottleneck_out = gamma_b * self.bottleneck(deep_features) + beta_b

        tgt_feats = list(tgt_feats)
        gamma_s2, beta_s2 = film['s2']
        tgt_feats[2] = gamma_s2 * tgt_feats[2] + beta_s2
        gamma_s1, beta_s1 = film['s1']
        tgt_feats[1] = gamma_s1 * tgt_feats[1] + beta_s1
        gamma_s0, beta_s0 = film['s0']
        tgt_feats[0] = gamma_s0 * tgt_feats[0] + beta_s0

        out = self.decoder(tgt_feats, bottleneck_out)
        out = self.calib_a * out + self.calib_b
        out = _crop_to_length(out, orig_len)

        sbp_pred = dbp_pred = None
        if USE_MULTITASK_SBP_DBP and self.sbp_head is not None and self.dbp_head is not None:
            pooled_bn = torch.mean(bottleneck_out, dim=2)
            sbp_pred = self.sbp_head(pooled_bn).squeeze(-1)
            dbp_pred = self.dbp_head(pooled_bn).squeeze(-1)
        return (out, sbp_pred, dbp_pred) if USE_MULTITASK_SBP_DBP else out

    def extract_bottleneck_embedding(self, x_target, x_cal1, x_cal2, cal_bp,
                                     signal_stats: Optional[torch.Tensor] = None):
        tgt_feats, pooled_tgt, _ = self._encode_and_pool(x_target)
        _, pooled_cal1, _ = self._encode_and_pool(x_cal1)
        _, pooled_cal2, _ = self._encode_and_pool(x_cal2)

        film = self._build_film_params(
            pooled_tgt, pooled_cal1, pooled_cal2, cal_bp, signal_stats)
        deep_features = tgt_feats[-1]
        gamma_b, beta_b = film['b']
        bottleneck_out = gamma_b * self.bottleneck(deep_features) + beta_b
        return _mean_std_pool(bottleneck_out, dim=2)

    def extract_decoder_embedding(self, x_target, x_cal1, x_cal2, cal_bp,
                                  signal_stats: Optional[torch.Tensor] = None):
        tgt_feats, pooled_tgt, orig_len = self._encode_and_pool(x_target)
        _, pooled_cal1, _ = self._encode_and_pool(x_cal1)
        _, pooled_cal2, _ = self._encode_and_pool(x_cal2)

        film = self._build_film_params(
            pooled_tgt, pooled_cal1, pooled_cal2, cal_bp, signal_stats)

        deep_features = tgt_feats[-1]
        gamma_b, beta_b = film['b']
        bottleneck_out = gamma_b * self.bottleneck(deep_features) + beta_b

        tgt_feats = list(tgt_feats)
        gamma_s2, beta_s2 = film['s2']
        tgt_feats[2] = gamma_s2 * tgt_feats[2] + beta_s2
        gamma_s1, beta_s1 = film['s1']
        tgt_feats[1] = gamma_s1 * tgt_feats[1] + beta_s1
        gamma_s0, beta_s0 = film['s0']
        tgt_feats[0] = gamma_s0 * tgt_feats[0] + beta_s0

        last_feat, _ = self.decoder.forward_with_last_feature(tgt_feats, bottleneck_out)
        if orig_len is not None:
            last_feat = last_feat[..., :orig_len]
        return _mean_std_pool(last_feat, dim=2)


# 3-level UNet decoder with plain concat / identity skips only.
class PlainUNetDecoder3L(nn.Module):

    def __init__(self, base_channels: int = BASE_CHANNELS):
        super().__init__()
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

        x = self.up3(x)                          # [B, 4*base, 256]
        x = torch.cat([x, f2], dim=1)
        x = self.conv3(x)                        # [B, 2*base, 256]

        x = self.up2(x)                          # [B, 2*base, 512]
        x = torch.cat([x, f1], dim=1)
        x = self.conv2(x)                        # [B, base, 512]

        x = self.up1(x)                          # [B, base, 1024]
        skip0 = self.up1(f0)                     # [B, base, 1024]
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


class PlainUNet3LNoCal(nn.Module):
    """No-calibration counterpart of CalibrationUNet for fair comparison."""
    def __init__(self, input_channels: int = 3, base_channels: int = BASE_CHANNELS,
                 use_signal_stats: bool = True):
        super().__init__()
        self.use_signal_stats = bool(use_signal_stats)

        self.encoder = ResNetEncoder3L(input_channels, base_channels)
        bottleneck_c = base_channels * 4
        self.bottleneck = ClassicUNetBottleneck3L(bottleneck_c)
        self.decoder = PlainUNetDecoder3L(base_channels)

        self.calib_a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.calib_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        if self.use_signal_stats:
            stats_embed_dim = base_channels
            self.stats_encoder = nn.Sequential(
                nn.Linear(4, stats_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(stats_embed_dim, stats_embed_dim),
                nn.ReLU(inplace=True),
            )
            # Light fusion: bias the bottleneck channels with a learned shift derived from signal_stats.
            self.stats_to_bias = nn.Linear(stats_embed_dim, bottleneck_c)
        else:
            self.stats_encoder = None
            self.stats_to_bias = None

        if USE_MULTITASK_SBP_DBP:
            self.sbp_head = nn.Sequential(
                nn.LayerNorm(bottleneck_c),
                nn.Linear(bottleneck_c, 1),
            )
            self.dbp_head = nn.Sequential(
                nn.LayerNorm(bottleneck_c),
                nn.Linear(bottleneck_c, 1),
            )
        else:
            self.sbp_head = None
            self.dbp_head = None

    def _encode(self, x):
        x, orig_len = _pad_to_multiple(x, 8)
        features = self.encoder(x)
        return features, orig_len

    def _bottleneck_with_stats(self, features, signal_stats: Optional[torch.Tensor]):
        deep = features[-1]
        b = self.bottleneck(deep)
        if self.stats_encoder is not None and signal_stats is not None:
            stats_embed = self.stats_encoder(signal_stats)
            bias = self.stats_to_bias(stats_embed).unsqueeze(-1)
            b = b + bias
        return b

    def forward(self, x_target, signal_stats: Optional[torch.Tensor] = None):
        features, orig_len = self._encode(x_target)
        bottleneck_out = self._bottleneck_with_stats(features, signal_stats)

        out = self.decoder(features, bottleneck_out)
        out = self.calib_a * out + self.calib_b
        out = _crop_to_length(out, orig_len)

        sbp_pred = dbp_pred = None
        if USE_MULTITASK_SBP_DBP and self.sbp_head is not None and self.dbp_head is not None:
            pooled_bn = torch.mean(bottleneck_out, dim=2)
            sbp_pred = self.sbp_head(pooled_bn).squeeze(-1)
            dbp_pred = self.dbp_head(pooled_bn).squeeze(-1)
        return (out, sbp_pred, dbp_pred) if USE_MULTITASK_SBP_DBP else out

    def extract_bottleneck_embedding(self, x_target,
                                     signal_stats: Optional[torch.Tensor] = None):
        features, _ = self._encode(x_target)
        bottleneck_out = self._bottleneck_with_stats(features, signal_stats)
        return _mean_std_pool(bottleneck_out, dim=2)

    def extract_decoder_embedding(self, x_target,
                                  signal_stats: Optional[torch.Tensor] = None):
        features, orig_len = self._encode(x_target)
        bottleneck_out = self._bottleneck_with_stats(features, signal_stats)
        last_feat, _ = self.decoder.forward_with_last_feature(features, bottleneck_out)
        if orig_len is not None:
            last_feat = last_feat[..., :orig_len]
        return _mean_std_pool(last_feat, dim=2)


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


class WaveformLoss(nn.Module):
    """Weighted SmoothL1 on the waveform and its first difference; peaks and valleys weigh more."""
    def __init__(self, alpha=0.7, beta=0.3, peak_weight=2.5, valley_weight=1.8, radius_samples=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.peak_weight = peak_weight
        self.valley_weight = valley_weight
        self.radius = radius_samples if radius_samples is not None else max(1, int(0.04 * FS))
        self.huber = nn.SmoothL1Loss(reduction="none", beta=1.0)

    def forward(self, pred, target):
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() == 0:
            return (pred.sum() * 0.0) + (target.sum() * 0.0)

        t = target
        B, L = t.shape
        if L < 3:
            elem = self.huber(pred[mask], target[mask]).mean()
            pred_grad = pred[:, 1:] - pred[:, :-1]
            target_grad = target[:, 1:] - target[:, :-1]
            mask_g = torch.isfinite(pred_grad) & torch.isfinite(target_grad)
            grad = self.huber(pred_grad[mask_g], target_grad[mask_g]).mean() if mask_g.any() else elem.detach() * 0.0
            return self.alpha * elem + self.beta * grad

        g = t[:, 1:] - t[:, :-1]
        s = torch.sign(g)
        s1, s2 = s[:, :-1], s[:, 1:]
        peaks_idx = (s1 > 0) & (s2 <= 0)
        valleys_idx = (s1 < 0) & (s2 >= 0)

        peaks = torch.zeros(B, L, device=t.device, dtype=t.dtype)
        valleys = torch.zeros_like(peaks)
        peaks[:, 1:-1] = peaks_idx.float()
        valleys[:, 1:-1] = valleys_idx.float()

        rad = int(self.radius)
        k = torch.ones(1, 1, 2 * rad + 1, device=t.device, dtype=t.dtype)
        pk_d = (F.conv1d(peaks.unsqueeze(1), k, padding=rad) > 0).squeeze(1)
        vl_d = (F.conv1d(valleys.unsqueeze(1), k, padding=rad) > 0).squeeze(1)

        w = torch.ones(B, L, device=t.device, dtype=t.dtype)
        w = w + (self.peak_weight - 1.0) * pk_d.float()
        w = w + (self.valley_weight - 1.0) * vl_d.float()
        w = w * mask.float()
        w_sum = w.sum().clamp_min(1.0)

        huber_per = self.huber(pred, target)
        huber_loss = (huber_per * w).sum() / w_sum

        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        mask_g = torch.isfinite(pred_grad) & torch.isfinite(target_grad)
        if mask_g.any():
            wg = 0.5 * (w[:, 1:] + w[:, :-1]) * mask_g.float()
            wg_sum = wg.sum().clamp_min(1.0)
            grad_per = self.huber(pred_grad, target_grad)
            grad_loss = (grad_per * wg).sum() / wg_sum
        else:
            grad_loss = huber_loss.detach() * 0.0

        return self.alpha * huber_loss + self.beta * grad_loss
