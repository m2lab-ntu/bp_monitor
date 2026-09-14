# No-calibration U-Net with refiners inside the decoder path (not on the skips):
# Mamba (MambaUNet3LNoCal) or Transformer (TransformerUNet3LNoCal).
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

# Identity-safe residual initialisation for the decoder refiners: zero-init the value.
SKIP_REFINER_ZERO_INIT_VALUE = True
SKIP_REFINER_INIT_ALPHA_BIAS = -4.0

# When False, the shallowest (highest-resolution, L=1024) decoder stage gets no refiner.
SKIP_REFINER_USE_SHALLOW_SKIP = False


# Mamba self-refiner inside the decoder path; causal (forward-only) by default.
class MambaDecoderRefiner(nn.Module):

    def __init__(self, channels: int,
                 d_state: int = 8, d_conv: int = 4, expand: int = 2,
                 dropout: float = DROPOUT_P,
                 bidirectional: bool = False,
                 use_channel_attn: bool = True,
                 init_alpha_bias: Optional[float] = None,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaDecoderRefiner requires mamba_ssm.Mamba. Install "
                "mamba_ssm or use the plain / transformer decoder variant."
            )
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        self.channels = int(channels)
        self.bidir = bool(bidirectional)
        self.use_channel_attn = bool(use_channel_attn)

        self.norm = nn.LayerNorm(channels)
        self.mamba_fwd = Mamba(d_model=channels, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = (
            Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
            if self.bidir else None
        )

        self.proj_v = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.psi_t = nn.Conv1d(channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(channels // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, channels, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)             # [B, L, C]
        seq = self.norm(seq)
        # Causal forward-only path: no time reversal, no bidirectional fusion.
        seq = self.mamba_fwd(seq)
        seq = self.dropout(seq)
        seq = seq.transpose(1, 2)           # [B, C, L]

        v = self.proj_v(seq)
        a_t = torch.sigmoid(self.psi_t(seq))
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq))
            alpha = a_t * a_c
        else:
            alpha = a_t
        return x + alpha * v


# Multi-head self-attention + FFN refiner applied INSIDE the decoder.
class TransformerDecoderRefiner(nn.Module):

    def __init__(self, channels: int,
                 n_heads: int = 4, ff_mult: int = 2,
                 dropout: float = DROPOUT_P,
                 use_channel_attn: bool = True,
                 init_alpha_bias: Optional[float] = None,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        while n_heads > 1 and (channels % n_heads != 0):
            n_heads //= 2
        self.channels = int(channels)
        self.n_heads = n_heads
        self.use_channel_attn = bool(use_channel_attn)

        self.pos_enc = SinusoidalPositionalEncoding1D(channels)

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        ff_hidden = channels * int(ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, channels),
        )

        self.proj_v = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.psi_t = nn.Conv1d(channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(channels // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, channels, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)             # [B, L, C]
        seq = self.pos_enc(seq)
        h = self.norm1(seq)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.dropout(attn_out)
        h = self.norm2(seq)
        seq = seq + self.dropout(self.ff(h))
        seq = seq.transpose(1, 2)           # [B, C, L]

        v = self.proj_v(seq)
        a_t = torch.sigmoid(self.psi_t(seq))
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq))
            alpha = a_t * a_c
        else:
            alpha = a_t
        return x + alpha * v


# In-decoder Transformer refiner with pooled internal attention.
class PooledTransformerDecoderRefiner(nn.Module):

    def __init__(self, channels: int,
                 n_heads: int = 2, ff_mult: int = 2,
                 pool_factor: int = 4,
                 dropout: float = DROPOUT_P,
                 use_channel_attn: bool = True,
                 init_alpha_bias: Optional[float] = None,
                 zero_init_value: Optional[bool] = None):
        super().__init__()
        if init_alpha_bias is None:
            init_alpha_bias = float(SKIP_REFINER_INIT_ALPHA_BIAS)
        if zero_init_value is None:
            zero_init_value = bool(SKIP_REFINER_ZERO_INIT_VALUE)
        while n_heads > 1 and (channels % n_heads != 0):
            n_heads //= 2
        self.channels = int(channels)
        self.n_heads = n_heads
        self.pool_factor = max(1, int(pool_factor))
        self.use_channel_attn = bool(use_channel_attn)

        if self.pool_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=self.pool_factor,
                                     stride=self.pool_factor)
            self.upsample = nn.Upsample(scale_factor=self.pool_factor,
                                        mode='linear', align_corners=True)
        else:
            self.pool = nn.Identity()
            self.upsample = nn.Identity()

        self.pos_enc = SinusoidalPositionalEncoding1D(channels)

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        ff_hidden = channels * int(ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, channels),
        )

        self.proj_v = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.psi_t = nn.Conv1d(channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.psi_t.bias, init_alpha_bias)

        if self.use_channel_attn:
            mid = max(channels // 2, 4)
            self.psi_c = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(mid, channels, kernel_size=1),
            )
            nn.init.zeros_(self.psi_c[-1].bias)
        else:
            self.psi_c = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if zero_init_value:
            nn.init.zeros_(self.proj_v.weight)
            nn.init.zeros_(self.proj_v.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = int(x.size(-1))
        if self.pool_factor > 1:
            pad_amount = 0
            if L % self.pool_factor != 0:
                target_L = int(math.ceil(L / self.pool_factor)
                               * self.pool_factor)
                pad_amount = target_L - L
                x_padded = F.pad(x, (0, pad_amount))
            else:
                x_padded = x
            x_pooled = self.pool(x_padded)
        else:
            pad_amount = 0
            x_pooled = x

        seq = x_pooled.transpose(1, 2)
        seq = self.pos_enc(seq)
        h = self.norm1(seq)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.dropout(attn_out)
        h = self.norm2(seq)
        seq = seq + self.dropout(self.ff(h))
        seq_pooled = seq.transpose(1, 2)

        if self.pool_factor > 1:
            seq_up = self.upsample(seq_pooled)
            if pad_amount:
                seq_up = seq_up[..., :L]
        else:
            seq_up = seq_pooled

        v = self.proj_v(seq_up)
        a_t = torch.sigmoid(self.psi_t(seq_up))
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq_up))
            alpha = a_t * a_c
        else:
            alpha = a_t
        return x + alpha * v


# 3-level UNet decoder with in-decoder Mamba self-refiners.
class MambaRefinerDecoder3LNoCal(nn.Module):

    def __init__(self, base_channels: int = BASE_CHANNELS,
                 ag_d_state: int = 8, ag_d_conv: int = 4, ag_expand: int = 2,
                 use_shallow_skip: Optional[bool] = None):
        super().__init__()
        if use_shallow_skip is None:
            use_shallow_skip = bool(SKIP_REFINER_USE_SHALLOW_SKIP)
        self.use_shallow_skip = bool(use_shallow_skip)

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)
        # Refiner after conv3 (output: [B, 2*base, 256]).
        self.ag2 = MambaDecoderRefiner(
            channels=base_channels * 2,
            d_state=ag_d_state, d_conv=ag_d_conv, expand=ag_expand,
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)
        # Refiner after conv2 (output: [B, base, 512]).
        self.ag1 = MambaDecoderRefiner(
            channels=base_channels,
            d_state=ag_d_state, d_conv=ag_d_conv, expand=ag_expand,
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)
        # Optional refiner after conv1 (output: [B, base, 1024]).
        if self.use_shallow_skip:
            self.ag0 = MambaDecoderRefiner(
                channels=base_channels,
                d_state=ag_d_state, d_conv=ag_d_conv, expand=ag_expand,
            )
        else:
            self.ag0 = None

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        f0, f1, f2, _ = features
        x = bottleneck_out                       # [B, 4*base, 128]

        # Stage 1: 128 -> 256.
        x = self.up3(x)                          # [B, 4*base, 256]
        x = torch.cat([x, f2], dim=1)            # plain skip concat
        x = self.conv3(x)                        # [B, 2*base, 256]
        x = self.ag2(x)                          # in-decoder Mamba refine

        # Stage 2: 256 -> 512.
        x = self.up2(x)                          # [B, 2*base, 512]
        x = torch.cat([x, f1], dim=1)            # plain skip concat
        x = self.conv2(x)                        # [B, base, 512]
        x = self.ag1(x)                          # in-decoder Mamba refine

        # Stage 3: 512 -> 1024.
        x = self.up1(x)                          # [B, base, 1024]
        skip0 = self.up1(f0)                     # [B, base, 1024] (identity skip)
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)                        # [B, base, 1024]
        if self.ag0 is not None:
            x = self.ag0(x)                      # optional in-decoder Mamba refine
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        # Normalised BP waveform lies in [0, 1]; sigmoid keeps the linear projection from overshooting.
        return torch.sigmoid(x.squeeze(1))

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        # Normalised BP waveform lies in [0, 1]; sigmoid keeps the linear projection from overshooting.
        return x, torch.sigmoid(out.squeeze(1))


# 3-level UNet decoder with in-decoder Transformer self-refiners.
class TransformerRefinerDecoder3LNoCal(nn.Module):

    def __init__(self, base_channels: int = BASE_CHANNELS,
                 n_heads: int = 2, ff_mult: int = 1,
                 ag0_pool_factor: int = 4,
                 use_shallow_skip: Optional[bool] = None):
        super().__init__()
        if use_shallow_skip is None:
            use_shallow_skip = bool(SKIP_REFINER_USE_SHALLOW_SKIP)
        self.use_shallow_skip = bool(use_shallow_skip)

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)
        # Refiner after conv3 (output: [B, 2*base, 256]).
        self.ag2 = TransformerDecoderRefiner(
            channels=base_channels * 2,
            n_heads=n_heads, ff_mult=ff_mult,
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)
        # Refiner after conv2 (output: [B, base, 512]).
        self.ag1 = TransformerDecoderRefiner(
            channels=base_channels,
            n_heads=max(1, n_heads // 2), ff_mult=ff_mult,
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)
        # Optional pooled refiner after conv1 (output: [B, base, 1024]).
        if self.use_shallow_skip:
            self.ag0 = PooledTransformerDecoderRefiner(
                channels=base_channels,
                n_heads=max(1, n_heads // 2), ff_mult=ff_mult,
                pool_factor=int(ag0_pool_factor),
            )
        else:
            self.ag0 = None

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        f0, f1, f2, _ = features
        x = bottleneck_out

        # Stage 1: 128 -> 256.
        x = self.up3(x)                          # [B, 4*base, 256]
        x = torch.cat([x, f2], dim=1)            # plain skip concat
        x = self.conv3(x)                        # [B, 2*base, 256]
        x = self.ag2(x)                          # in-decoder transformer refine

        # Stage 2: 256 -> 512.
        x = self.up2(x)                          # [B, 2*base, 512]
        x = torch.cat([x, f1], dim=1)            # plain skip concat
        x = self.conv2(x)                        # [B, base, 512]
        x = self.ag1(x)                          # in-decoder transformer refine

        # Stage 3: 512 -> 1024.
        x = self.up1(x)                          # [B, base, 1024]
        skip0 = self.up1(f0)                     # [B, base, 1024] (identity skip)
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)                        # [B, base, 1024]
        if self.ag0 is not None:
            x = self.ag0(x)                      # optional in-decoder transformer refine
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        # Normalised BP waveform lies in [0, 1]; sigmoid keeps the linear projection from overshooting.
        return torch.sigmoid(x.squeeze(1))

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        # Normalised BP waveform lies in [0, 1]; sigmoid keeps the linear projection from overshooting.
        return x, torch.sigmoid(out.squeeze(1))


class MambaUNet3LNoCal(PlainUNet3LNoCal):
    """No-calibration UNet with Mamba refiners inside the decoder."""
    def __init__(self, input_channels: int = 3,
                 base_channels: int = BASE_CHANNELS,
                 use_signal_stats: bool = True,
                 ag_d_state: int = 8, ag_d_conv: int = 4, ag_expand: int = 2):
        super().__init__(input_channels=input_channels,
                         base_channels=base_channels,
                         use_signal_stats=use_signal_stats)
        self.decoder = MambaRefinerDecoder3LNoCal(
            base_channels=base_channels,
            ag_d_state=ag_d_state, ag_d_conv=ag_d_conv, ag_expand=ag_expand,
        )


class TransformerUNet3LNoCal(PlainUNet3LNoCal):
    """No-calibration UNet with Transformer refiners inside the decoder."""
    def __init__(self, input_channels: int = 3,
                 base_channels: int = BASE_CHANNELS,
                 use_signal_stats: bool = True,
                 n_heads: int = 4, ff_mult: int = 2):
        super().__init__(input_channels=input_channels,
                         base_channels=base_channels,
                         use_signal_stats=use_signal_stats)
        self.decoder = TransformerRefinerDecoder3LNoCal(
            base_channels=base_channels,
            n_heads=n_heads, ff_mult=ff_mult,
        )
