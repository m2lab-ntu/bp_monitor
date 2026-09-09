import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # A mamba refiner then raises.

# Model configuration defaults (overridden by train.py)
FS = 125
USE_MULTITASK_SBP_DBP = True
DROPOUT_P = 0.2
BASE_CHANNELS = 32

# Skip refiner: "transformer" | "mamba" | "conv"; train.py may override model_lib.SKIP_REFINER.
SKIP_REFINER = "transformer"

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


# Fixed sinusoidal positional encoding for the transformer refiner.
class _SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, d_model]
        return x + self.pe[:, : x.size(1)]


# Attention Gate with additive residual refinement.
class MambaAttentionGate(nn.Module):

    def __init__(self, c_skip: int, c_gate: int, c_inter: Optional[int] = None,
                 d_state: int = 8, d_conv: int = 4, expand: int = 2,
                 dropout: float = DROPOUT_P, init_alpha_bias: float = -2.0,
                 bidirectional: bool = True, use_channel_attn: bool = True,
                 refiner: Optional[str] = None, n_heads: int = 4,
                 ffn_mult: int = 2):
        super().__init__()
        if c_inter is None:
            c_inter = max(c_skip // 2, 8)
        self.c_inter = c_inter
        self.bidir = bool(bidirectional)
        self.use_channel_attn = bool(use_channel_attn)

        # Read SKIP_REFINER at construction time (not as a def-time default) so train.py's.
        refiner = (refiner or SKIP_REFINER or "mamba").lower()
        self.W_x = nn.Conv1d(c_skip, c_inter, kernel_size=1, bias=False)
        self.W_g = nn.Conv1d(c_gate, c_inter, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c_inter)

        # Defaults; the active branch below overrides what it needs.
        self.mamba_fwd = None
        self.mamba_bwd = None
        self.transformer = None
        self.pos_enc = None
        self.fallback = None

        if refiner == "transformer":
            self.refiner = "transformer"
            self.norm = nn.LayerNorm(c_inter)
            # Pick the largest head count <= n_heads that divides c_inter.
            nhead = max(1, int(n_heads))
            while nhead > 1 and (c_inter % nhead) != 0:
                nhead -= 1
            self.pos_enc = _SinusoidalPositionalEncoding(c_inter)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=c_inter, nhead=nhead,
                dim_feedforward=max(c_inter * int(ffn_mult), c_inter),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
        elif refiner == "mamba" and Mamba is not None:
            self.refiner = "mamba"
            self.norm = nn.LayerNorm(c_inter)
            self.mamba_fwd = Mamba(d_model=c_inter, d_state=d_state,
                                   d_conv=d_conv, expand=expand)
            self.mamba_bwd = (
                Mamba(d_model=c_inter, d_state=d_state, d_conv=d_conv, expand=expand)
                if self.bidir else None
            )
        else:
            if refiner == "mamba":
                raise ImportError(
                "mamba_ssm is not installed; a Mamba skip refiner cannot be built."
            )
            self.refiner = "conv"
            self.norm = nn.Identity()
            self.fallback = nn.Sequential(
                nn.Conv1d(c_inter, c_inter, kernel_size=7, padding=3, groups=c_inter),
                nn.GELU(),
                nn.Conv1d(c_inter, c_inter, kernel_size=1),
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

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        xg = F.relu(self.bn(self.W_x(skip) + self.W_g(gate)))   # [B, c_inter, L]

        if self.refiner == "transformer":
            seq = xg.transpose(1, 2)            # [B, L, c_inter]
            seq = self.norm(seq)
            seq = self.pos_enc(seq)
            seq = self.transformer(seq)         # [B, L, c_inter] (self-attn is bidirectional)
            seq = self.dropout(seq)
            seq = seq.transpose(1, 2)           # [B, c_inter, L]
        elif self.mamba_fwd is not None:
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
        else:
            seq = self.dropout(self.fallback(xg))

        v = self.proj_v(seq)                                  # [B, c_skip, L]
        a_t = torch.sigmoid(self.psi_t(seq))                  # [B, 1,      L]
        if self.psi_c is not None:
            a_c = torch.sigmoid(self.psi_c(seq))              # [B, c_skip, 1]
            alpha = a_t * a_c
        else:
            alpha = a_t
        return skip + alpha * v


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
