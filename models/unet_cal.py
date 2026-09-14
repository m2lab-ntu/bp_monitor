# Calibration U-Net: two calibration segments and their BP condition the decoder through
# FiLM. The skip refiner is switchable ("mamba" | "transformer" | "conv"); the per-master
# residual adapter in the decoder is identity until train.py fine-tunes it.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import (
    Mamba, USE_MULTITASK_SBP_DBP, DROPOUT_P, BASE_CHANNELS,
    _pad_to_multiple, _crop_to_length, _mean_std_pool,
    ResNetEncoder3L, DoubleConv1D, ClassicUNetBottleneck3L, SinusoidalPositionalEncoding1D,
    PlainUNetDecoder3L, PlainUNet3LNoCal, WaveformLoss, ResidualFeatureAdapter1D,
)

__all__ = ["CalibrationUNet", "PlainUNet3LNoCal", "WaveformLoss", "ResidualFeatureAdapter1D"]

# Skip refiner: "transformer" | "mamba" | "conv". The registry sets it per model key.
SKIP_REFINER = "mamba"

# Whether the CalibrationUNet decoder uses skip Attention Gates.
USE_SKIP_ATTENTION_GATE = True


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

        # Read SKIP_REFINER at construction time (not as a def-time default) so the registry's value applies.
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
            self.pos_enc = SinusoidalPositionalEncoding1D(c_inter)
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


# 3-level UNet decoder with Attention Gates (refiner per SKIP_REFINER) on the two deep skips.
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
        # Decoder: attention-gated skips (CalibAGDecoder3L) or plain skips (PlainUNetDecoder3L).
        self.use_skip_attention_gate = bool(USE_SKIP_ATTENTION_GATE)
        if self.use_skip_attention_gate:
            self.decoder = CalibAGDecoder3L(base_channels)
        else:
            self.decoder = PlainUNetDecoder3L(base_channels)

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
