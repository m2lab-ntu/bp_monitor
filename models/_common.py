# Blocks shared by every model family: config defaults, ResNet encoder, plain decoder,
# no-calibration baseline U-Net, waveform loss, positional encoding, per-master adapter.
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # Modules that need Mamba raise at construction.

# Model configuration defaults; train.py overrides them through models.configure().
FS = 125
USE_MULTITASK_SBP_DBP = True
DROPOUT_P = 0.2
BASE_CHANNELS = 32


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
