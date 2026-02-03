import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ------------------------------------------------------------
# Model configuration defaults (overridden by train.py)
# ------------------------------------------------------------
FS = 125
USE_TRANSFORMER_SKIPS = 0
USE_TRANSFORMER_POSENC = 1
USE_MULTITASK_SBP_DBP = True
DROPOUT_P = 0.2
BASE_CHANNELS = 128

# ------------------------------------------------------------
# Helper functions for model internals
# ------------------------------------------------------------

def _build_sinusoidal_posenc(length: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """Create sinusoidal positional encoding of shape [1, length, dim].
    If dim is odd, pad the last dimension with 0.
    """
    device = device or (torch.device("cpu"))
    dtype = dtype or torch.float32
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
    # Even dims use sin, odd dims use cos
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, dim)))
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _pad_to_multiple(x: torch.Tensor, mult: int):
    """Right-pad temporal axis to be divisible by mult; return (padded, orig_len)."""
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


def _get_skip_mode() -> str:
    try:
        v = int(USE_TRANSFORMER_SKIPS)
    except Exception:
        v = 1
    if v == -1:
        return "identity"
    if v == 0:
        return "mamba"
    return "transformer"


# ------------------------------------------------------------
# (9) ResNet Encoder
# ------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
        # Dropout
        self.dropout = nn.Dropout(DROPOUT_P)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        # Dropout
        out = self.dropout(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS):
        super().__init__()

        # Initial convolution optimized for PPG/ECG signals
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 1024 -> 512
        )

        # ResNet blocks
        self.layer1 = self._make_layer(base_channels, base_channels, 2)  # 512
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)  # 256
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)  # 128
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)  # 64

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 3, 1024]
        features = []

        x = self.initial_conv(x)  # [batch, base_channels, 512]
        features.append(x)

        x = self.layer1(x)  # [batch, base_channels, 512]
        features.append(x)

        x = self.layer2(x)  # [batch, 128, 256]
        features.append(x)

        x = self.layer3(x)  # [batch, 256, 128]
        features.append(x)

        x = self.layer4(x)  # [batch, 512, 64]
        features.append(x)

        return features


class ResNetEncoder3L(nn.Module):
    """ResNet encoder with one fewer downsampling level (3-level UNet).
    Length path: 1024 -> 512 (initial pool) -> 256 -> 128
    Channels: base -> base -> 2*base -> 4*base
    Returns 4 feature maps for skips: f0(512), f1(512), f2(256), f3(128)
    """
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 1024 -> 512
        )
        # Keep layer1 at same length (512)
        self.layer1 = ResNetBlock(base_channels, base_channels)
        # Two downsampling stages to 256 and 128
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


# ------------------------------------------------------------
# (10-) Core modules: Double Conv and Mamba/Transformer Skip Blocks
# ------------------------------------------------------------
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


class IdentitySkipBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class MambaSkipBlock(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = DROPOUT_P):
        super().__init__()
        self.channels = channels
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed; MambaSkipBlock requires mamba_ssm.")
        self.norm = nn.LayerNorm(channels)
        self.core = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        residual = x
        seq = x.transpose(1, 2)           # [B, L, C]
        seq = self.norm(seq)
        seq = self.core(seq)
        seq = self.dropout(seq)
        seq = seq.transpose(1, 2)         # [B, C, L]
        return seq + residual


class TransformerSkipBlock(nn.Module):
    def __init__(self, channels: int, nhead: int | None = None, num_layers: int = 1,
                 dropout: float = DROPOUT_P, d_model_override: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self._cached_pe = None  # cache by (L, device, dtype)

        # Allow linear projection when d_model differs from channels
        enc_dim = int(d_model_override) if (isinstance(d_model_override, int) and d_model_override > 0) else int(channels)
        self.proj_in = nn.Linear(channels, enc_dim) if enc_dim != channels else None
        self.proj_out = nn.Linear(enc_dim, channels) if enc_dim != channels else None

        # Pick suitable number of heads; must divide enc_dim
        def _pick_head(c: int) -> int:
            for h in (8, 4, 2, 1):
                if c % h == 0:
                    return h
            return 1
        nhead = nhead if (isinstance(nhead, int) and nhead > 0 and enc_dim % nhead == 0) else _pick_head(enc_dim)

        self.ln = nn.LayerNorm(enc_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=nhead,
            dim_feedforward=max(128, 4 * enc_dim),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, C, L] -> [B, L, C]
        residual = x
        seq = x.transpose(1, 2)
        # Add sinusoidal positional encoding
        if USE_TRANSFORMER_POSENC:
            L = seq.size(1)
            C = seq.size(2)
            dev = seq.device
            dt = seq.dtype
            # Cache to avoid rebuilding
            if (
                self._cached_pe is None or
                self._cached_pe.shape[1] != L or
                self._cached_pe.shape[2] != C or
                self._cached_pe.device != dev or
                self._cached_pe.dtype != dt
            ):
                self._cached_pe = _build_sinusoidal_posenc(L, C, device=dev, dtype=dt)
            seq = seq + self._cached_pe

        # Project dimensions if needed
        if self.proj_in is not None:
            seq = self.proj_in(seq)

        seq = self.ln(seq)
        seq = self.encoder(seq)
        # Project back
        if self.proj_out is not None:
            seq = self.proj_out(seq)
        seq = self.dropout(seq)
        seq = seq.transpose(1, 2)
        return seq + residual


class UNetDecoder(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS, *,
                 skip_nhead: Optional[int] = None,
                 skip_d_model: Optional[int] = None,
                 skip_d_state: Optional[int] = None):
        super().__init__()

        skip_mode = _get_skip_mode()
        if skip_mode == "identity":
            SkipCls = IdentitySkipBlock
        elif skip_mode == "transformer":
            SkipCls = TransformerSkipBlock
        else:
            SkipCls = MambaSkipBlock
        def _mk_skip(ch: int):
            if skip_mode == "identity":
                return SkipCls(ch)
            if skip_mode == "transformer":
                return SkipCls(ch, nhead=skip_nhead, dropout=DROPOUT_P, d_model_override=skip_d_model)
            else:
                ds = int(skip_d_state) if (isinstance(skip_d_state, int) and skip_d_state > 0) else 16
                return SkipCls(ch, d_state=ds, dropout=DROPOUT_P)
        self.skip_blocks = nn.ModuleList([
            _mk_skip(base_channels * 4),
            _mk_skip(base_channels * 2),
            _mk_skip(base_channels),
            _mk_skip(base_channels),
        ])

        self.up4 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv4 = DoubleConv1D(base_channels * 8 + base_channels * 4, base_channels * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        # features: [f0(64,512), f1(64,512), f2(128,256), f3(256,128), f4(512,64)]
        f0, f1, f2, f3, _ = features

        x = bottleneck_out  # [B, 512, 64]

        x = self.up4(x)  # [B, 512, 128]
        skip3 = self.skip_blocks[0](f3)  # [B, 256, 128]
        x = torch.cat([x, skip3], dim=1)
        x = self.conv4(x)  # [B, 256, 128]

        x = self.up3(x)  # [B, 256, 256]
        skip2 = self.skip_blocks[1](f2)  # [B, 128, 256]
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)  # [B, 128, 256]

        x = self.up2(x)  # [B, 128, 512]
        skip1 = self.skip_blocks[2](f1)  # [B, 64, 512]
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)  # [B, 64, 512]

        x = self.up1(x)  # [B, 64, 1024]
        skip0 = self.up1(f0)  # [B, 64, 1024]
        skip0 = self.skip_blocks[3](skip0)
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)  # [B, 64, 1024]
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        return x.squeeze(1)  # [B, 1024]

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        return x, out.squeeze(1)


class UNetDecoder3L(nn.Module):
    """UNet decoder with 3 upsampling stages matching ResNetEncoder3L.
    Expects features: f0(base,512), f1(base,512), f2(2*base,256), deep/bottleneck(4*base,128)
    Skip refinement blocks mirror the original behavior.
    """
    def __init__(self, base_channels=BASE_CHANNELS):
        super().__init__()
        skip_mode = _get_skip_mode()
        if skip_mode == "identity":
            SkipCls = IdentitySkipBlock
        elif skip_mode == "transformer":
            SkipCls = TransformerSkipBlock
        else:
            SkipCls = MambaSkipBlock
        def _mk_skip(ch: int):
            if skip_mode == "identity":
                return SkipCls(ch)
            return SkipCls(ch)
        self.skip_blocks = nn.ModuleList([
            _mk_skip(base_channels * 2),  # for f2 (256)
            _mk_skip(base_channels),      # for f1 (512)
            _mk_skip(base_channels),      # for f0 (512 -> up to 1024)
        ])

        # up from 128 -> 256, concat with f2 (2*base)
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1D(base_channels * 4 + base_channels * 2, base_channels * 2)

        # up 256 -> 512, concat with f1 (base)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1D(base_channels * 2 + base_channels, base_channels)

        # up 512 -> 1024, concat with f0 (base)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1D(base_channels + base_channels, base_channels)

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        # features: [f0( base,512), f1( base,512), f2(2*base,256), f3(4*base,128)]
        f0, f1, f2, _ = features

        x = bottleneck_out  # [B, 4*base, 128]

        # 128 -> 256, join with f2
        x = self.up3(x)  # [B, 4*base, 256]
        skip2 = self.skip_blocks[0](f2)  # [B, 2*base, 256]
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)  # [B, 2*base, 256]

        # 256 -> 512, join with f1
        x = self.up2(x)  # [B, 2*base, 512]
        skip1 = self.skip_blocks[1](f1)  # [B, base, 512]
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)  # [B, base, 512]

        # 512 -> 1024, join with f0 (upsampled)
        x = self.up1(x)  # [B, base, 1024]
        skip0 = self.up1(f0)  # [B, base, 1024]
        skip0 = self.skip_blocks[2](skip0)
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)  # [B, base, 1024]
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        return x.squeeze(1)

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        return x, out.squeeze(1)


# ------------------------------------------------------------
# (10+) Mamba Skip modules and other components
# ------------------------------------------------------------
class ClassicUNetBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = DoubleConv1D(channels, channels)

    def forward(self, x):
        return self.block(x)


class ClassicUNetBottleneck3L(nn.Module):
    """Same as ClassicUNetBottleneck but serves as a semantic marker for 3L variant."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = DoubleConv1D(channels, channels)

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
# (11) Full ResNet-UNet model (Mamba skip refinement)
# ------------------------------------------------------------
class ResNetUNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS, bottleneck: str = "mamba",
                 skip_nhead: Optional[int] = None,
                 skip_d_model: Optional[int] = None,
                 skip_d_state: Optional[int] = None):
        super().__init__()
        bottleneck_type = str(bottleneck).lower()
        valid_types = {"mamba", "transformer", "identity", "classic", "unet"}
        if bottleneck_type not in valid_types:
            raise ValueError(f"Unsupported bottleneck type: {bottleneck}")
        if bottleneck_type == "transformer":
            warnings.warn(
                "Bottleneck uses Classic UNet; Transformer/Mamba/Identity only applied to decoder skip modules (controlled by USE_TRANSFORMER_SKIPS)",
                RuntimeWarning,
            )

        self.input_fuser = nn.Identity()
        self.encoder = ResNetEncoder(input_channels, base_channels)
        bottleneck_c = base_channels * 8  # encoder deepest channels (e.g., 512)

        self.bottleneck = ClassicUNetBottleneck(bottleneck_c)

        self.decoder = UNetDecoder(base_channels,
                                   skip_nhead=skip_nhead,
                                   skip_d_model=skip_d_model,
                                   skip_d_state=skip_d_state)
        # learned global affine calibration
        self.calib_a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.calib_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        stats_embed_dim = base_channels
        fusion_in_dim = bottleneck_c + stats_embed_dim
        self.stats_encoder = nn.Sequential(
            nn.Linear(4, stats_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(stats_embed_dim, stats_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, bottleneck_c),
            nn.ReLU(inplace=True),
        )

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

    def forward(self, x, signal_stats: Optional[torch.Tensor] = None):
        # x: [B, 3, L]
        x = self.input_fuser(x)
        x, orig_len = _pad_to_multiple(x, 16)
        features = self.encoder(x)
        deep_features = features[-1]              # [B, 512, 64]
        pooled = torch.mean(deep_features, dim=2)  # [B, 512]

        if signal_stats is None:
            stats_encoded = torch.zeros(
                pooled.size(0),
                self.stats_encoder[0].out_features,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            stats_encoded = self.stats_encoder(signal_stats)

        fusion_parts = [pooled, stats_encoded]
        fused_context = torch.cat(fusion_parts, dim=1)
        fusion_bias = self.context_fusion(fused_context).unsqueeze(-1)
        bottleneck_out = self.bottleneck(deep_features) + fusion_bias

        out = self.decoder(features, bottleneck_out)  # [B, 1024]
        out = self.calib_a * out + self.calib_b
        out = _crop_to_length(out, orig_len)

        sbp_pred = dbp_pred = None
        if USE_MULTITASK_SBP_DBP and self.sbp_head is not None and self.dbp_head is not None:
            pooled_bn = torch.mean(bottleneck_out, dim=2)
            sbp_pred = self.sbp_head(pooled_bn).squeeze(-1)
            dbp_pred = self.dbp_head(pooled_bn).squeeze(-1)
        return (out, sbp_pred, dbp_pred) if USE_MULTITASK_SBP_DBP else out

    def extract_bottleneck_embedding(self, x, signal_stats: Optional[torch.Tensor] = None):
        x = self.input_fuser(x)
        x, _ = _pad_to_multiple(x, 16)
        features = self.encoder(x)
        deep_features = features[-1]
        pooled = torch.mean(deep_features, dim=2)

        if signal_stats is None:
            stats_encoded = torch.zeros(
                pooled.size(0),
                self.stats_encoder[0].out_features,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            stats_encoded = self.stats_encoder(signal_stats)

        fusion_parts = [pooled, stats_encoded]
        fused_context = torch.cat(fusion_parts, dim=1)
        fusion_bias = self.context_fusion(fused_context).unsqueeze(-1)
        bottleneck_out = self.bottleneck(deep_features) + fusion_bias
        return _mean_std_pool(bottleneck_out, dim=2)

    def extract_decoder_embedding(self, x, signal_stats: Optional[torch.Tensor] = None):
        x = self.input_fuser(x)
        x, orig_len = _pad_to_multiple(x, 16)
        features = self.encoder(x)
        deep_features = features[-1]
        pooled = torch.mean(deep_features, dim=2)

        if signal_stats is None:
            stats_encoded = torch.zeros(
                pooled.size(0),
                self.stats_encoder[0].out_features,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            stats_encoded = self.stats_encoder(signal_stats)

        fusion_parts = [pooled, stats_encoded]
        fused_context = torch.cat(fusion_parts, dim=1)
        fusion_bias = self.context_fusion(fused_context).unsqueeze(-1)
        bottleneck_out = self.bottleneck(deep_features) + fusion_bias

        last_feat, _ = self.decoder.forward_with_last_feature(features, bottleneck_out)
        if orig_len is not None:
            last_feat = last_feat[..., :orig_len]
        return _mean_std_pool(last_feat, dim=2)


class ResNetUNet3L(nn.Module):
    """ResNet-UNet with 3 decoder levels (one fewer than default)."""
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS, bottleneck: str = "mamba"):
        super().__init__()
        bottleneck_type = str(bottleneck).lower()
        valid_types = {"mamba", "transformer", "identity", "classic", "unet"}
        if bottleneck_type not in valid_types:
            raise ValueError(f"Unsupported bottleneck type: {bottleneck}")
        if bottleneck_type == "transformer":
            warnings.warn(
                "Bottleneck uses Classic UNet; Transformer/Mamba/Identity only applied to decoder skip modules (controlled by USE_TRANSFORMER_SKIPS)",
                RuntimeWarning,
            )

        self.input_fuser = nn.Identity()
        self.encoder = ResNetEncoder3L(input_channels, base_channels)
        bottleneck_c = base_channels * 4  # deepest channels for 3L

        self.bottleneck = ClassicUNetBottleneck3L(bottleneck_c)
        self.decoder = UNetDecoder3L(base_channels)

        # learned global affine calibration
        self.calib_a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.calib_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        stats_embed_dim = base_channels
        fusion_in_dim = bottleneck_c + stats_embed_dim
        self.stats_encoder = nn.Sequential(
            nn.Linear(4, stats_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(stats_embed_dim, stats_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, bottleneck_c),
            nn.ReLU(inplace=True),
        )

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

    def forward(self, x, signal_stats: Optional[torch.Tensor] = None):
        x = self.input_fuser(x)
        x, orig_len = _pad_to_multiple(x, 8)
        features = self.encoder(x)
        deep_features = features[-1]              # [B, 4*base, 128]
        pooled = torch.mean(deep_features, dim=2)  # [B, 4*base]

        if signal_stats is None:
            stats_encoded = torch.zeros(
                pooled.size(0),
                self.stats_encoder[0].out_features,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            stats_encoded = self.stats_encoder(signal_stats)

        fusion_parts = [pooled, stats_encoded]
        fused_context = torch.cat(fusion_parts, dim=1)
        fusion_bias = self.context_fusion(fused_context).unsqueeze(-1)
        bottleneck_out = self.bottleneck(deep_features) + fusion_bias

        out = self.decoder(features, bottleneck_out)
        out = self.calib_a * out + self.calib_b
        out = _crop_to_length(out, orig_len)

        sbp_pred = dbp_pred = None
        if USE_MULTITASK_SBP_DBP and self.sbp_head is not None and self.dbp_head is not None:
            pooled_bn = torch.mean(bottleneck_out, dim=2)
            sbp_pred = self.sbp_head(pooled_bn).squeeze(-1)
            dbp_pred = self.dbp_head(pooled_bn).squeeze(-1)
        return (out, sbp_pred, dbp_pred) if USE_MULTITASK_SBP_DBP else out

    def extract_bottleneck_embedding(self, x, signal_stats: Optional[torch.Tensor] = None):
        x = self.input_fuser(x)
        x, _ = _pad_to_multiple(x, 8)
        features = self.encoder(x)
        deep_features = features[-1]
        pooled = torch.mean(deep_features, dim=2)

        if signal_stats is None:
            stats_encoded = torch.zeros(
                pooled.size(0),
                self.stats_encoder[0].out_features,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            stats_encoded = self.stats_encoder(signal_stats)

        fusion_parts = [pooled, stats_encoded]
        fused_context = torch.cat(fusion_parts, dim=1)
        fusion_bias = self.context_fusion(fused_context).unsqueeze(-1)
        bottleneck_out = self.bottleneck(deep_features) + fusion_bias
        return _mean_std_pool(bottleneck_out, dim=2)

    def extract_decoder_embedding(self, x, signal_stats: Optional[torch.Tensor] = None):
        x = self.input_fuser(x)
        x, orig_len = _pad_to_multiple(x, 8)
        features = self.encoder(x)
        deep_features = features[-1]
        pooled = torch.mean(deep_features, dim=2)

        if signal_stats is None:
            stats_encoded = torch.zeros(
                pooled.size(0),
                self.stats_encoder[0].out_features,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        else:
            stats_encoded = self.stats_encoder(signal_stats)

        fusion_parts = [pooled, stats_encoded]
        fused_context = torch.cat(fusion_parts, dim=1)
        fusion_bias = self.context_fusion(fused_context).unsqueeze(-1)
        bottleneck_out = self.bottleneck(deep_features) + fusion_bias

        last_feat, _ = self.decoder.forward_with_last_feature(features, bottleneck_out)
        if orig_len is not None:
            last_feat = last_feat[..., :orig_len]
        return _mean_std_pool(last_feat, dim=2)


# ------------------------------------------------------------
# (12) Loss functions
# ------------------------------------------------------------
class WaveformLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, peak_weight=2.5, valley_weight=1.8, radius_samples=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Peak-aware weights
        self.peak_weight = peak_weight
        self.valley_weight = valley_weight
        self.radius = radius_samples if radius_samples is not None else max(1, int(0.04 * FS))
        # Use elementwise loss to support weighting
        self.huber = nn.SmoothL1Loss(reduction="none", beta=1.0)

    def forward(self, pred, target):
        # finite mask
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() == 0:
            return (pred.sum() * 0.0) + (target.sum() * 0.0)

        # Peak/valley detection on target via gradient sign change
        t = target
        B, L = t.shape
        # guard for very short sequences
        if L < 3:
            elem = self.huber(pred[mask], target[mask]).mean()
            # simple gradient term fallback
            pred_grad = pred[:, 1:] - pred[:, :-1]
            target_grad = target[:, 1:] - target[:, :-1]
            mask_g = torch.isfinite(pred_grad) & torch.isfinite(target_grad)
            grad = self.huber(pred_grad[mask_g], target_grad[mask_g]).mean() if mask_g.any() else elem.detach() * 0.0
            return self.alpha * elem + self.beta * grad

        g = t[:, 1:] - t[:, :-1]                    # [B, L-1]
        s = torch.sign(g)                           # [-1, 0, 1]
        s1, s2 = s[:, :-1], s[:, 1:]               # [B, L-2], [B, L-2]
        peaks_idx = (s1 > 0) & (s2 <= 0)           # + -> -
        valleys_idx = (s1 < 0) & (s2 >= 0)         # - -> +

        peaks = torch.zeros(B, L, device=t.device, dtype=t.dtype)
        valleys = torch.zeros_like(peaks)
        peaks[:, 1:-1] = peaks_idx.float()         # center at i+1
        valleys[:, 1:-1] = valleys_idx.float()

        # Dilate neighborhoods around extrema
        rad = int(self.radius)
        k = torch.ones(1, 1, 2 * rad + 1, device=t.device, dtype=t.dtype)
        pk_d = (F.conv1d(peaks.unsqueeze(1), k, padding=rad) > 0).squeeze(1)    # [B, L]
        vl_d = (F.conv1d(valleys.unsqueeze(1), k, padding=rad) > 0).squeeze(1)  # [B, L]

        # Build weights
        w = torch.ones(B, L, device=t.device, dtype=t.dtype)
        w = w + (self.peak_weight - 1.0) * pk_d.float()
        w = w + (self.valley_weight - 1.0) * vl_d.float()
        w = w * mask.float()
        w_sum = w.sum().clamp_min(1.0)

        # Pointwise Huber with weights
        huber_per = self.huber(pred, target)       # [B, L]
        huber_loss = (huber_per * w).sum() / w_sum

        # Gradient loss with averaged neighbor weights
        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        mask_g = torch.isfinite(pred_grad) & torch.isfinite(target_grad)
        if mask_g.any():
            wg = 0.5 * (w[:, 1:] + w[:, :-1]) * mask_g.float()
            wg_sum = wg.sum().clamp_min(1.0)
            grad_per = self.huber(pred_grad, target_grad)   # [B, L-1]
            grad_loss = (grad_per * wg).sum() / wg_sum
        else:
            grad_loss = huber_loss.detach() * 0.0

        return self.alpha * huber_loss + self.beta * grad_loss
