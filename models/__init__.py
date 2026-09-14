# Model selector / registry.
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Mapping, Tuple

# Fallback only; the switch is MODEL in train.py.
MODEL = "unet_cal_mamba"

# Fine-tune objectives for the per-master residual adapter.
FT_ABP_WAVE = "abp_wave"    # WaveformLoss on the ABP waveform
FT_BP_SCALAR = "bp_scalar"  # L1 on scalar SBP/DBP top-K/bottom-K readouts

# Run modes — which variants one run trains, in order.
RUN_CALIBRATED_ONLY = "calibrated_only"    # [calibrated]
RUN_NO_CAL_ONLY = "no_cal_only"            # [no_calibration]
RUN_NO_CAL_THEN_CAL = "no_cal_then_cal"    # [no_calibration, *extras, calibrated]
RUN_NO_CAL_ABLATION = "no_cal_ablation"    # [*extras]  (no calibrated variant)
RUN_MODES = (RUN_CALIBRATED_ONLY, RUN_NO_CAL_ONLY,
             RUN_NO_CAL_THEN_CAL, RUN_NO_CAL_ABLATION)

# Decoder-ridge probe alpha grid.
_RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)

# Architecture families, one module each. Keys that share a module differ only in module_config.
CAL = "unet_cal"                      # CalibrationUNet with a switchable skip refiner
NOCAL_SKIP = "unet_nocal_skip"        # no-cal U-Net, Mamba / Transformer / Conformer gates on the skips
NOCAL_DECODER = "unet_nocal_decoder"  # no-cal U-Net, Mamba / Transformer refiners inside the decoder


@dataclass(frozen=True)
# Spec
class ModelSpec:
    """One selectable architecture plus the train/test settings it came with."""
    key: str
    module: str
    summary: str

    # Module constants set on the model module before anything is built, e.g. SKIP_REFINER.
    module_config: Mapping[str, Any] = field(default_factory=dict)

    # Learning rate for decoder.ag* (skip gates / in-decoder refiners) in train.py;
    # None = the whole model trains at learning_rate.
    gate_learning_rate: float | None = None

    # per-master residual adapter (train.py)
    use_per_master_adapter: bool = False
    adapter_ft_objective: str | None = None
    adapter_finetune_weight_decay: float = 1e-3
    # Only read when adapter_ft_objective == FT_BP_SCALAR.
    adapter_alpha_l2: float = 0.0
    adapter_bp_readout_frac: float = 0.05
    adapter_bp_readout_k_min: int = 8
    adapter_sbp_weight: float = 1.0
    adapter_dbp_weight: float = 1.0

    # decoder-ridge probe (test.py); same for every model
    ridge_alpha_grid: Tuple[float, ...] = _RIDGE_GRID

    # extra no-cal ablation baselines beyond calibrated / no_calibration
    no_cal_baselines: Tuple[str, ...] = ()

    # which variants a run trains (see RUN_* above)
    run_mode: str = RUN_NO_CAL_THEN_CAL

    def load(self) -> ModuleType:
        return importlib.import_module(f"{__name__}.{self.module}")


# use_per_master_adapter marks the keys whose runs fine-tune the per-master adapter.
_SPECS = (
    ModelSpec(
        key="unet_cal_mamba",
        module=CAL,
        summary="Calibration U-Net, Mamba attention gate on skips.",
        module_config={"SKIP_REFINER": "mamba"},
        run_mode=RUN_CALIBRATED_ONLY,
    ),
    ModelSpec(
        key="unet_cal",
        module=CAL,
        summary="Calibration U-Net with plain skips: no attention gate, no Mamba / Transformer.",
        # Gate off: plain skips; SKIP_REFINER is then unused.
        module_config={"SKIP_REFINER": "conv", "USE_SKIP_ATTENTION_GATE": False},
        run_mode=RUN_CALIBRATED_ONLY,
    ),
    ModelSpec(
        key="unet_cal_transformer",
        module=CAL,
        summary="Calibration U-Net with Transformer-refined skips.",
        module_config={"SKIP_REFINER": "transformer"},
        run_mode=RUN_CALIBRATED_ONLY,
    ),
    ModelSpec(
        key="unet_cal_abp_ft",
        module=CAL,
        summary="Calibration U-Net + per-master adapter fine-tuned on the ABP waveform.",
        module_config={"SKIP_REFINER": "mamba"},
        run_mode=RUN_CALIBRATED_ONLY,
        use_per_master_adapter=True,
        adapter_ft_objective=FT_ABP_WAVE,
        adapter_finetune_weight_decay=1e-3,
    ),
    ModelSpec(
        key="unet_cal_bp_ft",
        module=CAL,
        summary="Same architecture as unet_cal_abp_ft; adapter fine-tuned on scalar SBP/DBP.",
        module_config={"SKIP_REFINER": "mamba"},
        run_mode=RUN_CALIBRATED_ONLY,
        use_per_master_adapter=True,
        adapter_ft_objective=FT_BP_SCALAR,
        adapter_finetune_weight_decay=5e-3,
        adapter_alpha_l2=1e-2,
    ),
    ModelSpec(
        key="unet_mamba_skip",
        module=NOCAL_SKIP,
        summary="Mamba skip-refiner U-Net; no-cal Mamba ablation.",
        module_config={"SKIP_REFINER_INIT_ALPHA_BIAS": -4.0},
        gate_learning_rate=6e-5,
        run_mode=RUN_NO_CAL_ABLATION,
        use_per_master_adapter=False,
        no_cal_baselines=("mamba",),
    ),
    ModelSpec(
        key="unet_mamba_decoder",
        module=NOCAL_DECODER,
        summary="Mamba decoder-refiner U-Net; no-cal Mamba ablation.",
        gate_learning_rate=6e-5,
        run_mode=RUN_NO_CAL_ABLATION,
        use_per_master_adapter=False,
        no_cal_baselines=("mamba",),
    ),
    ModelSpec(
        key="unet_transformer_skip",
        module=NOCAL_SKIP,
        summary="Transformer skip-refiner U-Net; no-cal Transformer ablation.",
        module_config={"SKIP_REFINER_INIT_ALPHA_BIAS": -3.0},
        gate_learning_rate=1e-4,
        run_mode=RUN_NO_CAL_ABLATION,
        use_per_master_adapter=False,
        no_cal_baselines=("transformer",),
    ),
    ModelSpec(
        key="unet_transformer_decoder",
        module=NOCAL_DECODER,
        summary="Transformer decoder-refiner U-Net; no-cal Transformer ablation.",
        gate_learning_rate=6e-5,
        run_mode=RUN_NO_CAL_ABLATION,
        use_per_master_adapter=False,
        no_cal_baselines=("transformer",),
    ),
    # The shared no-calibration baseline, trained once and reused.
    ModelSpec(
        key="unet_plain_nocal",
        module=CAL,
        summary="Plain U-Net, no calibration and no Mamba/Transformer refiner (baseline).",
        run_mode=RUN_NO_CAL_ONLY,
    ),
)

REGISTRY = {s.key: s for s in _SPECS}


# Selection and lazy loading
def _selected_key() -> str:
    if MODEL not in REGISTRY:
        raise KeyError(
            f"Unknown MODEL {MODEL!r} (set in train.py). "
            f"Available: {', '.join(available())}"
        )
    return MODEL


def available() -> list[str]:
    return [s.key for s in _SPECS]


def get_spec(key: str | None = None) -> ModelSpec:
    return REGISTRY[key] if key is not None else REGISTRY[_selected_key()]


def load(key: str | None = None) -> ModuleType:
    return get_spec(key).load() if key is not None else _resolved()[1]


def describe(key: str | None = None) -> str:
    s = get_spec(key)
    bits = [f"model={s.key}", f"module={s.module}", s.summary]
    bits += [f"{name}={value}" for name, value in s.module_config.items()]
    if s.gate_learning_rate is not None:
        bits.append(f"gate_lr={s.gate_learning_rate}")
    if s.use_per_master_adapter:
        bits.append(f"adapter_ft={s.adapter_ft_objective}")
    if s.no_cal_baselines:
        bits.append("no_cal_baselines=" + ",".join(s.no_cal_baselines))
    return " | ".join(bits)


# The model module is imported on first use, not at import time.
_RESOLVED: tuple[ModelSpec, ModuleType] | None = None


def select(key: str) -> ModelSpec:
    global MODEL, _RESOLVED
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model {key!r}. Available: {', '.join(available())}"
        )
    if _RESOLVED is not None and _RESOLVED[0].key != key:
        raise RuntimeError(
            f"models.select({key!r}) came too late: {_RESOLVED[0].key!r} was "
            f"already loaded. Call select() before using any model class."
        )
    MODEL = key
    return REGISTRY[key]


def _resolved() -> tuple[ModelSpec, ModuleType]:
    global _RESOLVED
    if _RESOLVED is None:
        spec = REGISTRY[_selected_key()]
        module = spec.load()
        for name, value in spec.module_config.items():
            setattr(module, name, value)
        _RESOLVED = (spec, module)
    return _RESOLVED


# train.py pushes FS / DROPOUT_P / BASE_CHANNELS / USE_MULTITASK_SBP_DBP here. The classes read
# them as module globals, so every loaded models.* module gets them. Signature defaults such as
# dropout=DROPOUT_P are bound at import and keep the module's own value.
def configure(**values: Any) -> None:
    _resolved()
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith(f"{__name__}."):
            for attr, value in values.items():
                setattr(module, attr, value)


def __getattr__(name: str) -> Any:
    spec, module = _resolved()
    if name == "SPEC":
        return spec
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"{spec.key!r} (models/{spec.module}.py) has no attribute {name!r}"
        ) from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_resolved()[1])))
