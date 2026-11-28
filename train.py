import os, json, shutil, re
import gzip
import sys
import logging
import random
import math
import subprocess
from datetime import datetime


if __name__ == "__main__":
    sys.modules["train"] = sys.modules[__name__]
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import models as model_lib

# unet_cal_mamba  unet_cal  unet_cal_transformer  unet_cal_abp_ft  unet_cal_bp_ft
# unet_mamba_skip  unet_mamba_decoder  unet_transformer_skip  unet_transformer_decoder
# unet_plain_nocal
MODEL = "unet_cal_mamba"

model_lib.select(MODEL)

# Model classes are read off model_lib where built (see _VARIANT_MODELS); only the loss is imported.
from models import WaveformLoss
from models import FT_ABP_WAVE, FT_BP_SCALAR

import preprocess
# Functions may be imported by name. The manifest and the ABP stats change during a run, so
# they are always read through the module: preprocess.manifest(), preprocess.GLOBAL_ABP_MIN.
from preprocess import (
    load_segment, normalize_ppg_ecg_segment, normalize_abp, denormalize_abp,
    bp_targets_from_wave, get_sid_from_path, get_record_window_index, get_master_record,
    attach_record_and_subject_indices, install_manifest, save_split_json,
    build_lists_from_manifest, compute_subject_bp_stats_from_manifest,
    stratified_subject_kfold, make_kfold_split, compute_global_abp_stats, save_abp_stats,
)


# Data sources and the processed-segment cache
data_paths = [
    "/media/chen/disk2/Seg3min_Time",
    "/media/chen/disk2/Seg3min2_Time",
]

processed_dir = "/home/chen/Desktop/processed_minmax_Time_V3"
os.makedirs(processed_dir, exist_ok=True)

# True deletes the NPZ cache and rebuilds it from CSV. Otherwise existing NPZ are always reused
# (only the manifest is refreshed) and an empty cache is built without deleting anything.
FORCE_REBUILD_PROCESSED = False
NPZ_COMPRESS = False
WORKERS = max(1, min(32, (os.cpu_count() or 8) - 1))   # CSV -> NPZ process pool

SUBJECT_FILE_MAP_PATHS = [
    "/media/chen/disk2/Seg3min_Time/subject_file_map.csv",
    "/media/chen/disk2/Seg3min2_Time/subject_file_map2.csv",
]
SUBJECT_FILE_MAP_OUTPUT_PATH = os.path.join(processed_dir, "subject_file_map_result.csv")

data_usage_ratio = 1.0
DATA_SAMPLE_RATIO = 1.0
DATA_SAMPLE_SEED = 42

# Subject-wise split: k-fold (default), or one 70/10/20 split when USE_KFOLD = False.
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

# Subject-wise folds, so no subject spans train and test.
USE_KFOLD = True
KFOLD_N_SPLITS = 5
KFOLD_VAL_RATIO = 0.125
KFOLD_SEED = 42
KFOLD_STRATIFY_KEY = "sbp"
KFOLD_STRATIFY_MAX_WINDOWS = 40

# 1: folds, train/val/test lists, calibration pairs and ABP stats come from the shared bundle.
# 0: recompute the folds from the manifest.
USE_KFOLD_BUNDLE = 1
KFOLD_BUNDLE_DIR = "../kfold_N100_shared_bundle"
KFOLD_NPZ_ROOT = processed_dir
KFOLD_BUNDLE_FOLDS = (0, 1, 2, 3, 4)
KFOLD_BUNDLE_NPZ_CHECK = 5

# Optuna LR search
USE_OPTUNA_LR_SEARCH = 0
OPTUNA_LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
OPTUNA_EPOCHS = 5
OPTUNA_FOLD = 0            # which fold the search runs on when OPTUNA_PER_FOLD = 0
OPTUNA_PER_FOLD = 1        # 1 -> every fold searches its own lr
OPTUNA_THEN_KFOLD = 1      # 1 -> continue into the full k-fold run
# Models with a gate learning rate search (lr, gate_lr) jointly with TPE instead of the lr grid.
OPTUNA_N_TRIALS = 10
OPTUNA_LR_RANGE = (1e-6, 1e-3)        # log-uniform
OPTUNA_GATE_LR_RANGE = (1e-6, 3e-3)   # log-uniform
OPTUNA_SEED = 42

# Segments and training loop
SEGMENT_LEN, STEP_SIZE = 1024, 1024
APPLY_FILTERING, FS = True, 125
batch_size, epochs, learning_rate = 256, 30, 2e-5
DATALOADER_WORKERS = min(8, (os.cpu_count() or 8))
PIN_MEMORY = True
PERSISTENT_WORKERS = True
USE_COMPILE = True
GLOBAL_SEED = 42
SEED_DETERMINISTIC = 0

# Model
DROPOUT_P = 0.2
BASE_CHANNELS = 128
USE_SIGNAL_STATS = True

USE_MULTITASK_SBP_DBP = True
MT_SBP_LOSS_WEIGHT = 0.1
MT_DBP_LOSS_WEIGHT = 0.1

USE_BOTTLENECK_RIDGE_PROBE = True
BOTTLENECK_RIDGE_USE_TRAIN_VAL = True

MODEL_SPEC = model_lib.get_spec()
MODEL_NAME = MODEL_SPEC.key

# The refiner families train decoder.ag* (skip gates / in-decoder refiners) at their own
# learning rate; None = everything at learning_rate.
GATE_LEARNING_RATE = MODEL_SPEC.gate_learning_rate

# Deep-skip gate of the skip-refiner family (unet_mamba_skip / unet_transformer_skip):
# 0 = Transformer, 1 = ConvTransformer.
USE_CONV_TRANSFORMER_GATE = 0

_MODULE_HAS_GATE_SWITCH = hasattr(model_lib, "USE_CONV_TRANSFORMER_GATE")
_GATE_KWARGS = ({"use_conv_transformer_gate": int(USE_CONV_TRANSFORMER_GATE)}
                if _MODULE_HAS_GATE_SWITCH else {})

# Sync model module configuration
model_lib.configure(FS=FS, DROPOUT_P=DROPOUT_P, BASE_CHANNELS=BASE_CHANNELS,
                    USE_MULTITASK_SBP_DBP=USE_MULTITASK_SBP_DBP)

# Per-master residual adapter: objective and weights come from the spec, the schedule is here.
USE_PER_MASTER_ADAPTER = bool(MODEL_SPEC.use_per_master_adapter)
ADAPTER_FT_OBJECTIVE = MODEL_SPEC.adapter_ft_objective
ADAPTER_FINETUNE_WEIGHT_DECAY = float(MODEL_SPEC.adapter_finetune_weight_decay)
ADAPTER_ALPHA_L2 = float(MODEL_SPEC.adapter_alpha_l2)
ADAPTER_BP_READOUT_FRAC = float(MODEL_SPEC.adapter_bp_readout_frac)
ADAPTER_BP_READOUT_K_MIN = int(MODEL_SPEC.adapter_bp_readout_k_min)
ADAPTER_SBP_WEIGHT = float(MODEL_SPEC.adapter_sbp_weight)
ADAPTER_DBP_WEIGHT = float(MODEL_SPEC.adapter_dbp_weight)

ADAPTER_BOTTLENECK_DIM = 8
ADAPTER_FINETUNE_EPOCHS = 10
ADAPTER_FINETUNE_LR = 3e-4
ADAPTER_FINETUNE_BATCH_SIZE = 16
ADAPTER_INIT_ALPHA = 1.0
ADAPTER_VAL_FRAC = 0.2
ADAPTER_EARLY_STOP_PATIENCE = 5
ADAPTERS_BASENAME = "per_master_adapters.pt"

# Only the adapter models expose ResidualFeatureAdapter1D, so import it lazily.
if USE_PER_MASTER_ADAPTER:
    ResidualFeatureAdapter1D = model_lib.ResidualFeatureAdapter1D

# Calibration windows
CALIBRATION_WINDOW_COUNT = 100

N_VALUES = (100,)

# True: pool_end = max(t_min, t_max - max_gap_windows + 1); False: legacy first-N (pool_end = N).
CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET = True

CALIBRATION_MAX_GAP_MINUTES = 1440.0

# Clocks: "real" or "cumulative"; late targets: "drop_late" or "shift_pool".
CALIBRATION_TIME_MODE = "real"
CALIBRATION_GAP_POLICY = "drop_late"
CALIBRATION_POOL_TIME_MODE = "real"

# Variants a run trains, and their checkpoint files
NO_CAL_BASELINES = tuple(MODEL_SPEC.no_cal_baselines)

# Empty = the selected model's own run mode.
RUN_MODE_OVERRIDE = ""
RUN_MODE = RUN_MODE_OVERRIDE or MODEL_SPEC.run_mode
if RUN_MODE not in model_lib.RUN_MODES:
    raise RuntimeError(
        f"Unknown RUN_MODE {RUN_MODE!r}. Valid: {', '.join(model_lib.RUN_MODES)}"
    )

VARIANT_CALIBRATED = "calibrated"
VARIANT_NO_CALIBRATION = "no_calibration"
VARIANT_NO_CAL_UNET_MAMBA = "no_cal_unet_mamba"           # Mamba d_state=8
VARIANT_NO_CAL_UNET_TRANSFORMER = "no_cal_unet_transformer"

CHECKPOINT_BASENAME_CAL = "best_calibration_unet.pth"
CHECKPOINT_BASENAME_NO_CAL = "best_plain_unet_nocal.pth"
CHECKPOINT_BASENAME_NO_CAL_MAMBA = "best_no_cal_unet_mamba.pth"
CHECKPOINT_BASENAME_NO_CAL_TRANSFORMER = "best_no_cal_unet_transformer.pth"

_NO_CAL_BASELINE_VARIANTS = {
    "mamba": VARIANT_NO_CAL_UNET_MAMBA,
    "transformer": VARIANT_NO_CAL_UNET_TRANSFORMER,
}

# Test-only: reuse a run dir's artifacts, no training or re-split.
TEST_ONLY_MODE = 0
TEST_ONLY_RUN_DIR = ""              # e.g. "runs/calibration_20260101-120000"
TEST_ONLY_VISUALIZE = True
TEST_ONLY_RUN_RIDGE = True
TEST_ONLY_PLOT_TRUEPRED_SCATTER = True

# Run directories and plots
RUN_DIR: str = ""
RUN_ROOT = "runs"
RUN_DIR_TS_FMT = "%Y%m%d-%H%M%S"

ENABLE_PLOT_BA_SCATTER = True
ENABLE_PLOT_TRUEPRED_SCATTER = True
PLOT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"

# Hand the data-format and path settings to the preprocessing module.
preprocess.configure(
    fs=FS, segment_len=SEGMENT_LEN, step_size=STEP_SIZE, apply_filtering=APPLY_FILTERING,
    processed_dir=processed_dir, npz_compress=NPZ_COMPRESS, data_paths=data_paths,
    subject_file_map_paths=SUBJECT_FILE_MAP_PATHS, workers=WORKERS,
    data_sample_ratio=DATA_SAMPLE_RATIO, data_sample_seed=DATA_SAMPLE_SEED,
)

# Runtime state filled in during a run. The manifest and the ABP stats live in preprocess.
train_list, val_list, test_list = [], [], []
# target_npz_path -> cal pair dict.
_WINDOW_CAL_PAIRS: dict = {}


# Logging and plot output
def setup_logging(prefix="train"):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{ts}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logging.info(f"Logging to file: {log_file}")


def _sanitize_tag(tag: str) -> str:
    try:
        s = str(tag).strip()
    except Exception:
        s = ""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s.strip("_")


def save_plot(prefix: str, ext: str = "png") -> str:
    out_dir = RUN_DIR if RUN_DIR else "plots"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    safe_prefix = _sanitize_tag(prefix) or "plot"
    ts = datetime.now().strftime(PLOT_TIMESTAMP_FMT)
    out_path = os.path.join(out_dir, f"{safe_prefix}_{ts}.{ext}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved plot: {out_path}")
    return out_path


# Seeding and DataLoader determinism
def set_global_seed(seed: int, deterministic: bool = False):
    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
    except Exception:
        pass
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass


def _seed_worker(worker_id: int):
    base = int(GLOBAL_SEED)
    worker_seed = (base + worker_id) % (2**32)
    try:
        random.seed(worker_seed)
    except Exception:
        pass
    try:
        np.random.seed(worker_seed)
    except Exception:
        pass
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass


# Same RNG / cuDNN / matmul state at the start of the run and of every variant.
def _reset_torch_state():
    set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))
    if not bool(SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


# Calibration pair selection — per-master static first-N (no leakage from target ABP).
CALIBRATION_MODE_TAG = "per_master_record_static_first_n"
# Pool [pool_end - N, pool_end) is fixed per master; one cal pair shared by all its targets.
CALIBRATION_MODE_TAG_JUST_BEFORE_TARGET = "per_master_fixed_n_just_before_first_target"


def _active_calibration_mode_tag() -> str:
    return (CALIBRATION_MODE_TAG_JUST_BEFORE_TARGET
            if CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET
            else CALIBRATION_MODE_TAG)


# Per-master calibration pair selection
def select_calibration_pairs(file_list, output_pairs_path: Optional[str] = None,
                             output_summary_path: Optional[str] = None):
    global _WINDOW_CAL_PAIRS
    if preprocess.manifest() is None:
        raise RuntimeError("Manifest not installed; cannot select calibration pairs.")

    df = preprocess.manifest().copy()
    needs_attach = (
        "record_window_index" not in df.columns
        or "master_record" not in df.columns
        or "subject_window_index" not in df.columns
    )
    if needs_attach:
        df = attach_record_and_subject_indices(df)

    N = int(CALIBRATION_WINDOW_COUNT)

    _WINDOW_CAL_PAIRS = {}
    summary_rows = []

    n_master_total = 0
    n_master_target_producing = 0
    n_master_skipped_too_short = 0
    n_master_pool_empty = 0
    n_target_windows_total = 0

    sids = sorted(df["sid"].astype(str).unique().tolist())

    just_before_target = bool(CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET)
    # Convert the cal-to-target gap budget from real minutes into a window count using FS.
    max_gap_minutes = float(CALIBRATION_MAX_GAP_MINUTES)
    max_gap_windows = (int(math.ceil(max_gap_minutes * 60.0 * FS / SEGMENT_LEN))
                       if just_before_target and max_gap_minutes > 0 else None)
    time_mode = str(CALIBRATION_TIME_MODE).lower()
    gap_policy = str(CALIBRATION_GAP_POLICY).lower()
    pool_time_mode = str(CALIBRATION_POOL_TIME_MODE).lower()
    if (time_mode not in ("real", "cumulative") or gap_policy not in ("shift_pool", "drop_late")
            or pool_time_mode not in ("real", "cumulative")):
        raise ValueError(f"Unknown CALIBRATION_TIME_MODE={CALIBRATION_TIME_MODE!r}, "
                         f"CALIBRATION_GAP_POLICY={CALIBRATION_GAP_POLICY!r} "
                         f"or CALIBRATION_POOL_TIME_MODE={CALIBRATION_POOL_TIME_MODE!r}")
    gap_limit = max_gap_minutes * 60.0 * FS if time_mode == "real" else max_gap_windows
    desc_tag = (f"per-master fixed-{N} just-before-target "
                f"(gap<={max_gap_minutes:g}min)"
                if just_before_target else f"per-master static first-{N}")

    # Track per-master pool placement (offset of pool start from rwi 0).
    n_master_with_shifted_pool = 0
    n_master_gap_triggered_shift = 0
    n_targets_dropped_by_gap = 0
    eligible_target_set = set(file_list) if file_list else None

    for sid in tqdm(sids, desc=f"Cal: {desc_tag}"):
        g_sid = df[df["sid"].astype(str) == sid]
        # Each (sid, master_record) processed independently.
        for master, grp in g_sid.groupby("master_record", sort=True):
            n_master_total += 1
            rows = (grp.sort_values("record_window_index")
                       .reset_index(drop=True))
            n_total = len(rows)

            raw = rows["raw_start_idx"].to_numpy(np.int64)
            if (pool_time_mode == "real" or (max_gap_windows is not None and time_mode == "real")) \
                    and (raw < 0).any():
                raise ValueError(f"Missing raw_start_idx (sid={sid}, master={master}); "
                                 f"use the 'cumulative' modes.")
            n_pool0 = (int(np.searchsorted(raw + SEGMENT_LEN, raw[0] + N * SEGMENT_LEN, side="right"))
                       if pool_time_mode == "real" else N)

            # Master too short → unusable: no calibration pair, no targets.
            if n_total <= n_pool0:
                n_master_skipped_too_short += 1
                continue

            clock = raw if time_mode == "real" else np.arange(n_total)

            # Decide pool start position per master; static first-N means pool_start = 0.
            pool_start = 0
            n_dropped_this_master = 0
            if just_before_target:
                eligible_indices: list = []
                for i in range(n_pool0, n_total):
                    target_path = str(rows.loc[i, "npz_path"])
                    if (eligible_target_set is None
                            or target_path in eligible_target_set):
                        eligible_indices.append(i)
                if not eligible_indices:
                    # No eligible targets for this master in file_list.
                    n_master_skipped_too_short += 1
                    continue
                t_min_idx = eligible_indices[0]
                t_max_idx = eligible_indices[-1]
                t_min = int(rows.loc[t_min_idx, "record_window_index"])
                t_max = int(rows.loc[t_max_idx, "record_window_index"])

                if max_gap_windows is None or gap_policy == "drop_late":
                    pool_end_rwi = t_min
                elif time_mode == "cumulative":
                    pool_end_rwi = max(t_min, t_max - max_gap_windows + 1)
                else:
                    fits = clock[t_max_idx] - clock[:t_max_idx + 1] <= gap_limit
                    pool_end_rwi = max(t_min, int(np.argmax(fits)) + 1)
                if pool_end_rwi > t_min:
                    n_master_gap_triggered_shift += 1
                    n_dropped_this_master = sum(
                        1 for i in eligible_indices
                        if int(rows.loc[i, "record_window_index"]) < pool_end_rwi
                    )
                    n_targets_dropped_by_gap += n_dropped_this_master
                if pool_end_rwi > t_max:
                    n_master_skipped_too_short += 1
                    continue

                pool_start = (int(np.searchsorted(raw, raw[pool_end_rwi - 1] + SEGMENT_LEN - N * SEGMENT_LEN))
                              if pool_time_mode == "real" else pool_end_rwi - N)
                # Safety: index lookup uses sorted-rows position, which equals rwi for the canonical master.
                if pool_start < 0:
                    raise RuntimeError(
                        f"Computed pool_start={pool_start} < 0 for "
                        f"(sid={sid}, master={master}); this indicates "
                        f"non-contiguous rwis or a bug in placement."
                    )

            pool_end = pool_end_rwi if just_before_target else n_pool0  # exclusive

            # Read SBP/DBP for the pool windows ONLY (cheap, no target-period reads).
            pool_sbp: list = []
            pool_dbp: list = []
            pool_paths: list = []
            pool_rwis: list = []
            pool_raws: list = []
            for i in range(pool_start, pool_end):
                try:
                    with np.load(rows.loc[i, "npz_path"], allow_pickle=False) as d:
                        y = d["Y"]
                    s, db = bp_targets_from_wave(y)
                except Exception:
                    continue
                if not (np.isfinite(s) and np.isfinite(db)):
                    continue
                pool_sbp.append(float(s))
                pool_dbp.append(float(db))
                pool_paths.append(str(rows.loc[i, "npz_path"]))
                pool_rwis.append(int(rows.loc[i, "record_window_index"]))
                pool_raws.append(int(rows.loc[i, "raw_start_idx"]))

            if not pool_sbp:
                n_master_pool_empty += 1
                logging.warning(
                    f"(sid={sid}, master={master}) calibration pool empty "
                    f"despite n_total={n_total}>N={N} "
                    f"(pool=[{pool_start},{pool_end})); skipping targets."
                )
                continue

            sbp_arr = np.asarray(pool_sbp, dtype=np.float64)
            low = int(np.argmin(sbp_arr))
            high = int(np.argmax(sbp_arr))

            cal_low_sbp = float(pool_sbp[low])
            cal_low_dbp = float(pool_dbp[low])
            cal_high_sbp = float(pool_sbp[high])
            cal_high_dbp = float(pool_dbp[high])
            cal_low_path = pool_paths[low]
            cal_high_path = pool_paths[high]
            cal_low_rwi = pool_rwis[low]
            cal_high_rwi = pool_rwis[high]
            cal_low_raw = pool_raws[low]
            cal_high_raw = pool_raws[high]

            if pool_start > 0:
                n_master_with_shifted_pool += 1

            # One fixed (cal_low, cal_high) pair per master, pinned to every target at rwi >= pool_end.
            n_targets = 0
            for i in range(pool_end, n_total):
                if (max_gap_windows is not None and gap_policy == "drop_late"
                        and clock[i] - clock[pool_end - 1] > gap_limit):
                    n_targets_dropped_by_gap += 1
                    continue
                target_path = str(rows.loc[i, "npz_path"])
                # Defensive: a target must never use itself as a cal source.
                if target_path == cal_low_path or target_path == cal_high_path:
                    raise RuntimeError(
                        f"Target window {target_path} would use itself as "
                        f"calibration (sid={sid}, master={master})."
                    )
                _WINDOW_CAL_PAIRS[target_path] = {
                    "sid": str(sid),
                    "master_record": str(master),
                    "cal_low_path": cal_low_path,
                    "cal_high_path": cal_high_path,
                    "cal_low_sbp": cal_low_sbp,
                    "cal_low_dbp": cal_low_dbp,
                    "cal_high_sbp": cal_high_sbp,
                    "cal_high_dbp": cal_high_dbp,
                    "cal_low_record_window_index": cal_low_rwi,
                    "cal_high_record_window_index": cal_high_rwi,
                    "cal_low_raw_start_idx": cal_low_raw,
                    "cal_high_raw_start_idx": cal_high_raw,
                    # Per-master fixed pool: both cal sources live in the same master as every target they serve.
                    "cal_low_master_record": str(master),
                    "cal_high_master_record": str(master),
                    "pool_start_rwi": int(pool_start),
                    "pool_end_rwi": int(pool_end),
                }
                n_targets += 1

            if n_targets > 0:
                n_master_target_producing += 1
                n_target_windows_total += n_targets

            summary_rows.append({
                "sid": str(sid),
                "master_record": str(master),
                "n_windows": int(n_total),
                "pool_start_rwi": int(pool_start),
                "pool_end_rwi": int(pool_end),
                "cal_low_record_window_index": int(cal_low_rwi),
                "cal_high_record_window_index": int(cal_high_rwi),
                "cal_low_sbp": cal_low_sbp,
                "cal_low_dbp": cal_low_dbp,
                "cal_high_sbp": cal_high_sbp,
                "cal_high_dbp": cal_high_dbp,
                "target_count": int(n_targets),
                "n_calibration_windows": int(N),
                "n_pool_windows": int(len(pool_sbp)),
                "cal_low_npz_path": cal_low_path,
                "cal_high_npz_path": cal_high_path,
                "fixed_cal_sbp_span": float(cal_high_sbp - cal_low_sbp),
            })

    valid_sids = set(p["sid"] for p in _WINDOW_CAL_PAIRS.values())
    if just_before_target:
        logging.info(
            f"Calibration (per-master fixed-{N} just-before-target, "
            f"gap<={CALIBRATION_MAX_GAP_MINUTES:g}min ≈ {max_gap_windows} windows): "
            f"{n_master_target_producing}/{n_master_total} master_records "
            f"produced targets (>{N} windows); "
            f"{n_master_skipped_too_short} skipped (too short or no eligible target); "
            f"{n_master_pool_empty} skipped (empty pool); "
            f"{n_master_with_shifted_pool} master(s) had pool_start > 0 "
            f"(pool moved later than first-N), "
            f"of which {n_master_gap_triggered_shift} were shifted by the "
            f"gap budget; {n_targets_dropped_by_gap} target window(s) "
            f"dropped because they fell inside / before the shifted pool. "
            f"Total target windows: {n_target_windows_total}; "
            f"{len(valid_sids)} subjects represented. "
            f"Each master uses ONE fixed N-window pool; no per-target "
            f"updates; no cross-master sharing; no target-period ABP is "
            f"ever read."
        )
    else:
        logging.info(
            f"Calibration (per-master static first-{N}): "
            f"{n_master_target_producing}/{n_master_total} master_records "
            f"produced targets (>{N} windows); "
            f"{n_master_skipped_too_short} skipped (n_windows<=N); "
            f"{n_master_pool_empty} skipped (empty pool). "
            f"Total target windows: {n_target_windows_total}; "
            f"{len(valid_sids)} subjects represented. "
            f"Each master uses ONLY its own first {N} windows; no cross-master "
            f"sharing; no target-period ABP is ever read."
        )

    if output_pairs_path:
        payload = {
            "window_count": int(N),
            "mode": _active_calibration_mode_tag(),
            "pairs_by_target_path": _WINDOW_CAL_PAIRS,
        }
        with open(output_pairs_path, "w") as f:
            json.dump(payload, f, indent=2)
        logging.info(f"Saved calibration pairs to: {output_pairs_path}")

    if output_summary_path and summary_rows:
        try:
            pd.DataFrame(summary_rows).to_csv(output_summary_path, index=False)
            logging.info(f"Saved calibration summary: {output_summary_path}")
        except Exception as e:
            logging.warning(f"Failed to save calibration summary: {e}")

    return _WINDOW_CAL_PAIRS, valid_sids


def _validate_calibration_invariance_per_master(window_cal_pairs):
    by_master: dict = {}
    for tgt, p in window_cal_pairs.items():
        key = (str(p["sid"]), str(p["master_record"]))
        by_master.setdefault(key, set()).add(
            (str(p["cal_low_path"]), str(p["cal_high_path"]))
        )
    bad = {k: v for k, v in by_master.items() if len(v) != 1}
    if bad:
        raise RuntimeError(
            f"Per-master calibration invariance violated for "
            f"{len(bad)} master(s); first offender: {next(iter(bad.items()))}"
        )
    logging.info(
        f"Per-master calibration pair invariance verified across "
        f"{len(by_master)} (sid, master_record) groups."
    )


def _validate_calibration_pairs_random(window_cal_pairs, n_window_count: int,
                                       sample: int = 10):
    items = list(window_cal_pairs.items())
    if not items:
        logging.info("Cal pair sanity check: no targets to inspect.")
        return
    rng = random.Random(GLOBAL_SEED)
    chosen = rng.sample(items, min(sample, len(items)))
    for tgt, p in chosen:
        cl_rwi = int(p["cal_low_record_window_index"])
        ch_rwi = int(p["cal_high_record_window_index"])
        target_rwi = get_record_window_index(tgt)
        pool_start = int(p.get("pool_start_rwi", 0))
        pool_end = int(p.get("pool_end_rwi", n_window_count))

        if target_rwi >= 0 and target_rwi < pool_end:
            raise RuntimeError(
                f"target rwi={target_rwi} < pool_end={pool_end} for {tgt}"
            )
        if not (pool_start <= cl_rwi < pool_end):
            raise RuntimeError(
                f"cal_low rwi={cl_rwi} outside pool "
                f"[{pool_start},{pool_end}) for {tgt}"
            )
        if not (pool_start <= ch_rwi < pool_end):
            raise RuntimeError(
                f"cal_high rwi={ch_rwi} outside pool "
                f"[{pool_start},{pool_end}) for {tgt}"
            )
        if tgt == p["cal_low_path"] or tgt == p["cal_high_path"]:
            raise RuntimeError(
                f"Target {tgt} reused as its own calibration source."
            )
        cl_master = str(p["cal_low_master_record"])
        ch_master = str(p["cal_high_master_record"])
        m = str(p["master_record"])
        if cl_master != m or ch_master != m:
            raise RuntimeError(
                f"Cross-master leakage: target master={m}, "
                f"cal_low_master={cl_master}, cal_high_master={ch_master} "
                f"for {tgt}"
            )
    logging.info(
        f"Cal pair sanity check passed on {len(chosen)} random targets "
        f"(N={n_window_count}, mode="
        f"{'just_before_target' if CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET else 'static_first_n'})."
    )


def load_calibration_pairs_json(path: str):
    global _WINDOW_CAL_PAIRS
    with open(path, "r") as f:
        d = json.load(f)
    pairs = d.get("pairs_by_target_path", None)
    if pairs is None:
        raise RuntimeError(
            f"{path} does not contain 'pairs_by_target_path'. "
            f"Re-run training to regenerate the calibration pair JSON."
        )
    mode = d.get("mode", "")
    if mode == CALIBRATION_MODE_TAG:
        pass  # static first-N canonical mode
    elif mode == CALIBRATION_MODE_TAG_JUST_BEFORE_TARGET:
        pass  # per-master fixed pool just-before-target (current default)
    elif mode == "per_master_record_sequential_update":
        logging.warning(
            f"Loaded {path} produced by the legacy "
            f"'per_master_record_sequential_update' mode. That mode "
            f"updated cal_low/cal_high from target-window ABP, which "
            f"leaks ground truth into calibration. Re-run training to "
            f"regenerate pairs under '{CALIBRATION_MODE_TAG}'."
        )
    elif mode == "per_master_record_fixed_initial_window":
        logging.info(
            f"Loaded {path} produced by the legacy "
            f"'per_master_record_fixed_initial_window' mode — "
            f"behaviourally identical to '{CALIBRATION_MODE_TAG}'."
        )
    elif mode == "per_subject_cumulative_prefix_fixed":
        logging.warning(
            f"Loaded {path} produced by the older "
            f"'per_subject_cumulative_prefix_fixed' mode, which pooled "
            f"calibration windows across earlier masters of the same "
            f"subject. The current static-calibration experiments "
            f"require per-master independence; re-run training to "
            f"regenerate pairs under '{CALIBRATION_MODE_TAG}'."
        )
    elif mode:
        logging.warning(
            f"Loaded {path} with unknown calibration mode '{mode}'."
        )
    _WINDOW_CAL_PAIRS = pairs
    valid_sids = set(p.get("sid") for p in pairs.values() if isinstance(p, dict))
    saved_window_count = d.get("window_count", None)
    logging.info(
        f"Loaded calibration pairs from {path} "
        f"({len(pairs)} target windows, {len(valid_sids)} subjects, "
        f"window_count={saved_window_count}, mode={mode or 'unknown'})"
    )
    if saved_window_count is not None and int(saved_window_count) != int(CALIBRATION_WINDOW_COUNT):
        logging.warning(
            f"Calibration window count mismatch: JSON saved with N="
            f"{saved_window_count} but current CALIBRATION_WINDOW_COUNT="
            f"{CALIBRATION_WINDOW_COUNT}. Test-only mode will reuse the "
            f"saved pairs as-is; ensure this matches the experiment you "
            f"intend to evaluate."
        )
    return pairs, valid_sids


def filter_target_lists(train_l, val_l, test_l, valid_sids):
    def _keep(fp):
        if fp not in _WINDOW_CAL_PAIRS:
            return False
        if (str(CALIBRATION_POOL_TIME_MODE).lower() == "cumulative"
                and get_record_window_index(fp) < CALIBRATION_WINDOW_COUNT):
            return False
        return True

    return ([f for f in train_l if _keep(f)],
            [f for f in val_l if _keep(f)],
            [f for f in test_l if _keep(f)])


# Datasets
class CalibrationBPDataset(Dataset):
    def __init__(self, file_list, window_cal_pairs, transform=None):
        self.file_list = file_list
        self.window_cal_pairs = window_cal_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        X, Y, _, _ = load_segment(filepath)
        X, signal_stats = normalize_ppg_ecg_segment(X)
        Y_normalized = normalize_abp(Y)

        if self.transform is not None:
            X, Y_normalized, signal_stats = self.transform(X, Y_normalized, signal_stats)

        if filepath in self.window_cal_pairs:
            p = self.window_cal_pairs[filepath]
            cal_low_path = p["cal_low_path"]
            cal_high_path = p["cal_high_path"]
            sbp1, dbp1 = float(p["cal_low_sbp"]), float(p["cal_low_dbp"])
            sbp2, dbp2 = float(p["cal_high_sbp"]), float(p["cal_high_dbp"])
        else:
            raise RuntimeError(
                f"Missing calibration pair for target window: {filepath}. "
                f"This should never happen after filter_target_lists()."
            )

        X_cal1, _, _, _ = load_segment(cal_low_path)
        X_cal1, _ = normalize_ppg_ecg_segment(X_cal1)
        X_cal2, _, _, _ = load_segment(cal_high_path)
        X_cal2, _ = normalize_ppg_ecg_segment(X_cal2)

        X = torch.FloatTensor(X).transpose(0, 1)
        Y_normalized = torch.FloatTensor(Y_normalized)
        signal_stats = torch.FloatTensor(signal_stats)
        X_cal1 = torch.FloatTensor(X_cal1).transpose(0, 1)
        X_cal2 = torch.FloatTensor(X_cal2).transpose(0, 1)
        cal_bp = torch.FloatTensor([sbp1, dbp1, sbp2, dbp2])
        return X, Y_normalized, signal_stats, X_cal1, X_cal2, cal_bp


class BloodPressureDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        X, Y, _, _ = load_segment(filepath)
        X, signal_stats = normalize_ppg_ecg_segment(X)
        Y_normalized = normalize_abp(Y)

        if self.transform is not None:
            X, Y_normalized, signal_stats = self.transform(X, Y_normalized, signal_stats)

        X = torch.FloatTensor(X).transpose(0, 1)
        Y_normalized = torch.FloatTensor(Y_normalized)
        signal_stats = torch.FloatTensor(signal_stats)
        return X, Y_normalized, signal_stats


# Training loop
def _sbp_dbp_from_batch(y_norm: torch.Tensor):
    y_np = denormalize_abp(y_norm.detach().cpu().numpy())
    sbp_list, dbp_list = [], []
    for row in y_np:
        sbp, dbp = bp_targets_from_wave(row)
        sbp_list.append(sbp)
        dbp_list.append(dbp)
    sbp_t = torch.tensor(sbp_list, dtype=torch.float32, device=y_norm.device)
    dbp_t = torch.tensor(dbp_list, dtype=torch.float32, device=y_norm.device)
    sbp_mask = torch.isfinite(sbp_t)
    dbp_mask = torch.isfinite(dbp_t)
    return sbp_t, dbp_t, sbp_mask, dbp_mask


def _unpack_model_output(out):
    def _unwrap_first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, dict):
            for key in ("wave", "pred", "out", "y", "wave_pred"):
                if key in x:
                    return _unwrap_first_tensor(x[key])
            for v in x.values():
                return _unwrap_first_tensor(v)
            return None
        if isinstance(x, (tuple, list)):
            if len(x) == 0:
                return None
            return _unwrap_first_tensor(x[0])
        return x

    if isinstance(out, (tuple, list)):
        wave = _unwrap_first_tensor(out[0]) if len(out) >= 1 else None
        sbp = _unwrap_first_tensor(out[1]) if len(out) >= 2 else None
        dbp = _unwrap_first_tensor(out[2]) if len(out) >= 3 else None
        return wave, sbp, dbp
    if isinstance(out, dict):
        return _unwrap_first_tensor(out), None, None
    return _unwrap_first_tensor(out), None, None


def _forward_one_batch(model, batch_data, device, *, use_calibration: bool):
    if use_calibration:
        inputs, targets, signal_stats, x_cal1, x_cal2, cal_bp = batch_data
        x_tensor = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        x_cal1 = x_cal1.to(device, non_blocking=True)
        x_cal2 = x_cal2.to(device, non_blocking=True)
        cal_bp_t = cal_bp.to(device, non_blocking=True)
        stats_tensor = (signal_stats.to(device, non_blocking=True)
                        if USE_SIGNAL_STATS else None)
        out_all = model(x_tensor, x_cal1, x_cal2, cal_bp_t, stats_tensor)
    else:
        inputs, targets, signal_stats = batch_data
        x_tensor = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        stats_tensor = (signal_stats.to(device, non_blocking=True)
                        if USE_SIGNAL_STATS else None)
        out_all = model(x_tensor, stats_tensor)
    wave_pred, sbp_pred, dbp_pred = _unpack_model_output(out_all)
    return wave_pred, sbp_pred, dbp_pred, targets


def _is_gate_param(name: str) -> bool:
    return any(f"decoder.ag{i}." in name for i in (0, 1, 2))


# AdamW over the model. With a gate rate (GATE_LEARNING_RATE unless overridden), decoder.ag*
# form a second param group at that rate and the rest trains at lr; otherwise one group at lr.
def _make_optimizer(model, lr: float, gate_lr: Optional[float] = None):
    gate_lr = GATE_LEARNING_RATE if gate_lr is None else gate_lr
    gate_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if gate_lr is not None and _is_gate_param(name):
            gate_params.append(p)
        else:
            backbone_params.append(p)
    if not gate_params:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, eps=1e-8)
    gate_lr = float(gate_lr)
    logging.info(
        f"Optimizer param groups: backbone lr={lr}, skip-gate lr={gate_lr} "
        f"({sum(p.numel() for p in gate_params)} gate params, "
        f"{sum(p.numel() for p in backbone_params)} backbone params)"
    )
    return optim.AdamW(
        [{"params": backbone_params, "lr": lr}, {"params": gate_params, "lr": gate_lr}],
        weight_decay=1e-3, eps=1e-8,
    )


def train_model(model, train_loader, val_loader, num_epochs, device,
                checkpoint_path: str, plot_curve: bool = True,
                *, use_calibration: bool = True,
                history_title: Optional[str] = None,
                lr: Optional[float] = None,
                gate_lr: Optional[float] = None,
                return_best_val: bool = False):
    criterion = WaveformLoss()
    lr_used = float(lr) if lr is not None else float(learning_rate)
    optimizer = _make_optimizer(model, lr_used, gate_lr)
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr_used * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_train_batches = 0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    wave_pred, sbp_pred, dbp_pred, targets = _forward_one_batch(
                        model, batch_data, device, use_calibration=use_calibration
                    )
                    loss = criterion(wave_pred, targets)
                    if USE_MULTITASK_SBP_DBP and sbp_pred is not None and dbp_pred is not None:
                        sbp_t, dbp_t, sbp_mask, dbp_mask = _sbp_dbp_from_batch(targets)
                        scalar_loss = 0.0
                        if sbp_mask.any():
                            scalar_loss = scalar_loss + MT_SBP_LOSS_WEIGHT * nn.functional.l1_loss(sbp_pred[sbp_mask], sbp_t[sbp_mask])
                        if dbp_mask.any():
                            scalar_loss = scalar_loss + MT_DBP_LOSS_WEIGHT * nn.functional.l1_loss(dbp_pred[dbp_mask], dbp_t[dbp_mask])
                        loss = loss + scalar_loss
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
                valid_train_batches += 1
            except Exception as e:
                logging.warning(f"Skipping abnormal batch: {e}")
                continue

        if valid_train_batches == 0:
            logging.error("No valid training batches this epoch; stopping early.")
            break

        model.eval()
        val_loss_sum = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    wave_pred, sbp_pred, dbp_pred, targets = _forward_one_batch(
                        model, batch_data, device, use_calibration=use_calibration
                    )
                    loss = criterion(wave_pred, targets)
                    if USE_MULTITASK_SBP_DBP and sbp_pred is not None and dbp_pred is not None:
                        sbp_t, dbp_t, sbp_mask, dbp_mask = _sbp_dbp_from_batch(targets)
                        scalar_loss = 0.0
                        if sbp_mask.any():
                            scalar_loss = scalar_loss + MT_SBP_LOSS_WEIGHT * nn.functional.l1_loss(sbp_pred[sbp_mask], sbp_t[sbp_mask])
                        if dbp_mask.any():
                            scalar_loss = scalar_loss + MT_DBP_LOSS_WEIGHT * nn.functional.l1_loss(dbp_pred[dbp_mask], dbp_t[dbp_mask])
                        loss = loss + scalar_loss
                if torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    valid_val_batches += 1

        train_loss = train_loss / max(1, valid_train_batches)
        val_loss = (val_loss_sum / max(1, valid_val_batches)) if valid_val_batches > 0 else float('inf')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        improved = (val_loss < best_val_loss) and np.isfinite(val_loss)
        if improved:
            best_val_loss = val_loss
            to_save = getattr(model, "_orig_mod", model)
            state = to_save.state_dict()
            best_state = deepcopy(state)
            try:
                torch.save(state, checkpoint_path)
                logging.info(f"New best model saved: {checkpoint_path} (val_loss={val_loss:.6f})")
            except Exception as e:
                logging.error(f"Failed to save best model: {e}")

    if plot_curve:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(history_title or 'Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        save_plot("training_history")

    try:
        if best_state is not None:
            target = getattr(model, "_orig_mod", model)
            target.load_state_dict(best_state)
    except Exception as e:
        logging.warning(f"Failed to restore best weights to model: {e}")
    if return_best_val:
        return model, best_val_loss
    return model


# Run logging and checkpoint loading
def log_selected_model():
    logging.info("=" * 72)
    logging.info(f"SELECTED: {MODEL_NAME}  (models/{MODEL_SPEC.module}.py)")
    if int(USE_KFOLD_BUNDLE) == 1:
        logging.info(f"  k-fold            : FROM SHARED BUNDLE"
                     f"  dir={KFOLD_BUNDLE_DIR}"
                     f"  folds={tuple(KFOLD_BUNDLE_FOLDS)}"
                     f"  npz_root={KFOLD_NPZ_ROOT}")
        logging.info( "                      (no re-split / no resampling / "
                      "no re-selection of calibration)")
    else:
        logging.info(f"  k-fold            : {'ON' if USE_KFOLD else 'OFF'}"
                     f"  n_splits={KFOLD_N_SPLITS} val_ratio={KFOLD_VAL_RATIO}"
                     f" stratify={KFOLD_STRATIFY_KEY} seed={KFOLD_SEED}")
    search = (f"{OPTUNA_N_TRIALS} TPE trials over (lr, gate_lr)" if GATE_LEARNING_RATE is not None
              else f"{len(OPTUNA_LR_GRID)} fixed lr")
    logging.info(f"  optuna LR search  : {'ON' if USE_OPTUNA_LR_SEARCH else 'OFF'}"
                 + (f"  {search} x {OPTUNA_EPOCHS} epochs"
                    + ("  per fold" if OPTUNA_PER_FOLD
                       else f"  fold={OPTUNA_FOLD}  then_kfold={bool(OPTUNA_THEN_KFOLD)}")
                    if USE_OPTUNA_LR_SEARCH else ""))
    logging.info("=" * 72)


def _safe_load(model, path, device):
    sd = torch.load(path, map_location=device)
    target = getattr(model, "_orig_mod", model)
    try:
        target.load_state_dict(sd, strict=True)
        return True
    except Exception:
        try:
            cleaned = {(k.replace("_orig_mod.", "") if isinstance(k, str) else k): v
                       for k, v in sd.items()}
            target.load_state_dict(cleaned, strict=False)
            logging.warning("Loaded weights with cleaned _orig_mod prefix, strict=False")
            return True
        except Exception as e:
            logging.error(f"Failed to load weights: {e}")
            return False


_ADAPTER_NO_KEY_SENTINEL = "__UNSET__"


# Per-master residual adapter fine-tune
def _path_to_master_key(fp):
    meta = preprocess.meta_for(fp)
    if not meta:
        return None
    return (str(meta["sid"]), str(meta["master_record"]))


def _decoder_supports_adapter(model) -> bool:
    base = getattr(model, "_orig_mod", model)
    dec = getattr(base, "decoder", None)
    return dec is not None and hasattr(dec, "feature_adapter")


def _install_adapter_state(model, state_dict, device):
    if not _decoder_supports_adapter(model):
        return
    base = getattr(model, "_orig_mod", model)
    adapter = base.decoder.feature_adapter
    if not isinstance(adapter, ResidualFeatureAdapter1D):
        adapter = ResidualFeatureAdapter1D(
            channels=BASE_CHANNELS,
            bottleneck=ADAPTER_BOTTLENECK_DIM,
            init_alpha=ADAPTER_INIT_ALPHA,
        ).to(device)
        base.decoder.feature_adapter = adapter
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()


def _uninstall_adapter(model):
    if not _decoder_supports_adapter(model):
        return
    base = getattr(model, "_orig_mod", model)
    base.decoder.feature_adapter = nn.Identity()


def _attach_adapter_cache(model, adapter_cache, path_lookup=None,
                          diag_by_key=None):
    if not _decoder_supports_adapter(model):
        return
    base = getattr(model, "_orig_mod", model)
    base._adapter_cache = dict(adapter_cache) if adapter_cache else {}
    base._adapter_path_lookup = dict(path_lookup) if path_lookup else None
    base._adapter_diag_by_key = dict(diag_by_key) if diag_by_key else {}
    base._current_master_key = _ADAPTER_NO_KEY_SENTINEL


def _maybe_swap_adapter(model, fp, device):
    base = getattr(model, "_orig_mod", model)
    cache = getattr(base, "_adapter_cache", None)
    if cache is None:
        return
    lookup = getattr(base, "_adapter_path_lookup", None)
    if lookup:
        key = lookup.get(fp)
        if key is None:
            key = _path_to_master_key(fp)
    else:
        key = _path_to_master_key(fp)
    current = getattr(base, "_current_master_key", _ADAPTER_NO_KEY_SENTINEL)
    if current == key:
        return
    if key is not None and key in cache:
        _install_adapter_state(base, cache[key], device)
    else:
        _uninstall_adapter(base)
    base._current_master_key = key


def _reset_adapter(model):
    if not _decoder_supports_adapter(model):
        return
    base = getattr(model, "_orig_mod", model)
    _uninstall_adapter(base)
    base._current_master_key = _ADAPTER_NO_KEY_SENTINEL


# Per-segment adapter audit columns written to test_segment_metrics.csv.
_ADAPTER_FINETUNE_LOSS_TAGS = {
    FT_BP_SCALAR: "scalar_sbp_dbp_l1",
    FT_ABP_WAVE: "abp_waveform_loss",
}


def _adapter_ft_is_bp_scalar() -> bool:
    return ADAPTER_FT_OBJECTIVE == FT_BP_SCALAR


def _adapter_finetune_loss_tag() -> str:
    return _ADAPTER_FINETUNE_LOSS_TAGS.get(ADAPTER_FT_OBJECTIVE, "")
_ADAPTER_RECORD_NULL_FIELDS = {
    "used_per_master_adapter": False,
    "adapter_master_key": "",
    "adapter_n_pool": -1,
    "adapter_n_train": -1,
    "adapter_n_val": -1,
    "adapter_epochs_run": -1,
    "adapter_init_val_loss": float("nan"),
    "adapter_best_val_loss": float("nan"),
    "adapter_val_loss_delta": float("nan"),
    "adapter_alpha_final": float("nan"),
    "adapter_finetune_loss": "",
}


def _adapter_record_fields(model, fp, *, use_calibration: bool):
    fields = dict(_ADAPTER_RECORD_NULL_FIELDS)
    if not use_calibration:
        return fields
    base = getattr(model, "_orig_mod", model)
    cache = getattr(base, "_adapter_cache", None)
    if not cache:
        return fields
    lookup = getattr(base, "_adapter_path_lookup", None)
    key = lookup.get(fp) if lookup else None
    if key is None:
        key = _path_to_master_key(fp)
    if key is None or key not in cache:
        return fields
    diag = getattr(base, "_adapter_diag_by_key", None) or {}
    info = diag.get(key, {})
    init_v = float(info.get("init_val_loss", float("nan")))
    best_v = float(info.get("best_val_loss", float("nan")))
    if math.isfinite(init_v) and math.isfinite(best_v):
        delta = init_v - best_v   # positive = adapter improved the pool-val loss
    else:
        delta = float("nan")
    fields.update({
        "used_per_master_adapter": True,
        "adapter_master_key": f"{key[0]}||{key[1]}",
        "adapter_n_pool": int(info.get("n_pool", -1)),
        "adapter_n_train": int(info.get("n_train", -1)),
        "adapter_n_val": int(info.get("n_val", -1)),
        "adapter_epochs_run": int(info.get("epochs_run", -1)),
        "adapter_init_val_loss": init_v,
        "adapter_best_val_loss": best_v,
        "adapter_val_loss_delta": delta,
        "adapter_alpha_final": float(info.get("alpha_final", float("nan"))),
        "adapter_finetune_loss": str(info.get("finetune_loss", _adapter_finetune_loss_tag())),
    })
    return fields


def _build_pool_paths_by_master(window_cal_pairs):
    if preprocess.manifest() is None:
        raise RuntimeError("Manifest not installed; cannot derive pool paths.")
    master_pool = {}
    for p in window_cal_pairs.values():
        key = (str(p["sid"]), str(p["master_record"]))
        ps = int(p.get("pool_start_rwi", 0))
        pe = int(p.get("pool_end_rwi", CALIBRATION_WINDOW_COUNT))
        master_pool[key] = (ps, pe)
    df = preprocess.manifest()
    out = {}
    for key, (ps, pe) in master_pool.items():
        sid, master = key
        mask = ((df["sid"].astype(str) == sid)
                & (df["master_record"].astype(str) == master)
                & (df["record_window_index"] >= ps)
                & (df["record_window_index"] < pe))
        rows = df.loc[mask].sort_values("record_window_index")
        out[key] = rows["npz_path"].tolist()
    return out


def _build_cal_context_by_master(window_cal_pairs):
    out = {}
    for p in window_cal_pairs.values():
        key = (str(p["sid"]), str(p["master_record"]))
        if key in out:
            continue
        out[key] = {
            "cal_low_path": p["cal_low_path"],
            "cal_high_path": p["cal_high_path"],
            "cal_low_sbp": float(p["cal_low_sbp"]),
            "cal_low_dbp": float(p["cal_low_dbp"]),
            "cal_high_sbp": float(p["cal_high_sbp"]),
            "cal_high_dbp": float(p["cal_high_dbp"]),
        }
    return out


def _load_master_pool_tensors(pool_paths):
    Xs, Ys, stats_list = [], [], []
    for fp in pool_paths:
        X, Y, _, _ = load_segment(fp)
        if X is None or Y is None:
            continue
        X_norm, stats = normalize_ppg_ecg_segment(X)
        Y_norm = normalize_abp(Y)
        Xs.append(torch.from_numpy(np.ascontiguousarray(X_norm.T)).float())
        Ys.append(torch.from_numpy(np.ascontiguousarray(Y_norm)).float())
        stats_list.append(torch.from_numpy(
            np.ascontiguousarray(stats)).float())
    if not Xs:
        return None
    return {
        "X": torch.stack(Xs, dim=0),
        "Y": torch.stack(Ys, dim=0),
        "stats": torch.stack(stats_list, dim=0),
    }


def _load_cal_pair_tensors(cal_ctx):
    X_c1, _, _, _ = load_segment(cal_ctx["cal_low_path"])
    X_c1, _ = normalize_ppg_ecg_segment(X_c1)
    X_c2, _, _, _ = load_segment(cal_ctx["cal_high_path"])
    X_c2, _ = normalize_ppg_ecg_segment(X_c2)
    x_cal1 = torch.from_numpy(np.ascontiguousarray(X_c1.T)).float().unsqueeze(0)
    x_cal2 = torch.from_numpy(np.ascontiguousarray(X_c2.T)).float().unsqueeze(0)
    cal_bp = torch.tensor(
        [cal_ctx["cal_low_sbp"], cal_ctx["cal_low_dbp"],
         cal_ctx["cal_high_sbp"], cal_ctx["cal_high_dbp"]],
        dtype=torch.float32,
    ).unsqueeze(0)
    return x_cal1, x_cal2, cal_bp


def _bp_scalars_from_pred_normalized(pred_norm: torch.Tensor):
    abp_min, abp_max = preprocess.GLOBAL_ABP_MIN, preprocess.GLOBAL_ABP_MAX
    if abp_min is None or abp_max is None:
        raise RuntimeError(
            "ABP normalization stats not initialized; cannot "
            "denormalize predicted waveform for adapter scalar fine-tune."
        )
    span = float(abp_max) - float(abp_min)
    pred_mmhg = pred_norm * span + float(abp_min)
    L = int(pred_mmhg.size(-1))
    K = max(int(ADAPTER_BP_READOUT_K_MIN),
            int(round(L * float(ADAPTER_BP_READOUT_FRAC))))
    K = min(K, max(1, L))
    top_vals, _ = torch.topk(pred_mmhg, k=K, dim=-1, largest=True, sorted=False)
    bot_vals, _ = torch.topk(pred_mmhg, k=K, dim=-1, largest=False, sorted=False)
    sbp_pred = top_vals.mean(dim=-1)
    dbp_pred = bot_vals.mean(dim=-1)
    return sbp_pred, dbp_pred


def _finetune_one_master(eval_model, pool_tensors, cal_ctx, device,
                         *, epochs, lr, weight_decay, batch_size,
                         val_frac, patience):
    base = getattr(eval_model, "_orig_mod", eval_model)

    adapter = ResidualFeatureAdapter1D(
        channels=BASE_CHANNELS,
        bottleneck=ADAPTER_BOTTLENECK_DIM,
        init_alpha=ADAPTER_INIT_ALPHA,
    ).to(device)
    base.decoder.feature_adapter = adapter

    try:
        base.eval()
        for p in base.parameters():
            p.requires_grad = False
        for p in adapter.parameters():
            p.requires_grad = True

        X_all = pool_tensors["X"].to(device, non_blocking=True)
        Y_all = pool_tensors["Y"].to(device, non_blocking=True)
        stats_all = pool_tensors["stats"].to(device, non_blocking=True)
        n_total = int(X_all.size(0))

        # Scalar SBP/DBP labels are only needed by the FT_BP_SCALAR objective.
        sbp_lab_t = dbp_lab_t = None
        if _adapter_ft_is_bp_scalar():
            y_np_denorm = denormalize_abp(Y_all.detach().cpu().numpy())
            sbp_lab = np.empty(n_total, dtype=np.float32)
            dbp_lab = np.empty(n_total, dtype=np.float32)
            for i in range(n_total):
                s, d = bp_targets_from_wave(y_np_denorm[i])
                sbp_lab[i] = s
                dbp_lab[i] = d
            sbp_lab_t = torch.from_numpy(sbp_lab).to(device)
            dbp_lab_t = torch.from_numpy(dbp_lab).to(device)
            valid_mask = torch.isfinite(sbp_lab_t) & torch.isfinite(dbp_lab_t)
            n_valid = int(valid_mask.sum().item())
            if n_valid < 2:
                logging.info(
                    f"Adapter fine-tune skipped: only {n_valid} pool window(s) "
                    f"have detectable peaks AND valleys (need >=2); leaving "
                    f"this master on the Identity adapter."
                )
                return None, None
            if n_valid < n_total:
                keep = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
                X_all = X_all.index_select(0, keep)
                Y_all = Y_all.index_select(0, keep)
                stats_all = stats_all.index_select(0, keep)
                sbp_lab_t = sbp_lab_t.index_select(0, keep)
                dbp_lab_t = dbp_lab_t.index_select(0, keep)
                n_total = int(X_all.size(0))

        x_cal1_cpu, x_cal2_cpu, cal_bp_cpu = _load_cal_pair_tensors(cal_ctx)
        x_cal1 = x_cal1_cpu.to(device, non_blocking=True)
        x_cal2 = x_cal2_cpu.to(device, non_blocking=True)
        cal_bp = cal_bp_cpu.to(device, non_blocking=True)

        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(int(GLOBAL_SEED))
        perm = torch.randperm(n_total, generator=cpu_gen).tolist()
        n_val = (max(1, int(round(n_total * float(val_frac))))
                 if n_total > 1 else 0)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if not tr_idx:
            tr_idx = list(range(n_total))
            if not val_idx:
                val_idx = list(range(n_total))

        bp_scalar = _adapter_ft_is_bp_scalar()

        def _sel(t, idx_t):
            return None if t is None else t.index_select(0, idx_t)

        def _gather(idx_list):
            idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
            return (X_all.index_select(0, idx_t),
                    _sel(Y_all, idx_t),
                    stats_all.index_select(0, idx_t),
                    _sel(sbp_lab_t, idx_t),
                    _sel(dbp_lab_t, idx_t))

        Xt, Yt, St, Sbp_t_tr, Dbp_t_tr = _gather(tr_idx)
        if val_idx:
            Xv, Yv, Sv, Sbp_t_va, Dbp_t_va = _gather(val_idx)
        else:
            Xv = Yv = Sv = Sbp_t_va = Dbp_t_va = None

        sbp_w = float(ADAPTER_SBP_WEIGHT)
        dbp_w = float(ADAPTER_DBP_WEIGHT)
        # The alpha anchor is part of the FT_BP_SCALAR recipe.
        alpha_l2 = float(ADAPTER_ALPHA_L2) if bp_scalar else 0.0
        criterion = WaveformLoss() if not bp_scalar else None
        optimizer = optim.AdamW(adapter.parameters(), lr=float(lr),
                                weight_decay=float(weight_decay))

        def _expand_cal(B):
            return (x_cal1.expand(B, -1, -1).contiguous(),
                    x_cal2.expand(B, -1, -1).contiguous(),
                    cal_bp.expand(B, -1).contiguous())

        def _scalar_bp_loss(wave_pred, sbp_target, dbp_target):
            sbp_p, dbp_p = _bp_scalars_from_pred_normalized(wave_pred)
            l_s = nn.functional.l1_loss(sbp_p, sbp_target)
            l_d = nn.functional.l1_loss(dbp_p, dbp_target)
            # Reported val loss is the unweighted average of SBP/DBP L1 in mmHg.
            return (sbp_w * l_s + dbp_w * l_d), 0.5 * (l_s + l_d)

        # (optimized loss, reported loss) for the active objective.
        def _ft_loss(wave_pred, yb, sbp_b, dbp_b):
            if bp_scalar:
                return _scalar_bp_loss(wave_pred, sbp_b, dbp_b)
            loss = criterion(wave_pred, yb)
            return loss, loss

        @torch.no_grad()
        def _val_loss():
            adapter.eval()
            if Xv is None or Xv.size(0) == 0:
                return float("inf")
            n = int(Xv.size(0))
            c1, c2, cb = _expand_cal(n)
            stats_in = Sv if USE_SIGNAL_STATS else None
            out_all = base(Xv, c1, c2, cb, stats_in)
            wave_pred, _, _ = _unpack_model_output(out_all)
            _, report = _ft_loss(wave_pred, Yv, Sbp_t_va, Dbp_t_va)
            return float(report.item()) if torch.isfinite(report) else float("inf")

        n_tr = int(Xt.size(0))
        bs = max(1, int(batch_size))
        best_val = _val_loss()
        best_state = {k: v.detach().cpu().clone()
                      for k, v in adapter.state_dict().items()}
        no_improve = 0
        diag = {
            "n_pool": int(n_total),
            "n_train": int(n_tr),
            "n_val": int(0 if Xv is None else Xv.size(0)),
            "init_val_loss": best_val,
            "best_val_loss": best_val,
            "epochs_run": 0,
            "alpha_final": float(adapter.alpha.detach().item()),
            "finetune_loss": _adapter_finetune_loss_tag(),
        }

        for epoch in range(int(epochs)):
            adapter.train()
            order = torch.randperm(n_tr, generator=cpu_gen).tolist()
            for s in range(0, n_tr, bs):
                sel = torch.tensor(order[s:s + bs],
                                   dtype=torch.long, device=device)
                xb = Xt.index_select(0, sel)
                sb = St.index_select(0, sel) if USE_SIGNAL_STATS else None
                yb = _sel(Yt, sel)
                sbp_b = _sel(Sbp_t_tr, sel)
                dbp_b = _sel(Dbp_t_tr, sel)
                B = int(xb.size(0))
                c1, c2, cb = _expand_cal(B)
                optimizer.zero_grad(set_to_none=True)
                out_all = base(xb, c1, c2, cb, sb)
                wave_pred, _, _ = _unpack_model_output(out_all)
                loss, _ = _ft_loss(wave_pred, yb, sbp_b, dbp_b)
                # Anchor the residual gate towards the identity init so the adapter only moves when scalar BP.
                if alpha_l2 > 0.0:
                    loss = loss + alpha_l2 * (adapter.alpha - float(ADAPTER_INIT_ALPHA)) ** 2
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),
                                               max_norm=1.0)
                optimizer.step()
            cur_val = _val_loss()
            diag["epochs_run"] = epoch + 1
            if cur_val + 1e-8 < best_val:
                best_val = cur_val
                best_state = {k: v.detach().cpu().clone()
                              for k, v in adapter.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(patience):
                    break

        diag["best_val_loss"] = best_val
        # Snapshot alpha from the BEST-by-val state so the diag row matches what test inference will.
        alpha_t = best_state.get("alpha", None)
        if isinstance(alpha_t, torch.Tensor):
            diag["alpha_final"] = float(alpha_t.detach().item())
        return best_state, diag
    finally:
        # Critical: ALWAYS detach the per-master adapter from the shared eval_model so a failed master.
        base.decoder.feature_adapter = nn.Identity()


def finetune_per_master_adapters(eval_model, window_cal_pairs, device,
                                 *, used_masters=None):
    if not window_cal_pairs:
        return {}, []
    pool_by_master = _build_pool_paths_by_master(window_cal_pairs)
    ctx_by_master = _build_cal_context_by_master(window_cal_pairs)

    keys = sorted(pool_by_master.keys())
    if used_masters is not None:
        keys = [k for k in keys if k in used_masters]

    base = getattr(eval_model, "_orig_mod", eval_model)
    saved_training = base.training
    base.eval()

    adapters = {}
    diag_rows = []
    for key in tqdm(keys, desc="Adapter fine-tune (per master)"):
        pool_paths = pool_by_master.get(key, [])
        cal_ctx = ctx_by_master.get(key)
        if not pool_paths or cal_ctx is None:
            continue
        pool_tensors = _load_master_pool_tensors(pool_paths)
        if pool_tensors is None or pool_tensors["X"].size(0) < 2:
            continue
        try:
            state, diag = _finetune_one_master(
                eval_model, pool_tensors, cal_ctx, device,
                epochs=ADAPTER_FINETUNE_EPOCHS,
                lr=ADAPTER_FINETUNE_LR,
                weight_decay=ADAPTER_FINETUNE_WEIGHT_DECAY,
                batch_size=ADAPTER_FINETUNE_BATCH_SIZE,
                val_frac=ADAPTER_VAL_FRAC,
                patience=ADAPTER_EARLY_STOP_PATIENCE,
            )
        except Exception as e:
            logging.warning(
                f"Adapter fine-tune failed for (sid={key[0]}, master={key[1]}): {e}"
            )
            continue
        # _finetune_one_master returns (None, None) when the pool has fewer than 2 windows.
        if state is None:
            continue
        adapters[key] = state
        diag.update({"sid": key[0], "master_record": key[1]})
        diag_rows.append(diag)

    _reset_adapter(eval_model)
    # Re-enable grad on the base model so any downstream re-training paths work.
    for p in base.parameters():
        p.requires_grad = True
    if saved_training:
        base.train()
    return adapters, diag_rows


def _diag_rows_to_by_key(diag_rows):
    out = {}
    for r in diag_rows or []:
        sid = r.get("sid")
        master = r.get("master_record")
        if sid is None or master is None:
            continue
        clean = {k: v for k, v in r.items()
                 if k not in ("sid", "master_record")}
        out[(str(sid), str(master))] = clean
    return out


def save_per_master_adapters(adapters, diag_rows, run_dir):
    if not adapters:
        return ""
    out_path = os.path.join(run_dir, ADAPTERS_BASENAME)
    diag_by_key = _diag_rows_to_by_key(diag_rows)
    payload = {
        # Fine-tune supervision is scalar SBP/DBP L1, not the waveform loss.
        "schema": "per_master_residual_adapter_v3",
        "finetune_loss": "scalar_sbp_dbp_l1",
        "base_channels": int(BASE_CHANNELS),
        "bottleneck": int(ADAPTER_BOTTLENECK_DIM),
        "init_alpha": float(ADAPTER_INIT_ALPHA),
        "calibration_window_count": int(CALIBRATION_WINDOW_COUNT),
        "adapters": {f"{sid}||{master}": state
                     for (sid, master), state in adapters.items()},
        # Persist diagnostics inside the same artifact so test-only mode can stamp the same adapter audit.
        "diag": {f"{sid}||{master}": dict(row)
                 for (sid, master), row in diag_by_key.items()},
    }
    torch.save(payload, out_path)
    if diag_rows:
        try:
            diag_csv = os.path.join(run_dir, "per_master_adapter_diag.csv")
            pd.DataFrame(diag_rows).to_csv(diag_csv, index=False)
        except Exception as e:
            logging.warning(f"Failed to save adapter diagnostics: {e}")
    logging.info(
        f"Saved per-master adapters ({len(adapters)} masters): {out_path}"
    )
    return out_path


def load_per_master_adapters(path):
    if not path or not os.path.isfile(path):
        return {}, {}
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return {}, {}
    raw_adapters = payload.get("adapters", {})
    adapters = {}
    for k, v in raw_adapters.items():
        if "||" not in k:
            continue
        sid, master = k.split("||", 1)
        adapters[(sid, master)] = v
    diag = {}
    for k, row in (payload.get("diag", {}) or {}).items():
        if "||" not in k:
            continue
        sid, master = k.split("||", 1)
        diag[(sid, master)] = dict(row)
    logging.info(
        f"Loaded per-master adapters from {path}: "
        f"{len(adapters)} masters, {len(diag)} diag rows"
    )
    return adapters, diag


# Run-dir manifest persistence
def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return None


def save_data_lists(path, train_list, val_list, test_list):
    with open(path, "w") as f:
        json.dump({
            "train_list": list(train_list),
            "val_list": list(val_list),
            "test_list": list(test_list),
        }, f, indent=2)
    logging.info(f"Saved data lists to: {path}")


def load_data_lists(path):
    with open(path, "r") as f:
        d = json.load(f)
    return list(d["train_list"]), list(d["val_list"]), list(d["test_list"])


def save_run_manifest(path, *, run_dir, checkpoint_path, calibration_pairs_path,
                      processed_manifest_path, data_lists_path, abp_stats_path,
                      lr_sbp_path, lr_dbp_path,
                      n_train, n_val, n_test,
                      n_subj_train, n_subj_val, n_subj_test,
                      n_master_train=0, n_master_val=0, n_master_test=0,
                      calibration_summary_path="",
                      variant: str = VARIANT_CALIBRATED,
                      use_calibration: bool = True,
                      model_name: str = "CalibrationUNet",
                      parent_run_dir: str = "",
                      per_master_adapters_path: str = ""):
    manifest = {
        "variant": str(variant),
        "USE_CALIBRATION": bool(use_calibration),
        "model_name": str(model_name),
        "parent_run_dir": parent_run_dir,
        "GLOBAL_SEED": int(GLOBAL_SEED),
        "SEED_DETERMINISTIC": int(SEED_DETERMINISTIC),
        "FS": int(FS),
        "SEGMENT_LEN": int(SEGMENT_LEN),
        "STEP_SIZE": int(STEP_SIZE),
        "APPLY_FILTERING": bool(APPLY_FILTERING),
        "USE_SIGNAL_STATS": bool(USE_SIGNAL_STATS),
        "GLOBAL_ABP_MIN": preprocess.GLOBAL_ABP_MIN,
        "GLOBAL_ABP_MAX": preprocess.GLOBAL_ABP_MAX,
        "BASE_CHANNELS": int(BASE_CHANNELS),
        "DROPOUT_P": float(DROPOUT_P),
        # Which models/*.py this run used, and the model-owned settings that came.
        "MODEL": MODEL_NAME,
        "MODEL_MODULE": f"models/{MODEL_SPEC.module}.py",
        "USE_PER_MASTER_ADAPTER": bool(USE_PER_MASTER_ADAPTER),
        "ADAPTER_FT_OBJECTIVE": ADAPTER_FT_OBJECTIVE or "",
        "NO_CAL_BASELINES": list(NO_CAL_BASELINES),
        "USE_CONV_TRANSFORMER_GATE": (int(USE_CONV_TRANSFORMER_GATE)
                                      if _MODULE_HAS_GATE_SWITCH else None),
        "per_master_adapters_path": str(per_master_adapters_path or ""),
        "USE_MULTITASK_SBP_DBP": bool(USE_MULTITASK_SBP_DBP),
        "MT_SBP_LOSS_WEIGHT": float(MT_SBP_LOSS_WEIGHT),
        "MT_DBP_LOSS_WEIGHT": float(MT_DBP_LOSS_WEIGHT),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "GATE_LEARNING_RATE": GATE_LEARNING_RATE,
        "CALIBRATION_WINDOW_COUNT": int(CALIBRATION_WINDOW_COUNT),
        "USE_BOTTLENECK_RIDGE_PROBE": bool(USE_BOTTLENECK_RIDGE_PROBE),
        "BOTTLENECK_RIDGE_USE_TRAIN_VAL": bool(BOTTLENECK_RIDGE_USE_TRAIN_VAL),
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "calibration_pairs_path": calibration_pairs_path,
        "calibration_summary_path": calibration_summary_path,
        "calibration_mode": (_active_calibration_mode_tag()
                             if use_calibration else "none"),
        "CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET": bool(CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET),
        "CALIBRATION_MAX_GAP_MINUTES": float(CALIBRATION_MAX_GAP_MINUTES),
        "CALIBRATION_TIME_MODE": str(CALIBRATION_TIME_MODE),
        "CALIBRATION_GAP_POLICY": str(CALIBRATION_GAP_POLICY),
        "CALIBRATION_POOL_TIME_MODE": str(CALIBRATION_POOL_TIME_MODE),
        "processed_manifest_path": processed_manifest_path,
        "data_lists_path": data_lists_path,
        "abp_normalization_stats_path": abp_stats_path,
        "linear_regression_sbp_path": lr_sbp_path,
        "linear_regression_dbp_path": lr_dbp_path,
        "train_segment_count": int(n_train),
        "val_segment_count": int(n_val),
        "test_segment_count": int(n_test),
        "train_subject_count": int(n_subj_train),
        "val_subject_count": int(n_subj_val),
        "test_subject_count": int(n_subj_test),
        "train_master_record_count": int(n_master_train),
        "val_master_record_count": int(n_master_val),
        "test_master_record_count": int(n_master_test),
        "git_commit_hash": _git_commit_hash(),
        "data_paths": list(data_paths),
        "subject_file_map_paths": list(SUBJECT_FILE_MAP_PATHS),
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Saved run manifest: {path}")


# Model builders
def _make_dataloader_kwargs():
    gen = torch.Generator()
    gen.manual_seed(int(GLOBAL_SEED))
    dl_kwargs = dict(
        batch_size=batch_size,
        pin_memory=PIN_MEMORY,
        generator=gen,
        worker_init_fn=_seed_worker,
    )
    try:
        import multiprocessing as _mp
        all_methods = _mp.get_all_start_methods()
        use_fork = "fork" in all_methods
        if use_fork:
            try:
                torch.multiprocessing.set_start_method("fork", force=True)
            except RuntimeError:
                pass
        if DATALOADER_WORKERS > 0:
            if use_fork:
                ctx = _mp.get_context("fork")
                dl_kwargs.update(dict(
                    num_workers=DATALOADER_WORKERS,
                    persistent_workers=PERSISTENT_WORKERS,
                    prefetch_factor=2,
                    multiprocessing_context=ctx,
                ))
            else:
                logging.warning("Cannot use 'fork'; DataLoader set to num_workers=0")
                dl_kwargs.update(dict(num_workers=0))
    except Exception as e:
        logging.warning(f"DataLoader multiprocessing fallback to single process: {e}")
        dl_kwargs.update(dict(num_workers=0))
    return dl_kwargs


def _set_bn_momentum(m, momentum=0.01):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.momentum = momentum


# variant -> model name, class on the selected models/*.py, checkpoint file, extra ctor kwargs.
_VARIANT_MODELS = {
    VARIANT_CALIBRATED: {
        "model_name": "CalibrationUNet",
        "class_name": "CalibrationUNet",
        "checkpoint_basename": CHECKPOINT_BASENAME_CAL,
        "extra_kwargs": {},
    },
    VARIANT_NO_CALIBRATION: {
        "model_name": "PlainUNet3LNoCal",
        "class_name": "PlainUNet3LNoCal",
        "checkpoint_basename": CHECKPOINT_BASENAME_NO_CAL,
        "extra_kwargs": {},
    },
    VARIANT_NO_CAL_UNET_MAMBA: {
        "model_name": "MambaUNet3LNoCal",
        "class_name": "MambaUNet3LNoCal",
        "checkpoint_basename": CHECKPOINT_BASENAME_NO_CAL_MAMBA,
        "extra_kwargs": {"ag_d_state": 8},
    },
    VARIANT_NO_CAL_UNET_TRANSFORMER: {
        "model_name": "TransformerUNet3LNoCal",
        "class_name": "TransformerUNet3LNoCal",
        "checkpoint_basename": CHECKPOINT_BASENAME_NO_CAL_TRANSFORMER,
        # Recorded so a test-only re-run rebuilds the same gate the checkpoint was trained with.
        "extra_kwargs": dict(_GATE_KWARGS),
    },
}


def _variant_for_model_name(model_name, *, use_calibration):
    """Variant behind a run manifest's model_name; unknown no-cal names mean the plain baseline."""
    if use_calibration:
        return VARIANT_CALIBRATED
    for variant, spec in _VARIANT_MODELS.items():
        if variant != VARIANT_CALIBRATED and spec["model_name"] == model_name:
            return variant
    return VARIANT_NO_CALIBRATION


# Fresh, uncompiled model on device: the eval model, and the training model before compile.
def _new_model(variant, device):
    spec = _VARIANT_MODELS[variant]
    cls = getattr(model_lib, spec["class_name"])
    if variant == VARIANT_CALIBRATED:
        m = cls(input_channels=3, base_channels=BASE_CHANNELS)
    else:
        m = cls(input_channels=3, base_channels=BASE_CHANNELS,
                use_signal_stats=bool(USE_SIGNAL_STATS), **spec["extra_kwargs"])
    return m.to(device)


def _build_train_model(variant, device):
    m = _new_model(variant, device)
    _set_bn_momentum(m, momentum=0.01)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            m = torch.compile(m)
            logging.info(f"Enabled torch.compile acceleration ({_VARIANT_MODELS[variant]['model_name']})")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")
    return m


def _make_loaders(train_list, val_list, window_cal_pairs, *, use_calibration):
    if use_calibration:
        train_dataset = CalibrationBPDataset(train_list, window_cal_pairs)
        val_dataset = CalibrationBPDataset(val_list, window_cal_pairs)
    else:
        train_dataset = BloodPressureDataset(train_list)
        val_dataset = BloodPressureDataset(val_list)
    dl_kwargs = _make_dataloader_kwargs()
    return (DataLoader(train_dataset, shuffle=True, **dl_kwargs),
            DataLoader(val_dataset, shuffle=False, **dl_kwargs))


# Variant / N / fold orchestration
def _count_subjects(flist):
    sids = set()
    for fp in flist:
        sid = get_sid_from_path(fp)
        if sid is not None:
            sids.add(sid)
    return len(sids)


def _count_master_records(flist):
    masters = set()
    for fp in flist:
        sid = get_sid_from_path(fp)
        master = get_master_record(fp)
        if sid is not None and master:
            masters.add((sid, master))
    return len(masters)


def _resolve_compare_modes():
    # Extra no-calibration ablation baselines contributed by the selected model's spec.
    extra_no_cal = [_NO_CAL_BASELINE_VARIANTS[b]
                    for b in NO_CAL_BASELINES
                    if b in _NO_CAL_BASELINE_VARIANTS]

    if RUN_MODE == model_lib.RUN_CALIBRATED_ONLY:
        return [VARIANT_CALIBRATED]
    if RUN_MODE == model_lib.RUN_NO_CAL_ONLY:
        return [VARIANT_NO_CALIBRATION]
    if RUN_MODE == model_lib.RUN_NO_CAL_ABLATION:
        if not extra_no_cal:
            raise RuntimeError(
                f"RUN_MODE={RUN_MODE!r} needs at least one entry in the "
                f"model's no_cal_baselines; {MODEL_NAME!r} has none. Use "
                f"RUN_MODE_OVERRIDE to pick another mode."
            )
        return list(extra_no_cal)
    return [VARIANT_NO_CALIBRATION] + extra_no_cal + [VARIANT_CALIBRATED]


def _maybe_attach_calibration_columns(records, *, use_calibration: bool):
    if use_calibration:
        return records
    drop_keys = {
        "cal_low_record_window_index", "cal_high_record_window_index",
        "cal_low_raw_start_idx", "cal_high_raw_start_idx",
        "cal_low_master_record", "cal_high_master_record",
        "cal_low_is_cross_master", "cal_high_is_cross_master",
        "cal_distance_windows", "cal_distance_minutes",
        "cal_distance_samples", "cal_distance_time_minutes",
    }
    cleaned = []
    for r in records:
        cleaned.append({k: v for k, v in r.items() if k not in drop_keys})
    return cleaned


def _split_counts(train_list, val_list, test_list):
    return {
        "n_subj_train": _count_subjects(train_list),
        "n_subj_val": _count_subjects(val_list),
        "n_subj_test": _count_subjects(test_list),
        "n_master_train": _count_master_records(train_list),
        "n_master_val": _count_master_records(val_list),
        "n_master_test": _count_master_records(test_list),
    }


# Manifest copy with a split column: train / val / test, calibration (cal-only windows), unused.
def _label_manifest_split(manifest, train_list, val_list, test_list, window_cal_pairs):
    labelled = manifest.copy()
    labelled["split"] = "unused"
    target_paths = set(train_list) | set(val_list) | set(test_list)
    labelled.loc[labelled["npz_path"].isin(train_list), "split"] = "train"
    labelled.loc[labelled["npz_path"].isin(val_list), "split"] = "val"
    labelled.loc[labelled["npz_path"].isin(test_list), "split"] = "test"
    cal_only_paths = set()
    for p in (window_cal_pairs or {}).values():
        for k in ("cal_low_path", "cal_high_path"):
            cp = p.get(k)
            if cp and cp not in target_paths:
                cal_only_paths.add(cp)
    labelled.loc[labelled["npz_path"].isin(cal_only_paths), "split"] = "calibration"
    return labelled


# Train and test every variant under parent_run_dir, write the compare summary and bar plot,
# and return the summary rows.
def _run_variants(variants, *, parent_run_dir, N, train_list, val_list, test_list,
                  window_cal_pairs, counts, processed_manifest_df, device, test_mod,
                  cal_summary_path, split_json_path, log_tag):
    variant_results = []
    for v in variants:
        variant_results.append(_run_one_variant(
            variant=v,
            parent_run_dir=parent_run_dir,
            train_list=train_list, val_list=val_list, test_list=test_list,
            window_cal_pairs=window_cal_pairs,
            processed_manifest_df=processed_manifest_df,
            device=device, test_mod=test_mod,
            shared_cal_summary_path=cal_summary_path,
            shared_split_json_path=split_json_path,
            **counts,
        ))

    compare_csv = _save_compare_summary(parent_run_dir, variant_results)
    rows = []
    try:
        rows = pd.read_csv(compare_csv).to_dict("records")
    except Exception as e:
        logging.warning(f"[{log_tag}] compare summary fold-in failed: {e}")
    try:
        _plot_compare_bar(
            variant_results, N=int(N),
            out_path=os.path.join(parent_run_dir, "compare_bar.png"),
        )
    except Exception as e:
        logging.warning(f"[{log_tag}] comparison plot failed: {e}")
    return rows


def _run_one_variant(*, variant, parent_run_dir, train_list, val_list, test_list,
                     window_cal_pairs, n_subj_train, n_subj_val, n_subj_test,
                     n_master_train, n_master_val, n_master_test,
                     processed_manifest_df, device, test_mod,
                     shared_cal_summary_path: str = "",
                     shared_split_json_path: str = ""):
    global RUN_DIR

    use_calibration = (variant == VARIANT_CALIBRATED)
    sub_dir = os.path.join(parent_run_dir, variant)
    os.makedirs(sub_dir, exist_ok=True)
    RUN_DIR = sub_dir

    # Every variant starts from the same RNG / cuDNN state, whatever ran before it.
    _reset_torch_state()

    model_name = _VARIANT_MODELS[variant]["model_name"]
    ckpt_basename = _VARIANT_MODELS[variant]["checkpoint_basename"]

    logging.info("=" * 72)
    logging.info(f"Variant: {variant} (model={model_name})")
    logging.info(f"Variant run directory: {sub_dir}")
    logging.info("=" * 72)

    # Copy split_subjects.json from the parent dir so each variant subdir is self-contained.
    if shared_split_json_path:
        dst_split_path = os.path.join(sub_dir, "split_subjects.json")
        if os.path.isfile(shared_split_json_path):
            shutil.copy2(shared_split_json_path, dst_split_path)
            logging.info(f"Copied subject split JSON to: {dst_split_path}")
        else:
            logging.warning(f"Shared split JSON not found: {shared_split_json_path}")

    processed_manifest_path = os.path.join(sub_dir, "processed_manifest.csv")
    abp_stats_path = os.path.join(sub_dir, "abp_normalization_stats.json")
    data_lists_path = os.path.join(sub_dir, "data_lists.json")
    checkpoint_path = os.path.join(sub_dir, ckpt_basename)
    lr_sbp_path = os.path.join(sub_dir, "linear_regression_sbp.joblib")
    lr_dbp_path = os.path.join(sub_dir, "linear_regression_dbp.joblib")
    run_manifest_path = os.path.join(sub_dir, "run_manifest.json")

    if use_calibration:
        cal_pairs_path = os.path.join(sub_dir, "calibration_fixed_pairs.json")
        cal_summary_path = os.path.join(sub_dir, "calibration_fixed_summary.csv")
    else:
        cal_pairs_path = ""
        cal_summary_path = ""

    # Persist the manifest with split labels, and the data lists.
    if processed_manifest_df is not None:
        _label_manifest_split(
            processed_manifest_df, train_list, val_list, test_list, window_cal_pairs
        ).to_csv(processed_manifest_path, index=False)

    save_data_lists(data_lists_path, train_list, val_list, test_list)

    # Persist calibration JSON only for the calibrated variant.
    if use_calibration and window_cal_pairs:
        payload = {
            "window_count": int(CALIBRATION_WINDOW_COUNT),
            "mode": _active_calibration_mode_tag(),
            "pairs_by_target_path": window_cal_pairs,
        }
        with open(cal_pairs_path, "w") as f:
            json.dump(payload, f, indent=2)
        logging.info(f"Saved calibration pairs to: {cal_pairs_path}")

        # Copy the calibration summary produced once in the parent dir.
        if shared_cal_summary_path:
            if os.path.isfile(shared_cal_summary_path):
                shutil.copy2(shared_cal_summary_path, cal_summary_path)
                logging.info(f"Copied calibration summary to: {cal_summary_path}")
            else:
                logging.warning(
                    f"Shared calibration summary not found: {shared_cal_summary_path}"
                )

    # ABP normalization stats were computed by the caller (per fold, or once in main()).
    save_abp_stats(abp_stats_path)

    train_loader, val_loader = _make_loaders(
        train_list, val_list, window_cal_pairs, use_calibration=use_calibration
    )
    model = _build_train_model(variant, device)

    logging.info(f"Start training {model_name}...")
    model = train_model(
        model, train_loader, val_loader, epochs, device, checkpoint_path,
        use_calibration=use_calibration,
        history_title=f"Training History — {model_name}",
    )

    # Reload best weights into a fresh, uncompiled model for evaluation.
    eval_model = _new_model(variant, device)
    if os.path.isfile(checkpoint_path):
        _safe_load(eval_model, checkpoint_path, device)
    else:
        eval_model.load_state_dict(getattr(model, "_orig_mod", model).state_dict())
    eval_model.eval()

    # Per-master residual adapter fine-tune (calibrated only), on the uncompiled eval model.
    adapter_cache = {}
    adapter_diag_by_key = {}
    adapters_path = ""
    if use_calibration and USE_PER_MASTER_ADAPTER and window_cal_pairs:
        used_masters = set()
        for fp in (list(train_list) + list(val_list) + list(test_list)):
            key = _path_to_master_key(fp)
            if key is not None:
                used_masters.add(key)
        logging.info(
            f"[{variant}] Per-master adapter fine-tune "
            f"({ADAPTER_FT_OBJECTIVE}): {len(used_masters)} (sid, master) "
            f"keys with target windows."
        )
        try:
            adapter_cache, adapter_diag = finetune_per_master_adapters(
                eval_model, window_cal_pairs, device,
                used_masters=used_masters,
            )
            adapter_diag_by_key = _diag_rows_to_by_key(adapter_diag)
            adapters_path = save_per_master_adapters(
                adapter_cache, adapter_diag, sub_dir
            ) or ""
        except Exception as e:
            logging.warning(
                f"[{variant}] Per-master adapter fine-tune failed: {e}"
            )
            adapter_cache = {}
            adapter_diag_by_key = {}
            adapters_path = ""
    # Attach (possibly empty) cache + diag so test/eval loops can call the swap helper uniformly.
    _attach_adapter_cache(eval_model, adapter_cache,
                          diag_by_key=adapter_diag_by_key)

    logging.info(f"Final test on best {model_name} model...")
    test_results = test_mod.test_model(
        eval_model, test_list, device, use_calibration=use_calibration
    )
    test_results["records"] = _maybe_attach_calibration_columns(
        test_results["records"], use_calibration=use_calibration
    )
    test_mod.log_test_summary(test_results, tag=f"Test-Final[{variant}]")
    test_mod.save_test_segment_metrics(test_results, RUN_DIR)

    # Ridge probe (per variant)
    ridge_summary = None
    if USE_BOTTLENECK_RIDGE_PROBE:
        try:
            ridge_res = test_mod.fit_and_save_decoder_ridge(
                eval_model, train_list, val_list, test_list, device,
                lr_sbp_path=lr_sbp_path, lr_dbp_path=lr_dbp_path,
                summary_path=os.path.join(RUN_DIR, "linear_regression_summary.json"),
                use_calibration=use_calibration,
            )
            test_mod.log_ridge_summary(ridge_res, tag=f"Test-Final[{variant}]")
            test_mod.augment_segment_csv_with_ridge(
                run_dir=RUN_DIR, model=eval_model, test_list=test_list, device=device,
                lr_sbp_path=lr_sbp_path, lr_dbp_path=lr_dbp_path,
                use_calibration=use_calibration,
            )
            ridge_summary = ridge_res
        except Exception as e:
            logging.warning(f"Final ridge failed: {e}")

    # Distance analysis (only meaningful for calibrated variant)
    if use_calibration:
        try:
            test_mod.run_calibration_distance_analysis(RUN_DIR)
        except Exception as e:
            logging.warning(f"Calibration distance analysis failed: {e}")

    # Visualisations
    try:
        test_mod.visualize_results(eval_model, test_list, device,
                                   use_calibration=use_calibration)
        test_mod.plot_post_training_metrics(eval_model, test_list, device,
                                            use_calibration=use_calibration)
    except Exception as e:
        logging.warning(f"Visualization failed: {e}")

    save_run_manifest(
        run_manifest_path,
        run_dir=RUN_DIR,
        checkpoint_path=checkpoint_path,
        calibration_pairs_path=cal_pairs_path,
        calibration_summary_path=cal_summary_path,
        processed_manifest_path=processed_manifest_path,
        data_lists_path=data_lists_path,
        abp_stats_path=abp_stats_path,
        lr_sbp_path=lr_sbp_path,
        lr_dbp_path=lr_dbp_path,
        n_train=len(train_list), n_val=len(val_list), n_test=len(test_list),
        n_subj_train=n_subj_train, n_subj_val=n_subj_val, n_subj_test=n_subj_test,
        n_master_train=n_master_train, n_master_val=n_master_val,
        n_master_test=n_master_test,
        variant=variant, use_calibration=use_calibration,
        model_name=model_name, parent_run_dir=parent_run_dir,
        per_master_adapters_path=adapters_path,
    )

    # Leave the decoder adapter on Identity so the next variant / fold cannot inherit it.
    _reset_adapter(eval_model)

    return {
        "variant": variant,
        "use_calibration": use_calibration,
        "model_name": model_name,
        "run_dir": sub_dir,
        "test_summary": test_results.get("summary", {}),
        "ridge": ridge_summary,
        "n_train": len(train_list), "n_val": len(val_list), "n_test": len(test_list),
        "n_subj_train": n_subj_train, "n_subj_val": n_subj_val, "n_subj_test": n_subj_test,
        "n_master_train": n_master_train, "n_master_val": n_master_val,
        "n_master_test": n_master_test,
    }


def _plot_compare_bar(variant_results: list, *, N: int, out_path: str):
    rows = []
    for r in variant_results:
        s = r.get("test_summary", {}) or {}
        sbp = s.get("sbp_waveform", {}) or {}
        dbp = s.get("dbp_waveform", {}) or {}
        ridge = r.get("ridge") or {}
        ridge_sbp = ridge.get("sbp", {}) if isinstance(ridge, dict) else {}
        ridge_dbp = ridge.get("dbp", {}) if isinstance(ridge, dict) else {}
        rows.append({
            "variant": r["variant"],
            "wave_sbp_mae": float(sbp.get("MAE", float('nan'))),
            "wave_dbp_mae": float(dbp.get("MAE", float('nan'))),
            "ridge_sbp_mae": float(ridge_sbp.get("MAE", float('nan'))),
            "ridge_dbp_mae": float(ridge_dbp.get("MAE", float('nan'))),
        })
    if not rows:
        return
    variants = [r["variant"] for r in rows]
    metrics = ["wave_sbp_mae", "wave_dbp_mae", "ridge_sbp_mae", "ridge_dbp_mae"]
    metric_labels = ["Waveform SBP MAE", "Waveform DBP MAE",
                     "Ridge SBP MAE", "Ridge DBP MAE"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(rows):
        vals = [r[m] for m in metrics]
        offset = (i - (len(rows) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=variants[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15)
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title(f"Calibrated vs No-calibration — N = {N}")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=120)
        logging.info(f"Saved comparison plot: {out_path}")
    finally:
        plt.close(fig)


def _plot_cross_n_bar(cross_rows: list, out_path: str):
    if not cross_rows:
        return
    df = pd.DataFrame(cross_rows)
    if df.empty or "variant" not in df.columns:
        return
    n_col = "CALIBRATION_WINDOW_COUNT"
    df = df.sort_values([n_col, "variant"]).reset_index(drop=True)
    keys = [(int(r[n_col]), str(r["variant"])) for _, r in df.iterrows()]
    labels = [f"N={n}\n{v}" for (n, v) in keys]
    metrics = ["sbp_waveform_MAE", "dbp_waveform_MAE",
               "ridge_sbp_MAE", "ridge_dbp_MAE"]
    metric_labels = ["Waveform SBP MAE", "Waveform DBP MAE",
                     "Ridge SBP MAE", "Ridge DBP MAE"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    width = 0.2
    for j, m in enumerate(metrics):
        vals = [float(df.iloc[i].get(m, float('nan'))) for i in range(len(labels))]
        offset = (j - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=metric_labels[j])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Cross-N comparison — calibrated vs no-calibration")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=120)
        logging.info(f"Saved cross-N comparison plot: {out_path}")
    finally:
        plt.close(fig)


def _save_compare_summary(parent_run_dir: str, variant_results: list):
    rows = []
    for r in variant_results:
        s = r.get("test_summary", {}) or {}
        sg = s.get("sample_global", {}) or {}
        sbp = s.get("sbp_waveform", {}) or {}
        dbp = s.get("dbp_waveform", {}) or {}
        ridge = r.get("ridge") or {}
        ridge_sbp = ridge.get("sbp", {}) if isinstance(ridge, dict) else {}
        ridge_dbp = ridge.get("dbp", {}) if isinstance(ridge, dict) else {}

        rows.append({
            "variant": r["variant"],
            "model_name": r["model_name"],
            "use_calibration": bool(r["use_calibration"]),
            "CALIBRATION_WINDOW_COUNT": int(CALIBRATION_WINDOW_COUNT),
            "n_train": r["n_train"], "n_val": r["n_val"], "n_test": r["n_test"],
            "n_subj_train": r["n_subj_train"], "n_subj_val": r["n_subj_val"],
            "n_subj_test": r["n_subj_test"],
            "n_master_train": r["n_master_train"], "n_master_val": r["n_master_val"],
            "n_master_test": r["n_master_test"],
            # Waveform (per-segment averages)
            "wave_mae_mean": s.get("wave_mae_mean", float('nan')),
            "wave_R_mean": s.get("wave_R_mean", float('nan')),
            # Sample-global ABP residuals
            "sample_global_N": sg.get("N", 0),
            "sample_global_ME": sg.get("ME", float('nan')),
            "sample_global_MAE": sg.get("MAE", float('nan')),
            "sample_global_SD": sg.get("SD", float('nan')),
            "sample_global_RMSE": sg.get("RMSE", float('nan')),
            # SBP from waveform
            "sbp_waveform_N": sbp.get("N", 0),
            "sbp_waveform_MAE": sbp.get("MAE", float('nan')),
            "sbp_waveform_ME": sbp.get("ME", float('nan')),
            "sbp_waveform_SD": sbp.get("SD", float('nan')),
            "sbp_waveform_R": sbp.get("R", float('nan')),
            # DBP from waveform
            "dbp_waveform_N": dbp.get("N", 0),
            "dbp_waveform_MAE": dbp.get("MAE", float('nan')),
            "dbp_waveform_ME": dbp.get("ME", float('nan')),
            "dbp_waveform_SD": dbp.get("SD", float('nan')),
            "dbp_waveform_R": dbp.get("R", float('nan')),
            "ridge_sbp_MAE": ridge_sbp.get("MAE", float('nan')),
            "ridge_sbp_ME": ridge_sbp.get("ME", float('nan')),
            "ridge_sbp_SD": ridge_sbp.get("SD", float('nan')),
            "ridge_sbp_R": ridge_sbp.get("R", float('nan')),
            "ridge_dbp_MAE": ridge_dbp.get("MAE", float('nan')),
            "ridge_dbp_ME": ridge_dbp.get("ME", float('nan')),
            "ridge_dbp_SD": ridge_dbp.get("SD", float('nan')),
            "ridge_dbp_R": ridge_dbp.get("R", float('nan')),
            "run_dir": r["run_dir"],
        })

    out_path = os.path.join(parent_run_dir, "compare_summary.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info(f"Saved compare_summary: {out_path}")
    return out_path


def _run_cross_n(*, n_values, full_train_list, full_val_list, full_test_list,
                 base_manifest, run_root, split_path, variants, test_mod, device):
    global RUN_DIR, train_list, val_list, test_list, CALIBRATION_WINDOW_COUNT

    cross_rows: list = []

    # Per-N loop
    for N in n_values:
        CALIBRATION_WINDOW_COUNT = int(N)
        parent_run_dir = os.path.join(run_root, f"compare_N{int(N)}")
        os.makedirs(parent_run_dir, exist_ok=True)
        RUN_DIR = parent_run_dir

        logging.info("=" * 72)
        logging.info(f"### Static-first-N calibration: N = {N} ###")
        logging.info(f"### Parent run directory: {parent_run_dir} ###")
        logging.info("=" * 72)

        # Audit trail per-N: copy split + ABP stats into this N's parent dir.
        per_n_split_path = os.path.join(parent_run_dir, "split_subjects.json")
        if os.path.isfile(split_path):
            shutil.copy2(split_path, per_n_split_path)
        save_abp_stats(os.path.join(parent_run_dir,
                                    "abp_normalization_stats.json"))

        # Re-install the base (no-split-label) manifest so cal selection sees the full window set.
        if base_manifest is not None:
            install_manifest(base_manifest.copy())

        # Per-master static first-N calibration pairs + summary.
        cal_pairs_path = os.path.join(parent_run_dir,
                                      "calibration_fixed_pairs.json")
        cal_summary_path = os.path.join(parent_run_dir,
                                        "calibration_fixed_summary.csv")
        window_cal_pairs, valid_sids = select_calibration_pairs(
            full_train_list + full_val_list + full_test_list,
            output_pairs_path=cal_pairs_path,
            output_summary_path=cal_summary_path,
        )

        # Validation hooks: pairs invariant within each (sid, master), plus a random spot check.
        _validate_calibration_invariance_per_master(window_cal_pairs)
        _validate_calibration_pairs_random(window_cal_pairs, N, sample=20)

        # Filter target windows for THIS N.
        n_before = (len(full_train_list), len(full_val_list),
                    len(full_test_list))
        train_list, val_list, test_list = filter_target_lists(
            full_train_list, full_val_list, full_test_list, valid_sids
        )
        logging.info(
            f"[N={N}] Target filter (record_window_index >= {N}): "
            f"train {n_before[0]}->{len(train_list)}, "
            f"val {n_before[1]}->{len(val_list)}, "
            f"test {n_before[2]}->{len(test_list)}"
        )

        counts = _split_counts(train_list, val_list, test_list)
        logging.info(
            f"[N={N}] subjects: train={counts['n_subj_train']}, "
            f"val={counts['n_subj_val']}, test={counts['n_subj_test']}"
        )
        logging.info(
            f"[N={N}] master_records: train={counts['n_master_train']}, "
            f"val={counts['n_master_val']}, test={counts['n_master_test']}"
        )
        n_usable_masters_total = len({(p["sid"], p["master_record"])
                                      for p in window_cal_pairs.values()})
        logging.info(
            f"[N={N}] usable master_records across train/val/test: "
            f"{n_usable_masters_total}; total target windows: "
            f"{len(window_cal_pairs)}"
        )

        # Label the manifest with this N's split and re-install it so the variant subdirs inherit it.
        processed_manifest_df = None
        if base_manifest is not None:
            processed_manifest_df = _label_manifest_split(
                base_manifest, train_list, val_list, test_list, window_cal_pairs
            )
            processed_manifest_df.to_csv(
                os.path.join(parent_run_dir, "processed_manifest.csv"), index=False
            )
            install_manifest(processed_manifest_df)

        save_data_lists(os.path.join(parent_run_dir, "data_lists.json"),
                        train_list, val_list, test_list)

        # Run no-cal first then calibrated (or whatever variants resolved).
        cross_rows.extend(_run_variants(
            variants, parent_run_dir=parent_run_dir, N=N,
            train_list=train_list, val_list=val_list, test_list=test_list,
            window_cal_pairs=window_cal_pairs, counts=counts,
            processed_manifest_df=processed_manifest_df,
            device=device, test_mod=test_mod,
            cal_summary_path=cal_summary_path, split_json_path=per_n_split_path,
            log_tag=f"N={N}",
        ))

    # Cross-N artifacts
    if cross_rows:
        cross_csv = os.path.join(run_root, "cross_n_summary.csv")
        try:
            pd.DataFrame(cross_rows).to_csv(cross_csv, index=False)
            logging.info(f"Saved cross-N summary: {cross_csv}")
        except Exception as e:
            logging.warning(f"Failed to save cross-N summary CSV: {e}")
        try:
            _plot_cross_n_bar(cross_rows,
                              os.path.join(run_root, "cross_n_bar.png"))
        except Exception as e:
            logging.warning(f"Cross-N comparison plot failed: {e}")

    RUN_DIR = run_root
    return cross_rows


def _save_cv_summary(cross_root, cv_rows):
    df = pd.DataFrame(cv_rows)
    if df.empty:
        logging.warning("No CV rows to aggregate.")
        return None

    per_fold_path = os.path.join(cross_root, "cv_per_fold.csv")
    df.to_csv(per_fold_path, index=False)
    logging.info(f"Saved per-fold CV rows: {per_fold_path}")

    group_keys = [k for k in ("variant", "model_name", "CALIBRATION_WINDOW_COUNT")
                  if k in df.columns]
    metric_cols = [c for c in (
        "wave_mae_mean", "wave_R_mean",
        "sample_global_ME", "sample_global_MAE", "sample_global_SD",
        "sample_global_RMSE",
        "sbp_waveform_MAE", "sbp_waveform_ME", "sbp_waveform_SD", "sbp_waveform_R",
        "dbp_waveform_MAE", "dbp_waveform_ME", "dbp_waveform_SD", "dbp_waveform_R",
        "ridge_sbp_MAE", "ridge_sbp_ME", "ridge_sbp_SD", "ridge_sbp_R",
        "ridge_dbp_MAE", "ridge_dbp_ME", "ridge_dbp_SD", "ridge_dbp_R",
    ) if c in df.columns]

    if not group_keys or not metric_cols:
        logging.warning("CV aggregation skipped: missing group keys or metrics.")
        return per_fold_path

    agg = df.groupby(group_keys, dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    counts = (df.groupby(group_keys, dropna=False)
                .size().reset_index(name="n_folds"))
    agg = agg.merge(counts, on=group_keys, how="left")

    out_path = os.path.join(cross_root, "cv_summary.csv")
    agg.to_csv(out_path, index=False)
    logging.info(f"Saved CV summary (mean/std across {int(agg['n_folds'].max())} "
                 f"folds): {out_path}")

    # Headline log per group.
    for _, row in agg.iterrows():
        try:
            tag = (f"variant={row.get('variant')} "
                   f"N={row.get('CALIBRATION_WINDOW_COUNT')}")
            logging.info(
                f"[CV] {tag}: SBP MAE={row['sbp_waveform_MAE_mean']:.3f}"
                f"+/-{row['sbp_waveform_MAE_std']:.3f}, "
                f"DBP MAE={row['dbp_waveform_MAE_mean']:.3f}"
                f"+/-{row['dbp_waveform_MAE_std']:.3f} "
                f"(n_folds={int(row['n_folds'])})"
            )
        except Exception:
            pass
    return out_path


def _run_kfold_cv(*, n_values, cross_root, base_manifest, variants,
                  test_mod, device):
    if base_manifest is None:
        raise RuntimeError("K-fold CV requires an installed processed manifest.")

    all_subjects = sorted(base_manifest["sid"].astype(str).unique().tolist())
    n_subjects = len(all_subjects)
    n_splits = max(2, min(int(KFOLD_N_SPLITS), n_subjects))

    logging.info("=" * 72)
    logging.info(
        f"K-FOLD CROSS-VALIDATION: {n_splits} folds over {n_subjects} subjects "
        f"(stratify={KFOLD_STRATIFY_KEY}, val_ratio={KFOLD_VAL_RATIO}, "
        f"seed={KFOLD_SEED})"
    )
    logging.info("=" * 72)

    # Per-subject BP key for stratified folds (from NPZ; works in any mode).
    subj_stats = (compute_subject_bp_stats_from_manifest(KFOLD_STRATIFY_MAX_WINDOWS)
                  if KFOLD_STRATIFY_KEY else {})

    folds = stratified_subject_kfold(
        subj_stats, all_subjects, n_splits,
        sort_key=KFOLD_STRATIFY_KEY, seed=KFOLD_SEED,
    )

    # Persist the fold definition for reproducibility / audit.
    try:
        with open(os.path.join(cross_root, "kfold_subjects.json"), "w") as f:
            json.dump({
                "n_splits": n_splits,
                "stratify_key": KFOLD_STRATIFY_KEY,
                "val_ratio": KFOLD_VAL_RATIO,
                "seed": KFOLD_SEED,
                "folds": {f"fold_{i}": sorted(list(s))
                          for i, s in enumerate(folds)},
            }, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save kfold_subjects.json: {e}")

    cv_rows: list = []
    for fold_idx, test_subj in enumerate(folds):
        train_subj, val_subj, test_subj = make_kfold_split(
            all_subjects, test_subj, KFOLD_VAL_RATIO, subj_stats,
            sort_key=KFOLD_STRATIFY_KEY,
        )

        fold_root = os.path.join(cross_root, f"fold_{fold_idx}")
        os.makedirs(fold_root, exist_ok=True)

        logging.info("#" * 72)
        logging.info(f"### FOLD {fold_idx + 1}/{n_splits} ###")
        logging.info(
            f"### subjects: train={len(train_subj)} val={len(val_subj)} "
            f"test={len(test_subj)} ###"
        )
        logging.info(f"### fold directory: {fold_root} ###")
        logging.info("#" * 72)

        # Reinstall the unlabeled base manifest so list building sees every window.
        install_manifest(base_manifest.copy())

        fold_split_path = os.path.join(fold_root, "split_subjects.json")
        save_split_json(train_subj, val_subj, test_subj, fold_split_path)

        f_train, f_val, f_test = build_lists_from_manifest(
            train_subj, val_subj, test_subj
        )

        # Per-fold ABP normalization stats from TRAIN only -> no leakage.
        logging.info(f"[fold {fold_idx}] Compute training ABP stats "
                     f"(fold train_list)...")
        compute_global_abp_stats(f_train)
        save_abp_stats(os.path.join(fold_root, "abp_normalization_stats.json"))

        if int(USE_OPTUNA_LR_SEARCH) == 1 and int(OPTUNA_PER_FOLD) == 1:
            _install_tuned_lr(*_optuna_lr_for_fold(
                f_train, f_val, f_test, fold_idx=fold_idx,
                out_dir=fold_root, device=device,
            ))
            install_manifest(base_manifest.copy())
            compute_global_abp_stats(f_train)

        cross_rows = _run_cross_n(
            n_values=n_values,
            full_train_list=f_train, full_val_list=f_val, full_test_list=f_test,
            base_manifest=base_manifest, run_root=fold_root,
            split_path=fold_split_path, variants=variants,
            test_mod=test_mod, device=device,
        )
        for r in cross_rows:
            row = dict(r)
            row["fold"] = fold_idx
            cv_rows.append(row)

    _save_cv_summary(cross_root, cv_rows)


# Shared-bundle k-fold
def _kfold_bundle_dir() -> str:
    d = (KFOLD_BUNDLE_DIR or "").strip()
    if not d:
        raise RuntimeError("KFOLD_BUNDLE_DIR is empty.")
    if not os.path.isabs(d):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), d)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"K-fold bundle directory not found: {d}")
    return d


def _kfold_remap_path(path: str, npz_root: str) -> str:
    if not npz_root:
        return path
    filename = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return os.path.join(npz_root, filename)


def _read_bundle_json(path: str):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_kfold_bundle_info() -> dict:
    return _read_bundle_json(os.path.join(_kfold_bundle_dir(), "bundle_info.json"))


def load_kfold_fold(fold, npz_root: str, *, include_calibration: bool = True) -> dict:
    bundle_dir = _kfold_bundle_dir()
    fold_name = str(fold)
    if not fold_name.startswith("fold_"):
        fold_name = f"fold_{fold_name}"
    fold_dir = os.path.join(bundle_dir, fold_name)
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Unknown fold directory: {fold_dir}")

    # 1) Final target lists (order preserved exactly as in the bundle).
    lists = _read_bundle_json(os.path.join(fold_dir, "data_lists.json.gz"))
    train_l = [_kfold_remap_path(p, npz_root) for p in lists["train_list"]]
    val_l = [_kfold_remap_path(p, npz_root) for p in lists["val_list"]]
    test_l = [_kfold_remap_path(p, npz_root) for p in lists["test_list"]]

    # 2) ABP normalization stats computed on THIS fold's train set.
    abp = _read_bundle_json(os.path.join(fold_dir, "abp_normalization_stats.json"))
    abp_min = float(abp["GLOBAL_ABP_MIN"])
    abp_max = float(abp["GLOBAL_ABP_MAX"])

    # 3) Subject-level split (audit only) and processed manifest (needed for per-master pool lookups.
    split_subjects = _read_bundle_json(os.path.join(fold_dir, "split_subjects.json"))
    manifest_df = pd.read_csv(os.path.join(fold_dir, "processed_manifest.csv.gz"))
    manifest_df["npz_path"] = manifest_df["npz_path"].map(
        lambda p: _kfold_remap_path(p, npz_root)
    )

    result = {
        "fold": fold_name,
        "train_list": train_l,
        "val_list": val_l,
        "test_list": test_l,
        "abp_min": abp_min,
        "abp_max": abp_max,
        "split_subjects": split_subjects,
        "manifest_df": manifest_df,
        "cal_summary_path": os.path.join(fold_dir, "calibration_fixed_summary.csv"),
        "split_subjects_path": os.path.join(fold_dir, "split_subjects.json"),
        "calibration_pairs": {},
        "calibration_window_count": None,
        "calibration_mode": "",
    }

    # 4) Per-target calibration pairs (only when a calibrated variant runs).
    if include_calibration:
        cal = _read_bundle_json(
            os.path.join(fold_dir, "calibration_fixed_pairs.json.gz")
        )
        raw_pairs = cal["pairs_by_target_path"]
        remapped = {}
        for tgt_path, p in raw_pairs.items():
            q = dict(p)
            q["cal_low_path"] = _kfold_remap_path(p["cal_low_path"], npz_root)
            q["cal_high_path"] = _kfold_remap_path(p["cal_high_path"], npz_root)
            remapped[_kfold_remap_path(tgt_path, npz_root)] = q
        result["calibration_pairs"] = remapped
        result["calibration_window_count"] = cal.get("window_count")
        result["calibration_mode"] = cal.get("mode", "")

    return result


def _kfold_check_npz_exist(flist, *, tag: str, sample: int):
    if sample <= 0 or not flist:
        return
    probe = list(flist[:sample])
    missing = [p for p in probe if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"[{tag}] {len(missing)}/{len(probe)} probed .npz not found under "
            f"KFOLD_NPZ_ROOT. First missing: {missing[0]}. The bundle ships no "
            f".npz bodies — point KFOLD_NPZ_ROOT at the directory holding them."
        )


def run_kfold_bundle(*, device, test_mod):
    global RUN_DIR, train_list, val_list, test_list, CALIBRATION_WINDOW_COUNT
    global _WINDOW_CAL_PAIRS

    variants = _resolve_compare_modes()
    need_cal = VARIANT_CALIBRATED in variants

    bundle_dir = _kfold_bundle_dir()
    info = load_kfold_bundle_info()
    N = int(info.get("calibration_window_count", CALIBRATION_WINDOW_COUNT))
    CALIBRATION_WINDOW_COUNT = N
    npz_root = os.path.abspath(os.path.expanduser(KFOLD_NPZ_ROOT))

    ts = datetime.now().strftime(RUN_DIR_TS_FMT)
    cross_root = os.path.join(RUN_ROOT, f"kfold_bundle_N{N}_{ts}")
    os.makedirs(cross_root, exist_ok=True)
    RUN_DIR = cross_root

    logging.info("=" * 72)
    logging.info("K-FOLD FROM SHARED BUNDLE (no re-split, no resampling)")
    logging.info(f"  bundle        : {bundle_dir}")
    logging.info(f"  source run    : {info.get('source_run', '?')}")
    logging.info(f"  n_splits      : {info.get('n_splits', '?')}")
    logging.info(f"  N (cal windows): {N}")
    logging.info(f"  npz root      : {npz_root}")
    logging.info(f"  original root : {info.get('original_npz_root', '?')}")
    logging.info(f"  folds to run  : {tuple(KFOLD_BUNDLE_FOLDS)}")
    logging.info(f"  variants      : {', '.join(variants)}")
    logging.info(f"  run directory : {cross_root}")
    logging.info("=" * 72)

    # One search on bundle fold OPTUNA_FOLD, then every fold trains with the result.
    if int(USE_OPTUNA_LR_SEARCH) == 1 and int(OPTUNA_PER_FOLD) != 1:
        fold_data = load_kfold_fold(int(OPTUNA_FOLD), npz_root, include_calibration=need_cal)
        _kfold_check_npz_exist(fold_data["train_list"], tag=fold_data["fold"],
                               sample=int(KFOLD_BUNDLE_NPZ_CHECK))
        install_manifest(fold_data["manifest_df"].copy())
        preprocess.set_abp_stats(fold_data["abp_min"], fold_data["abp_max"])
        _WINDOW_CAL_PAIRS = fold_data["calibration_pairs"]
        _install_tuned_lr(*_optuna_lr_for_fold(
            fold_data["train_list"], fold_data["val_list"], fold_data["test_list"],
            fold_idx=int(OPTUNA_FOLD), out_dir=cross_root, device=device,
            window_cal_pairs=fold_data["calibration_pairs"],
        ))
        RUN_DIR = cross_root
        if int(OPTUNA_THEN_KFOLD) != 1:
            logging.info("OPTUNA_THEN_KFOLD=0 — stopping after the LR search.")
            return

    all_rows: list = []
    for fold in KFOLD_BUNDLE_FOLDS:
        fold_data = load_kfold_fold(fold, npz_root, include_calibration=need_cal)
        fold_name = fold_data["fold"]

        # Install this fold's manifest / calibration pairs / ABP stats as the module globals.
        install_manifest(fold_data["manifest_df"].copy())
        preprocess.set_abp_stats(fold_data["abp_min"], fold_data["abp_max"])
        window_cal_pairs = fold_data["calibration_pairs"]
        _WINDOW_CAL_PAIRS = window_cal_pairs

        train_list = fold_data["train_list"]
        val_list = fold_data["val_list"]
        test_list = fold_data["test_list"]

        # Cross-machine sanity: verify a few remapped .npz exist on disk.
        _kfold_check_npz_exist(train_list, tag=fold_name,
                               sample=int(KFOLD_BUNDLE_NPZ_CHECK))

        parent_run_dir = os.path.join(cross_root, fold_name, f"compare_N{N}")
        os.makedirs(parent_run_dir, exist_ok=True)
        RUN_DIR = parent_run_dir

        if need_cal:
            saved_wc = fold_data.get("calibration_window_count")
            if saved_wc is not None and int(saved_wc) != int(N):
                raise RuntimeError(
                    f"[{fold_name}] bundle calibration window_count "
                    f"{saved_wc} != bundle N {N}."
                )

        # Audit trail: copy subject split + ABP stats into this dir.
        per_n_split_path = os.path.join(parent_run_dir, "split_subjects.json")
        try:
            shutil.copyfile(fold_data["split_subjects_path"], per_n_split_path)
        except Exception as e:
            logging.warning(f"[{fold_name}] split_subjects copy failed: {e}")
        save_abp_stats(os.path.join(parent_run_dir,
                                    "abp_normalization_stats.json"))
        cal_summary_path = fold_data["cal_summary_path"]

        if int(USE_OPTUNA_LR_SEARCH) == 1 and int(OPTUNA_PER_FOLD) == 1:
            _install_tuned_lr(*_optuna_lr_for_fold(
                train_list, val_list, test_list, fold_idx=fold,
                out_dir=parent_run_dir, device=device,
                window_cal_pairs=window_cal_pairs,
            ))

        if need_cal and window_cal_pairs:
            with open(os.path.join(parent_run_dir,
                                   "calibration_fixed_pairs.json"), "w") as f:
                json.dump({
                    "window_count": int(N),
                    "mode": (fold_data.get("calibration_mode")
                             or _active_calibration_mode_tag()),
                    "pairs_by_target_path": window_cal_pairs,
                }, f, indent=2)

        counts = _split_counts(train_list, val_list, test_list)
        logging.info("#" * 72)
        logging.info(f"### {fold_name} (from bundle) ###")
        logging.info(f"### windows: train={len(train_list)}, "
                     f"val={len(val_list)}, test={len(test_list)} ###")
        logging.info(f"### subjects: train={counts['n_subj_train']}, "
                     f"val={counts['n_subj_val']}, test={counts['n_subj_test']} ###")
        logging.info(f"### master_records: train={counts['n_master_train']}, "
                     f"val={counts['n_master_val']}, test={counts['n_master_test']} ###")
        logging.info("#" * 72)

        processed_manifest_df = fold_data["manifest_df"].copy()
        processed_manifest_df.to_csv(
            os.path.join(parent_run_dir, "processed_manifest.csv"), index=False
        )
        save_data_lists(os.path.join(parent_run_dir, "data_lists.json"),
                        train_list, val_list, test_list)

        for row in _run_variants(
            variants, parent_run_dir=parent_run_dir, N=N,
            train_list=train_list, val_list=val_list, test_list=test_list,
            window_cal_pairs=window_cal_pairs, counts=counts,
            processed_manifest_df=processed_manifest_df,
            device=device, test_mod=test_mod,
            cal_summary_path=cal_summary_path, split_json_path=per_n_split_path,
            log_tag=fold_name,
        ):
            all_rows.append({"fold": fold_name, "N": int(N), **row})

    if all_rows:
        _save_cv_summary(cross_root, all_rows)

    RUN_DIR = cross_root
    logging.info("Done (k-fold from bundle)!")


# Optuna LR search
def _install_tuned_lr(best_lr, best_gate_lr):
    global learning_rate, GATE_LEARNING_RATE
    learning_rate = float(best_lr)
    if best_gate_lr is not None:
        GATE_LEARNING_RATE = float(best_gate_lr)
    logging.info(f"Tuned learning rates installed: lr={learning_rate:.2e}"
                 + (f", gate_lr={GATE_LEARNING_RATE:.2e}" if best_gate_lr is not None else ""))


def run_optuna_lr_search(*, cross_root, base_manifest, device):
    global RUN_DIR, train_list, val_list, test_list, _WINDOW_CAL_PAIRS

    if base_manifest is None:
        raise RuntimeError("Optuna LR search requires an installed processed manifest.")

    ts = datetime.now().strftime(RUN_DIR_TS_FMT)
    run_dir = os.path.join(RUN_ROOT, f"optuna_lr_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    RUN_DIR = run_dir

    # rebuild fold OPTUNA_FOLD exactly as _run_kfold_cv would
    all_subjects = sorted(base_manifest["sid"].astype(str).unique().tolist())
    n_splits = max(2, min(int(KFOLD_N_SPLITS), len(all_subjects)))
    fold_idx = int(OPTUNA_FOLD)
    if not 0 <= fold_idx < n_splits:
        raise RuntimeError(f"OPTUNA_FOLD={fold_idx} out of range for {n_splits} folds.")

    subj_stats = (compute_subject_bp_stats_from_manifest(KFOLD_STRATIFY_MAX_WINDOWS)
                  if KFOLD_STRATIFY_KEY else {})
    folds = stratified_subject_kfold(
        subj_stats, all_subjects, n_splits,
        sort_key=KFOLD_STRATIFY_KEY, seed=KFOLD_SEED,
    )
    train_subj, val_subj, test_subj = make_kfold_split(
        all_subjects, folds[fold_idx], KFOLD_VAL_RATIO, subj_stats,
        sort_key=KFOLD_STRATIFY_KEY,
    )
    save_split_json(train_subj, val_subj, test_subj,
                    os.path.join(run_dir, "split_subjects.json"))

    install_manifest(base_manifest.copy())
    f_train, f_val, f_test = build_lists_from_manifest(train_subj, val_subj, test_subj)

    compute_global_abp_stats(f_train)
    save_abp_stats(os.path.join(run_dir, "abp_normalization_stats.json"))

    return _optuna_lr_for_fold(f_train, f_val, f_test, fold_idx=fold_idx,
                               out_dir=run_dir, device=device)


# window_cal_pairs given (bundle mode) -> lists already filtered, nothing recomputed.
# Returns (best_lr, best_gate_lr); best_gate_lr is None when the model has no gate learning rate.
def _optuna_lr_for_fold(f_train, f_val, f_test, *, fold_idx, out_dir, device,
                        window_cal_pairs=None):
    global RUN_DIR, train_list, val_list, test_list, _WINDOW_CAL_PAIRS

    import optuna

    variant = _resolve_compare_modes()[0]
    use_calibration = (variant == VARIANT_CALIBRATED)
    N = int(N_VALUES[0])
    run_dir = os.path.join(out_dir, "optuna_lr")
    os.makedirs(run_dir, exist_ok=True)
    prev_run_dir = RUN_DIR
    RUN_DIR = run_dir

    joint = GATE_LEARNING_RATE is not None
    logging.info("=" * 72)
    if joint:
        logging.info(f"OPTUNA (lr, gate_lr) SEARCH — fold {fold_idx}, {OPTUNA_N_TRIALS} TPE trials, "
                     f"{OPTUNA_EPOCHS} epochs each  [{variant}]")
    else:
        logging.info(f"OPTUNA LR SEARCH — fold {fold_idx}, {len(OPTUNA_LR_GRID)} rates, "
                     f"{OPTUNA_EPOCHS} epochs each  [{variant}]")
    logging.info("=" * 72)

    if window_cal_pairs is None:
        window_cal_pairs, valid_sids = select_calibration_pairs(
            f_train + f_val + f_test,
            output_pairs_path=os.path.join(run_dir, "calibration_fixed_pairs.json"),
            output_summary_path=os.path.join(run_dir, "calibration_fixed_summary.csv"),
        )
        train_list, val_list, test_list = filter_target_lists(
            f_train, f_val, f_test, valid_sids
        )
        _WINDOW_CAL_PAIRS = window_cal_pairs
    else:
        train_list, val_list, test_list = f_train, f_val, f_test
    logging.info(f"[optuna fold {fold_idx}] targets: train={len(train_list)}, "
                 f"val={len(val_list)}, test={len(test_list)}; "
                 f"cal_pairs={len(window_cal_pairs)}")

    # Loaders are built once: they do not depend on lr.
    train_loader, val_loader = _make_loaders(
        train_list, val_list, window_cal_pairs, use_calibration=use_calibration
    )

    scratch_ckpt = os.path.join(run_dir, "optuna_trial_scratch.pth")
    trial_rows: list = []

    def objective(trial):
        if joint:
            lr = trial.suggest_float("lr", *OPTUNA_LR_RANGE, log=True)
            gate_lr = trial.suggest_float("gate_lr", *OPTUNA_GATE_LR_RANGE, log=True)
        else:
            lr = trial.suggest_categorical("lr", OPTUNA_LR_GRID)
            gate_lr = None
        tag = f"lr={lr:.2e}" + (f" gate_lr={gate_lr:.2e}" if joint else "")
        # Fresh seed + fresh model per trial so the learning rates are the ONLY thing varying.
        set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))
        model = _build_train_model(variant, device)
        logging.info(f"[trial {trial.number}] {tag} — training {OPTUNA_EPOCHS} epochs")
        _, best_val = train_model(
            model, train_loader, val_loader, OPTUNA_EPOCHS, device,
            scratch_ckpt, plot_curve=False,
            use_calibration=use_calibration,
            history_title=f"Optuna trial {trial.number} {tag}",
            lr=lr, gate_lr=gate_lr, return_best_val=True,
        )
        del model
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        best_val = float(best_val)
        row = {"trial": int(trial.number), "lr": float(lr), "best_val_loss": best_val}
        if joint:
            row["gate_lr"] = float(gate_lr)
        trial_rows.append(row)
        logging.info(f"[trial {trial.number}] {tag} -> best_val_loss={best_val:.6f}")
        return best_val

    if joint:
        sampler = optuna.samplers.TPESampler(seed=int(OPTUNA_SEED))
        n_trials = int(OPTUNA_N_TRIALS)
    else:
        sampler = optuna.samplers.GridSampler({"lr": OPTUNA_LR_GRID})
        n_trials = len(OPTUNA_LR_GRID)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"lr_{'tpe' if joint else 'grid'}_fold{fold_idx}")
    study.optimize(objective, n_trials=n_trials)

    results_df = pd.DataFrame(trial_rows).sort_values("lr").reset_index(drop=True)
    csv_path = os.path.join(run_dir, "optuna_lr_search.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Saved Optuna LR results: {csv_path}")

    best_lr = float(study.best_trial.params["lr"])
    best_gate_lr = float(study.best_trial.params["gate_lr"]) if joint else None
    with open(os.path.join(run_dir, "optuna_lr_search_best.json"), "w") as f:
        json.dump({
            "best_lr": best_lr,
            "best_gate_lr": best_gate_lr,
            "best_val_loss": float(study.best_trial.value),
            "epochs_per_trial": int(OPTUNA_EPOCHS),
            "fold": fold_idx,
            "N": N,
            "variant": variant,
            "model": MODEL_NAME,
            "search": ("tpe_joint" if joint else "lr_grid"),
            "lr_grid": (None if joint else [float(x) for x in OPTUNA_LR_GRID]),
            "lr_range": (list(OPTUNA_LR_RANGE) if joint else None),
            "gate_lr_range": (list(OPTUNA_GATE_LR_RANGE) if joint else None),
            "all_trials": trial_rows,
        }, f, indent=2)

    # Grid: one bar per lr. Joint: (lr, gate_lr) scatter coloured by loss. Winner highlighted.
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        if joint:
            sc = ax.scatter(results_df["lr"], results_df["gate_lr"],
                            c=results_df["best_val_loss"], cmap="viridis", s=60)
            ax.scatter([best_lr], [best_gate_lr], s=200, facecolors="none",
                       edgecolors="#d62728", linewidths=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("backbone learning rate")
            ax.set_ylabel("gate learning rate")
            fig.colorbar(sc, ax=ax, label=f"best val loss ({OPTUNA_EPOCHS} epochs)")
        else:
            lrs = results_df["lr"].tolist()
            colors = ["#d62728" if abs(v - best_lr) < 1e-12 else "#1f77b4" for v in lrs]
            ax.bar([f"{v:.0e}" for v in lrs], results_df["best_val_loss"].tolist(),
                   color=colors)
            ax.set_xlabel("learning rate")
            ax.set_ylabel(f"best val loss ({OPTUNA_EPOCHS} epochs)")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_title(f"Optuna LR search — {MODEL_NAME} [{variant}], fold {fold_idx}")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "optuna_lr_search.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Optuna plot failed: {e}")

    logging.info(f"[fold {fold_idx}] OPTUNA BEST lr = {best_lr:.2e}"
                 + (f", gate_lr = {best_gate_lr:.2e}" if joint else "")
                 + f" (best_val_loss={study.best_trial.value:.6f})")
    RUN_DIR = prev_run_dir
    return best_lr, best_gate_lr


# Entry point
def main():
    global RUN_DIR

    variants = _resolve_compare_modes()

    _reset_torch_state()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = datetime.now().strftime(RUN_DIR_TS_FMT)
    n_values = tuple(int(n) for n in N_VALUES)
    if not n_values:
        raise RuntimeError("N_VALUES must contain at least one entry.")

    # Top-level dir holding one parent_run_dir per N (single split) or one fold_<k>/ per CV fold.
    cross_root = os.path.join(RUN_ROOT, f"static_cal_xN_{ts}")
    os.makedirs(cross_root, exist_ok=True)
    RUN_DIR = cross_root

    setup_logging(prefix="train")
    log_selected_model()
    logging.info(f"Using device: {device}")
    logging.info(f"Cross-N root directory: {cross_root}")
    logging.info(f"N values to evaluate (in order): {n_values}")
    logging.info(f"Compare variants per N (in order): {variants}")

    import test as test_mod

    # The bundle is the split, so preprocessing and calibration selection are skipped.
    if int(USE_KFOLD_BUNDLE) == 1:
        run_kfold_bundle(device=device, test_mod=test_mod)
        return

    # Shared (split-independent) preprocessing: NPZ + manifest built once.
    shared_split_path = os.path.join(cross_root, "split_subjects.json")
    shared_manifest_path = os.path.join(cross_root, "processed_manifest.csv")
    shared_abp_stats_path = os.path.join(cross_root, "abp_normalization_stats.json")

    logging.info("Start data preprocessing (shared across N values and variants)...")
    full_train_list, full_val_list, full_test_list = preprocess.data_preprocessing(
        shared_split_path, shared_manifest_path,
        force_rebuild=FORCE_REBUILD_PROCESSED, data_usage_ratio=data_usage_ratio,
        ratios=(train_ratio, val_ratio, test_ratio),
        summary_csv_path=SUBJECT_FILE_MAP_OUTPUT_PATH,
        min_windows=CALIBRATION_WINDOW_COUNT,
    )

    manifest = preprocess.manifest()
    base_manifest = manifest.copy() if manifest is not None else None

    # Optional LR search on one fold before the real run.
    if int(USE_OPTUNA_LR_SEARCH) == 1 and int(OPTUNA_PER_FOLD) != 1:
        tuned = run_optuna_lr_search(
            cross_root=cross_root, base_manifest=base_manifest, device=device,
        )
        if int(OPTUNA_THEN_KFOLD) != 1:
            logging.info("OPTUNA_THEN_KFOLD=0 — stopping after the LR search.")
            return
        _install_tuned_lr(*tuned)
        RUN_DIR = cross_root

    if USE_KFOLD:
        # Subject-wise stratified k-fold; ABP stats recomputed per fold.
        _run_kfold_cv(
            n_values=n_values, cross_root=cross_root,
            base_manifest=base_manifest, variants=variants,
            test_mod=test_mod, device=device,
        )
    else:
        # Single 70/10/20 split (original behaviour).
        logging.info("Compute training ABP stats (pre-filter, full train_list)...")
        compute_global_abp_stats(full_train_list)
        save_abp_stats(shared_abp_stats_path)
        _run_cross_n(
            n_values=n_values,
            full_train_list=full_train_list, full_val_list=full_val_list,
            full_test_list=full_test_list, base_manifest=base_manifest,
            run_root=cross_root, split_path=shared_split_path,
            variants=variants, test_mod=test_mod, device=device,
        )

    RUN_DIR = cross_root
    logging.info("Done!")


if __name__ == "__main__":
    if int(TEST_ONLY_MODE) == 1:
        import test as test_mod
        test_mod.main_test_only()
    else:
        main()
