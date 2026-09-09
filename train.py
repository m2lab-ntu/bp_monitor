import os, glob, json, shutil, re
import gzip
import sys
import logging
import random
import hashlib
import math
import subprocess
from datetime import datetime


if __name__ == "__main__":
    sys.modules["train"] = sys.modules[__name__]
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import joblib

import models as model_lib

# unet_cal_mamba  unet_cal  unet_cal_transformer  unet_cal_abp_ft  unet_cal_bp_ft
# unet_mamba_skip  unet_mamba_decoder  unet_transformer_skip  unet_transformer_decoder
# unet_plain_nocal
MODEL = "unet_cal_mamba"

model_lib.select(MODEL)

from models import CalibrationUNet, PlainUNet3LNoCal, WaveformLoss
from models import FT_ABP_WAVE, FT_BP_SCALAR


data_paths = [
    "/media/chen/disk2/Seg3min_Time",
    "/media/chen/disk2/Seg3min2_Time",
]

processed_dir = "/home/chen/Desktop/processed_minmax_Time_V3"
os.makedirs(processed_dir, exist_ok=True)

REUSE_PROCESSED = True
SPLIT_MODE = "subject"            
DATALOADER_WORKERS = min(8, (os.cpu_count() or 8))
PIN_MEMORY = True
PERSISTENT_WORKERS = True

USE_COMPILE = True

SUBJECT_FILE_MAP_PATHS = [
    "/media/chen/disk2/Seg3min_Time/subject_file_map.csv",
    "/media/chen/disk2/Seg3min2_Time/subject_file_map2.csv",
]
SUBJECT_FILE_MAP_OUTPUT_PATH = os.path.join(processed_dir, "subject_file_map_result.csv")

data_usage_ratio = 1.0
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
DATA_SAMPLE_RATIO = 1.0
DATA_SAMPLE_SEED = 42

# Subject-wise folds, so no subject spans train and test.
USE_KFOLD = True
KFOLD_N_SPLITS = 5
KFOLD_VAL_RATIO = 0.125            
KFOLD_SEED = 42
KFOLD_STRATIFY_KEY = "sbp"
KFOLD_STRATIFY_MAX_WINDOWS = 40

# USE_KFOLD_BUNDLE = 1 reads the fold definition from a bundle instead of computing it.
USE_KFOLD_BUNDLE = 0
KFOLD_BUNDLE_DIR = "../kfold_N100_shared_bundle"
KFOLD_NPZ_ROOT = processed_dir
KFOLD_BUNDLE_FOLDS = (0, 1, 2, 3, 4)
KFOLD_BUNDLE_NPZ_CHECK = 5

USE_OPTUNA_LR_SEARCH = 0
OPTUNA_LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
OPTUNA_EPOCHS = 5          
OPTUNA_FOLD = 0            # which fold the search runs on when OPTUNA_PER_FOLD = 0
OPTUNA_PER_FOLD = 1        # 1 -> every fold searches its own lr
OPTUNA_THEN_KFOLD = 1      # 1 -> continue into the full k-fold run

SEGMENT_LEN, STEP_SIZE = 1024, 1024
batch_size, epochs, learning_rate = 256, 30, 2e-5
APPLY_FILTERING, FS = True, 125

WORKERS = max(1, min(32, (os.cpu_count() or 8) - 1))

train_list, val_list, test_list = [], [], []

GLOBAL_ABP_MIN = None
GLOBAL_ABP_MAX = None
NPZ_COMPRESS = False

DROPOUT_P = 0.2
BASE_CHANNELS = 128

REGRESSION_TARGETS = ("sbp", "dbp")

USE_MULTITASK_SBP_DBP = True
MT_SBP_LOSS_WEIGHT = 0.1
MT_DBP_LOSS_WEIGHT = 0.1

USE_BOTTLENECK_RIDGE_PROBE = True
BOTTLENECK_RIDGE_USE_TRAIN_VAL = True

MODEL_SPEC = model_lib.get_spec()
MODEL_NAME = MODEL_SPEC.key

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

NO_CAL_BASELINES = tuple(MODEL_SPEC.no_cal_baselines)

# unet_transformer_skip deep-skip gate: 0 = Transformer, 1 = Conformer.
USE_CONFORMER_GATE = 0

# Gates read their module's SKIP_REFINER at construction, so set it before use.
SKIP_REFINER = MODEL_SPEC.skip_refiner
if SKIP_REFINER is not None:
    setattr(model_lib.load(), "SKIP_REFINER", SKIP_REFINER)

_MODULE_HAS_GATE_SWITCH = hasattr(model_lib, "USE_CONFORMER_GATE")
_GATE_KWARGS = ({"use_conformer_gate": int(USE_CONFORMER_GATE)}
                if _MODULE_HAS_GATE_SWITCH else {})

# Only the adapter models expose ResidualFeatureAdapter1D, so import it lazily.
if USE_PER_MASTER_ADAPTER:
    ResidualFeatureAdapter1D = model_lib.ResidualFeatureAdapter1D

CALIBRATION_WINDOW_COUNT = 100

N_VALUES = (100,)

# True: pool_end = max(t_min, t_max - max_gap_windows + 1); False: legacy first-N (pool_end = N).
CALIBRATION_POOL_JUST_BEFORE_FIRST_TARGET = True

CALIBRATION_MAX_GAP_MINUTES = 1440.0

# Empty = the selected model's own run mode.
RUN_MODE_OVERRIDE = ""
RUN_MODE = RUN_MODE_OVERRIDE or MODEL_SPEC.run_mode
if RUN_MODE not in model_lib.RUN_MODES:
    raise RuntimeError(
        f"Unknown RUN_MODE {RUN_MODE!r}. Valid: {', '.join(model_lib.RUN_MODES)}"
    )

# Every variant reuses the same filtered target list; no-cal models ignore the cal inputs.

CHECKPOINT_BASENAME_CAL = "best_calibration_unet.pth"
CHECKPOINT_BASENAME_NO_CAL = "best_plain_unet_nocal.pth"

VARIANT_CALIBRATED = "calibrated"
VARIANT_NO_CALIBRATION = "no_calibration"

VARIANT_NO_CAL_UNET_MAMBA = "no_cal_unet_mamba"           # Mamba d_state=8
VARIANT_NO_CAL_UNET_MAMBA_D16 = "no_cal_unet_mamba_d16"   # Mamba d_state=16
VARIANT_NO_CAL_UNET_TRANSFORMER = "no_cal_unet_transformer"

CHECKPOINT_BASENAME_NO_CAL_MAMBA = "best_no_cal_unet_mamba.pth"
CHECKPOINT_BASENAME_NO_CAL_MAMBA_D16 = "best_no_cal_unet_mamba_d16.pth"
CHECKPOINT_BASENAME_NO_CAL_TRANSFORMER = "best_no_cal_unet_transformer.pth"

_NO_CAL_BASELINE_VARIANTS = {
    "mamba": VARIANT_NO_CAL_UNET_MAMBA,
    "mamba_d16": VARIANT_NO_CAL_UNET_MAMBA_D16,
    "transformer": VARIANT_NO_CAL_UNET_TRANSFORMER,
}

# Test-only: reuse a run dir's artifacts, no training or re-split.
TEST_ONLY_MODE = 0
TEST_ONLY_RUN_DIR = ""              # e.g. "runs/calibration_20260101-120000"
TEST_ONLY_VISUALIZE = True
TEST_ONLY_RUN_RIDGE = True
TEST_ONLY_PLOT_TRUEPRED_SCATTER = True

RUN_DIR: str = ""
RUN_ROOT = "runs"
RUN_DIR_PREFIX = "calibration_"
RUN_DIR_TS_FMT = "%Y%m%d-%H%M%S"

# Backwards-compat: kept so code paths that read CHECKPOINT_BASENAME still resolve.
CHECKPOINT_BASENAME = CHECKPOINT_BASENAME_CAL

GLOBAL_SEED = 42
SEED_DETERMINISTIC = 0

USE_SIGNAL_STATS = True

ENABLE_PLOT_PRETRAIN_DISTRIBUTIONS = False
ENABLE_PLOT_BA_SCATTER = True
ENABLE_PLOT_TRUEPRED_SCATTER = True

PLOT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"

# Sync model module configuration
model_lib.FS = FS
model_lib.DROPOUT_P = DROPOUT_P
model_lib.BASE_CHANNELS = BASE_CHANNELS
model_lib.USE_MULTITASK_SBP_DBP = USE_MULTITASK_SBP_DBP

# Runtime caches. target_npz_path -> cal pair dict.
_WINDOW_CAL_PAIRS: dict = {}
# pandas DataFrame; columns: npz_path, sid, source_csv, master_record, local_window_index_in_csv.
_PROCESSED_MANIFEST: Optional[pd.DataFrame] = None
_PATH_TO_META: dict = {}


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


# Signal filtering, normalization, alignment
def normalize_abp(abp):
    return (abp - GLOBAL_ABP_MIN) / (GLOBAL_ABP_MAX - GLOBAL_ABP_MIN)


def denormalize_abp(abp_norm):
    return abp_norm * (GLOBAL_ABP_MAX - GLOBAL_ABP_MIN) + GLOBAL_ABP_MIN


def create_bandpass_filter(lowcut=0.5, highcut=8.0, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


b_bp, a_bp = create_bandpass_filter()


def hampel_filter(ppg, window_size=7, n_std=3):
    df = pd.DataFrame(ppg, columns=["ppg"])
    med = df["ppg"].rolling(window_size, center=True, min_periods=1).median()
    std = df["ppg"].rolling(window_size, center=True, min_periods=1).std()
    diff = np.abs(df["ppg"] - med)
    df.loc[diff > n_std * std, "ppg"] = med
    return df["ppg"].values


def cross_correlation_align(ABP, PPG, ECG, dPPG, max_lag_sec=0.6,
                            raw_idx=None, master_idx=None):
    length = min(len(ABP), len(PPG), len(ECG), len(dPPG))
    if length < 2:
        return ABP, PPG, ECG, dPPG, raw_idx, master_idx
    ABP_, PPG_ = ABP[:length].astype(np.float32), PPG[:length].astype(np.float32)
    ECG_, dPPG_ = ECG[:length].astype(np.float32), dPPG[:length].astype(np.float32)
    raw_ = raw_idx[:length] if raw_idx is not None else None
    master_ = master_idx[:length] if master_idx is not None else None

    max_lag = int(max(1, round(max_lag_sec * FS)))
    abp_c = ABP_ - ABP_.mean()
    ppg_c = PPG_ - PPG_.mean()
    abp_std = np.std(abp_c) + 1e-6
    ppg_std = np.std(ppg_c) + 1e-6

    best_lag, best_score = 0, -1e9
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = abp_c[lag:]
            p = ppg_c[:len(a)]
        else:
            p = ppg_c[-lag:]
            a = abp_c[:len(p)]
        if len(a) < 8:
            continue
        score = float(np.dot(a, p) / (len(a) * abp_std * ppg_std))
        if score > best_score:
            best_score, best_lag = score, lag

    if best_lag > 0:
        # Search used a = abp_c[lag:], p = ppg_c[:len(a)]: drop first best_lag samples of ABP.
        ABP_ = ABP_[best_lag:]
        L_a = len(ABP_)
        PPG_, ECG_, dPPG_ = PPG_[:L_a], ECG_[:L_a], dPPG_[:L_a]
        if raw_ is not None:
            raw_ = raw_[best_lag:best_lag + L_a]
        if master_ is not None:
            master_ = master_[best_lag:best_lag + L_a]
    elif best_lag < 0:
        # Search used p = ppg_c[-lag:], a = abp_c[:len(p)]: drop first -best_lag samples of PPG/ECG/dPPG.
        s = -best_lag
        PPG_, ECG_, dPPG_ = PPG_[s:], ECG_[s:], dPPG_[s:]
        L_a = len(PPG_)
        ABP_ = ABP_[:L_a]
        if raw_ is not None:
            raw_ = raw_[:L_a]
        if master_ is not None:
            master_ = master_[:L_a]

    L = min(len(ABP_), len(PPG_), len(ECG_), len(dPPG_))
    ABP_, PPG_, ECG_, dPPG_ = ABP_[:L], PPG_[:L], ECG_[:L], dPPG_[:L]
    if raw_ is not None:
        raw_ = raw_[:L]
    if master_ is not None:
        master_ = master_[:L]
    return ABP_, PPG_, ECG_, dPPG_, raw_, master_


# subject_file_map I/O
def read_subject_file_map(map_csv_path):
    if not os.path.isfile(map_csv_path):
        raise FileNotFoundError(f"subject_file_map file not found: {map_csv_path}")
    df = pd.read_csv(map_csv_path)
    if "subject_id" not in df.columns:
        raise ValueError("subject_file_map missing column: subject_id")
    if ("input_file" not in df.columns) and ("output_csv" not in df.columns):
        raise ValueError("subject_file_map requires at least one column: input_file or output_csv")

    base_dirs = [os.path.dirname(os.path.abspath(map_csv_path))]
    for d in (data_paths if isinstance(data_paths, (list, tuple)) else []):
        if isinstance(d, str):
            base_dirs.append(d)
    base_dirs = list(dict.fromkeys(base_dirs))

    def resolve_path(name):
        if isinstance(name, str) and os.path.isabs(name) and os.path.isfile(name):
            return os.path.abspath(name)
        if isinstance(name, str):
            for b in base_dirs:
                cand = os.path.abspath(os.path.join(b, name))
                if os.path.isfile(cand):
                    return cand
        return None

    rows = []
    for _, r in df.iterrows():
        sid = str(r.get("subject_id", "")).strip()
        if not sid or sid.lower() == "unknown":
            continue
        inp = str(r.get("input_file", "")).strip() if "input_file" in df.columns else ""
        out = str(r.get("output_csv", "")).strip() if "output_csv" in df.columns else ""

        csv_name = out if out else (os.path.splitext(inp)[0] + ".csv" if inp else "")
        if not csv_name:
            continue

        csv_path = resolve_path(csv_name)
        if csv_path is None and inp:
            csv_path = resolve_path(os.path.splitext(inp)[0] + ".csv")

        if csv_path is None:
            continue
        if not out:
            out = os.path.basename(csv_path)

        rows.append({
            "subject_id": sid,
            "input_file": inp,
            "output_csv": out,
            "csv_path": csv_path,
        })

    if not rows:
        raise RuntimeError(
            "No valid CSV paths parsed in subject_file_map.csv; check output_csv/filename and "
            f"directories: {base_dirs}"
        )

    used_df = pd.DataFrame(rows, columns=["subject_id", "input_file", "output_csv", "csv_path"]).drop_duplicates("csv_path")
    # Sort for deterministic downstream processing
    used_df = used_df.sort_values("csv_path").reset_index(drop=True)
    mapping = dict(zip(used_df["csv_path"], used_df["subject_id"]))
    return mapping, used_df


def read_subject_file_maps(map_csv_paths):
    rows = []
    merged_mapping = {}
    used_list = []
    for p in map_csv_paths:
        try:
            mp_map, mp_df = read_subject_file_map(p)
            for k, v in mp_map.items():
                if k in merged_mapping and merged_mapping[k] != v:
                    logging.warning(f"Mapping conflict: {k} -> {merged_mapping[k]} vs {v}, keeping previous value")
                    continue
                merged_mapping[k] = v
            used_list.append(mp_df)
        except Exception as e:
            logging.warning(f"Failed to read {p}: {e}")
    if not merged_mapping:
        raise RuntimeError("No valid records read from any subject_file_map")
    used_df = pd.concat(used_list, axis=0, ignore_index=True).drop_duplicates("csv_path")
    used_df = used_df.sort_values("csv_path").reset_index(drop=True)
    return merged_mapping, used_df


def _read_master_record_lookup(map_csv_paths):
    lookup = {}
    for p in map_csv_paths or []:
        try:
            if not p or not os.path.isfile(p):
                continue
            df = pd.read_csv(p)
            if "subject_id" not in df.columns or "master_record" not in df.columns:
                continue
            csv_col = "output_csv" if "output_csv" in df.columns else (
                "input_file" if "input_file" in df.columns else None)
            if csv_col is None:
                continue
            for _, r in df.iterrows():
                sid = str(r.get("subject_id", "")).strip()
                csv_name = str(r.get(csv_col, "")).strip()
                master = str(r.get("master_record", "")).strip()
                if not sid or not csv_name or not master or master.lower() == "nan":
                    continue
                if csv_col == "input_file":
                    csv_name = os.path.splitext(csv_name)[0] + ".csv"
                csv_name = os.path.basename(csv_name)
                key = (sid, csv_name)
                if key in lookup and lookup[key] != master:
                    logging.warning(
                        f"master_record conflict for {key}: "
                        f"{lookup[key]} vs {master}, keeping previous"
                    )
                    continue
                lookup[key] = master
        except Exception as e:
            logging.warning(f"Failed to read master_record from {p}: {e}")
    return lookup


# Per-subject BP stats and stratified split
def _csv_avg_bp(csv_path):
    try:
        abp = pd.read_csv(csv_path, usecols=["ABP"], dtype=np.float32,
                          engine="c", memory_map=True).values.squeeze()
        abp = abp[np.isfinite(abp)]
        if len(abp) < 200:
            return (np.nan, np.nan)
        abp_s = np.convolve(abp, np.ones(5)/5, mode="same")
        dist = int(0.32 * FS)
        sys_idx, _ = find_peaks(abp_s, distance=dist)
        dia_idx, _ = find_peaks(-abp_s, distance=dist)
        if len(sys_idx) < 3 or len(dia_idx) < 3:
            return (np.nan, np.nan)
        return (float(np.mean(abp[sys_idx])), float(np.mean(abp[dia_idx])))
    except Exception:
        return (np.nan, np.nan)


def compute_subject_bp_stats(csv_files, file2subj):
    subj_sbp, subj_dbp, subj_cnt = {}, {}, {}
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        results = list(exe.map(_csv_avg_bp, csv_files))
        for csv_path, result in tqdm(zip(csv_files, results), total=len(csv_files),
                desc="Phase-A: compute subject BP stats"):
            sbp_m, dbp_m = result
            if not np.isfinite(sbp_m):
                continue
            subj = file2subj[csv_path]
            subj_sbp[subj] = subj_sbp.get(subj, 0.0) + sbp_m
            subj_dbp[subj] = subj_dbp.get(subj, 0.0) + dbp_m
            subj_cnt[subj] = subj_cnt.get(subj, 0) + 1
    stats = {}
    for s, c in subj_cnt.items():
        stats[s] = (subj_sbp[s] / c, subj_dbp[s] / c)
    return stats


def stratified_snake_split(subj_stats, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, sort_key="sbp"):
    subjects = sorted(subj_stats.keys())  # deterministic input order
    n = len(subjects)
    if n == 0:
        return set(), set(), set()
    ratios = np.array([max(0.0, float(train_ratio)), max(0.0, float(val_ratio)), max(0.0, float(test_ratio))], dtype=np.float64)
    s = ratios.sum()
    if s <= 0:
        ratios = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ratios /= s
    n_train = int(np.floor(n * ratios[0]))
    n_val   = int(np.floor(n * ratios[1]))
    n_test  = max(0, n - n_train - n_val)
    quotas = np.array([n_train, n_val, n_test], dtype=np.int64)

    def key_fn(sid):
        sbp, dbp = subj_stats.get(sid, (np.nan, np.nan))
        if sort_key == "dbp":
            return dbp if np.isfinite(dbp) else np.inf
        elif sort_key == "mean":
            m = (sbp + dbp) / 2.0 if (np.isfinite(sbp) and np.isfinite(dbp)) else np.nan
            return m if np.isfinite(m) else np.inf
        else:
            return sbp if np.isfinite(sbp) else np.inf

    valid = [sid for sid in subjects if np.isfinite(key_fn(sid))]
    invalid = [sid for sid in subjects if not np.isfinite(key_fn(sid))]
    valid.sort(key=lambda x: (key_fn(x), x))
    invalid.sort()
    ordered = valid + invalid

    credits = np.zeros(3, dtype=np.float64)
    bins = [[], [], []]
    for sid in ordered:
        for _ in range(3):
            credits += ratios
            masked = credits.copy()
            for i in range(3):
                if quotas[i] <= 0:
                    masked[i] = -1e18
            idx = int(np.argmax(masked))
            if quotas[idx] > 0:
                bins[idx].append(sid)
                quotas[idx] -= 1
                credits[idx] -= 1.0
                break
        else:
            idx = int(np.argmax([len(b) for b in bins]))
            bins[idx].append(sid)
    return set(bins[0]), set(bins[1]), set(bins[2])


# CSV reading + NPZ creation with metadata
_REQUIRED_CSV_COLS = ["PLETH", "II", "dPPG", "ABP"]
_OPTIONAL_CSV_COLS = ["raw_sample_idx", "master_record_raw_sample_idx"]


# CSV -> NPZ conversion
def read_and_align_csv(fp):
    has_raw_warn_msg = ""
    try:
        # Detect available columns first (cheap header read)
        header = pd.read_csv(fp, nrows=0)
        avail = set(header.columns)
        has_raw = "raw_sample_idx" in avail
        has_master = "master_record_raw_sample_idx" in avail
        usecols = list(_REQUIRED_CSV_COLS) + [c for c in _OPTIONAL_CSV_COLS if c in avail]
        dtypes = {c: np.float32 for c in _REQUIRED_CSV_COLS}
        if has_raw:
            dtypes["raw_sample_idx"] = np.int64
        if has_master:
            dtypes["master_record_raw_sample_idx"] = np.int64

        df = pd.read_csv(fp, usecols=usecols, dtype=dtypes, engine="c",
                         memory_map=True, low_memory=False)
        df.dropna(subset=_REQUIRED_CSV_COLS, inplace=True)
        df = df[(df[_REQUIRED_CSV_COLS] != 0).all(axis=1)]
        if len(df) < SEGMENT_LEN:
            return np.array([]), np.array([]), None, None, ""

        ppg = df["PLETH"].values
        ecg = df["II"].values
        dppg = df["dPPG"].values
        abp = df["ABP"].values
        raw_idx = df["raw_sample_idx"].values.astype(np.int64) if has_raw else None
        master_idx = df["master_record_raw_sample_idx"].values.astype(np.int64) if has_master else None

        if not has_raw:
            has_raw_warn_msg = (f"WARN: {os.path.basename(fp)} missing raw_sample_idx; "
                                f"falling back without time anchor")

        a, p, e, d, r_idx, m_idx = cross_correlation_align(
            abp, ppg, ecg, dppg, raw_idx=raw_idx, master_idx=master_idx
        )
        if len(a) < SEGMENT_LEN:
            return np.array([]), np.array([]), None, None, has_raw_warn_msg

        X = np.column_stack([p, e, d])  # shape [N,3]
        Y = a
        return X, Y, r_idx, m_idx, has_raw_warn_msg
    except Exception as err:
        logging.error(f"Failed to read {os.path.basename(fp)}: {err}")
        return np.array([]), np.array([]), None, None, ""


def chunk_data_with_index(X, Y, raw_idx, master_idx,
                          seg_len=SEGMENT_LEN, step=STEP_SIZE):
    N = len(Y)
    for i in range(0, N - seg_len + 1, step):
        seg_X = X[i:i + seg_len]
        seg_Y = Y[i:i + seg_len]
        if not np.isfinite(seg_X).all() or not np.isfinite(seg_Y).all():
            continue
        if np.std(seg_X[:, 0]) <= 0.01:
            continue
        end_pos = i + seg_len - 1
        raw_s = int(raw_idx[i]) if raw_idx is not None else None
        raw_e = int(raw_idx[end_pos]) if raw_idx is not None else None
        m_s = int(master_idx[i]) if master_idx is not None else None
        m_e = int(master_idx[end_pos]) if master_idx is not None else None
        yield seg_X, seg_Y, i, end_pos, raw_s, raw_e, m_s, m_e


def _short_hash(s: str) -> str:
    try:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest()
    except Exception:
        return hex(abs(hash(s)) & 0xFFFFFFFF)[2:]


def process_single_csv(args):
    csv_path, subj_id = args
    out_paths = []
    X_all, Y_all, raw_idx_all, master_idx_all, raw_warn = read_and_align_csv(csv_path)
    if Y_all.size == 0:
        return out_paths, raw_warn

    source_csv_basename = os.path.basename(csv_path)
    safe_subj = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(subj_id))
    h = _short_hash(os.path.abspath(csv_path))

    local_id = 0
    for seg_X, seg_Y, _, _, raw_s, raw_e, m_s, m_e in chunk_data_with_index(
            X_all, Y_all, raw_idx_all, master_idx_all):
        abp = seg_Y
        mn, mx = float(abp.min()), float(abp.max())
        if mx - mn < 5.0 or mn < 30 or mx > 250:
            continue
        abp_cv = float(np.std(abp) / (np.mean(abp) + 1e-6))
        if abp_cv < 0.05:
            continue

        out_p = os.path.join(processed_dir,
                             f"{safe_subj}_{source_csv_basename[:-4]}_{h}_{local_id}.npz")

        npz_payload = dict(
            X=seg_X.astype(np.float32),
            Y=seg_Y.astype(np.float32),
            abp_min=np.float32(mn),
            abp_max=np.float32(mx),
            sid=np.array(str(subj_id)),
            source_csv=np.array(source_csv_basename),
            local_window_index_in_csv=np.int64(local_id),
            subject_window_index=np.int64(-1),  # filled by manifest pass
            raw_start_idx=np.int64(raw_s) if raw_s is not None else np.int64(-1),
            raw_end_idx=np.int64(raw_e) if raw_e is not None else np.int64(-1),
            master_raw_start_idx=np.int64(m_s) if m_s is not None else np.int64(-1),
            master_raw_end_idx=np.int64(m_e) if m_e is not None else np.int64(-1),
            window_start_time_sec=(np.float64(raw_s) / FS) if raw_s is not None else np.float64(-1.0),
            window_end_time_sec=(np.float64(raw_e) / FS) if raw_e is not None else np.float64(-1.0),
            has_raw_idx=np.int64(1 if raw_s is not None else 0),
        )

        if NPZ_COMPRESS:
            np.savez_compressed(out_p, **npz_payload)
        else:
            np.savez(out_p, **npz_payload)
        out_paths.append(out_p)
        local_id += 1

    return out_paths, raw_warn


# Split files and processed manifest
def save_split_json(train_subj, val_subj, test_subj, path):
    try:
        with open(path, "w") as f:
            json.dump({
                "train": sorted(list(train_subj)),
                "val": sorted(list(val_subj)),
                "test": sorted(list(test_subj)),
            }, f)
        logging.info(f"Saved subject splits to: {path}")
    except Exception as e:
        logging.warning(f"Failed to save splits: {e}")


def load_split_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return set(d.get("train", [])), set(d.get("val", [])), set(d.get("test", []))
    except Exception as e:
        logging.warning(f"Failed to read splits: {e}")
        return None


def _npz_meta_minimal(npz_path):
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            keys = set(getattr(d, 'files', []))
            sid = str(d["sid"].item()) if d["sid"].ndim == 0 else str(d["sid"])
            source_csv = (str(d["source_csv"].item()) if "source_csv" in keys
                          and d["source_csv"].ndim == 0
                          else (str(d["source_csv"]) if "source_csv" in keys else ""))
            local_idx = int(d["local_window_index_in_csv"]) if "local_window_index_in_csv" in keys else -1
            raw_start = int(d["raw_start_idx"]) if "raw_start_idx" in keys else -1
            raw_end = int(d["raw_end_idx"]) if "raw_end_idx" in keys else -1
            m_start = int(d["master_raw_start_idx"]) if "master_raw_start_idx" in keys else -1
            m_end = int(d["master_raw_end_idx"]) if "master_raw_end_idx" in keys else -1
        return {
            "sid": sid,
            "source_csv": source_csv,
            "local_window_index_in_csv": local_idx,
            "raw_start_idx": raw_start,
            "raw_end_idx": raw_end,
            "master_raw_start_idx": m_start,
            "master_raw_end_idx": m_end,
            "_has_meta": (source_csv != "" and local_idx >= 0),
        }
    except Exception as e:
        logging.warning(f"Failed to read NPZ metadata for {npz_path}: {e}")
        return None


def _attach_master_record_column(df):
    if df is None or len(df) == 0:
        return df
    lookup = _read_master_record_lookup(SUBJECT_FILE_MAP_PATHS)
    if lookup:
        master_col = []
        miss = 0
        for s, c in zip(df["sid"].astype(str).tolist(),
                        df["source_csv"].astype(str).tolist()):
            base = os.path.basename(c)
            v = lookup.get((s, base))
            if v is None:
                miss += 1
                v = base  # fallback: 1 source_csv = 1 master
            master_col.append(v)
        if miss:
            logging.warning(
                f"master_record lookup missing for {miss} manifest rows; "
                f"falling back to source_csv as master_record for those rows."
            )
        df = df.copy()
        df["master_record"] = master_col
    else:
        logging.warning(
            "No master_record column found in subject_file_map.csv; "
            "treating each source_csv as its own master_record."
        )
        df = df.copy()
        df["master_record"] = df["source_csv"].astype(str).apply(os.path.basename)
    return df


def _attach_record_and_subject_indices(df):
    if df is None or len(df) == 0:
        return df
    if "master_record" not in df.columns:
        df = _attach_master_record_column(df)

    df = df.copy()
    if "master_raw_start_idx" not in df.columns:
        df["master_raw_start_idx"] = -1
    df["_sort_master"] = df["master_raw_start_idx"].where(
        df["master_raw_start_idx"] >= 0, other=2**62
    )
    df["_sort_raw"] = df["raw_start_idx"].where(df["raw_start_idx"] >= 0,
                                                other=2**62)

    # record_window_index within (sid, master_record), master-timeline first.
    df = df.sort_values(
        by=["sid", "master_record", "_sort_master", "_sort_raw",
            "source_csv", "local_window_index_in_csv", "npz_path"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["record_window_index"] = (
        df.groupby(["sid", "master_record"]).cumcount().astype(np.int64)
    )

    # subject_window_index across all master_records for that subject.
    df = df.sort_values(
        by=["sid", "_sort_raw", "master_record", "_sort_master",
            "source_csv", "local_window_index_in_csv", "npz_path"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["subject_window_index"] = (
        df.groupby("sid").cumcount().astype(np.int64)
    )

    df = df.drop(columns=["_sort_master", "_sort_raw"])
    return df


def build_or_update_processed_manifest(npz_files, output_manifest_path):
    npz_files_sorted = sorted(npz_files)
    rows = []
    for fp in tqdm(npz_files_sorted, desc="Manifest: scan NPZ"):
        meta = _npz_meta_minimal(fp)
        if meta is None:
            continue
        if not meta["_has_meta"]:
            raise RuntimeError(
                f"NPZ '{fp}' missing required metadata (source_csv / "
                f"local_window_index_in_csv). Legacy local_id is NOT a true "
                f"cumulative record index; per-subject cumulative-prefix "
                f"calibration requires processed_manifest "
                f"record_window_index. Set REUSE_PROCESSED=False to "
                f"rebuild NPZ."
            )
        rows.append({
            "npz_path": fp,
            "sid": meta["sid"],
            "source_csv": meta["source_csv"],
            "local_window_index_in_csv": meta["local_window_index_in_csv"],
            "raw_start_idx": meta["raw_start_idx"],
            "raw_end_idx": meta["raw_end_idx"],
            "master_raw_start_idx": meta["master_raw_start_idx"],
            "master_raw_end_idx": meta["master_raw_end_idx"],
        })

    if not rows:
        raise RuntimeError(f"No NPZ metadata could be read from {len(npz_files)} files")

    df = pd.DataFrame(rows)

    # Attach master_record (from subject_file_map.csv, falling back to source_csv basename).
    df = _attach_master_record_column(df)
    df = _attach_record_and_subject_indices(df)

    cols = ["npz_path", "sid", "source_csv", "master_record",
            "local_window_index_in_csv",
            "subject_window_index", "record_window_index",
            "raw_start_idx", "raw_end_idx",
            "master_raw_start_idx", "master_raw_end_idx"]
    df = df[cols]

    df.to_csv(output_manifest_path, index=False)

    n_rows = int(len(df))
    n_subjects = int(df["sid"].nunique())
    n_masters = int(df.groupby(["sid", "master_record"]).ngroups)
    sizes = df.groupby(["sid", "master_record"]).size()
    n_masters_usable = int((sizes >= CALIBRATION_WINDOW_COUNT).sum())
    logging.info(
        f"Saved processed manifest: {output_manifest_path} "
        f"({n_rows} windows, {n_subjects} subjects, {n_masters} master_records; "
        f"usable with >={CALIBRATION_WINDOW_COUNT} windows: {n_masters_usable})"
    )
    return df


def load_processed_manifest(manifest_path):
    if not os.path.isfile(manifest_path):
        return None
    df = pd.read_csv(manifest_path)
    required_cols = {"npz_path", "sid", "source_csv", "local_window_index_in_csv",
                     "raw_start_idx", "raw_end_idx"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"processed_manifest.csv missing columns: {missing}")
    # Older manifests may lack master_record / record_window_index / subject_window_index.
    if "master_record" not in df.columns:
        df["master_record"] = df["source_csv"].astype(str).apply(os.path.basename)
    if "record_window_index" not in df.columns or "subject_window_index" not in df.columns:
        df = _attach_record_and_subject_indices(df)
    return df


def install_manifest(df: pd.DataFrame):
    global _PROCESSED_MANIFEST, _PATH_TO_META
    _PROCESSED_MANIFEST = df
    _PATH_TO_META = {}
    has_master_record = "master_record" in df.columns
    has_record_idx = "record_window_index" in df.columns
    has_subject_idx = "subject_window_index" in df.columns
    for r in df.itertuples(index=False):
        _PATH_TO_META[r.npz_path] = {
            "sid": str(r.sid),
            "source_csv": str(r.source_csv),
            "master_record": str(getattr(r, "master_record", r.source_csv)) if has_master_record else str(r.source_csv),
            "local_window_index_in_csv": int(r.local_window_index_in_csv),
            "subject_window_index": int(getattr(r, "subject_window_index", -1)) if has_subject_idx else -1,
            "record_window_index": int(getattr(r, "record_window_index", -1)) if has_record_idx else -1,
            "raw_start_idx": int(r.raw_start_idx),
            "raw_end_idx": int(r.raw_end_idx),
            "master_raw_start_idx": int(getattr(r, "master_raw_start_idx", -1)) if hasattr(r, "master_raw_start_idx") else -1,
            "master_raw_end_idx": int(getattr(r, "master_raw_end_idx", -1)) if hasattr(r, "master_raw_end_idx") else -1,
        }


def _get_sid_from_path(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath]["sid"]
    # Fallback: read NPZ
    try:
        with np.load(fpath, allow_pickle=False) as d:
            sid = d["sid"]
            if isinstance(sid, np.ndarray) and sid.shape == ():
                return str(sid.item())
            return str(sid)
    except Exception:
        return None


def _get_subject_window_index(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath]["subject_window_index"]
    return -1


def _get_record_window_index(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath].get("record_window_index", -1)
    return -1


def _get_master_record(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath].get("master_record", "")
    return ""


def _get_raw_start_idx(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath]["raw_start_idx"]
    return -1


# Data lists from the manifest, k-fold partition
def _sample_list(lst, ratio, seed=DATA_SAMPLE_SEED):
    ratio = float(ratio)
    if ratio >= 1.0 or ratio <= 0.0 or len(lst) == 0:
        return lst
    n = max(1, int(len(lst) * ratio))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lst), size=n, replace=False)
    return [lst[i] for i in idx]


def build_lists_from_manifest(train_subj, val_subj, test_subj):
    if _PROCESSED_MANIFEST is None:
        raise RuntimeError("Manifest not installed; call install_manifest() first.")
    df = _PROCESSED_MANIFEST
    # Rows already sorted by (sid, subject_window_index)
    train_list = df.loc[df["sid"].isin(train_subj), "npz_path"].tolist()
    val_list = df.loc[df["sid"].isin(val_subj), "npz_path"].tolist()
    test_list = df.loc[df["sid"].isin(test_subj), "npz_path"].tolist()

    train_sids = df.loc[df["sid"].isin(train_subj), "sid"].nunique()
    val_sids = df.loc[df["sid"].isin(val_subj), "sid"].nunique()
    test_sids = df.loc[df["sid"].isin(test_subj), "sid"].nunique()
    logging.info(
        "Lists from manifest: "
        f"Train={len(train_list)} segments ({train_sids} subjects), "
        f"Val={len(val_list)} segments ({val_sids} subjects), "
        f"Test={len(test_list)} segments ({test_sids} subjects)"
    )

    train_list = _sample_list(train_list, DATA_SAMPLE_RATIO)
    val_list = _sample_list(val_list, DATA_SAMPLE_RATIO)
    test_list = _sample_list(test_list, DATA_SAMPLE_RATIO)
    return train_list, val_list, test_list


def infer_subject_splits_from_manifest(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    if _PROCESSED_MANIFEST is None:
        raise RuntimeError("Manifest not installed.")
    subjects = sorted(_PROCESSED_MANIFEST["sid"].unique().tolist())
    n = len(subjects)
    if n == 0:
        return set(), set(), set()
    ratios = np.array([max(0.0, float(train_ratio)),
                       max(0.0, float(val_ratio)),
                       max(0.0, float(test_ratio))], dtype=np.float64)
    s = ratios.sum()
    ratios = ratios / s if s > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n_train = int(np.floor(n * ratios[0]))
    n_val   = int(np.floor(n * ratios[1]))
    n_test  = max(0, n - n_train - n_val)
    quotas = np.array([n_train, n_val, n_test], dtype=np.int64)
    credits = np.zeros(3, dtype=np.float64)
    bins = [[], [], []]
    for sid in subjects:
        for _ in range(3):
            credits += ratios
            masked = credits.copy()
            for i in range(3):
                if quotas[i] <= 0:
                    masked[i] = -1e18
            idx = int(np.argmax(masked))
            if quotas[idx] > 0:
                bins[idx].append(sid)
                quotas[idx] -= 1
                credits[idx] -= 1.0
                break
        else:
            idx = int(np.argmax([len(b) for b in bins]))
            bins[idx].append(sid)
    return set(bins[0]), set(bins[1]), set(bins[2])


def _subject_bp_key_fn(subj_stats, sort_key="sbp"):
    def key_fn(sid):
        sbp, dbp = subj_stats.get(sid, (np.nan, np.nan))
        if sort_key == "dbp":
            v = dbp
        elif sort_key == "mean":
            v = ((sbp + dbp) / 2.0
                 if (np.isfinite(sbp) and np.isfinite(dbp)) else np.nan)
        else:  # 'sbp' (default)
            v = sbp
        return float(v) if np.isfinite(v) else np.inf
    return key_fn


def compute_subject_bp_stats_from_manifest(max_windows_per_subject=None):
    if _PROCESSED_MANIFEST is None:
        raise RuntimeError("Manifest not installed; cannot compute subject BP stats.")
    df = _PROCESSED_MANIFEST
    stats = {}
    for sid, grp in tqdm(df.groupby("sid", sort=True),
                         desc="K-fold: per-subject BP stats"):
        paths = grp["npz_path"].tolist()
        if max_windows_per_subject and len(paths) > int(max_windows_per_subject):
            idx = np.unique(
                np.linspace(0, len(paths) - 1, int(max_windows_per_subject))
                .astype(int)
            )
            paths = [paths[i] for i in idx]
        sbps, dbps = [], []
        for p in paths:
            try:
                with np.load(p) as d:
                    y = d["Y"]
                sbp_m, dbp_m, _ = _segment_bp_means_from_wave(y)
                if np.isfinite(sbp_m):
                    sbps.append(sbp_m)
                if np.isfinite(dbp_m):
                    dbps.append(dbp_m)
            except Exception:
                continue
        if sbps:
            stats[str(sid)] = (
                float(np.mean(sbps)),
                float(np.mean(dbps)) if dbps else np.nan,
            )
    return stats


def stratified_subject_kfold(subj_stats, all_subjects, n_splits,
                             sort_key="sbp", seed=KFOLD_SEED):
    subjects = sorted({str(s) for s in all_subjects})
    n = len(subjects)
    n_splits = max(2, min(int(n_splits), n)) if n >= 2 else 1

    if sort_key:
        key_fn = _subject_bp_key_fn(subj_stats, sort_key)
        ordered = sorted(subjects, key=lambda x: (key_fn(x), x))
    else:
        rng = random.Random(int(seed))
        ordered = subjects[:]
        rng.shuffle(ordered)

    folds = [[] for _ in range(n_splits)]
    for i, sid in enumerate(ordered):
        folds[i % n_splits].append(sid)
    return [set(f) for f in folds]


def make_kfold_split(all_subjects, test_subjects, val_ratio, subj_stats,
                     sort_key="sbp"):
    test_subj = {str(s) for s in test_subjects}
    non_test = [str(s) for s in all_subjects if str(s) not in test_subj]
    non_test_stats = {s: subj_stats.get(s, (np.nan, np.nan)) for s in non_test}
    vr = max(0.0, min(0.9, float(val_ratio)))
    train_subj, val_subj, leftover = stratified_snake_split(
        non_test_stats,
        train_ratio=1.0 - vr, val_ratio=vr, test_ratio=0.0,
        sort_key=(sort_key or "sbp"),
    )
    # stratified_snake_split assigns integer-floored quotas.
    train_subj = set(train_subj) | set(leftover)
    return train_subj, set(val_subj), set(test_subj)


# Preprocessing driver
def data_preprocessing(split_json_path: str, processed_manifest_path: str):
    old_npz = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))

    if REUSE_PROCESSED:
        if not old_npz:
            raise RuntimeError(f"REUSE_PROCESSED=True but no .npz in folder: {processed_dir}")

        # Try loading existing manifest
        manifest = load_processed_manifest(processed_manifest_path)
        if manifest is None:
            logging.info("No processed_manifest.csv found; rebuilding from NPZ metadata...")
            manifest = build_or_update_processed_manifest(old_npz, processed_manifest_path)
        else:
            logging.info(f"Loaded existing manifest: {processed_manifest_path}")
            # If new NPZ files exist that aren't in manifest, rebuild
            manifest_paths = set(manifest["npz_path"].tolist())
            missing = [p for p in old_npz if p not in manifest_paths]
            if missing:
                logging.info(f"{len(missing)} new NPZ files not in manifest; rebuilding...")
                manifest = build_or_update_processed_manifest(old_npz, processed_manifest_path)

        install_manifest(manifest)

        loaded = load_split_json(split_json_path)
        if loaded is not None:
            train_subj, val_subj, test_subj = loaded
            logging.info("Using existing subject splits.")
        else:
            logging.info("No subject splits found; infer from manifest.")
            train_subj, val_subj, test_subj = infer_subject_splits_from_manifest(
                train_ratio, val_ratio, test_ratio)
            save_split_json(train_subj, val_subj, test_subj, split_json_path)
        return build_lists_from_manifest(train_subj, val_subj, test_subj)

    # Full rebuild path
    mapping, used_df = read_subject_file_maps(SUBJECT_FILE_MAP_PATHS)
    csv_files = sorted(mapping.keys())
    if not csv_files:
        raise FileNotFoundError("No usable CSV files in subject_file_map.csv")
    csv_files = csv_files[:int(len(csv_files) * data_usage_ratio)]
    used_df = used_df[used_df["csv_path"].isin(csv_files)].copy()
    file2subj = mapping

    loaded = load_split_json(split_json_path)
    if loaded is not None and old_npz:
        train_subj, val_subj, test_subj = loaded
        logging.info("Using saved subject splits")
    else:
        subj_stats = compute_subject_bp_stats(csv_files, file2subj)
        train_subj, val_subj, test_subj = stratified_snake_split(
            subj_stats, train_ratio, val_ratio, test_ratio
        )
        save_split_json(train_subj, val_subj, test_subj, split_json_path)

    if old_npz:
        logging.info(f"Deleting {len(old_npz)} old NPZ files...")
        for f in old_npz:
            try:
                os.remove(f)
            except Exception:
                pass

    logging.info(f"Start processing {len(csv_files)} CSV files using {WORKERS} workers...")
    seg_counts = {fp: 0 for fp in csv_files}
    raw_warns = []

    with tqdm(total=len(csv_files), desc="Phase-B: processing progress") as pbar, \
         ProcessPoolExecutor(max_workers=WORKERS) as exe:
        future_to_file = {
            exe.submit(process_single_csv, (fp, file2subj[fp])): fp
            for fp in csv_files
        }
        for future in as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                paths, warn = future.result()
                seg_counts[fp] = len(paths)
                if warn:
                    raw_warns.append(warn)
            except Exception as e:
                logging.error(f"Error processing file {fp}: {str(e)}")
            pbar.update(1)

    if raw_warns:
        for w in raw_warns[:10]:
            logging.warning(w)
        logging.warning(f"Total CSVs missing raw_sample_idx: {len(raw_warns)}")

    summary_df = used_df.copy()
    summary_df["segments_saved"] = summary_df["csv_path"].map(seg_counts).fillna(0).astype(int)
    summary_df[["subject_id", "input_file", "output_csv", "segments_saved"]].to_csv(
        SUBJECT_FILE_MAP_OUTPUT_PATH, index=False
    )
    logging.info(f"Processing summary saved to: {SUBJECT_FILE_MAP_OUTPUT_PATH}")

    new_npz = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    if not new_npz:
        raise RuntimeError("No segments produced")
    manifest = build_or_update_processed_manifest(new_npz, processed_manifest_path)
    install_manifest(manifest)
    return build_lists_from_manifest(train_subj, val_subj, test_subj)


# ABP normalization stats
def compute_global_abp_stats(train_files):
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX
    all_mins, all_maxs = [], []
    for filepath in tqdm(train_files, desc="Compute training ABP stats"):
        try:
            data = np.load(filepath)
            Y = data['Y']
            all_mins.append(np.min(Y))
            all_maxs.append(np.max(Y))
        except Exception as e:
            logging.warning(f"Error processing file {filepath}: {e}")
            continue
    if not all_mins or not all_maxs:
        raise RuntimeError("Unable to compute ABP stats; data may be invalid")
    GLOBAL_ABP_MIN = float(np.min(all_mins))
    GLOBAL_ABP_MAX = float(np.max(all_maxs))
    logging.info(f"ABP range: {GLOBAL_ABP_MIN:.2f} - {GLOBAL_ABP_MAX:.2f} mmHg")
    return GLOBAL_ABP_MIN, GLOBAL_ABP_MAX


def save_abp_stats(path):
    with open(path, "w") as f:
        json.dump({"GLOBAL_ABP_MIN": GLOBAL_ABP_MIN,
                   "GLOBAL_ABP_MAX": GLOBAL_ABP_MAX}, f)
    logging.info(f"Saved normalization stats to {path}")


def _load_saved_norm_stats(path: str) -> bool:
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as f:
            d = json.load(f)
        GLOBAL_ABP_MIN = d.get("GLOBAL_ABP_MIN", GLOBAL_ABP_MIN)
        GLOBAL_ABP_MAX = d.get("GLOBAL_ABP_MAX", GLOBAL_ABP_MAX)
        if GLOBAL_ABP_MIN is not None and GLOBAL_ABP_MAX is not None:
            logging.info(f"Loaded normalization stats: {path}")
        return True
    except Exception as e:
        logging.warning(f"Failed to read normalization stats: {e}")
        return False


# Segment loading, datasets, BP readout
def load_segment(filepath, apply_filter=True):
    try:
        with np.load(filepath) as data:
            X = data['X']
            Y = data['Y']
            keys = set(getattr(data, 'files', []))
            abp_min = float(data['abp_min']) if 'abp_min' in keys else float(np.min(Y))
            abp_max = float(data['abp_max']) if 'abp_max' in keys else float(np.max(Y))

        if apply_filter and APPLY_FILTERING and X is not None:
            try:
                # Spike/outlier removal BEFORE bandpass: hampel suppresses impulse-like artifacts that would.
                X[:, 0] = hampel_filter(X[:, 0])
                X[:, 0] = filtfilt(b_bp, a_bp, X[:, 0])
            except Exception:
                pass

        if X is not None and Y is not None:
            target_len = max(1, (len(Y) // 16) * 16)
            if target_len < len(Y):
                X = X[:target_len, :]
                Y = Y[:target_len]
        return X, Y, abp_min, abp_max
    except Exception as e:
        logging.error(f"Failed to load {filepath}: {e}")
        return None, None, 0, 0


def normalize_ppg_ecg_segment(signals: np.ndarray):
    if signals is None or signals.ndim != 2 or signals.shape[1] < 2:
        stats = np.zeros(4, dtype=np.float32)
        return signals, stats
    normalized = signals.astype(np.float32).copy()
    stats = []
    for ch in (0, 1):
        sig = normalized[:, ch]
        s_min = float(np.min(sig))
        s_max = float(np.max(sig))
        denom = s_max - s_min
        if denom < 1e-6:
            normalized[:, ch] = 0.0
        else:
            normalized[:, ch] = (sig - s_min) / denom
        stats.extend([s_min, s_max])
    return normalized, np.asarray(stats, dtype=np.float32)


def smooth_signal(sig, win=5):
    if win < 2:
        return sig
    return np.convolve(sig, np.ones(win) / win, mode="same")


def detect_sbp_dbp(sig, fs=FS):
    s = smooth_signal(sig)
    dist = int(0.32 * fs)
    sys_idx, _ = find_peaks(s, distance=dist)
    dia_idx, _ = find_peaks(-s, distance=dist)
    return sig[sys_idx], sig[dia_idx]


def _bp_targets_from_wave(abp_wave: np.ndarray):
    sbp_vals, dbp_vals = detect_sbp_dbp(abp_wave)
    sbp_mean = float(np.mean(sbp_vals)) if len(sbp_vals) > 0 else float(np.max(abp_wave))
    dbp_mean = float(np.mean(dbp_vals)) if len(dbp_vals) > 0 else float(np.min(abp_wave))
    return sbp_mean, dbp_mean


def _segment_bp_means_from_wave(y: np.ndarray):
    try:
        sbp_vals, dbp_vals = detect_sbp_dbp(y)
        sbp_mean = float(np.mean(sbp_vals)) if len(sbp_vals) > 0 else np.nan
        dbp_mean = float(np.mean(dbp_vals)) if len(dbp_vals) > 0 else np.nan
        abp_mean = float(np.mean(y)) if y is not None and y.size > 0 else np.nan
        return sbp_mean, dbp_mean, abp_mean
    except Exception:
        return np.nan, np.nan, np.nan


# Calibration pair selection — per-master static first-N (no leakage from target ABP.
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
    if _PROCESSED_MANIFEST is None:
        raise RuntimeError("Manifest not installed; cannot select calibration pairs.")

    df = _PROCESSED_MANIFEST.copy()
    needs_attach = (
        "record_window_index" not in df.columns
        or "master_record" not in df.columns
        or "subject_window_index" not in df.columns
    )
    if needs_attach:
        df = _attach_record_and_subject_indices(df)

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

            # Master too short → unusable: no calibration pair, no targets.
            if n_total <= N:
                n_master_skipped_too_short += 1
                continue

            # Decide pool start position per master; static first-N means pool_start = 0.
            pool_start = 0
            n_dropped_this_master = 0
            if just_before_target:
                eligible_indices: list = []
                for i in range(N, n_total):
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

                if max_gap_windows is None:
                    pool_end_rwi = t_min
                else:
                    pool_end_rwi = max(t_min, t_max - max_gap_windows + 1)
                if pool_end_rwi > t_min:
                    n_master_gap_triggered_shift += 1
                    n_dropped_this_master = sum(
                        1 for i in eligible_indices
                        if int(rows.loc[i, "record_window_index"]) < pool_end_rwi
                    )
                    n_targets_dropped_by_gap += n_dropped_this_master

                pool_start = pool_end_rwi - N
                # Safety: index lookup uses sorted-rows position, which equals rwi for the canonical master.
                if pool_start < 0:
                    raise RuntimeError(
                        f"Computed pool_start={pool_start} < 0 for "
                        f"(sid={sid}, master={master}); this indicates "
                        f"non-contiguous rwis or a bug in placement."
                    )

            pool_end = pool_start + N  # exclusive

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
                    s, db = _bp_targets_from_wave(y)
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
        target_rwi = _get_record_window_index(tgt)
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
        if _get_record_window_index(fp) < CALIBRATION_WINDOW_COUNT:
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
        sbp, dbp = _bp_targets_from_wave(row)
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


def train_model(model, train_loader, val_loader, num_epochs, device,
                checkpoint_path: str, plot_curve: bool = True,
                *, use_calibration: bool = True,
                history_title: Optional[str] = None,
                lr: Optional[float] = None,
                return_best_val: bool = False):
    criterion = WaveformLoss()
    lr_used = float(lr) if lr is not None else float(learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=lr_used, weight_decay=1e-3, eps=1e-8)
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
    logging.info(f"  optuna LR search  : {'ON' if USE_OPTUNA_LR_SEARCH else 'OFF'}"
                 + (f"  {len(OPTUNA_LR_GRID)} fixed lr x {OPTUNA_EPOCHS} epochs"
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
    meta = _PATH_TO_META.get(fp)
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
    if _PROCESSED_MANIFEST is None:
        raise RuntimeError("Manifest not installed; cannot derive pool paths.")
    master_pool = {}
    for p in window_cal_pairs.values():
        key = (str(p["sid"]), str(p["master_record"]))
        ps = int(p.get("pool_start_rwi", 0))
        pe = int(p.get("pool_end_rwi", CALIBRATION_WINDOW_COUNT))
        master_pool[key] = (ps, pe)
    df = _PROCESSED_MANIFEST
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
    if GLOBAL_ABP_MIN is None or GLOBAL_ABP_MAX is None:
        raise RuntimeError(
            "GLOBAL_ABP_MIN / GLOBAL_ABP_MAX not initialized; cannot "
            "denormalize predicted waveform for adapter scalar fine-tune."
        )
    span = float(GLOBAL_ABP_MAX) - float(GLOBAL_ABP_MIN)
    pred_mmhg = pred_norm * span + float(GLOBAL_ABP_MIN)
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
                s, d = _bp_targets_from_wave(y_np_denorm[i])
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
        "GLOBAL_ABP_MIN": GLOBAL_ABP_MIN,
        "GLOBAL_ABP_MAX": GLOBAL_ABP_MAX,
        "BASE_CHANNELS": int(BASE_CHANNELS),
        "DROPOUT_P": float(DROPOUT_P),
        # Which models/*.py this run used, and the model-owned settings that came.
        "MODEL": MODEL_NAME,
        "MODEL_MODULE": f"models/{MODEL_SPEC.module}.py",
        "MODEL_SOURCE_PROJECT": MODEL_SPEC.source_project,
        "USE_PER_MASTER_ADAPTER": bool(USE_PER_MASTER_ADAPTER),
        "ADAPTER_FT_OBJECTIVE": ADAPTER_FT_OBJECTIVE or "",
        "NO_CAL_BASELINES": list(NO_CAL_BASELINES),
        "USE_CONFORMER_GATE": (int(USE_CONFORMER_GATE)
                               if _MODULE_HAS_GATE_SWITCH else None),
        "per_master_adapters_path": str(per_master_adapters_path or ""),
        "USE_MULTITASK_SBP_DBP": bool(USE_MULTITASK_SBP_DBP),
        "MT_SBP_LOSS_WEIGHT": float(MT_SBP_LOSS_WEIGHT),
        "MT_DBP_LOSS_WEIGHT": float(MT_DBP_LOSS_WEIGHT),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
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


def _build_calibration_unet(device):
    m = CalibrationUNet(input_channels=3, base_channels=BASE_CHANNELS).to(device)
    _set_bn_momentum(m, momentum=0.01)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            m = torch.compile(m)
            logging.info("Enabled torch.compile acceleration")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")
    return m


def _build_plain_unet_no_cal(device):
    m = PlainUNet3LNoCal(input_channels=3, base_channels=BASE_CHANNELS,
                         use_signal_stats=bool(USE_SIGNAL_STATS)).to(device)
    _set_bn_momentum(m, momentum=0.01)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            m = torch.compile(m)
            logging.info("Enabled torch.compile acceleration (no-cal model)")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")
    return m


def _compile_no_cal(m, tag):
    _set_bn_momentum(m, momentum=0.01)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            m = torch.compile(m)
            logging.info(f"Enabled torch.compile acceleration ({tag})")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")
    return m


def _build_mamba_unet_no_cal(device):
    m = model_lib.MambaUNet3LNoCal(
        input_channels=3, base_channels=BASE_CHANNELS,
        use_signal_stats=bool(USE_SIGNAL_STATS),
    ).to(device)
    return _compile_no_cal(m, "mamba no-cal model")


def _build_mamba_unet_no_cal_d16(device):
    m = model_lib.MambaUNet3LNoCal(
        input_channels=3, base_channels=BASE_CHANNELS,
        use_signal_stats=bool(USE_SIGNAL_STATS),
        ag_d_state=16,
    ).to(device)
    return _compile_no_cal(m, "mamba no-cal model, d_state=16")


def _build_transformer_unet_no_cal(device):
    m = model_lib.TransformerUNet3LNoCal(
        input_channels=3, base_channels=BASE_CHANNELS,
        use_signal_stats=bool(USE_SIGNAL_STATS),
        **_GATE_KWARGS,
    ).to(device)
    return _compile_no_cal(m, "transformer no-cal model")


# variant -> how to build it, what to call it, where its checkpoint goes.
_NO_CAL_VARIANT_SPEC = {
    VARIANT_NO_CAL_UNET_MAMBA: {
        "model_name": "MambaUNet3LNoCal",
        "class_name": "MambaUNet3LNoCal",
        "checkpoint_basename": CHECKPOINT_BASENAME_NO_CAL_MAMBA,
        "extra_kwargs": {"ag_d_state": 8},
        "train_builder": _build_mamba_unet_no_cal,
        "eval_factory": lambda device: model_lib.MambaUNet3LNoCal(
            input_channels=3, base_channels=BASE_CHANNELS,
            use_signal_stats=bool(USE_SIGNAL_STATS),
        ).to(device),
    },
    VARIANT_NO_CAL_UNET_MAMBA_D16: {
        "model_name": "MambaUNet3LNoCal_d16",
        "class_name": "MambaUNet3LNoCal",
        "checkpoint_basename": CHECKPOINT_BASENAME_NO_CAL_MAMBA_D16,
        "extra_kwargs": {"ag_d_state": 16},
        "train_builder": _build_mamba_unet_no_cal_d16,
        "eval_factory": lambda device: model_lib.MambaUNet3LNoCal(
            input_channels=3, base_channels=BASE_CHANNELS,
            use_signal_stats=bool(USE_SIGNAL_STATS),
            ag_d_state=16,
        ).to(device),
    },
    VARIANT_NO_CAL_UNET_TRANSFORMER: {
        "model_name": "TransformerUNet3LNoCal",
        "class_name": "TransformerUNet3LNoCal",
        "checkpoint_basename": CHECKPOINT_BASENAME_NO_CAL_TRANSFORMER,
        # Carried into test.py's test-only dispatch so a re-test rebuilds the same gate the checkpoint.
        "extra_kwargs": dict(_GATE_KWARGS),
        "train_builder": _build_transformer_unet_no_cal,
        "eval_factory": lambda device: model_lib.TransformerUNet3LNoCal(
            input_channels=3, base_channels=BASE_CHANNELS,
            use_signal_stats=bool(USE_SIGNAL_STATS),
            **_GATE_KWARGS,
        ).to(device),
    },
}


# Variant / N / fold orchestration
def _count_subjects(flist):
    sids = set()
    for fp in flist:
        sid = _get_sid_from_path(fp)
        if sid is not None:
            sids.add(sid)
    return len(sids)


def _count_master_records(flist):
    masters = set()
    for fp in flist:
        sid = _get_sid_from_path(fp)
        master = _get_master_record(fp)
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


def _run_one_variant(*, variant, parent_run_dir, train_list, val_list, test_list,
                     window_cal_pairs, n_subj_train, n_subj_val, n_subj_test,
                     n_master_train, n_master_val, n_master_test,
                     processed_manifest_df, device, test_mod,
                     shared_cal_summary_path: str = "",
                     shared_split_json_path: str = ""):
    global RUN_DIR, GLOBAL_ABP_MIN, GLOBAL_ABP_MAX

    use_calibration = (variant == VARIANT_CALIBRATED)
    sub_dir = os.path.join(parent_run_dir, variant)
    os.makedirs(sub_dir, exist_ok=True)
    RUN_DIR = sub_dir

    # Reset seed and CUDA/torch state per variant so the second variant in a no-cal-then-cal run.
    set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))
    if not bool(SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    no_cal_spec = _NO_CAL_VARIANT_SPEC.get(variant)
    if use_calibration:
        model_name = "CalibrationUNet"
        ckpt_basename = CHECKPOINT_BASENAME_CAL
    elif no_cal_spec is not None:
        model_name = no_cal_spec["model_name"]
        ckpt_basename = no_cal_spec["checkpoint_basename"]
    else:
        model_name = "PlainUNet3LNoCal"
        ckpt_basename = CHECKPOINT_BASENAME_NO_CAL

    logging.info("=" * 72)
    logging.info(f"Variant: {variant} (model={model_name})")
    logging.info(f"Variant run directory: {sub_dir}")
    logging.info("=" * 72)

    # Copy split_subjects.json from the shared parent dir into the variant subdir so each subdir.
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

    # Persist a copy of processed manifest annotated with split labels and the data lists.
    if processed_manifest_df is not None:
        manifest_run = processed_manifest_df.copy()
        manifest_run["split"] = "unused"
        target_paths = set(train_list) | set(val_list) | set(test_list)
        manifest_run.loc[manifest_run["npz_path"].isin(train_list), "split"] = "train"
        manifest_run.loc[manifest_run["npz_path"].isin(val_list), "split"] = "val"
        manifest_run.loc[manifest_run["npz_path"].isin(test_list), "split"] = "test"
        cal_only_paths = set()
        for _tgt, _p in (window_cal_pairs or {}).items():
            for k in ("cal_low_path", "cal_high_path"):
                cp = _p.get(k)
                if cp and cp not in target_paths:
                    cal_only_paths.add(cp)
        manifest_run.loc[manifest_run["npz_path"].isin(cal_only_paths), "split"] = "calibration"
        manifest_run.to_csv(processed_manifest_path, index=False)

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

        # Copy the rich calibration summary from the shared parent dir (produced once.
        if shared_cal_summary_path:
            if os.path.isfile(shared_cal_summary_path):
                shutil.copy2(shared_cal_summary_path, cal_summary_path)
                logging.info(f"Copied calibration summary to: {cal_summary_path}")
            else:
                logging.warning(
                    f"Shared calibration summary not found: {shared_cal_summary_path}"
                )

    # Synchronise calibration pair globals so downstream test/visualisation code reads the right.

    # ABP normalization stats are computed once in main() using the FULL (pre-filter) train_list.
    save_abp_stats(abp_stats_path)

    # Datasets / DataLoaders
    if use_calibration:
        train_dataset = CalibrationBPDataset(train_list, window_cal_pairs)
        val_dataset = CalibrationBPDataset(val_list, window_cal_pairs)
    else:
        train_dataset = BloodPressureDataset(train_list)
        val_dataset = BloodPressureDataset(val_list)
    dl_kwargs = _make_dataloader_kwargs()
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    if use_calibration:
        model = _build_calibration_unet(device)
    elif no_cal_spec is not None:
        model = no_cal_spec["train_builder"](device)
    else:
        model = _build_plain_unet_no_cal(device)

    logging.info(f"Start training {model_name}...")
    model = train_model(
        model, train_loader, val_loader, epochs, device, checkpoint_path,
        use_calibration=use_calibration,
        history_title=f"Training History — {model_name}",
    )

    # Reload best weights into a fresh, uncompiled model for evaluation.
    if use_calibration:
        eval_model = CalibrationUNet(input_channels=3, base_channels=BASE_CHANNELS).to(device)
    elif no_cal_spec is not None:
        eval_model = no_cal_spec["eval_factory"](device)
    else:
        eval_model = PlainUNet3LNoCal(input_channels=3, base_channels=BASE_CHANNELS,
                                      use_signal_stats=bool(USE_SIGNAL_STATS)).to(device)
    if os.path.isfile(checkpoint_path):
        _safe_load(eval_model, checkpoint_path, device)
    else:
        eval_model.load_state_dict(getattr(model, "_orig_mod", model).state_dict())
    eval_model.eval()

    # Per-master Residual Feature Adapter fine-tune (calibrated only) Performed on the uncompiled.
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

    # Leave the decoder on Identity so the next variant / fold starts from a clean slate and cannot.
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

        # Validation hooks (per the static-calibration spec): * cal pair invariance per (sid.
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

        n_subj_train = _count_subjects(train_list)
        n_subj_val = _count_subjects(val_list)
        n_subj_test = _count_subjects(test_list)
        n_master_train = _count_master_records(train_list)
        n_master_val = _count_master_records(val_list)
        n_master_test = _count_master_records(test_list)
        logging.info(
            f"[N={N}] subjects: train={n_subj_train}, "
            f"val={n_subj_val}, test={n_subj_test}"
        )
        logging.info(
            f"[N={N}] master_records: train={n_master_train}, "
            f"val={n_master_val}, test={n_master_test}"
        )
        n_usable_masters_total = sum(
            1 for _ in {(p["sid"], p["master_record"])
                        for p in window_cal_pairs.values()}
        )
        logging.info(
            f"[N={N}] usable master_records across train/val/test: "
            f"{n_usable_masters_total}; total target windows: "
            f"{len(window_cal_pairs)}"
        )

        # Annotate manifest copy with split labels per N and re-install so the per-variant subdir.
        processed_manifest_df = None
        if base_manifest is not None:
            manifest_run = base_manifest.copy()
            manifest_run["split"] = "unused"
            target_paths = set(train_list) | set(val_list) | set(test_list)
            manifest_run.loc[manifest_run["npz_path"].isin(train_list), "split"] = "train"
            manifest_run.loc[manifest_run["npz_path"].isin(val_list), "split"] = "val"
            manifest_run.loc[manifest_run["npz_path"].isin(test_list), "split"] = "test"
            cal_only_paths = set()
            for _tgt, _p in window_cal_pairs.items():
                for k in ("cal_low_path", "cal_high_path"):
                    cp = _p.get(k)
                    if cp and cp not in target_paths:
                        cal_only_paths.add(cp)
            manifest_run.loc[manifest_run["npz_path"].isin(cal_only_paths),
                             "split"] = "calibration"
            manifest_run.to_csv(
                os.path.join(parent_run_dir, "processed_manifest.csv"),
                index=False,
            )
            install_manifest(manifest_run)
            processed_manifest_df = manifest_run

        save_data_lists(os.path.join(parent_run_dir, "data_lists.json"),
                        train_list, val_list, test_list)

        # Run no-cal first then calibrated (or whatever variants resolved).
        variant_results = []
        for v in variants:
            res = _run_one_variant(
                variant=v,
                parent_run_dir=parent_run_dir,
                train_list=train_list, val_list=val_list, test_list=test_list,
                window_cal_pairs=window_cal_pairs,
                n_subj_train=n_subj_train, n_subj_val=n_subj_val,
                n_subj_test=n_subj_test,
                n_master_train=n_master_train, n_master_val=n_master_val,
                n_master_test=n_master_test,
                processed_manifest_df=processed_manifest_df,
                device=device, test_mod=test_mod,
                shared_cal_summary_path=cal_summary_path,
                shared_split_json_path=per_n_split_path,
            )
            variant_results.append(res)

        compare_csv = _save_compare_summary(parent_run_dir, variant_results)
        try:
            cross_rows.extend(pd.read_csv(compare_csv).to_dict("records"))
        except Exception as e:
            logging.warning(f"Failed to fold compare summary into cross-N: {e}")

        try:
            _plot_compare_bar(
                variant_results, N=int(N),
                out_path=os.path.join(parent_run_dir, "compare_bar.png"),
            )
        except Exception as e:
            logging.warning(f"Per-N comparison plot failed: {e}")

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
    global learning_rate
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX

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
            learning_rate = _optuna_lr_for_fold(
                f_train, f_val, f_test, fold_idx=fold_idx,
                out_dir=fold_root, device=device,
            )
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
    global RUN_DIR, train_list, val_list, test_list
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX, CALIBRATION_WINDOW_COUNT
    global _WINDOW_CAL_PAIRS, learning_rate

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

    all_rows: list = []
    for fold in KFOLD_BUNDLE_FOLDS:
        fold_data = load_kfold_fold(fold, npz_root, include_calibration=need_cal)
        fold_name = fold_data["fold"]

        # Install this fold's manifest / calibration pairs / ABP stats as the module globals.
        install_manifest(fold_data["manifest_df"].copy())
        GLOBAL_ABP_MIN = fold_data["abp_min"]
        GLOBAL_ABP_MAX = fold_data["abp_max"]
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
            learning_rate = _optuna_lr_for_fold(
                train_list, val_list, test_list, fold_idx=fold,
                out_dir=parent_run_dir, device=device,
                window_cal_pairs=window_cal_pairs,
            )

        if need_cal and window_cal_pairs:
            with open(os.path.join(parent_run_dir,
                                   "calibration_fixed_pairs.json"), "w") as f:
                json.dump({
                    "window_count": int(N),
                    "mode": (fold_data.get("calibration_mode")
                             or _active_calibration_mode_tag()),
                    "pairs_by_target_path": window_cal_pairs,
                }, f, indent=2)

        n_subj_train = _count_subjects(train_list)
        n_subj_val = _count_subjects(val_list)
        n_subj_test = _count_subjects(test_list)
        n_master_train = _count_master_records(train_list)
        n_master_val = _count_master_records(val_list)
        n_master_test = _count_master_records(test_list)
        logging.info("#" * 72)
        logging.info(f"### {fold_name} (from bundle) ###")
        logging.info(f"### windows: train={len(train_list)}, "
                     f"val={len(val_list)}, test={len(test_list)} ###")
        logging.info(f"### subjects: train={n_subj_train}, "
                     f"val={n_subj_val}, test={n_subj_test} ###")
        logging.info(f"### master_records: train={n_master_train}, "
                     f"val={n_master_val}, test={n_master_test} ###")
        logging.info("#" * 72)

        processed_manifest_df = fold_data["manifest_df"].copy()
        processed_manifest_df.to_csv(
            os.path.join(parent_run_dir, "processed_manifest.csv"), index=False
        )
        save_data_lists(os.path.join(parent_run_dir, "data_lists.json"),
                        train_list, val_list, test_list)

        variant_results = []
        for v in variants:
            res = _run_one_variant(
                variant=v,
                parent_run_dir=parent_run_dir,
                train_list=train_list, val_list=val_list, test_list=test_list,
                window_cal_pairs=window_cal_pairs,
                n_subj_train=n_subj_train, n_subj_val=n_subj_val,
                n_subj_test=n_subj_test,
                n_master_train=n_master_train, n_master_val=n_master_val,
                n_master_test=n_master_test,
                processed_manifest_df=processed_manifest_df,
                device=device, test_mod=test_mod,
                shared_cal_summary_path=cal_summary_path,
                shared_split_json_path=per_n_split_path,
            )
            variant_results.append(res)

        compare_csv = _save_compare_summary(parent_run_dir, variant_results)
        try:
            for row in pd.read_csv(compare_csv).to_dict("records"):
                all_rows.append({"fold": fold_name, "N": int(N), **row})
        except Exception as e:
            logging.warning(f"[{fold_name}] compare summary fold-in failed: {e}")

        try:
            _plot_compare_bar(
                variant_results, N=int(N),
                out_path=os.path.join(parent_run_dir, "compare_bar.png"),
            )
        except Exception as e:
            logging.warning(f"[{fold_name}] comparison plot failed: {e}")

    if all_rows:
        _save_cv_summary(cross_root, all_rows)

    RUN_DIR = cross_root
    logging.info("Done (k-fold from bundle)!")


# Optuna LR search
def run_optuna_lr_search(*, cross_root, base_manifest, device):
    global RUN_DIR, train_list, val_list, test_list
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX, _WINDOW_CAL_PAIRS

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
def _optuna_lr_for_fold(f_train, f_val, f_test, *, fold_idx, out_dir, device,
                        window_cal_pairs=None):
    global RUN_DIR, train_list, val_list, test_list, _WINDOW_CAL_PAIRS

    import optuna

    variant = _resolve_compare_modes()[0]
    use_calibration = (variant == VARIANT_CALIBRATED)
    no_cal_spec = _NO_CAL_VARIANT_SPEC.get(variant)
    N = int(N_VALUES[0])
    run_dir = os.path.join(out_dir, "optuna_lr")
    os.makedirs(run_dir, exist_ok=True)
    prev_run_dir = RUN_DIR
    RUN_DIR = run_dir

    logging.info("=" * 72)
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

    # loaders built ONCE: they do not depend on lr
    if use_calibration:
        train_dataset = CalibrationBPDataset(train_list, window_cal_pairs)
        val_dataset = CalibrationBPDataset(val_list, window_cal_pairs)
    else:
        train_dataset = BloodPressureDataset(train_list)
        val_dataset = BloodPressureDataset(val_list)
    dl_kwargs = _make_dataloader_kwargs()
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    scratch_ckpt = os.path.join(run_dir, "optuna_trial_scratch.pth")
    trial_rows: list = []

    def objective(trial):
        lr = trial.suggest_categorical("lr", OPTUNA_LR_GRID)
        # Fresh seed + fresh model per trial so lr is the ONLY thing varying.
        set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))
        if use_calibration:
            model = _build_calibration_unet(device)
        elif no_cal_spec is not None:
            model = no_cal_spec["train_builder"](device)
        else:
            model = _build_plain_unet_no_cal(device)
        logging.info(f"[trial {trial.number}] lr={lr:.2e} — "
                     f"training {OPTUNA_EPOCHS} epochs")
        _, best_val = train_model(
            model, train_loader, val_loader, OPTUNA_EPOCHS, device,
            scratch_ckpt, plot_curve=False,
            use_calibration=use_calibration,
            history_title=f"Optuna trial {trial.number} lr={lr:.2e}",
            lr=lr, return_best_val=True,
        )
        del model
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        best_val = float(best_val)
        trial_rows.append({"trial": int(trial.number), "lr": float(lr),
                           "best_val_loss": best_val})
        logging.info(f"[trial {trial.number}] lr={lr:.2e} -> "
                     f"best_val_loss={best_val:.6f}")
        return best_val

    sampler = optuna.samplers.GridSampler({"lr": OPTUNA_LR_GRID})
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"lr_grid_fold{fold_idx}")
    study.optimize(objective, n_trials=len(OPTUNA_LR_GRID))

    results_df = pd.DataFrame(trial_rows).sort_values("lr").reset_index(drop=True)
    csv_path = os.path.join(run_dir, "optuna_lr_search.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Saved Optuna LR results: {csv_path}")

    best_lr = float(study.best_trial.params["lr"])
    with open(os.path.join(run_dir, "optuna_lr_search_best.json"), "w") as f:
        json.dump({
            "best_lr": best_lr,
            "best_val_loss": float(study.best_trial.value),
            "epochs_per_trial": int(OPTUNA_EPOCHS),
            "fold": fold_idx,
            "N": N,
            "variant": variant,
            "model": MODEL_NAME,
            "lr_grid": [float(x) for x in OPTUNA_LR_GRID],
            "all_trials": trial_rows,
        }, f, indent=2)

    # Bar plot: lr -> best val loss (winner highlighted).
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        lrs = results_df["lr"].tolist()
        colors = ["#d62728" if abs(v - best_lr) < 1e-12 else "#1f77b4" for v in lrs]
        ax.bar([f"{v:.0e}" for v in lrs], results_df["best_val_loss"].tolist(),
               color=colors)
        ax.set_xlabel("learning rate")
        ax.set_ylabel(f"best val loss ({OPTUNA_EPOCHS} epochs)")
        ax.set_title(f"Optuna LR search — {MODEL_NAME} [{variant}], fold {fold_idx}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "optuna_lr_search.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Optuna bar plot failed: {e}")

    logging.info(f"[fold {fold_idx}] OPTUNA BEST lr = {best_lr:.2e} "
                 f"(best_val_loss={study.best_trial.value:.6f})")
    RUN_DIR = prev_run_dir
    return best_lr


# Entry point
def main():
    global RUN_DIR, train_list, val_list, test_list
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX, CALIBRATION_WINDOW_COUNT
    global learning_rate

    variants = _resolve_compare_modes()

    set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))
    if not bool(SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

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
    full_train_list, full_val_list, full_test_list = data_preprocessing(
        shared_split_path, shared_manifest_path
    )

    base_manifest = (_PROCESSED_MANIFEST.copy()
                     if _PROCESSED_MANIFEST is not None else None)

    # Optional fixed-grid LR search before the real run.
    if int(USE_OPTUNA_LR_SEARCH) == 1 and int(OPTUNA_PER_FOLD) != 1:
        best_lr = run_optuna_lr_search(
            cross_root=cross_root, base_manifest=base_manifest, device=device,
        )
        if int(OPTUNA_THEN_KFOLD) != 1:
            logging.info("OPTUNA_THEN_KFOLD=0 — stopping after the LR search.")
            return
        learning_rate = float(best_lr)
        RUN_DIR = cross_root
        logging.info(f"Continuing the full run with lr={learning_rate:.2e}")

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
