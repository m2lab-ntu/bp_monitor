import os, glob, gc, re, json, shutil, warnings
import random
from datetime import datetime
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import math
import hashlib
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import tempfile

import model as model_lib
from model import ResNetUNet, ResNetUNet3L, WaveformLoss, _get_skip_mode



# ------------------------------------------------------------
# (1) Global parameters and paths
# ------------------------------------------------------------
data_paths = [
    "/media/chen/EXTERNAL_USB/Seg3min",
    "/media/chen/EXTERNAL_USB/Seg3min2",
]

processed_dir = "/home/chen/Desktop/processed_minmax"
os.makedirs(processed_dir, exist_ok=True)

# Fast start and performance/stability options
REUSE_PROCESSED = True # Force no re-processing; if True, load existing .npz and splits
# Split mode 'subject' or 'segment' (random segment split)
SPLIT_MODE = "subject"  # Options: "subject" or "segment"
SPLIT_JSON_PATH = os.path.join(processed_dir, "split_subjects.json")  # Save/load subject split
USE_COMPILE = True  # PyTorch 2.x can speed up
DATALOADER_WORKERS = min(8, (os.cpu_count() or 8))
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Keep training after validation target; only keep the best model

# Model saving: add timestamp and custom remark while keeping original filename
SAVE_REMARK = "Unet"  # Optional remark (leave empty to skip)
SAVE_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"  # e.g. 20250131-235959

# Backup timestamp: fixed to the time when the best model first appears,
# later best models overwrite the same timestamped file to avoid many backups.
BACKUP_TS_FIRST: Optional[str] = None
# Optuna run timestamp (for study_name)
_OPTUNA_RUN_TS: Optional[str] = None

# Support multiple subject_file_map files
SUBJECT_FILE_MAP_PATHS = [
    "/media/chen/EXTERNAL_USB/Seg3min/subject_file_map.csv",
    "/media/chen/EXTERNAL_USB/Seg3min2/subject_file_map2.csv",
]
SUBJECT_FILE_MAP_OUTPUT_PATH = os.path.join(processed_dir, "subject_file_map_result.csv")

data_usage_ratio = 1.0
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
# Quick end-to-end sanity check: when <1.0, randomly subsample train/val/test segments (avoid full data)
DATA_SAMPLE_RATIO = 1.0
DATA_SAMPLE_SEED = 42

SEGMENT_LEN, STEP_SIZE = 1024, 1024
batch_size, epochs, learning_rate = 256, 30, 5e-5
APPLY_FILTERING, FS = True, 125

# Limit worker count (avoid oversubscription, default CPU-1, max 32)
WORKERS = max(1, min(32, (os.cpu_count() or 8) - 1))

train_list, val_list, test_list = [], [] , []

# Normalization parameters
GLOBAL_ABP_MIN = None
GLOBAL_ABP_MAX = None
# Phase-B I/O toggle: uncompressed NPZ is faster; set True to use compressed NPZ
NPZ_COMPRESS = False

# Global dropout rate (regularization, no data augmentation)
DROPOUT_P = 0.2
BASE_CHANNELS = 128  # UNet/ResNet base channel count

# Optuna hyperparameter search settings
#  - OPTUNA_USE: 1=enable Optuna, 0=disable (keep normal training)
OPTUNA_USE = 0
OPTUNA_TRIALS = 5    
OPTUNA_TRIAL_EPOCHS = 8
OPTUNA_DASHBOARD = 1
OPTUNA_DASHBOARD_PORT = 8080
OPTUNA_STORAGE = "sqlite:///optuna_study.db"
# 1=auto new study_name per run (append timestamp); 0=fixed study_name
OPTUNA_AUTO_NEW_STUDY_NAME = 1
OPTUNA_STUDY_NAME_TS_FMT = "%Y%m%d-%H%M%S"
# 1=resume to target total trials (across runs); 0=run OPTUNA_TRIALS more this time
OPTUNA_CONTINUE = 0

# UNet depth: 1=use 3-layer (one less) version; 0=use original 4-layer version
USE_UNET_3L = 1

# Replace all Mamba skips with Transformer (1=Transformer; 0=Mamba; -1=identity)
USE_TRANSFORMER_SKIPS = 0

# Transformer positional encoding (1=sinusoidal on; 0=off)
USE_TRANSFORMER_POSENC = 1

REGRESSION_TARGETS = ("sbp", "dbp")

# Multi-task head settings (waveform + SBP/DBP scalars)
USE_MULTITASK_SBP_DBP = True
MT_SBP_LOSS_WEIGHT = 0.1
MT_DBP_LOSS_WEIGHT = 0.1

# Ridge regression on decoder final features (after skip/transformer, before ABP head) for SBP/DBP
USE_BOTTLENECK_RIDGE_PROBE = True
BOTTLENECK_RIDGE_USE_TRAIN_VAL = True  

# ------------------------------------------------------------
# Test-only mode (no training, test only)
# ------------------------------------------------------------
TEST_ONLY_MODE = 0          # 1=enable test-only; 0=normal train/val/test
TEST_ONLY_CHECKPOINT = "Setpth.pth"         # Set .pth weight path (empty uses best_{run_name}.pth)
TEST_ONLY_USE_SAVED_STATS = True  # Prefer loading saved normalization stats
TEST_ONLY_STATS_PATH = "abp_normalization_stats.json"
TEST_ONLY_VISUALIZE = True         # Whether to plot predictions for some test segments
TEST_ONLY_RUN_RIDGE = True
TEST_ONLY_PLOT_TRUEPRED_SCATTER = True

# ------------------------------------------------------------
# Global random seed settings
# ------------------------------------------------------------
GLOBAL_SEED = 42            # Set seed here
SEED_DETERMINISTIC = 0      # 1=force full determinism (may be slower); 0=best performance

# Whether to fuse 4-point stats into model input (default on).
# When off, still train on existing data but model forward ignores these four points (use zero vector).
USE_SIGNAL_STATS = True

# ------------------------------------------------------------
# Plot toggles
# ------------------------------------------------------------
# Plot pre-train SBP/DBP/mean ABP distributions (histograms)
ENABLE_PLOT_PRETRAIN_DISTRIBUTIONS = False
# Plot post-train Bland-Altman scatter
ENABLE_PLOT_BA_SCATTER = False
# Plot post-train True vs Pred scatter
ENABLE_PLOT_TRUEPRED_SCATTER = False

# Plot output settings
PLOT_OUTPUT_DIR = "plots"
PLOT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"

# Sync model module configuration (model.py reads these globals at init/forward)
model_lib.FS = FS
model_lib.DROPOUT_P = DROPOUT_P
model_lib.BASE_CHANNELS = BASE_CHANNELS
model_lib.USE_TRANSFORMER_SKIPS = USE_TRANSFORMER_SKIPS
model_lib.USE_TRANSFORMER_POSENC = USE_TRANSFORMER_POSENC
model_lib.USE_MULTITASK_SBP_DBP = USE_MULTITASK_SBP_DBP

# ------------------------------------------------------------
# (2) Utilities (filtering, Hampel, alignment)
# ------------------------------------------------------------
def normalize_abp(abp):
    """Normalize ABP using statistics."""
    return (abp - GLOBAL_ABP_MIN) / (GLOBAL_ABP_MAX - GLOBAL_ABP_MIN)

def denormalize_abp(abp_norm):
    """Denormalize ABP using statistics."""
    return abp_norm * (GLOBAL_ABP_MAX - GLOBAL_ABP_MIN) + GLOBAL_ABP_MIN

def create_bandpass_filter(lowcut=0.5, highcut=8.0, fs=FS, order=4):
    # 0.5-8 Hz filter for PPG only
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

def _sanitize_tag(tag: str) -> str:
    """Convert remarks to a safe filename token (alnum, underscore, hyphen only)."""
    try:
        s = str(tag).strip()
    except Exception:
        s = ""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s.strip("_")

def save_plot(prefix: str, ext: str = "png") -> str:
    """Save the current matplotlib figure with timestamped filename."""
    try:
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    except Exception:
        pass
    safe_prefix = _sanitize_tag(prefix) or "plot"
    ts = datetime.now().strftime(PLOT_TIMESTAMP_FMT)
    out_path = os.path.join(PLOT_OUTPUT_DIR, f"{safe_prefix}_{ts}.{ext}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Info] Saved plot: {out_path}")
    return out_path

def _sync_test_globals(test_mod) -> None:
    """Sync normalization stats into test.py's train_mod instance."""
    try:
        test_mod.train_mod.GLOBAL_ABP_MIN = GLOBAL_ABP_MIN
        test_mod.train_mod.GLOBAL_ABP_MAX = GLOBAL_ABP_MAX
    except Exception:
        pass

def _get_optuna_study_name(bn_key: str) -> str:
    base = f"bp_study_{bn_key}"
    if int(OPTUNA_AUTO_NEW_STUDY_NAME) != 1:
        return base
    global _OPTUNA_RUN_TS
    if _OPTUNA_RUN_TS is None:
        _OPTUNA_RUN_TS = datetime.now().strftime(OPTUNA_STUDY_NAME_TS_FMT)
    return f"{base}_{_OPTUNA_RUN_TS}"

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

def set_global_seed(seed: int, deterministic: bool = False):
    """Set global random seeds for Python/NumPy/Torch.
    When deterministic=True, enable PyTorch deterministic mode (may reduce performance).
    """
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
    """Seed DataLoader workers: ensure each worker has a fixed but distinct seed."""
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

def cross_correlation_align(ABP, PPG, ECG, dPPG, max_lag_sec=0.6):
    """
    Use limited-lag normalized cross-correlation to speed up alignment, limited to +/- max_lag_sec.
    """
    length = min(len(ABP), len(PPG), len(ECG), len(dPPG))
    if length < 2:
        return ABP, PPG, ECG, dPPG
    ABP_, PPG_ = ABP[:length].astype(np.float32), PPG[:length].astype(np.float32)
    ECG_, dPPG_ = ECG[:length].astype(np.float32), dPPG[:length].astype(np.float32)

    max_lag = int(max(1, round(max_lag_sec * FS)))
    abp_c = ABP_ - ABP_.mean()
    ppg_c = PPG_ - PPG_.mean()
    abp_std = np.std(abp_c) + 1e-6
    ppg_std = np.std(ppg_c) + 1e-6

    best_lag, best_score = 0, -1e9
    # Scan [-max_lag, max_lag]
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
        PPG_, ECG_, dPPG_ = PPG_[best_lag:], ECG_[best_lag:], dPPG_[best_lag:]
        ABP_ = ABP_[:len(PPG_)]
    elif best_lag < 0:
        s = abs(best_lag)
        ABP_ = ABP_[s:]
        PPG_, ECG_, dPPG_ = PPG_[:len(ABP_)], ECG_[:len(ABP_)], dPPG_[:len(ABP_)]

    L = min(len(ABP_), len(PPG_), len(ECG_), len(dPPG_))
    return ABP_[:L], PPG_[:L], ECG_[:L], dPPG_[:L]

# ------------------------------------------------------------
# (3) Build record -> subject mapping (read from master table)
# ------------------------------------------------------------
def read_subject_file_map(map_csv_path):
    """
    Read subject_file_map.csv to speed up pairing.
    Expected columns: subject_id, input_file, output_csv(preferred), segments_saved(optional)
    - Keep subject_id as string
    - If only a filename is provided, search in base_dirs:
        base_dirs = [dirname(map_csv_path)] + data_paths
    - Resolution order: prefer output_csv; otherwise change input_file extension to .csv
    Returns:
      mapping: dict[csv_abs_path -> subject_id]
      used_df: DataFrame[subject_id, input_file, output_csv, csv_path]
    """
    if not os.path.isfile(map_csv_path):
        raise FileNotFoundError(f"subject_file_map file not found: {map_csv_path}")
    df = pd.read_csv(map_csv_path)
    if "subject_id" not in df.columns:
        raise ValueError("subject_file_map missing column: subject_id")
    # Require at least one of input_file or output_csv
    if ("input_file" not in df.columns) and ("output_csv" not in df.columns):
        raise ValueError("subject_file_map requires at least one column: input_file or output_csv")

    # Prepare search dirs: map file dir + data_paths
    base_dirs = []
    map_dir = os.path.dirname(os.path.abspath(map_csv_path))
    base_dirs.append(map_dir)
    for d in (data_paths if isinstance(data_paths, (list, tuple)) else []):
        if isinstance(d, str):
            base_dirs.append(d)
    # De-duplicate
    base_dirs = list(dict.fromkeys(base_dirs))

    def resolve_path(name):
        # Already absolute path
        if isinstance(name, str) and os.path.isabs(name) and os.path.isfile(name):
            return os.path.abspath(name)
        # Try joining with base_dirs
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

        # Target CSV name
        csv_name = out if out else (os.path.splitext(inp)[0] + ".csv" if inp else "")
        if not csv_name:
            continue

        # Resolve to actual CSV path
        csv_path = resolve_path(csv_name)
        # If it fails, try changing input_file extension to .csv
        if csv_path is None and inp:
            csv_path = resolve_path(os.path.splitext(inp)[0] + ".csv")

        if csv_path is None:
            continue

        # If output_csv missing, fill with filename
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
            "No valid CSV paths parsed in subject_file_map.csv; check output_csv/filename and directories:"
            f"{base_dirs}"
        )

    used_df = pd.DataFrame(rows, columns=["subject_id", "input_file", "output_csv", "csv_path"]).drop_duplicates("csv_path")
    mapping = dict(zip(used_df["csv_path"], used_df["subject_id"]))
    return mapping, used_df

# Read multiple mappings and merge
def read_subject_file_maps(map_csv_paths):
    rows = []
    merged_mapping = {}
    used_list = []
    for p in map_csv_paths:
        try:
            mp_map, mp_df = read_subject_file_map(p)
            # Merge (if same csv_path maps to different subject_id, keep first and warn)
            for k, v in mp_map.items():
                if k in merged_mapping and merged_mapping[k] != v:
                    print(f"[Warning] Mapping conflict: {k} -> {merged_mapping[k]} vs {v}, keeping previous value")
                    continue
                merged_mapping[k] = v
            used_list.append(mp_df)
        except Exception as e:
            print(f"[Warning] Failed to read {p}: {e}")
    if not merged_mapping:
        raise RuntimeError("No valid records read from any subject_file_map")
    used_df = pd.concat(used_list, axis=0, ignore_index=True).drop_duplicates("csv_path")
    return merged_mapping, used_df

# ------------------------------------------------------------
# (4) Compute per-subject mean SBP/DBP
# ------------------------------------------------------------
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
    """
    Sort subjects by SBP (default) from low to high, then assign to train/val/test with weighted round-robin (snake).
    - subj_stats: dict[subject_id -> (sbp_mean, dbp_mean)]
    - Returns: (train_set, val_set, test_set)
    """
    subjects = list(subj_stats.keys())
    n = len(subjects)
    if n == 0:
        return set(), set(), set()
    # Normalize ratios
    ratios = np.array([max(0.0, float(train_ratio)), max(0.0, float(val_ratio)), max(0.0, float(test_ratio))], dtype=np.float64)
    s = ratios.sum()
    if s <= 0:
        ratios = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ratios /= s
    # Target quotas (ensure sum equals n)
    n_train = int(np.floor(n * ratios[0]))
    n_val   = int(np.floor(n * ratios[1]))
    n_test  = max(0, n - n_train - n_val)
    quotas = np.array([n_train, n_val, n_test], dtype=np.int64)
    # Sort by key (default sbp)
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
    valid.sort(key=key_fn)  # ascending
    ordered = valid + invalid  # Unsortable at the end
    # Weighted round-robin (snake): credits accumulate ratios, pick max non-full fold
    credits = np.zeros(3, dtype=np.float64)
    bins = [[], [], []]  # train, val, test
    for sid in ordered:
        # Try up to 3 times to find an available fold
        for _ in range(3):
            credits += ratios
            # Mask folds that are already full
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
            # If all folds are full (should not happen), put in the largest fold
            idx = int(np.argmax([len(b) for b in bins]))
            bins[idx].append(sid)
    return set(bins[0]), set(bins[1]), set(bins[2])

# ------------------------------------------------------------
# (2+) Physiological feature extraction (removed, no longer used)
# Keep original function if needed; not called in pipeline
# ------------------------------------------------------------
def _smooth_local(sig, win=5):
    if win < 2: return sig
    return np.convolve(sig, np.ones(win) / win, mode="same")

# ------------------------------------------------------------
# (5) Read CSV, align, segment, output .npz
# ------------------------------------------------------------
def read_and_align_csv(fp):
    cols = ["PLETH", "II", "dPPG", "ABP"]
    try:
        df = pd.read_csv(fp, usecols=cols, dtype=np.float32, engine="c", memory_map=True, low_memory=False)
        df.dropna(subset=cols, inplace=True)
        df = df[(df[cols] != 0).all(axis=1)]
        if len(df) < SEGMENT_LEN: return np.array([])
        ppg, ecg, dppg, abp = (df["PLETH"].values, df["II"].values,
                               df["dPPG"].values, df["ABP"].values)
        a, p, e, d = cross_correlation_align(abp, ppg, ecg, dppg)
        if len(a) < SEGMENT_LEN: return np.array([])
        return np.column_stack([p, e, d, a])
    except Exception as err:
        print(f"[Error] Failed to read {os.path.basename(fp)}: {err}")
        return np.array([])

def chunk_data(arr, seg_len=SEGMENT_LEN, step=STEP_SIZE):
    for i in range(0, len(arr) - seg_len + 1, step):
        segment = arr[i:i + seg_len]
        if np.isfinite(segment).all() and np.std(segment[:, 0]) > 0.01:
            yield segment

def _short_hash(s: str) -> str:
    try:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest()
    except Exception:
        return hex(abs(hash(s)) & 0xFFFFFFFF)[2:]

def process_single_csv(args):
    csv_path, subj_id = args
    out_paths, local_id = [], 0
    data = read_and_align_csv(csv_path)
    if data.size == 0:
        return out_paths

    segments = list(chunk_data(data, step=STEP_SIZE))

    for i, seg in enumerate(segments):
        abp = seg[:, 3]
        # PPG-ABP correlation filtering removed

        mn, mx = abp.min(), abp.max()
        if mx - mn < 5.0 or mn < 30 or mx > 250:
            continue

        abp_cv = np.std(abp) / (np.mean(abp) + 1e-6)
        if abp_cv < 0.05:
            continue

        # Physiological features removed; no longer extract or filter
        X, Y = seg[:, :3], abp  # Store raw ABP values, no normalization

        # Keep subject_id as string and sanitize for filename
        safe_subj = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(subj_id))
        h = _short_hash(os.path.abspath(csv_path))
        out_p = os.path.join(processed_dir, f"{safe_subj}_{os.path.basename(csv_path)[:-4]}_{h}_{local_id}.npz")
        if NPZ_COMPRESS:
            np.savez_compressed(
                out_p,
                X=X.astype(np.float32),
                Y=Y.astype(np.float32),
                abp_min=np.float32(mn),
                abp_max=np.float32(mx),
                sid=np.array(str(subj_id)),
            )
        else:
            np.savez(
                out_p,
                X=X.astype(np.float32),
                Y=Y.astype(np.float32),
                abp_min=np.float32(mn),
                abp_max=np.float32(mx),
                sid=np.array(str(subj_id)),
            )
        out_paths.append(out_p)
        local_id += 1

    return out_paths

# ------------------------------------------------------------
# (6+) Helpers: save/load subject splits and build lists from existing npz
# ------------------------------------------------------------
def save_split_json(train_subj, val_subj, test_subj, path=SPLIT_JSON_PATH):
    try:
        with open(path, "w") as f:
            json.dump({
                "train": sorted(list(train_subj)),
                "val": sorted(list(val_subj)),
                "test": sorted(list(test_subj)),
            }, f)
        print(f"[Info] Saved subject splits to: {path}")
    except Exception as e:
        print(f"[Warning] Failed to save splits: {e}")

def load_split_json(path=SPLIT_JSON_PATH):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return set(d.get("train", [])), set(d.get("val", [])), set(d.get("test", []))
    except Exception as e:
        print(f"[Warning] Failed to read splits: {e}")
        return None

def _npz_sid(npz_path):
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            sid = d["sid"]
            # sid stored as 0-d array
            if isinstance(sid, np.ndarray) and sid.shape == ():
                return str(sid.item())
            return str(sid)
    except Exception:
        return None

def build_lists_from_existing_npz(train_subj, val_subj, test_subj):
    files = glob.glob(os.path.join(processed_dir, "*.npz"))
    train_list, val_list, test_list = [], [], []
    train_sids, val_sids, test_sids = set(), set(), set()  # Track unique subject counts per split
    for f in files:
        sid = _npz_sid(f)
        if sid is None:
            continue
        if sid in train_subj:
            train_list.append(f)
            train_sids.add(sid)
        elif sid in val_subj:
            val_list.append(f)
            val_sids.add(sid)
        elif sid in test_subj:
            test_list.append(f)
            test_sids.add(sid)
        else:
            # If subject not in splits, default to train
            train_list.append(f)
            train_sids.add(sid)
    print(
        "[Info] Build data lists directly from existing .npz:"
        f" Train={len(train_list)} segments ({len(train_sids)} subjects),"
        f" Val={len(val_list)} segments ({len(val_sids)} subjects),"
        f" Test={len(test_list)} segments ({len(test_sids)} subjects)"
    )
    # Subsample for quick trial
    train_list = _sample_list(train_list, DATA_SAMPLE_RATIO)
    val_list = _sample_list(val_list, DATA_SAMPLE_RATIO)
    test_list = _sample_list(test_list, DATA_SAMPLE_RATIO)
    return train_list, val_list, test_list


def _sample_list(lst, ratio, seed=DATA_SAMPLE_SEED):
    """Randomly subsample list for quick trials; enabled when ratio<1.0."""
    ratio = float(ratio)
    if ratio >= 1.0 or ratio <= 0.0 or len(lst) == 0:
        return lst
    n = max(1, int(len(lst) * ratio))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lst), size=n, replace=False)
    return [lst[i] for i in idx]

def infer_subject_splits_from_npz(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Infer subject splits only from existing .npz (no CSV, no Phase-A).
    Assign in snake order by subject lexicographic order, honoring ratio quotas.
    """
    files = glob.glob(os.path.join(processed_dir, "*.npz"))
    sid_counts = {}
    for f in files:
        sid = _npz_sid(f)
        if sid is not None:
            sid_counts[sid] = sid_counts.get(sid, 0) + 1

    subjects = sorted(sid_counts.keys())
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
    bins = [[], [], []]  # train, val, test

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

# Random segment split from existing .npz (ignore subject)
def random_segment_split_from_existing_npz(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    files = glob.glob(os.path.join(processed_dir, "*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {processed_dir}")
    ratios = np.array([
        max(0.0, float(train_ratio)),
        max(0.0, float(val_ratio)),
        max(0.0, float(test_ratio))
    ], dtype=np.float64)
    s = ratios.sum()
    if s <= 0:
        ratios = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ratios /= s
    n = len(files)
    n_train = int(np.floor(n * ratios[0]))
    n_val   = int(np.floor(n * ratios[1]))
    n_test  = max(0, n - n_train - n_val)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    train_list = files[:n_train]
    val_list   = files[n_train:n_train + n_val]
    test_list  = files[n_train + n_val: n_train + n_val + n_test]
    print(f"[Info] Random segment split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)} (total {n})")
    return train_list, val_list, test_list

# ------------------------------------------------------------
# (6) Subject-wise stratified split + preprocessing (use subject_file_map.csv)
# ------------------------------------------------------------
def data_preprocessing():
    # Forced fast path: if REUSE_PROCESSED=True, skip CSV/Phase-A/segmentation
    old_npz = glob.glob(os.path.join(processed_dir, "*.npz"))
    if REUSE_PROCESSED:
        if not old_npz:
            raise RuntimeError(f"REUSE_PROCESSED=True but no .npz in folder: {processed_dir}")
        # Random segment split: build train/val/test directly from existing .npz
        if str(SPLIT_MODE).lower() == "segment":
            return random_segment_split_from_existing_npz(train_ratio, val_ratio, test_ratio)
        # Subject split: reuse existing splits or infer from .npz
        loaded = load_split_json()
        if loaded is not None:
            train_subj, val_subj, test_subj = loaded
            print("[Info] Using existing subject splits and .npz to build data lists")
        else:
            print("[Info] No subject splits found; infer from existing .npz (no rebuild)")
            train_subj, val_subj, test_subj = infer_subject_splits_from_npz(train_ratio, val_ratio, test_ratio)
            save_split_json(train_subj, val_subj, test_subj)
        return build_lists_from_existing_npz(train_subj, val_subj, test_subj)

    # Read multiple master tables (rebuild splits if needed)
    mapping, used_df = read_subject_file_maps(SUBJECT_FILE_MAP_PATHS)
    csv_files = list(mapping.keys())
    if not csv_files:
        raise FileNotFoundError("No usable CSV files in subject_file_map.csv")
    csv_files = csv_files[:int(len(csv_files) * data_usage_ratio)]
    used_df = used_df[used_df["csv_path"].isin(csv_files)].copy()
    file2subj = mapping

    # If splits exist and reuse is chosen, skip Phase-A recomputation
    loaded = load_split_json()
    if REUSE_PROCESSED and loaded is not None and old_npz:
        train_subj, val_subj, test_subj = loaded
        print("[Info] Using saved subject splits")
    else:
        # Phase-A: compute subject BP stats for stratified split
        subj_stats = compute_subject_bp_stats(csv_files, file2subj)
        train_subj, val_subj, test_subj = stratified_snake_split(
            subj_stats, train_ratio, val_ratio, test_ratio
        )
        save_split_json(train_subj, val_subj, test_subj)

    # If .npz exists and reuse allowed, skip rebuild; use .npz + splits
    if REUSE_PROCESSED and old_npz:
        print("[Info] Detected existing .npz; skip re-segmentation and saving")
        if str(SPLIT_MODE).lower() == "segment":
            return random_segment_split_from_existing_npz(train_ratio, val_ratio, test_ratio)
        else:
            return build_lists_from_existing_npz(train_subj, val_subj, test_subj)

    # Otherwise, delete old files and rebuild
    old_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    if old_files:
        print(f"[Info] Deleting {len(old_files)} old files...")
        for f in old_files:
            os.remove(f)

    print(f"[Info] Start processing {len(csv_files)} CSV files using {WORKERS} workers...")

    seg_infos = []
    seg_counts = {fp: 0 for fp in csv_files}
    subj_seg_counts = {}
    total_files = len(csv_files)

    with tqdm(total=total_files, desc="Phase-B: processing progress") as pbar, \
         ProcessPoolExecutor(max_workers=WORKERS) as exe:
        future_to_file = {
            exe.submit(process_single_csv, (fp, file2subj[fp])): fp
            for fp in csv_files
        }
        for future in as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                paths = future.result()
                subj = file2subj[fp]
                tag = ("train" if subj in train_subj else
                       "val"   if subj in val_subj   else
                       "test")
                for p in paths:
                    seg_infos.append((p, tag))
                seg_counts[fp] = len(paths)
                subj_seg_counts[subj] = subj_seg_counts.get(subj, 0) + len(paths)
            except Exception as e:
                print(f"[Error] Error processing file {fp}: {str(e)}")
            pbar.update(1)

    if not seg_infos:
        raise RuntimeError("No segments were produced")

    summary_df = used_df.copy()
    summary_df["segments_saved"] = summary_df["csv_path"].map(seg_counts).fillna(0).astype(int)
    summary_df[["subject_id", "input_file", "output_csv", "segments_saved"]].to_csv(
        SUBJECT_FILE_MAP_OUTPUT_PATH, index=False
    )
    print(f"[Info] Processing summary saved to: {SUBJECT_FILE_MAP_OUTPUT_PATH}")

    # Build final lists by mode
    if str(SPLIT_MODE).lower() == "segment":
        # Ignore subject tag and randomly split all segments
        all_segments = [p for p, _ in seg_infos]
        ratios = np.array([
            max(0.0, float(train_ratio)),
            max(0.0, float(val_ratio)),
            max(0.0, float(test_ratio))
        ], dtype=np.float64)
        s = ratios.sum()
        if s <= 0:
            ratios = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            ratios /= s
        n = len(all_segments)
        n_train = int(np.floor(n * ratios[0]))
        n_val   = int(np.floor(n * ratios[1]))
        n_test  = max(0, n - n_train - n_val)
        rng = np.random.default_rng(42)
        rng.shuffle(all_segments)
        train_list = all_segments[:n_train]
        val_list   = all_segments[n_train:n_train + n_val]
        test_list  = all_segments[n_train + n_val: n_train + n_val + n_test]
    else:
        train_list = [p for p, t in seg_infos if t == "train"]
        val_list   = [p for p, t in seg_infos if t == "val"]
        test_list  = [p for p, t in seg_infos if t == "test"]

    # For quick trial, subsample each split (keep non-empty)
    train_list = _sample_list(train_list, DATA_SAMPLE_RATIO)
    val_list = _sample_list(val_list, DATA_SAMPLE_RATIO)
    test_list = _sample_list(test_list, DATA_SAMPLE_RATIO)

    train_subj_eff = {s for s in train_subj if subj_seg_counts.get(s, 0) > 0}
    val_subj_eff   = {s for s in val_subj   if subj_seg_counts.get(s, 0) > 0}
    test_subj_eff  = {s for s in test_subj  if subj_seg_counts.get(s, 0) > 0}

    if str(SPLIT_MODE).lower() == "segment":
        print(f"\n[Info] Segment counts (random segments): Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
    else:
        print(f"\n[Info] Segment counts: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}  | Subjects (valid):"
              f"{len(train_subj_eff)}/{len(val_subj_eff)}/{len(test_subj_eff)}")
    return train_list, val_list, test_list

# ------------------------------------------------------------
# (7) Compute training stats and normalize
# ------------------------------------------------------------
def compute_global_abp_stats(train_files):
    """Compute min and max ABP across training set."""
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX

    all_mins, all_maxs = [], []
    for filepath in tqdm(train_files, desc="Compute training ABP stats"):
        try:
            data = np.load(filepath)
            Y = data['Y']  # raw ABP values
            all_mins.append(np.min(Y))
            all_maxs.append(np.max(Y))
        except Exception as e:
            print(f"[Warning] Error processing file {filepath}: {e}")
            continue

    if not all_mins or not all_maxs:
        raise RuntimeError("Unable to compute ABP stats; data may be invalid")

    GLOBAL_ABP_MIN = float(np.min(all_mins))
    GLOBAL_ABP_MAX = float(np.max(all_maxs))

    print(f"[Info] ABP range: {GLOBAL_ABP_MIN:.2f} - {GLOBAL_ABP_MAX:.2f} mmHg")
    return GLOBAL_ABP_MIN, GLOBAL_ABP_MAX

def _load_saved_norm_stats(path: str) -> bool:
    """Load saved normalization stats into globals."""
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as f:
            d = json.load(f)
        GLOBAL_ABP_MIN = d.get("GLOBAL_ABP_MIN", GLOBAL_ABP_MIN)
        GLOBAL_ABP_MAX = d.get("GLOBAL_ABP_MAX", GLOBAL_ABP_MAX)
        if GLOBAL_ABP_MIN is not None and GLOBAL_ABP_MAX is not None:
            print(f"[Info] Loaded normalization stats: {path}")
        return True
    except Exception as e:
        print(f"[Warn] Failed to read normalization stats: {e}")
        return False

# ------------------------------------------------------------
# (8) Load segments and Dataset
# ------------------------------------------------------------
def load_segment(filepath, apply_filter=True):
    try:
        with np.load(filepath) as data:
            X = data['X']  # [SEGMENT_LEN, 3] PPG, ECG, dPPG
            Y = data['Y']  # [SEGMENT_LEN] raw ABP values, unnormalized
            keys = set(getattr(data, 'files', []))
            if 'abp_min' in keys:
                abp_min = float(data['abp_min'])
            else:
                abp_min = float(np.min(Y))
            if 'abp_max' in keys:
                abp_max = float(data['abp_max'])
            else:
                abp_max = float(np.max(Y))

        # Backward compatibility: ignore 'physio' if present

        if apply_filter and APPLY_FILTERING and X is not None:
            # Only apply 0.5-8 Hz filter + Hampel to PPG (channel 0); no filtering for ECG/dPPG
            try:
                X[:, 0] = filtfilt(b_bp, a_bp, X[:, 0])
                X[:, 0] = hampel_filter(X[:, 0])
            except Exception:
                pass

        # UNet expects length divisible by 16; 1250 -> 1248 (drop last 2 points)
        if X is not None and Y is not None:
            target_len = max(1, (len(Y) // 16) * 16)
            if target_len < len(Y):
                X = X[:target_len, :]
                Y = Y[:target_len]

        return X, Y, abp_min, abp_max

    except Exception as e:
        print(f"[Error] Failed to load {filepath}: {e}")
        return None, None, 0, 0


def normalize_ppg_ecg_segment(signals: np.ndarray):
    """Normalize PPG/ECG and record scaling params.
    - Per-segment min-max (two values per channel)
    """
    if signals is None or signals.ndim != 2 or signals.shape[1] < 2:
        stats = np.zeros(4, dtype=np.float32)
        return signals, stats

    normalized = signals.astype(np.float32).copy()
    stats = []
    for ch in (0, 1):  # Only normalize PPG, ECG
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
        # Normalize ABP using statistics
        Y_normalized = normalize_abp(Y)

        # If transform provided, apply custom processing to X/Y/stats
        if self.transform is not None:
            X, Y_normalized, signal_stats = self.transform(X, Y_normalized, signal_stats)

        X = torch.FloatTensor(X).transpose(0, 1)  # [3, 1024]
        Y_normalized = torch.FloatTensor(Y_normalized)  # [1024]
        signal_stats = torch.FloatTensor(signal_stats)  # [4]

        return X, Y_normalized, signal_stats


def _bp_targets_from_wave(abp_wave: np.ndarray):
    """Return (SBP, DBP) using peak detection; fallback to max/min if peaks missing."""
    sbp_vals, dbp_vals = detect_sbp_dbp(abp_wave)
    sbp_mean = float(np.mean(sbp_vals)) if len(sbp_vals) > 0 else float(np.max(abp_wave))
    dbp_mean = float(np.mean(dbp_vals)) if len(dbp_vals) > 0 else float(np.min(abp_wave))
    return sbp_mean, dbp_mean


class BPRegressionDataset(Dataset):
    """Dataset for SBP/DBP scalar regression targets."""
    def __init__(self, file_list, targets=REGRESSION_TARGETS, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.targets = tuple(t.lower() for t in targets)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        X, Y, _, _ = load_segment(filepath)

        X, signal_stats = normalize_ppg_ecg_segment(X)
        Y_normalized = normalize_abp(Y)

        if self.transform is not None:
            X, Y_normalized, signal_stats = self.transform(X, Y_normalized, signal_stats)

        sbp_mean, dbp_mean = _bp_targets_from_wave(Y)

        target_vals = []
        mask_vals = []
        for key in self.targets:
            if key == "sbp":
                target_vals.append(np.float32(sbp_mean))
                mask_vals.append(1.0 if np.isfinite(sbp_mean) else 0.0)
            elif key == "dbp":
                target_vals.append(np.float32(dbp_mean))
                mask_vals.append(1.0 if np.isfinite(dbp_mean) else 0.0)

        if not target_vals:
            raise ValueError("Regression targets not set (sbp/dbp)")

        X = torch.FloatTensor(X).transpose(0, 1)  # [C, T]
        targets = torch.tensor(target_vals, dtype=torch.float32)
        mask = torch.tensor(mask_vals, dtype=torch.float32)
        signal_stats = torch.FloatTensor(signal_stats)  # [4]
        return X, targets, mask, signal_stats

def _sbp_dbp_from_batch(y_norm: torch.Tensor):
    """Compute SBP/DBP scalar targets from normalized ABP batch."""
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

# ------------------------------------------------------------
# (12) Training
# ------------------------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs, device, run_name: str,
                save_best: bool = True, plot_curve: bool = True):
    criterion = WaveformLoss()
    lr_used = learning_rate
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
            inputs, targets, signal_stats = batch_data
            # inputs are now tensors
            x_tensor = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            stats_tensor = (signal_stats.to(device, non_blocking=True)
                            if USE_SIGNAL_STATS else None)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out_all = model(x_tensor, stats_tensor)
                    wave_pred, sbp_pred, dbp_pred = _unpack_model_output(out_all)
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

                # Update learning rate every step (CosineAnnealingLR)
                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
                valid_train_batches += 1
            except Exception as e:
                print(f"[Warn] Skipping abnormal batch: {e}")
                continue

        if valid_train_batches == 0:
            print("[Error] No valid training batches this epoch; stopping early.")
            break

        # Validation
        model.eval()
        val_loss_sum = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets, signal_stats = batch_data
                x_tensor = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                stats_tensor = (signal_stats.to(device, non_blocking=True)
                                if USE_SIGNAL_STATS else None)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out_all = model(x_tensor, stats_tensor)
                    wave_pred, sbp_pred, dbp_pred = _unpack_model_output(out_all)
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # CosineAnnealingLR doesn't use val; no need to step at epoch end

        improved = (val_loss < best_val_loss) and np.isfinite(val_loss)
        if improved:
            best_val_loss = val_loss
            to_save = getattr(model, "_orig_mod", model)
            state = to_save.state_dict()
            best_state = deepcopy(state)

            if save_best:
                ckpt_path = f'best_{run_name}.pth'
                # Generate filename with timestamp and remark (timestamp fixed to first improvement)
                global BACKUP_TS_FIRST
                if not BACKUP_TS_FIRST:
                    BACKUP_TS_FIRST = datetime.now().strftime(SAVE_TIMESTAMP_FMT)
                remark_tag = _sanitize_tag(SAVE_REMARK)
                extra = f"_{remark_tag}" if remark_tag else ""
                ckpt_ts_path = f"best_{run_name}_{BACKUP_TS_FIRST}{extra}.pth"

                # Save timestamped file first, then overwrite original filename to keep existing flow
                try:
                    torch.save(state, ckpt_ts_path)
                    shutil.copy2(ckpt_ts_path, ckpt_path)
                    print(
                        f"New best model saved: {ckpt_ts_path} (also overwrote {ckpt_path})\n"
                        f"Validation loss: {val_loss:.6f}"
                    )
                except Exception as e:
                    # If it fails, at least ensure original filename exists
                    try:
                        torch.save(state, ckpt_path)
                        print(
                            f"[Warn] Failed to save timestamped file ({e}); saved only {ckpt_path}."
                        )
                    except Exception as e2:
                        print(f"[Error] Failed to save best model: {e2}")

    # Plot training curves
    if plot_curve:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Training History - {run_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        save_plot(f"training_history_{run_name}")

    # Finish with best weights (best_*.pth already saved)
    try:
        if best_state is not None:
            target = getattr(model, "_orig_mod", model)
            target.load_state_dict(best_state)
    except Exception as e:
        print(f"[Warn] Failed to restore best weights to model: {e}")

    # Attach best val loss to function attribute for Optuna (no API change)
    try:
        train_model._last_best_val_loss = float(best_val_loss)
    except Exception:
        pass

    return model


def _regression_loss(preds: dict, targets: torch.Tensor, mask: torch.Tensor,
                     target_keys, criterion):
    losses = []
    for idx, key in enumerate(target_keys):
        if key not in preds:
            continue
        active = mask[:, idx] > 0.5
        if active.any():
            losses.append(criterion(preds[key][active], targets[:, idx][active]))
    if not losses:
        return None
    return sum(losses) / len(losses)


def train_regression_model(model, train_loader, val_loader, num_epochs, device, run_name: str,
                           target_keys=None):
    target_keys = tuple(target_keys or REGRESSION_TARGETS)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate, weight_decay=1e-3, eps=1e-8)
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_train_batches = 0
        for batch_data in tqdm(train_loader, desc=f"[Reg] Epoch {epoch+1}/{num_epochs}"):
            inputs, targets, mask, signal_stats = batch_data
            x_tensor = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            stats_tensor = (signal_stats.to(device, non_blocking=True)
                            if USE_SIGNAL_STATS else None)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(x_tensor, stats_tensor)
                loss = _regression_loss(preds, targets, mask, target_keys, criterion)
            if loss is None or not torch.isfinite(loss):
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

        if valid_train_batches == 0:
            print("[Error] No valid training batches this epoch (regression); stopping early.")
            break

        model.eval()
        val_loss_sum = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets, mask, signal_stats = batch_data
                x_tensor = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                stats_tensor = (signal_stats.to(device, non_blocking=True)
                                if USE_SIGNAL_STATS else None)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    preds = model(x_tensor, stats_tensor)
                    loss = _regression_loss(preds, targets, mask, target_keys, criterion)
                if loss is not None and torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    valid_val_batches += 1

        train_loss = train_loss / max(1, valid_train_batches)
        val_loss = (val_loss_sum / max(1, valid_val_batches)) if valid_val_batches > 0 else float('inf')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'[Reg] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # CosineAnnealingLR doesn't use val; no need to step at epoch end

        improved = (val_loss < best_val_loss) and np.isfinite(val_loss)
        if improved:
            best_val_loss = val_loss
            ckpt_path = f'best_{run_name}.pth'
            to_save = getattr(model, "_orig_mod", model)
            state = deepcopy(to_save.state_dict())
            best_state = state

            global BACKUP_TS_FIRST
            if not BACKUP_TS_FIRST:
                BACKUP_TS_FIRST = datetime.now().strftime(SAVE_TIMESTAMP_FMT)
            remark_tag = _sanitize_tag(SAVE_REMARK)
            extra = f"_{remark_tag}" if remark_tag else ""
            ckpt_ts_path = f"best_{run_name}_{BACKUP_TS_FIRST}{extra}.pth"

            try:
                torch.save(state, ckpt_ts_path)
                shutil.copy2(ckpt_ts_path, ckpt_path)
                print(
                    f"[Reg] New best model saved: {ckpt_ts_path} (also overwrote {ckpt_path})\n"
                    f"Validation loss: {val_loss:.6f}"
                )
            except Exception as e:
                try:
                    torch.save(state, ckpt_path)
                    print(f"[Reg-Warn] Failed to save timestamped file ({e}); saved only {ckpt_path}.")
                except Exception as e2:
                    print(f"[Reg-Error] Failed to save best model: {e2}")

    try:
        if best_state is not None:
            target = getattr(model, "_orig_mod", model)
            target.load_state_dict(best_state)
    except Exception as e:
        print(f"[Reg-Warn] Failed to restore best weights to model: {e}")

    try:
        train_regression_model._last_best_val_loss = float(best_val_loss)
    except Exception:
        pass

    return model

# ------------------------------------------------------------
# (13) Evaluation and main
# ------------------------------------------------------------
def smooth_signal(sig, win=5):
    if win < 2: return sig
    return np.convolve(sig, np.ones(win) / win, mode="same")

def detect_sbp_dbp(sig, fs=FS):
    s = smooth_signal(sig)
    dist = int(0.32 * fs)
    sys_idx, _ = find_peaks(s, distance=dist)
    dia_idx, _ = find_peaks(-s, distance=dist)
    return sig[sys_idx], sig[dia_idx]


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

# ------------------------------------------------------------
# (13+) Segment-level BP distributions and post-train plots
# ------------------------------------------------------------
def _segment_bp_means_from_wave(y: np.ndarray):
    """Return (sbp_mean, dbp_mean, abp_mean) for one 1024-sample segment."""
    try:
        sbp_vals, dbp_vals = detect_sbp_dbp(y)
        sbp_mean = float(np.mean(sbp_vals)) if len(sbp_vals) > 0 else np.nan
        dbp_mean = float(np.mean(dbp_vals)) if len(dbp_vals) > 0 else np.nan
        abp_mean = float(np.mean(y)) if y is not None and y.size > 0 else np.nan
        return sbp_mean, dbp_mean, abp_mean
    except Exception:
        return np.nan, np.nan, np.nan

def collect_pretrain_bp_distributions(file_list):
    """Iterate .npz segments; compute per-seg SBP/DBP means and average ABP."""
    sbp_means, dbp_means, abp_means = [], [], []
    for p in tqdm(file_list, desc="Pre-train: collect SBP/DBP/ABP distributions"):
        try:
            with np.load(p) as d:
                y = d['Y']
            s_m, d_m, a_m = _segment_bp_means_from_wave(y)
            if np.isfinite(s_m): sbp_means.append(s_m)
            if np.isfinite(d_m): dbp_means.append(d_m)
            if np.isfinite(a_m): abp_means.append(a_m)
        except Exception:
            continue
    return np.asarray(sbp_means, dtype=np.float32), np.asarray(dbp_means, dtype=np.float32), np.asarray(abp_means, dtype=np.float32)

def plot_pretrain_distributions(train_files, val_files=None, test_files=None):
    """Plot three histograms: SBP, DBP, mean ABP. One point per 1024-seg (segment mean)."""
    if not ENABLE_PLOT_PRETRAIN_DISTRIBUTIONS:
        return
    use_files = list(train_files)
    if val_files: use_files.extend(list(val_files))
    if test_files: use_files.extend(list(test_files))
    sbp_m, dbp_m, abp_m = collect_pretrain_bp_distributions(use_files)
    if sbp_m.size == 0 and dbp_m.size == 0 and abp_m.size == 0:
        print("[Warn] Unable to plot distribution: no valid SBP/DBP/ABP stats")
        return
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(sbp_m, bins=40, color="#4C78A8", alpha=0.85)
    plt.title("SBP Distribution")
    plt.xlabel("SBP (mmHg)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)

    plt.subplot(1, 3, 2)
    plt.hist(dbp_m, bins=40, color="#F58518", alpha=0.85)
    plt.title("DBP Distribution")
    plt.xlabel("DBP (mmHg)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)

    plt.subplot(1, 3, 3)
    plt.hist(abp_m, bins=40, color="#54A24B", alpha=0.85)
    plt.title("Mean ABP Distribution")
    plt.xlabel("Mean ABP (mmHg)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    save_plot("pretrain_bp_distributions")

def main():
    # Declare here to avoid Python treating assignment in function as local (UnboundLocalError)
    global train_list, val_list, test_list, GLOBAL_ABP_MIN, GLOBAL_ABP_MAX, learning_rate, DROPOUT_P, batch_size

    # Global seed: set before any random operations
    set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))

    # Acceleration settings
    if not bool(SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    import test as test_mod

    # Data preprocessing (can be skipped quickly)

    print("[Info] Start data preprocessing...")
    train_list, val_list, test_list = data_preprocessing()

    # Compute training stats (ABP)
    print("[Info] Compute training ABP stats...")
    compute_global_abp_stats(train_list)
    _sync_test_globals(test_mod)

    # Pre-training: plot SBP/DBP/mean ABP distributions (one point per segment mean)
    if ENABLE_PLOT_PRETRAIN_DISTRIBUTIONS:
        try:
            plot_pretrain_distributions(train_list, val_list, test_list)
        except Exception as e:
            print(f"[Warn] Failed to plot pre-train distributions: {e}")

    # Physiological feature stats removed
    # print("[Info] Compute training Physio stats...")
    # compute_global_physio_stats(train_list)

    # Create datasets and dataloaders (faster)
    train_dataset = BloodPressureDataset(train_list)
    val_dataset = BloodPressureDataset(val_list)

    # Avoid pickle __main__ failures under spawn; use fork on Linux for multiprocess efficiency
    # Make DataLoader shuffle deterministic: provide generator and worker init
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
        # Prefer setting at PyTorch level for DataLoader reuse
        if use_fork:
            try:
                torch.multiprocessing.set_start_method("fork", force=True)
            except RuntimeError:
                pass  # Already set
        if DATALOADER_WORKERS > 0:
            if use_fork:
                ctx = _mp.get_context("fork")
                dl_kwargs.update(dict(
                    num_workers=DATALOADER_WORKERS,
                    persistent_workers=PERSISTENT_WORKERS,
                    prefetch_factor=2,
                    multiprocessing_context=ctx
                ))
            else:
                # On spawn-only platforms (Windows/macOS), fall back to single process to avoid crashes
                print("[Warn] Cannot use 'fork'; DataLoader set to num_workers=0 to avoid pickle errors")
                dl_kwargs.update(dict(num_workers=0))
    except Exception as e:
        print(f"[Warn] Issue setting DataLoader multiprocessing; fallback to single process: {e}")
        dl_kwargs.update(dict(num_workers=0))

    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    stats = {
        'GLOBAL_ABP_MIN': GLOBAL_ABP_MIN,
        'GLOBAL_ABP_MAX': GLOBAL_ABP_MAX,
    }
    with open('abp_normalization_stats.json', 'w') as f:
        json.dump(stats, f)
    print(f"[Info] Saved normalization stats to abp_normalization_stats.json")

    # Lower BN momentum to stabilize stats (helps val)
    def _set_bn_momentum(m, momentum=0.01):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.momentum = momentum
    # Save stats (Physio removed)
    stats = {
        'GLOBAL_ABP_MIN': GLOBAL_ABP_MIN,
        'GLOBAL_ABP_MAX': GLOBAL_ABP_MAX,
    }
    with open('abp_normalization_stats.json', 'w') as f:
        json.dump(stats, f)
    print(f"[Info] Saved normalization stats to abp_normalization_stats.json")

    # ----------------------
    # Optuna helpers (build loaders/model, compute val loss)
    # ----------------------
    def _make_loaders_with(bs: int):
        # Rebuild datasets (per batch size)
        t_ds = BloodPressureDataset(train_list)
        v_ds = BloodPressureDataset(val_list)
        kw = dict(dl_kwargs)
        kw['batch_size'] = int(max(1, bs))
        t_loader = DataLoader(t_ds, shuffle=True, **kw)
        v_loader = DataLoader(v_ds, shuffle=False, **kw)
        return t_loader, v_loader

    def _build_model_for_variant(bn_key: str, *, tr_nhead=None, tr_dmodel=None, mb_dstate=None):
        if USE_UNET_3L:
            rn = f"resnet_unet3l_{bn_key}"
            m = ResNetUNet3L(input_channels=3, base_channels=BASE_CHANNELS, bottleneck=bn_key).to(device)
        else:
            rn = f"resnet_unet_{bn_key}"
            m = ResNetUNet(input_channels=3, base_channels=BASE_CHANNELS, bottleneck=bn_key).to(device)
        _set_bn_momentum(m, momentum=0.01)
        if USE_COMPILE and hasattr(torch, "compile"):
            try:
                m = torch.compile(m)
                print("[Info] Enabled torch.compile acceleration")
            except Exception as e:
                print(f"[Warn] torch.compile failed: {e}")
        return m, rn

    def _compute_val_loss(model, v_loader):
        model.eval()
        criterion = WaveformLoss()
        loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for inputs, targets, signal_stats in v_loader:
                x = inputs.to(device, non_blocking=True)
                y = targets.to(device, non_blocking=True)
                s = (signal_stats.to(device, non_blocking=True) if USE_SIGNAL_STATS else None)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred = model(x, s)
                    wave_pred, _, _ = _unpack_model_output(pred)
                    loss = criterion(wave_pred, y)
                if torch.isfinite(loss):
                    loss_sum += float(loss.item())
                    n += 1
        return (loss_sum / max(1, n)) if n > 0 else float('inf')

    # Train and compare bottleneck models in order (single config: Mamba / Transformer / Identity)
    skip_mode = _get_skip_mode()
    if skip_mode == "identity":
        variants = [("Identity", "identity")]
    elif skip_mode == "transformer":
        variants = [("Transformer", "transformer")]
    else:
        variants = [("Mamba", "mamba")]

    best_key = None
    best_val_loss = float('inf')
    best_run_name = None

    def _safe_load(model, path):
        sd = torch.load(path, map_location=device)
        try:
            getattr(model, "load_state_dict")(sd, strict=True)
            return True
        except Exception:
            try:
                # Strip _orig_mod. prefix
                cleaned = { (k.replace("_orig_mod.", "") if isinstance(k, str) else k): v for k, v in sd.items() }
                getattr(model, "load_state_dict")(cleaned, strict=False)
                print("[Warn] Weight keys include _orig_mod. prefix; converted and loaded (strict=False)")
                return True
            except Exception as e:
                print(f"[Error] Failed to load weights: {e}")
                return False

    for pretty_name, bn_key in variants:
        if int(OPTUNA_USE) == 1:
            # --- Optuna: build/start dashboard (if available) ---
            if int(OPTUNA_DASHBOARD) == 1:
                try:
                    import subprocess, sys
                    dash_cmd = [sys.executable, "-m", "optuna_dashboard", "--storage", OPTUNA_STORAGE,
                                "--host", "0.0.0.0", "--port", str(int(OPTUNA_DASHBOARD_PORT))]
                    subprocess.Popen(dash_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[Info] optuna-dashboard started at http://127.0.0.1:{int(OPTUNA_DASHBOARD_PORT)} (storage={OPTUNA_STORAGE})")
                except Exception as e:
                    print(f"[Warn] Unable to start optuna-dashboard: {e}")

            try:
                import optuna
            except Exception as e:
                print(f"[Warn] optuna not installed or failed to load ({e}); fall back to normal training.")
                use_optuna = False
            else:
                use_optuna = True

            if use_optuna:
                study_name = _get_optuna_study_name(bn_key)
                try:
                    study = optuna.create_study(direction="minimize", study_name=study_name,
                                                storage=OPTUNA_STORAGE, load_if_exists=True)
                except Exception as e:
                    print(f"[Warn] Failed to create Optuna study ({e}); fall back to normal training.")
                    use_optuna = False

            if use_optuna:
                def objective(trial: 'optuna.trial.Trial'):
                    # Suggest hyperparameters
                    lr_val = trial.suggest_categorical("lr", [1e-5, 2e-5, 5e-5, 1e-4, 5e-4])

                    # Apply learning rate (global)
                    global learning_rate
                    learning_rate = float(lr_val)

                    # Build loaders (fixed batch size)
                    t_loader, v_loader = _make_loaders_with(int(batch_size))

                    # Build model
                    model, run_name = _build_model_for_variant(bn_key)

                    # Shorten epochs per trial (configurable via OPTUNA_TRIAL_EPOCHS)
                    model = train_model(model, t_loader, v_loader, int(OPTUNA_TRIAL_EPOCHS), device, run_name,
                                        save_best=False, plot_curve=False)

                    # Estimate val loss using best weights (if available)
                    val_best = getattr(train_model, "_last_best_val_loss", None)
                    if isinstance(val_best, float) and np.isfinite(val_best):
                        val_loss = float(val_best)
                    else:
                        val_loss = float(_compute_val_loss(model, v_loader))

                    # Reclaim GPU memory (if any)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return val_loss

                # Use OPTUNA_CONTINUE to decide whether to reach total target or just run OPTUNA_TRIALS more
                total_target = int(OPTUNA_TRIALS)
                n_to_run = total_target
                if int(OPTUNA_CONTINUE) == 1:
                    try:
                        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.get_trials())
                        n_to_run = max(0, total_target - completed)
                        print(f"[Info] Optuna resume mode: completed {completed} / target {total_target}, running {n_to_run} trials")
                    except Exception as e:
                        print(f"[Warn] Failed to get completed trials ({e}); using n_trials={total_target}")
                        n_to_run = total_target

                print(f"[Info] Start Optuna search (n_trials={n_to_run}): {bn_key}")
                try:
                    if n_to_run > 0:
                        study.optimize(objective, n_trials=int(n_to_run), show_progress_bar=False)
                    else:
                        print("[Info] Target trial count reached; skipping this search.")
                except Exception as e:
                    print(f"[Warn] Optuna search failed: {e}; fall back to normal training.")
                    use_optuna = False

            if use_optuna:
                best_params = study.best_trial.params
                lr_best = float(best_params.get("lr", learning_rate))

                print("[Info] Optuna best parameters: ", best_params)

                # Retrain fully with best parameters
                learning_rate = lr_best
                train_loader_best, val_loader_best = _make_loaders_with(int(batch_size))
                model, run_name = _build_model_for_variant(bn_key)
                print(f"[Info] Full training with best parameters (epochs={epochs})...")
                model = train_model(model, train_loader_best, val_loader_best, epochs, device, run_name,
                                    save_best=True, plot_curve=True)
                val_loader_used = val_loader_best
            else:
                # Fallback: keep original training flow
                model, run_name = _build_model_for_variant(bn_key)
                print(f"[Info] Start training {pretty_name} bottleneck model (no hyperparameter search)...")
                model = train_model(model, train_loader, val_loader, epochs, device, run_name)
                val_loader_used = val_loader
        else:
            # Use original flow directly
            model, run_name = _build_model_for_variant(bn_key)
            print(f"[Info] Start training {pretty_name} bottleneck model...")
            model = train_model(model, train_loader, val_loader, epochs, device, run_name)
            val_loader_used = val_loader

        # Load best weights
        best_path = f"best_{run_name}.pth"
        if os.path.isfile(best_path):
            _safe_load(model, best_path)

        # Evaluate on validation set to select best model (avoid test leakage)
        val_loss = _compute_val_loss(model, val_loader_used)

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_key = pretty_name
            best_run_name = run_name

    # Visualize best model
    if best_key is not None:
        print(f"\n[Info] Final test and visualization with best model ({best_key})...")
        # Use run_name to choose architecture/filename; avoid 3L mismatch
        run_name_for_best = best_run_name or ""
        best_path = f"best_{run_name_for_best}.pth"

        # Infer bottleneck type and depth
        if "identity" in run_name_for_best:
            bn_key = "identity"
        elif "transformer" in run_name_for_best:
            bn_key = "transformer"
        elif "mamba" in run_name_for_best:
            bn_key = "mamba"
        elif "classic" in run_name_for_best:
            bn_key = "classic"
        else:
            bn_key = "unet"

        if "unet3l" in run_name_for_best:
            best_model = ResNetUNet3L(input_channels=3, base_channels=BASE_CHANNELS, bottleneck=bn_key).to(device)
        else:
            best_model = ResNetUNet(input_channels=3, base_channels=BASE_CHANNELS, bottleneck=bn_key).to(device)

        if os.path.isfile(best_path):
            _safe_load(best_model, best_path)
        else:
            print(f"[Warn] Best model checkpoint not found: {best_path}")

        # Only run test once here at the end
        final_test_results = test_mod.test_model(best_model, test_list, device, collect_errors=False)
        final_w_mae = float(np.mean(final_test_results['wave_mae'])) if final_test_results['wave_mae'] else float('inf')
        final_w_r = float(np.mean(final_test_results['R_all'])) if final_test_results['R_all'] else float('nan')
        print(
            f"[Test-Final] Waveform: MAE {final_w_mae:.4f}, R {final_w_r:.4f}"
        )
        if USE_BOTTLENECK_RIDGE_PROBE:
            try:
                ridge_res_test = test_mod.run_decoder_official_ridge(best_model, train_list, val_list, test_list, device)
                ridge_sbp = ridge_res_test.get("sbp", {})
                ridge_dbp = ridge_res_test.get("dbp", {})
                print(
                    f"[Test-Final][Ridge-DEC] "
                    f"SBP: MAE {ridge_sbp.get('MAE', float('nan')):.4f}, ME {ridge_sbp.get('ME', float('nan')):.4f}, "
                    f"SD {ridge_sbp.get('SD', float('nan')):.4f}, R {ridge_sbp.get('R', float('nan')):.4f} | "
                    f"DBP: MAE {ridge_dbp.get('MAE', float('nan')):.4f}, ME {ridge_dbp.get('ME', float('nan')):.4f}, "
                    f"SD {ridge_dbp.get('SD', float('nan')):.4f}, R {ridge_dbp.get('R', float('nan')):.4f}"
                )
            except Exception as e:
                print(f"[Warn] Final test ridge failed: {e}")

        test_mod.visualize_results(best_model, test_list, device)

    print("\n[Info] Done!")

if __name__ == "__main__":
    if int(TEST_ONLY_MODE) == 1:
        import test as test_mod
        test_mod.main_test_only()
    else:
        main()
