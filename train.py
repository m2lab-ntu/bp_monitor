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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import clone

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from mamba_ssm import Mamba
import tempfile



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
# New: split mode 'subject' or 'segment' (random segment split)
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

# Replace all Mamba skips with Transformer (1=Transformer; 0=Mamba, fallback to conv if not installed; -1=identity pass-through)
USE_TRANSFORMER_SKIPS = -1

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
TEST_ONLY_CHECKPOINT = "pth/best_resnet_unet3l_transformer_20251221-210708_papagei_regression.pth"         # Set .pth weight path (empty uses best_{run_name}.pth)
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

# New: read multiple mappings and merge
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

# New: random segment split from existing .npz (ignore subject)
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
        # New: dropout
        self.dropout = nn.Dropout(DROPOUT_P)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        # New: dropout
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
        self.layer2 = self._make_layer(base_channels, base_channels*2, 2, stride=2)  # 256
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, 2, stride=2)  # 128
        self.layer4 = self._make_layer(base_channels*4, base_channels*8, 2, stride=2)  # 64

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
            ResNetBlock(base_channels, base_channels*2, stride=2,
                        downsample=nn.Sequential(
                            nn.Conv1d(base_channels, base_channels*2, 1, stride=2, bias=False),
                            nn.BatchNorm1d(base_channels*2)
                        )),
            ResNetBlock(base_channels*2, base_channels*2)
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(base_channels*2, base_channels*4, stride=2,
                        downsample=nn.Sequential(
                            nn.Conv1d(base_channels*2, base_channels*4, 1, stride=2, bias=False),
                            nn.BatchNorm1d(base_channels*4)
                        )),
            ResNetBlock(base_channels*4, base_channels*4)
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
# (10-) Core modules: Double Conv and Mamba Skip Block
# ------------------------------------------------------------
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout: float = DROPOUT_P):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout1d(dropout))
        layers.extend([
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0:
            layers.append(nn.Dropout1d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DoubleConv1DNoBN(nn.Module):
    """Two conv + ReLU blocks without BatchNorm/Dropout for pure UNet mode."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

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
# (10=) Simple U-Net encoder/decoder (for PURE mode)
# ------------------------------------------------------------
class SimpleUNetEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS):
        super().__init__()
        self.enc0 = DoubleConv1DNoBN(input_channels, base_channels)
        self.pool0 = nn.MaxPool1d(2)  # 1024 -> 512

        self.enc1 = DoubleConv1DNoBN(base_channels, base_channels)  # 512
        self.pool1 = nn.MaxPool1d(2)  # 512 -> 256

        self.enc2 = DoubleConv1DNoBN(base_channels, base_channels * 2)  # 256
        self.pool2 = nn.MaxPool1d(2)  # 256 -> 128

        self.enc3 = DoubleConv1DNoBN(base_channels * 2, base_channels * 4)  # 128
        self.pool3 = nn.MaxPool1d(2)  # 128 -> 64

        self.enc4 = DoubleConv1DNoBN(base_channels * 4, base_channels * 8)  # 64

    def forward(self, x):
        features = []
        x = self.enc0(x)
        x = self.pool0(x)
        features.append(x)  # f0: [B, base, 512]

        x = self.enc1(x)
        features.append(x)  # f1: [B, base, 512]

        x = self.pool1(x)
        x = self.enc2(x)
        features.append(x)  # f2: [B, 2*base, 256]

        x = self.pool2(x)
        x = self.enc3(x)
        features.append(x)  # f3: [B, 4*base, 128]

        x = self.pool3(x)
        x = self.enc4(x)
        features.append(x)  # f4: [B, 8*base, 64]
        return features


class SimpleUNetDecoder(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS):
        super().__init__()
        # Identity skip refinement blocks (no Mamba)
        self.skip_blocks = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        ])

        self.up4 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv4 = DoubleConv1DNoBN(base_channels * 8 + base_channels * 4, base_channels * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1DNoBN(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1DNoBN(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1DNoBN(base_channels + base_channels, base_channels)

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        f0, f1, f2, f3, _ = features
        x = bottleneck_out  # [B, 8*base, 64]

        x = self.up4(x)  # -> 128
        skip3 = self.skip_blocks[0](f3)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)  # -> 256
        skip2 = self.skip_blocks[1](f2)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)  # -> 512
        skip1 = self.skip_blocks[2](f1)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)  # -> 1024
        skip0 = self.up1(f0)
        skip0 = self.skip_blocks[3](skip0)
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)
        return x

    def forward(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        x = self.final_conv(x)
        return x.squeeze(1)

    def forward_with_last_feature(self, features, bottleneck_out):
        x = self._forward_to_last_feature(features, bottleneck_out)
        out = self.final_conv(x)
        return x, out.squeeze(1)

class SimpleUNetEncoder3L(nn.Module):
    def __init__(self, input_channels=3, base_channels=BASE_CHANNELS):
        super().__init__()
        self.enc0 = DoubleConv1DNoBN(input_channels, base_channels)
        self.pool0 = nn.MaxPool1d(2)  # 1024 -> 512

        self.enc1 = DoubleConv1DNoBN(base_channels, base_channels)  # 512
        self.pool1 = nn.MaxPool1d(2)  # 512 -> 256

        self.enc2 = DoubleConv1DNoBN(base_channels, base_channels * 2)  # 256
        self.pool2 = nn.MaxPool1d(2)  # 256 -> 128

        self.enc3 = DoubleConv1DNoBN(base_channels * 2, base_channels * 4)  # 128

    def forward(self, x):
        features = []
        x = self.enc0(x)
        x = self.pool0(x)
        features.append(x)  # f0: [B, base, 512]

        x = self.enc1(x)
        features.append(x)  # f1: [B, base, 512]

        x = self.pool1(x)
        x = self.enc2(x)
        features.append(x)  # f2: [B, 2*base, 256]

        x = self.pool2(x)
        x = self.enc3(x)
        features.append(x)  # f3: [B, 4*base, 128]
        return features

class SimpleUNetDecoder3L(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS):
        super().__init__()
        self.skip_blocks = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity()
        ])

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = DoubleConv1DNoBN(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = DoubleConv1DNoBN(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = DoubleConv1DNoBN(base_channels + base_channels, base_channels)

        self.final_conv = nn.Conv1d(base_channels, 1, 1)

    def _forward_to_last_feature(self, features, bottleneck_out):
        f0, f1, f2, _ = features
        x = bottleneck_out  # [B, 4*base, 128]

        x = self.up3(x)
        skip2 = self.skip_blocks[0](f2)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        skip1 = self.skip_blocks[1](f1)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        skip0 = self.up1(f0)
        skip0 = self.skip_blocks[2](skip0)
        x = torch.cat([x, skip0], dim=1)
        x = self.conv1(x)
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
# (12) Loss functions and training
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
        plt.show()

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

def calc_ME_SD(t, p):
    e = p - t
    n = e.size if hasattr(e, 'size') else len(e)
    if n <= 1:
        # Definition: sample SD undefined for n<=1; return 0.0 to avoid NaN in mean
        return float(np.mean(e)), 0.0
    return float(np.mean(e)), float(np.std(e, ddof=1))

def calc_MASE(t, p):
    if len(t) < 2: return np.nan
    mae = np.mean(np.abs(p - t))
    denom = np.mean(np.abs(t[1:] - t[:-1]))
    return mae / denom if denom >= 1e-8 else np.nan


def _finalize_regression_metrics(true_list, pred_list):
    if len(true_list) == 0 or len(pred_list) == 0:
        return {
            "MAE": float('nan'),
            "ME": float('nan'),
            "SD": float('nan'),
            "RMSE": float('nan'),
            "N": 0,
        }
    t = np.asarray(true_list, dtype=np.float32)
    p = np.asarray(pred_list, dtype=np.float32)
    mae = mean_absolute_error(t, p)
    me, sd = calc_ME_SD(t, p)
    rmse = float(np.sqrt(np.mean((p - t) ** 2))) if t.size > 0 else float('nan')
    return {
        "MAE": float(mae),
        "ME": float(me),
        "SD": float(sd),
        "RMSE": rmse,
        "N": int(len(t)),
    }


def test_regression_model(model, file_list, device, target_keys=None):
    target_keys = tuple(target_keys or REGRESSION_TARGETS)
    model.eval()
    stores = {k: {"true": [], "pred": []} for k in target_keys}

    with torch.no_grad():
        for p in tqdm(file_list, desc="Testing (regression)"):
            X, Y, _, _ = load_segment(p)
            if X is None:
                continue
            X_norm, stats = normalize_ppg_ecg_segment(X)
            sbp_true, dbp_true = _bp_targets_from_wave(Y)
            target_map = {"sbp": sbp_true, "dbp": dbp_true}

            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if USE_SIGNAL_STATS else None)
            preds = model(x_tensor, stats_tensor)

            for key in target_keys:
                if key not in preds:
                    continue
                true_val = target_map.get(key, np.nan)
                if not np.isfinite(true_val):
                    continue
                pred_val = float(preds[key].detach().cpu().view(-1)[0])
                stores[key]["true"].append(float(true_val))
                stores[key]["pred"].append(pred_val)

    metrics = {k: _finalize_regression_metrics(v["true"], v["pred"]) for k, v in stores.items()}
    return {"metrics": metrics, "raw": stores}


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


def _sklearn_regression_probe(estimator, param_grid, X_train, y_train, X_val, y_val, X_test, y_test,
                              *, refit_with_train_val: bool = False):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) if X_val is not None else None
    X_test = scaler.transform(X_test)

    has_val = X_val is not None and len(y_val) > 0
    best_params = {}
    best_mae = float('inf')

    for params in ParameterGrid(param_grid):
        model = clone(estimator)
        model.set_params(**params)
        model.fit(X_train, y_train)
        if not has_val:
            best_params = params
            break
        y_val_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_val_pred))
        if mae < best_mae:
            best_mae = mae
            best_params = params

    final_model = clone(estimator)
    if best_params:
        final_model.set_params(**best_params)
    if refit_with_train_val and has_val:
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
    else:
        X_fit, y_fit = X_train, y_train
    final_model.fit(X_fit, y_fit)
    y_pred = final_model.predict(X_test)

    y_t = np.asarray(y_test, dtype=np.float32)
    y_p = np.asarray(y_pred, dtype=np.float32)
    err = y_p - y_t
    n = int(err.size)
    if n > 0:
        me = float(np.mean(err))
        sd = float(np.std(err, ddof=1)) if n > 1 else 0.0
    else:
        me = float('nan')
        sd = float('nan')
    if n > 1 and (np.std(y_t) >= 1e-8) and (np.std(y_p) >= 1e-8):
        r = float(np.corrcoef(y_t, y_p)[0, 1])
    else:
        r = 0.0

    return {
        "best_params": best_params,
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "ME": me,
        "SD": sd,
        "R": r,
        "R2": float(r2_score(y_test, y_pred)),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


def _collect_decoder_embeddings_xy(file_list, model, device, desc_tag="train"):
    """Collect decoder-tail embeddings (post skip/transformer, pre-ABP conv) and segment-mean SBP/DBP."""
    base_model = getattr(model, "_orig_mod", model)
    if not hasattr(base_model, "extract_decoder_embedding"):
        raise AttributeError("Model lacks extract_decoder_embedding(); cannot run decoder ridge probe")
    base_model.eval()

    X_list = []
    y_sbp = []
    y_dbp = []
    skipped = 0

    with torch.no_grad():
        for fp in tqdm(file_list, desc=f"Extracting bottleneck embeddings ({desc_tag})"):
            X, Y, _, _ = load_segment(fp)
            if X is None or Y is None:
                skipped += 1
                continue
            X_norm, stats = normalize_ppg_ecg_segment(X)
            sbp_m_t, dbp_m_t, _ = _segment_bp_means_from_wave(Y)
            if not (np.isfinite(sbp_m_t) and np.isfinite(dbp_m_t)):
                skipped += 1
                continue

            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if USE_SIGNAL_STATS else None)
            try:
                emb = base_model.extract_decoder_embedding(x_tensor, stats_tensor)
            except Exception:
                skipped += 1
                continue
            emb_np = emb.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            X_list.append(emb_np)
            y_sbp.append(float(sbp_m_t))
            y_dbp.append(float(dbp_m_t))

    embed_dim = int(X_list[0].shape[0]) if X_list else 0

    def _stack_embeddings_safe(arr_list, dim, tag):
        if not arr_list or dim <= 0:
            return np.empty((0, dim), dtype=np.float32)
        try:
            return np.stack(arr_list, axis=0).astype(np.float32, copy=False)
        except MemoryError:
            tmp_path = os.path.join(processed_dir, f"tmp_decoder_{tag}_{len(arr_list)}.dat")
            mm = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(len(arr_list), dim))
            for i, arr in enumerate(arr_list):
                mm[i] = arr.astype(np.float32, copy=False)
            mm.flush()
            print(f"[Reg-Info] Using memmap to cache decoder embeddings: {tmp_path}")
            return mm

    X_arr = _stack_embeddings_safe(X_list, embed_dim, desc_tag)
    y_sbp_arr = np.asarray(y_sbp, dtype=np.float32) if y_sbp else np.empty((0,), dtype=np.float32)
    y_dbp_arr = np.asarray(y_dbp, dtype=np.float32) if y_dbp else np.empty((0,), dtype=np.float32)
    return X_arr, y_sbp_arr, y_dbp_arr, skipped


def run_decoder_official_ridge(model, train_files, val_files, test_files, device):
    """
    Use decoder-tail features (after skip/transformer, right before ABP head) as Ridge inputs,
    following the official-style Ridge (val-based selection + StandardScaler) for SBP/DBP separately.
    This uses decoder-tail features only.
    """
    X_train, y_sbp_tr, y_dbp_tr, skip_tr = _collect_decoder_embeddings_xy(train_files, model, device, desc_tag="train")
    X_val, y_sbp_val, y_dbp_val, skip_val = _collect_decoder_embeddings_xy(val_files, model, device, desc_tag="val")
    X_test, y_sbp_te, y_dbp_te, skip_te = _collect_decoder_embeddings_xy(test_files, model, device, desc_tag="test")

    estimator = Ridge()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'cholesky', 'sparse_cg'],
    }

    results = {
        "skipped_train": int(skip_tr),
        "skipped_val": int(skip_val),
        "skipped_test": int(skip_te),
        "embed_dim": int(X_train.shape[1]) if hasattr(X_train, "shape") and X_train.ndim == 2 else 0,
    }

    if len(y_sbp_tr) >= 4 and len(y_sbp_val) >= 1 and len(y_sbp_te) >= 1:
        res_sbp = _sklearn_regression_probe(
            estimator, param_grid,
            X_train, y_sbp_tr, X_val, y_sbp_val, X_test, y_sbp_te,
            refit_with_train_val=bool(BOTTLENECK_RIDGE_USE_TRAIN_VAL)
        )
        results["sbp"] = res_sbp | {"train_N": int(len(y_sbp_tr)), "val_N": int(len(y_sbp_val)), "test_N": int(len(y_sbp_te))}
    else:
        results["sbp"] = {"MAE": float('nan'), "R2": float('nan'), "best_params": {}, "train_N": int(len(y_sbp_tr)), "val_N": int(len(y_sbp_val)), "test_N": int(len(y_sbp_te))}

    if len(y_dbp_tr) >= 4 and len(y_dbp_val) >= 1 and len(y_dbp_te) >= 1:
        res_dbp = _sklearn_regression_probe(
            estimator, param_grid,
            X_train, y_dbp_tr, X_val, y_dbp_val, X_test, y_dbp_te,
            refit_with_train_val=bool(BOTTLENECK_RIDGE_USE_TRAIN_VAL)
        )
        results["dbp"] = res_dbp | {"train_N": int(len(y_dbp_tr)), "val_N": int(len(y_dbp_val)), "test_N": int(len(y_dbp_te))}
    else:
        results["dbp"] = {"MAE": float('nan'), "R2": float('nan'), "best_params": {}, "train_N": int(len(y_dbp_tr)), "val_N": int(len(y_dbp_val)), "test_N": int(len(y_dbp_te))}

    return results

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
    plt.show()

def evaluate_segment_level_predictions(model, file_list, device):
    """
    For each segment, predict ABP, compute per-seg SBP/DBP means and average ABP for
    both true and predicted signals. Returns dict of arrays.
    """
    sbp_t, sbp_p = [], []
    dbp_t, dbp_p = [], []
    abp_t, abp_p = [], []
    model.eval()
    with torch.no_grad():
        for p in tqdm(file_list, desc="Post-train: collect segment-level SBP/DBP/ABP"):
            X, Y, _, _ = load_segment(p)
            if X is None:
                continue
            X_norm, stats = normalize_ppg_ecg_segment(X)
            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if USE_SIGNAL_STATS else None)
            out_all = model(x_tensor, stats_tensor)
            pred_norm, _, _ = _unpack_model_output(out_all)
            pred_norm = pred_norm.cpu().numpy().ravel()
            pred = denormalize_abp(pred_norm)

            s_m, d_m, a_m = _segment_bp_means_from_wave(Y)
            s_pm, d_pm, a_pm = _segment_bp_means_from_wave(pred)

            if np.isfinite(s_m) and np.isfinite(s_pm):
                sbp_t.append(s_m); sbp_p.append(s_pm)
            if np.isfinite(d_m) and np.isfinite(d_pm):
                dbp_t.append(d_m); dbp_p.append(d_pm)
            if np.isfinite(a_m) and np.isfinite(a_pm):
                abp_t.append(a_m); abp_p.append(a_pm)

    return {
        'sbp_true': np.asarray(sbp_t, dtype=np.float32),
        'sbp_pred': np.asarray(sbp_p, dtype=np.float32),
        'dbp_true': np.asarray(dbp_t, dtype=np.float32),
        'dbp_pred': np.asarray(dbp_p, dtype=np.float32),
        'abp_true': np.asarray(abp_t, dtype=np.float32),
        'abp_pred': np.asarray(abp_p, dtype=np.float32),
    }

def _binned_mae(x_true: np.ndarray, err: np.ndarray, bin_width=10.0):
    if x_true.size == 0:
        return np.array([]), np.array([])
    x_min, x_max = float(np.floor(np.nanmin(x_true))), float(np.ceil(np.nanmax(x_true)))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.array([]), np.array([])
    edges = np.arange(x_min, x_max + bin_width, bin_width)
    inds = np.digitize(x_true, edges) - 1
    centers, maes = [], []
    for b in range(len(edges) - 1):
        m = inds == b
        if not np.any(m):
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        maes.append(float(np.nanmean(err[m])))
    return np.asarray(centers, dtype=np.float32), np.asarray(maes, dtype=np.float32)

def plot_post_training_metrics(model, file_list, device):
    """Plot per-segment BA scatter and true vs pred scatter for SBP/DBP/ABP (controlled by flags)."""
    if not (ENABLE_PLOT_BA_SCATTER or ENABLE_PLOT_TRUEPRED_SCATTER):
        return
    seg = evaluate_segment_level_predictions(model, file_list, device)
    sbp_t, sbp_p = seg['sbp_true'], seg['sbp_pred']
    dbp_t, dbp_p = seg['dbp_true'], seg['dbp_pred']
    abp_t, abp_p = seg['abp_true'], seg['abp_pred']

    sbp_err = np.abs(sbp_p - sbp_t) if sbp_t.size and sbp_p.size else np.array([])
    dbp_err = np.abs(dbp_p - dbp_t) if dbp_t.size and dbp_p.size else np.array([])

    # Figure 1: Bland–Altman plots (per segment)
    if ENABLE_PLOT_BA_SCATTER:
        plt.figure(figsize=(12, 4))
        # SBP BA
        plt.subplot(1, 2, 1)
        if sbp_t.size and sbp_p.size:
            sbp_mean = 0.5 * (sbp_t + sbp_p)
            sbp_diff = sbp_p - sbp_t
            md = float(np.nanmean(sbp_diff))
            sd = float(np.nanstd(sbp_diff, ddof=1)) if sbp_diff.size > 1 else 0.0
            loa_low, loa_high = md - 1.96 * sd, md + 1.96 * sd
            plt.scatter(sbp_mean, sbp_diff, s=8, alpha=0.35, color="#4C78A8", edgecolors='none')
            xmin, xmax = np.nanmin(sbp_mean), np.nanmax(sbp_mean)
            plt.hlines([md, loa_low, loa_high], xmin, xmax, colors=['k', 'r', 'r'], linestyles=['-', '--', '--'], linewidths=1)
        plt.title("SBP Bland–Altman (per segment)")
        plt.xlabel("Mean SBP (mmHg)")
        plt.ylabel("Difference (Pred - True) (mmHg)")
        plt.grid(True, alpha=0.3)

        # DBP BA
        plt.subplot(1, 2, 2)
        if dbp_t.size and dbp_p.size:
            dbp_mean = 0.5 * (dbp_t + dbp_p)
            dbp_diff = dbp_p - dbp_t
            md = float(np.nanmean(dbp_diff))
            sd = float(np.nanstd(dbp_diff, ddof=1)) if dbp_diff.size > 1 else 0.0
            loa_low, loa_high = md - 1.96 * sd, md + 1.96 * sd
            plt.scatter(dbp_mean, dbp_diff, s=8, alpha=0.35, color="#E45756", edgecolors='none')
            xmin, xmax = np.nanmin(dbp_mean), np.nanmax(dbp_mean)
            plt.hlines([md, loa_low, loa_high], xmin, xmax, colors=['k', 'r', 'r'], linestyles=['-', '--', '--'], linewidths=1)
        plt.title("DBP Bland–Altman (per segment)")
        plt.xlabel("Mean DBP (mmHg)")
        plt.ylabel("Difference (Pred - True) (mmHg)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Figure 2: True vs Pred distributions for SBP/DBP/ABP
    if ENABLE_PLOT_TRUEPRED_SCATTER:
        plt.figure(figsize=(15, 4))
        lims = None
        if sbp_t.size:
            lims = (float(np.nanmin(sbp_t)), float(np.nanmax(sbp_t)))
        if dbp_t.size:
            dl = (float(np.nanmin(dbp_t)), float(np.nanmax(dbp_t)))
            if lims is None:
                lims = dl
            else:
                lims = (min(lims[0], dl[0]), max(lims[1], dl[1]))

        def _scatter(ax_idx, xt, xp, title, color):
            plt.subplot(1, 3, ax_idx)
            if xt.size and xp.size:
                plt.scatter(xt, xp, s=8, alpha=0.35, color=color, edgecolors='none')
                m = min(float(np.nanmin(xt)), float(np.nanmin(xp)))
                M = max(float(np.nanmax(xt)), float(np.nanmax(xp)))
                plt.plot([m, M], [m, M], 'k--', linewidth=1)
                plt.xlim(m, M)
                plt.ylim(m, M)
            plt.title(title)
            plt.xlabel("True (mmHg)")
            plt.ylabel("Predicted (mmHg)")
            plt.grid(True, alpha=0.2)

        _scatter(1, sbp_t, sbp_p, "SBP DBP: Actual vs. Predicted", "#4C78A8")
        _scatter(2, dbp_t, dbp_p, "DBP: Actual vs. Predicted", "#F58518")
        _scatter(3, abp_t, abp_p, "ABP: Actual vs. Predicted", "#54A24B")
        plt.tight_layout()
        plt.show()

def plot_segmean_truepred_scatter(sbp_t, sbp_p, dbp_t, dbp_p, title_prefix: str = ""):
    sbp_t = np.asarray(sbp_t, dtype=np.float32)
    sbp_p = np.asarray(sbp_p, dtype=np.float32)
    dbp_t = np.asarray(dbp_t, dtype=np.float32)
    dbp_p = np.asarray(dbp_p, dtype=np.float32)
    if sbp_t.size == 0 and dbp_t.size == 0:
        print("[Warn] No valid SBP/DBP data to plot scatter")
        return

    def _scatter(ax_idx, xt, xp, title, color):
        plt.subplot(1, 2, ax_idx)
        if xt.size and xp.size:
            plt.scatter(xt, xp, s=10, alpha=0.4, color=color, edgecolors='none')
            m = min(float(np.nanmin(xt)), float(np.nanmin(xp)))
            M = max(float(np.nanmax(xt)), float(np.nanmax(xp)))
            plt.plot([m, M], [m, M], 'k--', linewidth=1)
            plt.xlim(m, M)
            plt.ylim(m, M)
        plt.title(title)
        plt.xlabel("True (mmHg)")
        plt.ylabel("Predicted (mmHg)")
        plt.grid(True, alpha=0.2)

    title_prefix = f"{title_prefix} " if title_prefix else ""
    plt.figure(figsize=(10, 4))
    _scatter(1, sbp_t, sbp_p, f"{title_prefix}SBP_mean: True vs Pred", "#4C78A8")
    _scatter(2, dbp_t, dbp_p, f"{title_prefix}DBP_mean: True vs Pred", "#F58518")
    plt.tight_layout()
    plt.show()

def test_model(model, test_list, device, collect_errors: bool = False):
    model.eval()

    # Per-segment metrics (current algorithm)
    wave_mae, wave_me, wave_sd, wave_mase, R_all = [], [], [], [], []
    head_sbp_mae_list, head_dbp_mae_list = [], []
    sbp_mean_true_list, sbp_mean_pred_list = [], []
    dbp_mean_true_list, dbp_mean_pred_list = [], []

    # Global residual accumulators (across all segments, all samples)
    total_n = 0
    sum_e = 0.0
    sum_abs_e = 0.0
    sum_e2 = 0.0
    err_parts = [] if collect_errors else None

    # Filtered/Strict accumulators (ABP samples, per-segment gate)
    total_n_filt = 0
    sum_e_filt = 0.0
    sum_abs_e_filt = 0.0
    sum_e2_filt = 0.0

    total_n_strict = 0
    sum_e_strict = 0.0
    sum_abs_e_strict = 0.0
    sum_e2_strict = 0.0

    # Global residual accumulators using per-segment mean SBP/DBP (one value per segment)
    sbp_mean_total_n = 0
    sbp_mean_sum_e = 0.0
    sbp_mean_sum_abs_e = 0.0
    sbp_mean_sum_e2 = 0.0

    dbp_mean_total_n = 0
    dbp_mean_sum_e = 0.0
    dbp_mean_sum_abs_e = 0.0
    dbp_mean_sum_e2 = 0.0

    # Per-segment mean SBP filtered accumulators (new: SBP<=170 and SBP<=160)
    sbp_mean_total_n_filt170 = 0
    sbp_mean_sum_e_filt170 = 0.0
    sbp_mean_sum_abs_e_filt170 = 0.0
    sbp_mean_sum_e2_filt170 = 0.0

    sbp_mean_total_n_filt160 = 0
    sbp_mean_sum_e_filt160 = 0.0
    sbp_mean_sum_abs_e_filt160 = 0.0
    sbp_mean_sum_e2_filt160 = 0.0

    with torch.no_grad():
        for p in tqdm(test_list, desc="Testing"):
            X, Y, _, _ = load_segment(p)
            if X is None:
                continue
            # Standardize inputs; normalize/denormalize handled outside predictions
            X_norm, stats = normalize_ppg_ecg_segment(X)
            X_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if USE_SIGNAL_STATS else None)

            # Model inference (prediction on normalized scale -> denormalize to mmHg)
            out_all = model(X_tensor, stats_tensor)
            wave_pred_norm, sbp_head_pred, dbp_head_pred = _unpack_model_output(out_all)
            pred_norm = wave_pred_norm.cpu().numpy().ravel()
            pred = denormalize_abp(pred_norm)

            # Per-segment waveform metrics
            mae = mean_absolute_error(Y, pred)
            me, sd = calc_ME_SD(Y, pred)
            R = np.corrcoef(Y, pred)[0, 1] if (
                np.std(Y) >= 1e-8 and np.std(pred) >= 1e-8) else 0.0
            mase = calc_MASE(Y, pred)

            # Per-segment mean SBP/DBP residuals (one per segment if valid)
            # Also compute filter gates for filtered/strict variants
            seg_cond_filtered = False
            seg_cond_strict = False
            try:
                sbp_m_t, dbp_m_t, _ = _segment_bp_means_from_wave(Y)
                sbp_m_p, dbp_m_p, _ = _segment_bp_means_from_wave(pred)

                # Original accumulators
                if np.isfinite(sbp_m_t) and np.isfinite(sbp_m_p):
                    e_sbp = float(sbp_m_p - sbp_m_t)
                    sbp_mean_total_n += 1
                    sbp_mean_sum_e += e_sbp
                    sbp_mean_sum_abs_e += abs(e_sbp)
                    sbp_mean_sum_e2 += e_sbp * e_sbp
                    sbp_mean_true_list.append(float(sbp_m_t))
                    sbp_mean_pred_list.append(float(sbp_m_p))
                else:
                    e_sbp = None

                if np.isfinite(dbp_m_t) and np.isfinite(dbp_m_p):
                    e_dbp = float(dbp_m_p - dbp_m_t)
                    dbp_mean_total_n += 1
                    dbp_mean_sum_e += e_dbp
                    dbp_mean_sum_abs_e += abs(e_dbp)
                    dbp_mean_sum_e2 += e_dbp * e_dbp
                    dbp_mean_true_list.append(float(dbp_m_t))
                    dbp_mean_pred_list.append(float(dbp_m_p))
                else:
                    e_dbp = None

                if USE_MULTITASK_SBP_DBP and sbp_head_pred is not None and dbp_head_pred is not None:
                    if np.isfinite(sbp_m_t):
                        sbp_head_val = float(sbp_head_pred.detach().cpu().view(-1)[0])
                        head_sbp_mae_list.append(abs(sbp_head_val - sbp_m_t))
                    if np.isfinite(dbp_m_t):
                        dbp_head_val = float(dbp_head_pred.detach().cpu().view(-1)[0])
                        head_dbp_mae_list.append(abs(dbp_head_val - dbp_m_t))

                # Determine filter gates using true segment-level means
                if np.isfinite(sbp_m_t):
                    seg_cond_filtered = (sbp_m_t <= 170.0)
                if np.isfinite(sbp_m_t) and np.isfinite(dbp_m_t):
                    seg_cond_strict = (sbp_m_t <= 160.0) and (dbp_m_t >= 50.0)

                # Update new per-segment-mean SBP filtered accumulators (by SBP only)
                if (e_sbp is not None):
                    if np.isfinite(sbp_m_t) and (sbp_m_t <= 170.0):
                        sbp_mean_total_n_filt170 += 1
                        sbp_mean_sum_e_filt170 += e_sbp
                        sbp_mean_sum_abs_e_filt170 += abs(e_sbp)
                        sbp_mean_sum_e2_filt170 += e_sbp * e_sbp
                    if np.isfinite(sbp_m_t) and (sbp_m_t <= 160.0):
                        sbp_mean_total_n_filt160 += 1
                        sbp_mean_sum_e_filt160 += e_sbp
                        sbp_mean_sum_abs_e_filt160 += abs(e_sbp)
                        sbp_mean_sum_e2_filt160 += e_sbp * e_sbp
            except Exception:
                pass

            # Collect per-segment summaries
            wave_mae.append(mae)
            wave_me.append(me)
            wave_sd.append(sd)
            wave_mase.append(mase)
            R_all.append(R)

            # Update global residual accumulators
            e = pred - Y
            # ensure finite
            if np.any(np.isfinite(e)):
                msk = np.isfinite(e)
                e = e[msk]
                n = e.size
                if n > 0:
                    total_n += int(n)
                    sum_e += float(np.sum(e))
                    sum_abs_e += float(np.sum(np.abs(e)))
                    sum_e2 += float(np.sum(e * e))
                    if collect_errors:
                        err_parts.append(e.astype(np.float32, copy=False))

                    # Add to filtered/strict sample-level accumulators if this segment passes gates
                    if seg_cond_filtered:
                        total_n_filt += int(n)
                        sum_e_filt += float(np.sum(e))
                        sum_abs_e_filt += float(np.sum(np.abs(e)))
                        sum_e2_filt += float(np.sum(e * e))
                    if seg_cond_strict:
                        total_n_strict += int(n)
                        sum_e_strict += float(np.sum(e))
                        sum_abs_e_strict += float(np.sum(np.abs(e)))
                        sum_e2_strict += float(np.sum(e * e))

    # Compute global metrics over concatenated residuals (population stats)
    def _finalize_sample_metrics(N, s_e, s_abs, s_e2):
        if N > 1:
            me_ = s_e / N
            mae_ = s_abs / N
            var_num = s_e2 - (s_e * s_e) / N
            var_ = max(0.0, var_num / (N - 1))
            sd_ = math.sqrt(var_)
            rmse_ = math.sqrt(s_e2 / N)
        else:
            me_ = float('nan')
            sd_ = float('nan')
            mae_ = float('nan')
            rmse_ = float('nan')
        return me_, mae_, sd_, rmse_

    me_global, mae_global, sd_global, rmse_global = _finalize_sample_metrics(total_n, sum_e, sum_abs_e, sum_e2)
    me_global_filt, mae_global_filt, sd_global_filt, rmse_global_filt = _finalize_sample_metrics(total_n_filt, sum_e_filt, sum_abs_e_filt, sum_e2_filt)
    me_global_strict, mae_global_strict, sd_global_strict, rmse_global_strict = _finalize_sample_metrics(total_n_strict, sum_e_strict, sum_abs_e_strict, sum_e2_strict)

    # Compute per-segment mean SBP/DBP global residual metrics
    def _finalize_segmean_metrics(N, s_e, s_abs, s_e2):
        if N > 1:
            me_ = s_e / N
            mae_ = s_abs / N
            num_ = s_e2 - (s_e * s_e) / N
            var_ = max(0.0, num_ / (N - 1))
            sd_ = math.sqrt(var_)
            rmse_ = math.sqrt(s_e2 / N)
        else:
            me_ = float('nan')
            mae_ = float('nan')
            sd_ = float('nan')
            rmse_ = float('nan')
        return me_, mae_, sd_, rmse_

    def _finalize_segmean_r(true_list, pred_list):
        if not true_list or not pred_list:
            return float('nan')
        t = np.asarray(true_list, dtype=np.float32)
        p = np.asarray(pred_list, dtype=np.float32)
        if t.size < 2 or p.size < 2:
            return float('nan')
        if np.std(t) < 1e-8 or np.std(p) < 1e-8:
            return 0.0
        return float(np.corrcoef(t, p)[0, 1])

    sbp_mean_me_global, sbp_mean_mae_global, sbp_mean_sd_global, sbp_mean_rmse_global = _finalize_segmean_metrics(
        sbp_mean_total_n, sbp_mean_sum_e, sbp_mean_sum_abs_e, sbp_mean_sum_e2
    )
    sbp_mean_r_global = _finalize_segmean_r(sbp_mean_true_list, sbp_mean_pred_list)
    # New: finalize SBP seg-mean filtered metrics (SBP<=170 and SBP<=160)
    sbp_mean_me_global_filt170, sbp_mean_mae_global_filt170, sbp_mean_sd_global_filt170, sbp_mean_rmse_global_filt170 = _finalize_segmean_metrics(
        sbp_mean_total_n_filt170, sbp_mean_sum_e_filt170, sbp_mean_sum_abs_e_filt170, sbp_mean_sum_e2_filt170
    )
    sbp_mean_me_global_filt160, sbp_mean_mae_global_filt160, sbp_mean_sd_global_filt160, sbp_mean_rmse_global_filt160 = _finalize_segmean_metrics(
        sbp_mean_total_n_filt160, sbp_mean_sum_e_filt160, sbp_mean_sum_abs_e_filt160, sbp_mean_sum_e2_filt160
    )

    dbp_mean_me_global, dbp_mean_mae_global, dbp_mean_sd_global, dbp_mean_rmse_global = _finalize_segmean_metrics(
        dbp_mean_total_n, dbp_mean_sum_e, dbp_mean_sum_abs_e, dbp_mean_sum_e2
    )
    dbp_mean_r_global = _finalize_segmean_r(dbp_mean_true_list, dbp_mean_pred_list)

    result = {
        'wave_mae': wave_mae, 'wave_me': wave_me, 'wave_sd': wave_sd,
        'wave_mase': wave_mase, 'R_all': R_all,
        'global_metrics': {
            'N': total_n,
            'ME': me_global,
            'SD': sd_global,
            'MAE': mae_global,
            'RMSE': rmse_global,
        },
        'global_metrics_filtered': {
            'N': total_n_filt,
            'ME': me_global_filt,
            'SD': sd_global_filt,
            'MAE': mae_global_filt,
            'RMSE': rmse_global_filt,
        },
        'global_metrics_strict': {
            'N': total_n_strict,
            'ME': me_global_strict,
            'SD': sd_global_strict,
            'MAE': mae_global_strict,
            'RMSE': rmse_global_strict,
        },
        'head_sbp_mae_mean': float(np.mean(head_sbp_mae_list)) if head_sbp_mae_list else np.nan,
        'head_dbp_mae_mean': float(np.mean(head_dbp_mae_list)) if head_dbp_mae_list else np.nan,
        'global_sbp_mean_metrics': {
            'N': sbp_mean_total_n,
            'ME': sbp_mean_me_global,
            'SD': sbp_mean_sd_global,
            'MAE': sbp_mean_mae_global,
            'RMSE': sbp_mean_rmse_global,
            'R': sbp_mean_r_global,
        },
        # New: SBP seg-mean filtered groups
        'global_sbp_mean_metrics_filtered170': {
            'N': sbp_mean_total_n_filt170,
            'ME': sbp_mean_me_global_filt170,
            'SD': sbp_mean_sd_global_filt170,
            'MAE': sbp_mean_mae_global_filt170,
            'RMSE': sbp_mean_rmse_global_filt170,
        },
        'global_sbp_mean_metrics_filtered160': {
            'N': sbp_mean_total_n_filt160,
            'ME': sbp_mean_me_global_filt160,
            'SD': sbp_mean_sd_global_filt160,
            'MAE': sbp_mean_mae_global_filt160,
            'RMSE': sbp_mean_rmse_global_filt160,
        },
        # Removed: global_sbp_mean_metrics_filtered and _strict
        'global_dbp_mean_metrics': {
            'N': dbp_mean_total_n,
            'ME': dbp_mean_me_global,
            'SD': dbp_mean_sd_global,
            'MAE': dbp_mean_mae_global,
            'RMSE': dbp_mean_rmse_global,
            'R': dbp_mean_r_global,
        }
    }

    if collect_errors and err_parts is not None and len(err_parts) > 0:
        try:
            result['all_errors'] = np.concatenate(err_parts)
        except Exception:
            # fallback: skip storing if memory error
            result['all_errors'] = None

    return result

def visualize_results(model, test_list, device, num_samples=3):
    model.eval()
    sample_indices = np.linspace(0, len(test_list)-1, num_samples, dtype=int)
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            p = test_list[idx]
            X, Y, _, _ = load_segment(p)
            if X is None:
                continue
            Y_normalized = normalize_abp(Y)
            X_norm, stats = normalize_ppg_ecg_segment(X)
            X_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if USE_SIGNAL_STATS else None)
            out_all = model(X_tensor, stats_tensor)
            pred_norm, _, _ = _unpack_model_output(out_all)
            pred_norm = pred_norm.cpu().numpy().ravel()
            pred_sig = denormalize_abp(pred_norm)

            plt.figure(figsize=(15, 5))
            plt.plot(Y, label="True ABP", alpha=0.8, linewidth=1)
            plt.plot(pred_sig, label="Predicted ABP", alpha=0.8, linewidth=1)
            plt.title(f"Sample {i+1}: True Range {np.min(Y):.1f}-{np.max(Y):.1f}")
            plt.xlabel("Sample Index")
            plt.ylabel("Blood Pressure (mmHg)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

def main_test_only():
    global train_list, val_list, test_list
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX

    set_global_seed(GLOBAL_SEED, bool(SEED_DETERMINISTIC))
    if not bool(SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("[Info] Test-only mode: build data lists...")
    train_list, val_list, test_list = data_preprocessing()

    stats_loaded = False
    if TEST_ONLY_USE_SAVED_STATS:
        stats_loaded = _load_saved_norm_stats(TEST_ONLY_STATS_PATH)
    if not stats_loaded or GLOBAL_ABP_MIN is None or GLOBAL_ABP_MAX is None:
        print("[Info] Compute training ABP stats...")
        compute_global_abp_stats(train_list)

    def _build_test_model():
        try:
            depth_mode = int(USE_UNET_3L)
        except Exception:
            depth_mode = 0
        if depth_mode not in (0, 1, 2):
            depth_mode = 0

        bn_key = _get_skip_mode()
        if depth_mode == 1:
            return ResNetUNet3L(input_channels=3, base_channels=BASE_CHANNELS, bottleneck=bn_key).to(device), f"resnet_unet3l_{bn_key}"
        return ResNetUNet(input_channels=3, base_channels=BASE_CHANNELS, bottleneck=bn_key).to(device), f"resnet_unet_{bn_key}"

    def _safe_load(model, path):
        sd = torch.load(path, map_location=device)
        try:
            getattr(model, "load_state_dict")(sd, strict=True)
            return True
        except Exception:
            try:
                cleaned = { (k.replace("_orig_mod.", "") if isinstance(k, str) else k): v for k, v in sd.items() }
                getattr(model, "load_state_dict")(cleaned, strict=False)
                print("[Warn] Weight keys include _orig_mod. prefix; converted and loaded (strict=False)")
                return True
            except Exception as e:
                print(f"[Error] Failed to load weights: {e}")
                return False

    model, run_name = _build_test_model()
    ckpt_path = str(TEST_ONLY_CHECKPOINT).strip()
    if not ckpt_path:
        ckpt_path = f"best_{run_name}.pth"
    if not os.path.isfile(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return
    _safe_load(model, ckpt_path)

    # Test-only: evaluate test only
    final_test_results = test_model(model, test_list, device, collect_errors=False)
    final_w_mae = float(np.mean(final_test_results['wave_mae'])) if final_test_results['wave_mae'] else float('inf')
    final_w_r = float(np.mean(final_test_results['R_all'])) if final_test_results['R_all'] else float('nan')

    print(
        f"[Test-Only] Waveform: MAE {final_w_mae:.4f}, R {final_w_r:.4f}"
    )

    ridge_res_test = None
    if USE_BOTTLENECK_RIDGE_PROBE and TEST_ONLY_RUN_RIDGE:
        try:
            ridge_res_test = run_decoder_official_ridge(model, train_list, val_list, test_list, device)
            ridge_sbp = ridge_res_test.get("sbp", {})
            ridge_dbp = ridge_res_test.get("dbp", {})
            print(
                f"[Test-Only][Ridge-DEC] "
                f"SBP: MAE {ridge_sbp.get('MAE', float('nan')):.4f}, ME {ridge_sbp.get('ME', float('nan')):.4f}, "
                f"SD {ridge_sbp.get('SD', float('nan')):.4f}, R {ridge_sbp.get('R', float('nan')):.4f} | "
                f"DBP: MAE {ridge_dbp.get('MAE', float('nan')):.4f}, ME {ridge_dbp.get('ME', float('nan')):.4f}, "
                f"SD {ridge_dbp.get('SD', float('nan')):.4f}, R {ridge_dbp.get('R', float('nan')):.4f}"
            )
        except Exception as e:
            print(f"[Warn] Test-only ridge failed: {e}")

    if TEST_ONLY_PLOT_TRUEPRED_SCATTER:
        try:
            seg = evaluate_segment_level_predictions(model, test_list, device)
            plot_segmean_truepred_scatter(
                seg.get('sbp_true', []), seg.get('sbp_pred', []),
                seg.get('dbp_true', []), seg.get('dbp_pred', []),
                title_prefix="Waveform"
            )
        except Exception as e:
            print(f"[Warn] Test-only waveform scatter failed: {e}")

        if ridge_res_test is not None:
            try:
                ridge_sbp = ridge_res_test.get("sbp", {})
                ridge_dbp = ridge_res_test.get("dbp", {})
                sbp_t = ridge_sbp.get("y_test", [])
                sbp_p = ridge_sbp.get("y_pred", [])
                dbp_t = ridge_dbp.get("y_test", [])
                dbp_p = ridge_dbp.get("y_pred", [])
                if sbp_t and sbp_p and dbp_t and dbp_p:
                    plot_segmean_truepred_scatter(
                        sbp_t, sbp_p, dbp_t, dbp_p, title_prefix="Ridge-DEC"
                    )
                else:
                    print("[Warn] Ridge-DEC scatter data insufficient; skipped")
            except Exception as e:
                print(f"[Warn] Test-only Ridge-DEC scatter failed: {e}")

    if TEST_ONLY_VISUALIZE:
        visualize_results(model, test_list, device)

    print("\n[Info] Test-only complete!")

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

    # Data preprocessing (can be skipped quickly)

    print("[Info] Start data preprocessing...")
    train_list, val_list, test_list = data_preprocessing()

    # Compute training stats (ABP)
    print("[Info] Compute training ABP stats...")
    compute_global_abp_stats(train_list)

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
        final_test_results = test_model(best_model, test_list, device, collect_errors=False)
        final_w_mae = float(np.mean(final_test_results['wave_mae'])) if final_test_results['wave_mae'] else float('inf')
        final_w_r = float(np.mean(final_test_results['R_all'])) if final_test_results['R_all'] else float('nan')
        print(
            f"[Test-Final] Waveform: MAE {final_w_mae:.4f}, R {final_w_r:.4f}"
        )
        if USE_BOTTLENECK_RIDGE_PROBE:
            try:
                ridge_res_test = run_decoder_official_ridge(best_model, train_list, val_list, test_list, device)
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

        visualize_results(best_model, test_list, device)

    print("\n[Info] Done!")

if __name__ == "__main__":
    if int(TEST_ONLY_MODE) == 1:
        main_test_only()
    else:
        main()
