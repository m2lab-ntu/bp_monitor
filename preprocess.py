# CSV -> NPZ segments, the processed manifest, subject splits and k-fold, ABP normalization
# stats, segment loading and BP readout. train.py hands the data-format and path settings over
# once through configure(). The manifest and the ABP stats live here: read them through the
# module (preprocess.manifest(), preprocess.GLOBAL_ABP_MIN), never import them by name.
import os, glob, json, re
import hashlib
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks

# Settings. train.py overrides them through configure() before anything runs.
FS = 125
SEGMENT_LEN, STEP_SIZE = 1024, 1024
APPLY_FILTERING = True
NPZ_COMPRESS = False
PROCESSED_DIR = ""
DATA_PATHS = ()
SUBJECT_FILE_MAP_PATHS = ()
WORKERS = 1
DATA_SAMPLE_RATIO = 1.0
DATA_SAMPLE_SEED = 42

_REQUIRED_CSV_COLS = ["PLETH", "II", "dPPG", "ABP"]
_OPTIONAL_CSV_COLS = ["raw_sample_idx", "master_record_raw_sample_idx"]

# Runtime state, owned here.
GLOBAL_ABP_MIN = None
GLOBAL_ABP_MAX = None
_PROCESSED_MANIFEST: Optional[pd.DataFrame] = None
# npz_path -> manifest row as a dict.
_PATH_TO_META: dict = {}


def configure(*, fs, segment_len, step_size, apply_filtering, processed_dir, npz_compress,
              data_paths, subject_file_map_paths, workers, data_sample_ratio, data_sample_seed):
    global FS, SEGMENT_LEN, STEP_SIZE, APPLY_FILTERING, NPZ_COMPRESS, PROCESSED_DIR
    global DATA_PATHS, SUBJECT_FILE_MAP_PATHS, WORKERS, DATA_SAMPLE_RATIO, DATA_SAMPLE_SEED
    global b_bp, a_bp
    FS = int(fs)
    SEGMENT_LEN, STEP_SIZE = int(segment_len), int(step_size)
    APPLY_FILTERING = bool(apply_filtering)
    NPZ_COMPRESS = bool(npz_compress)
    PROCESSED_DIR = str(processed_dir)
    DATA_PATHS = tuple(data_paths)
    SUBJECT_FILE_MAP_PATHS = tuple(subject_file_map_paths)
    WORKERS = int(workers)
    DATA_SAMPLE_RATIO = float(data_sample_ratio)
    DATA_SAMPLE_SEED = int(data_sample_seed)
    b_bp, a_bp = create_bandpass_filter()


def manifest() -> Optional[pd.DataFrame]:
    return _PROCESSED_MANIFEST


def meta_for(npz_path) -> Optional[dict]:
    return _PATH_TO_META.get(npz_path)


def set_abp_stats(abp_min, abp_max):
    global GLOBAL_ABP_MIN, GLOBAL_ABP_MAX
    GLOBAL_ABP_MIN, GLOBAL_ABP_MAX = abp_min, abp_max


# Signal filtering, normalization, alignment
def normalize_abp(abp):
    return (abp - GLOBAL_ABP_MIN) / (GLOBAL_ABP_MAX - GLOBAL_ABP_MIN)


def denormalize_abp(abp_norm):
    return abp_norm * (GLOBAL_ABP_MAX - GLOBAL_ABP_MIN) + GLOBAL_ABP_MIN


def create_bandpass_filter(lowcut=0.5, highcut=8.0, fs=None, order=4):
    fs = FS if fs is None else fs
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
    for d in (DATA_PATHS if isinstance(DATA_PATHS, (list, tuple)) else []):
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


def chunk_data_with_index(X, Y, raw_idx, master_idx, seg_len=None, step=None):
    seg_len = SEGMENT_LEN if seg_len is None else seg_len
    step = STEP_SIZE if step is None else step
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

        out_p = os.path.join(PROCESSED_DIR,
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


def attach_record_and_subject_indices(df):
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

    # Order windows within each (sid, master_record) along the master timeline; record_window_index follows it.
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


def build_or_update_processed_manifest(npz_files, output_manifest_path, min_windows=None):
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
                f"record_window_index. Set FORCE_REBUILD_PROCESSED=True to "
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
    df = attach_record_and_subject_indices(df)

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
    usable = (f"; usable with >={min_windows} windows: {int((sizes >= min_windows).sum())}"
              if min_windows else "")
    logging.info(
        f"Saved processed manifest: {output_manifest_path} "
        f"({n_rows} windows, {n_subjects} subjects, {n_masters} master_records{usable})"
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
        df = attach_record_and_subject_indices(df)
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


def get_sid_from_path(fpath):
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


def get_record_window_index(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath].get("record_window_index", -1)
    return -1


def get_master_record(fpath):
    if fpath in _PATH_TO_META:
        return _PATH_TO_META[fpath].get("master_record", "")
    return ""


# Data lists from the manifest, k-fold partition
def _sample_list(lst, ratio, seed):
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

    train_list = _sample_list(train_list, DATA_SAMPLE_RATIO, DATA_SAMPLE_SEED)
    val_list = _sample_list(val_list, DATA_SAMPLE_RATIO, DATA_SAMPLE_SEED)
    test_list = _sample_list(test_list, DATA_SAMPLE_RATIO, DATA_SAMPLE_SEED)
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
                sbp_m, dbp_m, _ = segment_bp_means_from_wave(y)
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


def stratified_subject_kfold(subj_stats, all_subjects, n_splits, sort_key="sbp", *, seed):
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


# Preprocessing driver. Existing NPZ are always reused and only the manifest is refreshed; the
# cache is deleted and rebuilt from CSV only with force_rebuild=True. An empty cache is built
# from CSV without deleting anything. ratios: (train, val, test) subject shares for a fresh
# split; min_windows only feeds the manifest log.
def data_preprocessing(split_json_path: str, processed_manifest_path: str, *,
                       force_rebuild: bool, data_usage_ratio: float, ratios,
                       summary_csv_path: str, min_windows: int):
    train_ratio, val_ratio, test_ratio = ratios
    old_npz = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.npz")))

    if old_npz and not force_rebuild:
        # Try loading existing manifest
        manifest = load_processed_manifest(processed_manifest_path)
        if manifest is None:
            logging.info("No processed_manifest.csv found; rebuilding from NPZ metadata...")
            manifest = build_or_update_processed_manifest(old_npz, processed_manifest_path, min_windows)
        else:
            logging.info(f"Loaded existing manifest: {processed_manifest_path}")
            # If new NPZ files exist that aren't in manifest, rebuild
            manifest_paths = set(manifest["npz_path"].tolist())
            missing = [p for p in old_npz if p not in manifest_paths]
            if missing:
                logging.info(f"{len(missing)} new NPZ files not in manifest; rebuilding...")
                manifest = build_or_update_processed_manifest(old_npz, processed_manifest_path, min_windows)

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

    # Build from CSV: no NPZ yet, or force_rebuild.
    if not old_npz:
        logging.info(f"No .npz in {PROCESSED_DIR}; building from CSV.")
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
        logging.info(f"force_rebuild=True: deleting {len(old_npz)} old NPZ files...")
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
        summary_csv_path, index=False
    )
    logging.info(f"Processing summary saved to: {summary_csv_path}")

    new_npz = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.npz")))
    if not new_npz:
        raise RuntimeError("No segments produced")
    manifest = build_or_update_processed_manifest(new_npz, processed_manifest_path, min_windows)
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


def load_saved_norm_stats(path: str) -> bool:
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


def detect_sbp_dbp(sig, fs=None):
    fs = FS if fs is None else fs
    s = smooth_signal(sig)
    dist = int(0.32 * fs)
    sys_idx, _ = find_peaks(s, distance=dist)
    dia_idx, _ = find_peaks(-s, distance=dist)
    return sig[sys_idx], sig[dia_idx]


def bp_targets_from_wave(abp_wave: np.ndarray):
    sbp_vals, dbp_vals = detect_sbp_dbp(abp_wave)
    sbp_mean = float(np.mean(sbp_vals)) if len(sbp_vals) > 0 else float(np.max(abp_wave))
    dbp_mean = float(np.mean(dbp_vals)) if len(dbp_vals) > 0 else float(np.min(abp_wave))
    return sbp_mean, dbp_mean


def segment_bp_means_from_wave(y: np.ndarray):
    try:
        sbp_vals, dbp_vals = detect_sbp_dbp(y)
        sbp_mean = float(np.mean(sbp_vals)) if len(sbp_vals) > 0 else np.nan
        dbp_mean = float(np.mean(dbp_vals)) if len(dbp_vals) > 0 else np.nan
        abp_mean = float(np.mean(y)) if y is not None and y.size > 0 else np.nan
        return sbp_mean, dbp_mean, abp_mean
    except Exception:
        return np.nan, np.nan, np.nan
