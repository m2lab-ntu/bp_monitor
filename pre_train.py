import os
import wfdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import find_peaks, correlate  # added correlate
from multiprocessing import Manager  # added

# =============================================================================
# (A) Parameter and folder settings
# ============================================================================
# Only specify p01 ~ p05 under the matched folder
base_dir = "/media/chen/EXTERNAL_USB/matched/matched"
input_dirs = [os.path.join(base_dir, f"p{str(i).zfill(2)}") for i in range(0, 5)]
#input_dirs = [os.path.join(base_dir, f"p{str(i).zfill(2)}") for i in range(5, 7)]
# Output folder: each converted CSV file will be stored here
output_dir = "/media/chen/EXTERNAL_USB/Seg3min"
# output_dir = "/media/chen/EXTERNAL_USB/Seg3min2"
os.makedirs(output_dir, exist_ok=True)

# Required signals (make sure the names match with the data file)
required_signals = ["PLETH", "II", "ABP"]

# =============================================================================
# (B) Preprocessing and flatline filtering functions
# =============================================================================
def remove_flatlines(ppg, abp, min_run=3, max_ratio=0.2, diff_eps=None):
    """
    Identify flat runs in ABP where the first difference is exactly zero (Δ == 0)
    for at least min_run samples, and return the non-flat complement segments.
    """
    n = len(abp)
    if n == 0:
        return []
    # Use exact zero derivative instead of near-zero threshold
    d = np.diff(abp)
    flat = np.zeros_like(d, dtype=bool)
    finite_mask = np.isfinite(d)
    flat[finite_mask] = d[finite_mask] == 0

    # Find runs of True in 'flat' (diff-based), translate to sample index runs
    runs = []
    i, L = 0, len(flat)
    while i < L:
        if flat[i]:
            j = i
            while j + 1 < L and flat[j + 1]:
                j += 1
            # diff indices i..j correspond to sample indices i..(j+1)
            run_len_samples = (j - i + 1) + 1
            if run_len_samples >= min_run:
                runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1

    if not runs:
        return [(0, n - 1)]

    keep_segments = []
    cur = 0
    for rs, re in runs:
        if cur <= rs - 1:
            keep_segments.append((cur, rs - 1))
        cur = re + 1
    if cur <= n - 1:
        keep_segments.append((cur, n - 1))
    keep_segments = [(s, e) for s, e in keep_segments if e >= s]
    return keep_segments

def detect_flat_peaks_in_abp(abp, min_run=3, max_ratio=0.2, diff_eps=None):
    """
    Mark as bad if more than max_ratio of peaks sit on exactly-flat regions
    (based on first differences Δ == 0 around the peak).
    """
    n = len(abp)
    if n < min_run:
        return True

    # require finite for peak detection (segment-level NaN filtered earlier)
    peaks, _ = find_peaks(abp)
    if len(peaks) == 0:
        return False

    # Use exact zero derivative instead of near-zero threshold
    d = np.diff(abp)
    flat_diff = (d == 0)

    # Map diff-flatness to sample-flatness around each index
    sample_flat = np.zeros(n, dtype=bool)
    sample_flat[1:-1] = flat_diff[:-1] & flat_diff[1:]
    sample_flat[0] = flat_diff[0] if len(flat_diff) > 0 else False
    sample_flat[-1] = flat_diff[-1] if len(flat_diff) > 0 else False

    def run_len_at(idx):
        # contiguous flat run length covering idx
        if not sample_flat[idx]:
            return 0
        l = idx
        while l - 1 >= 0 and sample_flat[l - 1]:
            l -= 1
        r = idx
        while r + 1 < n and sample_flat[r + 1]:
            r += 1
        return (r - l + 1)

    flat_count = 0
    for pk in peaks:
        if run_len_at(pk) >= min_run:
            flat_count += 1
    return (flat_count / len(peaks)) > max_ratio

def max_corr_with_lag(x, y, fs, max_seconds=2.0):
    """
    Optimized: compute best Pearson correlation across integer-sample lags
    within ±max_seconds using FFT-based cross-correlation and prefix sums.
    Returns (best_corr, best_lag).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 2:
        return 0.0, 0
    x = x[:n]
    y = y[:n]

    # Fill NaNs by linear interpolation to avoid NaN propagation in FFT
    def _interp_nan(z):
        if np.all(np.isfinite(z)):
            return z
        z = z.copy()
        mask = np.isfinite(z)
        if mask.sum() < 2:
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.arange(z.size)
        z[~mask] = np.interp(idx[~mask], idx[mask], z[mask])
        return z

    x = _interp_nan(x)
    y = _interp_nan(y)

    max_lag = int(round(max_seconds * float(fs)))
    if max_lag <= 0:
        L = n
        sum_x = x.sum()
        sum_y = y.sum()
        sum_x2 = np.dot(x, x)
        sum_y2 = np.dot(y, y)
        sum_xy = np.dot(x, y)
        cov = sum_xy - (sum_x * sum_y) / L
        var_x = sum_x2 - (sum_x * sum_x) / L
        var_y = sum_y2 - (sum_y * sum_y) / L
        denom = np.sqrt(max(var_x, 0.0)) * np.sqrt(max(var_y, 0.0))
        if denom <= 0:
            return 0.0, 0
        return float(cov / denom), 0

    # Cross-correlation via FFT (full range lags)
    c = correlate(x, y, mode='full', method='fft')  # length 2n-1
    k_arr = np.arange(-max_lag, max_lag + 1, dtype=int)
    idx_arr = (n - 1) + k_arr
    sum_xy = c[idx_arr]

    # Prefix sums for window sums/variances
    Sx = np.concatenate(([0.0], np.cumsum(x)))
    Sy = np.concatenate(([0.0], np.cumsum(y)))
    Sx2 = np.concatenate(([0.0], np.cumsum(x * x)))
    Sy2 = np.concatenate(([0.0], np.cumsum(y * y)))

    L_arr = n - np.abs(k_arr)
    corr = np.full(k_arr.shape, -np.inf, dtype=float)
    mask_valid = L_arr >= 2

    if np.any(mask_valid):
        kv = k_arr[mask_valid]
        Lv = L_arr[mask_valid]
        sum_xy_v = sum_xy[mask_valid]

        sum_x_v = np.empty_like(sum_xy_v, dtype=float)
        sum_y_v = np.empty_like(sum_xy_v, dtype=float)
        sum_x2_v = np.empty_like(sum_xy_v, dtype=float)
        sum_y2_v = np.empty_like(sum_xy_v, dtype=float)

        pos = kv >= 0
        neg = ~pos

        if np.any(pos):
            kp = kv[pos]
            sum_x_v[pos] = Sx[n] - Sx[kp]
            sum_y_v[pos] = Sy[n - kp]
            sum_x2_v[pos] = Sx2[n] - Sx2[kp]
            sum_y2_v[pos] = Sy2[n - kp]

        if np.any(neg):
            kn = -kv[neg]  # abs(neg lags)
            sum_x_v[neg] = Sx[n - kn]
            sum_y_v[neg] = Sy[n] - Sy[kn]
            sum_x2_v[neg] = Sx2[n - kn]
            sum_y2_v[neg] = Sy2[n] - Sy2[kn]

        cov_v = sum_xy_v - (sum_x_v * sum_y_v) / Lv
        var_x_v = sum_x2_v - (sum_x_v * sum_x_v) / Lv
        var_y_v = sum_y2_v - (sum_y_v * sum_y_v) / Lv
        denom_v = np.sqrt(np.maximum(var_x_v, 0.0)) * np.sqrt(np.maximum(var_y_v, 0.0))

        corr_v = np.full_like(sum_xy_v, -np.inf)
        nz = denom_v > 0
        corr_v[nz] = cov_v[nz] / denom_v[nz]
        corr[mask_valid] = corr_v

    best_idx = int(np.argmax(corr))
    best_corr = corr[best_idx]
    best_lag = int(k_arr[best_idx])
    if not np.isfinite(best_corr):
        return 0.0, 0
    return float(best_corr), best_lag

def corr_at_lag(x, y, lag):
    """
    Pearson correlation at a fixed integer-sample lag (no search).
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    if lag >= 0:
        xa = x[lag:n]
        ya = y[:n - lag]
    else:
        l = -lag
        xa = x[:n - l]
        ya = y[l:n]
    if len(xa) < 2:
        return 0.0
    sx, sy = np.std(xa), np.std(ya)
    if sx == 0 or sy == 0:
        return 0.0
    try:
        c = np.corrcoef(xa, ya)[0, 1]
        return float(c) if np.isfinite(c) else 0.0
    except Exception:
        return 0.0

def extract_subject_id_from_path(path: str) -> str:
    """
    Try to extract MIMIC-III waveform subject id like 'p123456' from the path.
    Fallback to parent directory name if not found.
    """
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part.startswith('p') and part[1:].isdigit() and len(part) == 7:
            return part
    for part in parts:
        if part.startswith('p') and part[1:].isdigit() and len(part) >= 4:
            return part
    return os.path.basename(os.path.dirname(path))

# =============================================================================
# (C) Single .dat file processing function
# =============================================================================
def process_file(file_path, output_dir, required_signals, shared_counts, shared_lock):
    try:
        base_name = os.path.splitext(file_path)[0]
        output_file = os.path.join(output_dir, f"{os.path.basename(base_name)}.csv")

        samples, fields = wfdb.rdsamp(base_name)
        signal_names = fields['sig_name']
        if not all(sig in signal_names for sig in required_signals):
            return f"{file_path}: Missing required signals"
        fs = float(fields.get('fs', 0.0))
        if fs <= 0:
            return f"{file_path}: Missing or invalid sampling rate (fs)"
        min_samples = int(180 * fs)

        indices = [signal_names.index(sig) for sig in required_signals]
        data = samples[:, indices]
        df = pd.DataFrame(data, columns=required_signals)

        if len(df) < min_samples:
            return f"{file_path}: Not enough data points (< 3 minutes), skipped"

        ppg = df['PLETH'].values
        ecg = df['II'].values
        abp = df['ABP'].values

        # Compute fixed best lag for this file (±2s integer samples)
        _, file_best_lag = max_corr_with_lag(ppg, abp, fs, max_seconds=2.0)

        # Get non-flat (derivative-based) segments to keep
        segments = remove_flatlines(ppg, abp, min_run=3, max_ratio=0.2)
        if not segments:
            return f"{file_path}: No usable non-flat segments"

        subject_id = extract_subject_id_from_path(file_path)

        saved = 0
        out_parts = []
        stop_for_subject = False
        for seg_idx, (s, e) in enumerate(segments):
            if stop_for_subject:
                break
            seg_len = e - s + 1
            if seg_len < min_samples:
                continue

            seg_ppg = ppg[s:e+1]
            seg_ecg = ecg[s:e+1]
            seg_abp = abp[s:e+1]

            # 1) Low-cost: NaN/Zero check
            if (np.isnan(seg_ppg).any() or np.isnan(seg_ecg).any() or np.isnan(seg_abp).any()):
                continue
            if (np.any(seg_ppg == 0) or np.any(seg_ecg == 0) or np.any(seg_abp == 0)):
                continue

            # 2) Medium-cost: correlation check at fixed per-file lag
            c = corr_at_lag(seg_ppg, seg_abp, file_best_lag)
            if not np.isfinite(c) or c < 0.75:
                continue

            # 3) High-cost: flat-peak detection (only if correlation passes)
            if detect_flat_peaks_in_abp(seg_abp, min_run=3, max_ratio=0.2):
                continue

            # Enforce per-subject 100-segment cap (atomic check+increment)
            with shared_lock:
                current = shared_counts.get(subject_id, 0)
                if current >= 100:
                    stop_for_subject = True
                    continue
                shared_counts[subject_id] = current + 1

            dppg = np.gradient(seg_ppg)
            out_df = pd.DataFrame({
                "PLETH": seg_ppg,
                "II": seg_ecg,
                "dPPG": dppg,
                "ABP": seg_abp
            })
            out_parts.append(out_df)
            saved += 1

        if saved == 0:
            return f"{file_path}: No valid segments after 3-min filtering"

        combined_df = pd.concat(out_parts, axis=0, ignore_index=True)
        combined_df.to_csv(output_file, index=False)

        # Return stats for aggregation (subject and file mapping)
        return ("STATS", subject_id, file_path, output_file, saved)

    except Exception as e:
        return f"{file_path}: Conversion failed ({e})"

# =============================================================================
# (D) Collect .dat files from specified subfolders
# =============================================================================
def collect_dat_files(input_dirs):
    all_files = []
    for d in input_dirs:
        if not os.path.isdir(d):
            print(f"Warning: Folder not found {d}")
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith('.dat'):
                    all_files.append(os.path.join(root, f))
    return all_files

# =============================================================================
# (E) Main function: parallel conversion
# =============================================================================
def convert_dat_to_csv(input_dirs, output_dir, required_signals, max_workers=16):
    dat_files = collect_dat_files(input_dirs)
    if not dat_files:
        print("No .dat files found.")
        return

    print(f"Found {len(dat_files)} .dat files, starting processing...")
    failed = []
    subject_ids = set()
    total_segments = 0
    mapping_rows = []

    # Shared state for per-subject segment cap
    manager = Manager()
    subject_counts = manager.dict()
    subject_lock = manager.Lock()

    with tqdm(total=len(dat_files), desc="Processing progress", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, fp, output_dir, required_signals, subject_counts, subject_lock): fp
                for fp in dat_files
            }
            for future in as_completed(futures):
                res = future.result()
                if isinstance(res, tuple) and len(res) >= 5 and res[0] == "STATS":
                    _, sid, in_fp, out_fp, cnt = res
                    if cnt > 0:
                        subject_ids.add(sid)
                        total_segments += int(cnt)
                        mapping_rows.append((
                            sid,
                            os.path.basename(in_fp),
                            os.path.basename(out_fp) if out_fp else "",
                            int(cnt),
                        ))
                elif res:
                    failed.append(res)
                pbar.update(1)

    if failed:
        log_path = os.path.join(output_dir, "error_log.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("\n".join(failed))
        print(f"Some files failed to process, please check: {log_path}")
    else:
        print("All files have been successfully converted!")

    # Write subject-file mapping table
    if mapping_rows:
        map_df = pd.DataFrame(mapping_rows, columns=["subject_id", "input_file", "output_csv", "segments_saved"])
        map_path = os.path.join(output_dir, "subject_file_map.csv")
        map_df.to_csv(map_path, index=False)
        print(f"Subject-file mapping saved: {map_path}")

    # Final summary
    print(f"Summary - subjects: {len(subject_ids)}, segments: {total_segments}")

# =============================================================================
# (F) Execute
# =============================================================================
if __name__ == "__main__":
    cpu_count = os.cpu_count() or 4
    max_workers = min(cpu_count * 2, 4)
    convert_dat_to_csv(input_dirs, output_dir, required_signals, max_workers=max_workers)