import os
import wfdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import find_peaks

# =============================================================================
# (A) Parameter and folder settings
# =============================================================================
# Only specify p05 ~ p09 under the matched folder
base_dir = "/home/chen/Desktop/mnt/physionet.org/files/mimic3wdb/1.0/matched/matched"
input_dirs = [os.path.join(base_dir, f"p{str(i).zfill(2)}") for i in range(5, 10)]

# Output folder: each converted CSV file will be stored here
output_dir = "/media/chen/EXTERNAL_USB/Process_matched_p05_p09"
os.makedirs(output_dir, exist_ok=True)

# Required signals (make sure the names match with the data file)
required_signals = ["PLETH", "II", "ABP"]

# =============================================================================
# (B) Preprocessing and flatline filtering functions
# =============================================================================
def remove_flatlines(ppg, abp, min_run=3, max_ratio=0.2):
    n = len(abp)
    if n < min_run:
        return None

    def get_flat_indices(signal):
        diffs = np.diff(signal)
        flat_indices = []
        start = None
        for i, d in enumerate(diffs):
            if d == 0:
                if start is None:
                    start = i
            else:
                if start is not None and (i - start) >= (min_run - 1):
                    flat_indices.append((start, i))
                start = None
        if start is not None and (len(diffs) - start) >= (min_run - 1):
            flat_indices.append((start, len(diffs)))
        return flat_indices

    flat_a = get_flat_indices(abp)
    to_remove = set()
    for s, e in flat_a:
        to_remove.update(range(s, e+1))
    if len(to_remove) / n > max_ratio:
        return None
    abp_new = np.delete(abp, sorted(to_remove))
    return ppg, abp_new

def detect_flat_peaks_in_abp(abp, min_run=3, max_ratio=0.2):
    if len(abp) < min_run:
        return True
    peaks, _ = find_peaks(abp)
    if len(peaks) == 0:
        return False
    flat_count = 0
    for pk in peaks:
        val = abp[pk]
        run_len = 1
        i = pk + 1
        while i < len(abp) and abp[i] == val:
            run_len += 1; i += 1
        j = pk - 1
        while j >= 0 and abp[j] == val:
            run_len += 1; j -= 1
        if run_len >= min_run:
            flat_count += 1
    return (flat_count / len(peaks)) > max_ratio

# =============================================================================
# (C) Single .dat file processing function
# =============================================================================
def process_file(file_path, output_dir, required_signals):
    try:
        base_name = os.path.splitext(file_path)[0]
        output_file = os.path.join(output_dir, f"{os.path.basename(base_name)}.csv")
        if os.path.exists(output_file):
            return None  # Skip if already exists

        samples, fields = wfdb.rdsamp(base_name)
        signal_names = fields['sig_name']
        if not all(sig in signal_names for sig in required_signals):
            return f"{file_path}: Missing required signals"
        indices = [signal_names.index(sig) for sig in required_signals]
        data = samples[:, indices]
        df = pd.DataFrame(data, columns=required_signals)
        df.dropna(inplace=True)
        df = df[(df != 0).all(axis=1)]
        if len(df) < 1024:
            return f"{file_path}: Not enough data points (< 1024), skipped"

        ppg = df['PLETH'].values
        ecg = df['II'].values
        abp = df['ABP'].values

        res = remove_flatlines(ppg, abp, min_run=3, max_ratio=0.2)
        if res is None:
            return f"{file_path}: ABP has too many or insufficient flat segments"
        _, abp = res

        if detect_flat_peaks_in_abp(abp, min_run=3, max_ratio=0.2):
            return f"{file_path}: ABP flat peak ratio is too high"

        dppg = np.gradient(ppg)

        out_df = pd.DataFrame({
            "PLETH": ppg,
            "II": ecg,
            "dPPG": dppg,
            "ABP": abp
        })
        out_df.to_csv(output_file, index=False)
        return None

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

    with tqdm(total=len(dat_files), desc="Processing progress", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, fp, output_dir, required_signals): fp
                for fp in dat_files
            }
            for future in as_completed(futures):
                err = future.result()
                if err:
                    failed.append(err)
                pbar.update(1)

    if failed:
        log_path = os.path.join(output_dir, "error_log.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("\n".join(failed))
        print(f"Some files failed to process, please check: {log_path}")
    else:
        print("All files have been successfully converted!")

# =============================================================================
# (F) Execute
# =============================================================================
if __name__ == "__main__":
    cpu_count = os.cpu_count() or 4
    max_workers = min(cpu_count * 2, 16)
    convert_dat_to_csv(input_dirs, output_dir, required_signals, max_workers=max_workers)