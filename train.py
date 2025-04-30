# ------------------------------------------------------------
# (0) Package Imports
# ------------------------------------------------------------
import os, glob, gc, re, json, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import mean_absolute_error

# --- Keras(TF2) ---
from keras import layers, models, regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.losses import Huber

# --- Optuna ---
import optuna
import pickle
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice

# ------------------------------------------------------------
# (1) Global parameters and paths
# ------------------------------------------------------------
# Two data sources
data_paths = [
    "/media/chen/EXTERNAL_USB/Process_matched_all",
    "/media/chen/EXTERNAL_USB/Process_matched_p05_p09"
]
processed_dir = "processed_minmax"
os.makedirs(processed_dir, exist_ok=True)

# Corresponding mapping directories
input_dirs_for_mapping = [
    ["/media/chen/EXTERNAL_USB/matched/matched"],
    ["/home/chen/Desktop/mnt/physionet.org/files/mimic3wdb/1.0/matched/matched"]
]

# Control whether to skip the preprocessing stage - set to True to directly use preprocessed data for training
SKIP_PREPROCESSING = False

data_usage_ratio = 1
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

SEGMENT_LEN, STEP_SIZE = 1024, 256
batch_size, epochs, learning_rate = 64, 30, 1e-4
APPLY_FILTERING, FS = True, 125

TRAIN_MIN_MEAN = TRAIN_MIN_STD = TRAIN_MAX_MEAN = TRAIN_MAX_STD = 0.0

WORKERS = max(8, os.cpu_count())  # Increase default number of worker processes
IO_CHUNK_SIZE = 16  # Number of files processed per batch

train_list, val_list = [], []  # Global variables for Optuna usage

# Optuna-related settings
OPTUNA_TRIALS = 10  # Number of Optuna trials
OPTUNA_EPOCHS = 10  # Number of training epochs per Optuna trial
OPTUNA_TIMEOUT = 3600  # Optuna timeout in seconds
VISUALIZE_DIR = "optuna_plots"  # Directory to save Optuna visualization results
os.makedirs(VISUALIZE_DIR, exist_ok=True)

# ------------------------------------------------------------
# (2) Common tools (Filtering, Hampel, Align)
# ------------------------------------------------------------
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

def cross_correlation_align(ABP, PPG, ECG, dPPG):
    length = min(len(ABP), len(PPG), len(ECG), len(dPPG))
    if length < 2: return ABP, PPG, ECG, dPPG
    ABP_, PPG_ = ABP[:length], PPG[:length]
    ECG_, dPPG_ = ECG[:length], dPPG[:length]
    shift = np.argmax(np.correlate(
        ABP_ - ABP_.mean(), PPG_ - PPG_.mean(), mode="full")) - (length - 1)
    if shift > 0:
        PPG_, ECG_, dPPG_ = PPG_[shift:], ECG_[shift:], dPPG_[shift:]
        ABP_ = ABP_[:len(PPG_)]
    elif shift < 0:
        s = abs(shift)
        ABP_ = ABP_[s:]
        PPG_, ECG_, dPPG_ = PPG_[:len(ABP_)], ECG_[:len(ABP_)], dPPG_[:len(ABP_)]
    L = min(len(ABP_), len(PPG_), len(ECG_), len(dPPG_))
    return ABP_[:L], PPG_[:L], ECG_[:L], dPPG_[:L]

# ------------------------------------------------------------
# (3) Build record → subject mapping
# ------------------------------------------------------------
def build_record_subject_mapping(directories_list):
    mapping = {}
    for directories in directories_list:
        for base in directories:
            for root, _, files in os.walk(base):
                subj = os.path.basename(root)
                for f in files:
                    if f.lower().endswith(".dat"):
                        mapping[os.path.splitext(f)[0]] = subj
    return mapping

# ------------------------------------------------------------
# (3.5) Calculate each subject’s average SBP / DBP
# ------------------------------------------------------------
def _csv_avg_bp(csv_path):
    """Read only the ABP column and calculate systolic/diastolic averages; return (nan, nan) on failure"""
    try:
        abp = pd.read_csv(csv_path, usecols=["ABP"], dtype=np.float32,
                          engine="c", memory_map=True).values.squeeze()
        abp = abp[np.isfinite(abp)]
        if len(abp) < 200:  # Too short
            return (np.nan, np.nan)
        # Simple smoothing
        abp_s = np.convolve(abp, np.ones(5)/5, mode="same")
        dist = int(0.32 * FS)
        sys_idx, _ = find_peaks(abp_s, distance=dist)
        dia_idx, _ = find_peaks(-abp_s, distance=dist)
        if len(sys_idx) < 3 or len(dia_idx) < 3:
            return (np.nan, np.nan)
        return (float(np.mean(abp[sys_idx])), float(np.mean(abp[dia_idx]))
    except Exception:
        return (np.nan, np.nan)

def compute_subject_bp_stats(csv_files, file2subj):
    """Return subj -> (avg_sbp, avg_dbp)"""
    subj_sbp, subj_dbp, subj_cnt = {}, {}, {}
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        results = list(exe.map(_csv_avg_bp, csv_files))
        for csv_path, result in tqdm(zip(csv_files, results), total=len(csv_files),
                desc="Phase-A: Calculating subject blood pressure statistics"):
            sbp_m, dbp_m = result  # Unpack the result tuple
            if not np.isfinite(sbp_m):  # Failed
                continue
            subj = file2subj[csv_path]
            subj_sbp[subj] = subj_sbp.get(subj, 0.0) + sbp_m
            subj_dbp[subj] = subj_dbp.get(subj, 0.0) + dbp_m
            subj_cnt[subj] = subj_cnt.get(subj, 0) + 1
    stats = {}
    for s, c in subj_cnt.items():
        stats[s] = (subj_sbp[s] / c, subj_dbp[s] / c)
    return stats   # {subj: (avg_sbp, avg_dbp)}

# ------------------------------------------------------------
# (4) Read CSV, align, segment, output .npz
# ------------------------------------------------------------
def read_and_align_csv(fp):
    cols = ["PLETH", "II", "dPPG", "ABP"]
    try:
        df = pd.read_csv(fp, usecols=cols, dtype=np.float32)
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
        yield arr[i:i + seg_len]

def process_single_csv(args):
    csv_path, subj_id = args
    out_paths, local_id = [], 0
    data = read_and_align_csv(csv_path)
    if data.size == 0:
        return out_paths
    segments = list(chunk_data(data))
    # Process all segments at once instead of opening/closing files multiple times in a loop
    for i, seg in enumerate(segments):
        abp = seg[:, 3]
        mn, mx = abp.min(), abp.max()
        if mx - mn < 1e-5:
            continue
        X, Y = seg[:, :3], (abp - mn) / (mx - mn)
        out_p = os.path.join(
            processed_dir, f"{subj_id}_{os.path.basename(csv_path)[:-4]}_{local_id}.npz")
        np.savez_compressed(
            out_p, X=X.astype(np.float32), Y=Y.astype(np.float32),
            abp_min=np.float32(mn), abp_max=np.float32(mx),
            sid=np.int32(subj_id))
        out_paths.append(out_p)
        local_id += 1
    return out_paths

# ------------------------------------------------------------
# (5) subject-wise Stratified Split + Preprocessing
# ------------------------------------------------------------
def improved_stratified_split(subj_stats, file2subj, train_r, val_r, test_r, seed=42):
 
    rng = np.random.RandomState(seed)
    
    # 1. Create bins for SBP and DBP ranges
    sbp_values = [stats[0] for stats in subj_stats.values() if np.isfinite(stats[0])]
    dbp_values = [stats[1] for stats in subj_stats.values() if np.isfinite(stats[1])]
    
    # Calculate percentiles for binning
    sbp_bins = np.percentile(sbp_values, [0, 25, 50, 75, 100])
    dbp_bins = np.percentile(dbp_values, [0, 25, 50, 75, 100])
    
    # 2. Identify data sources for each subject
    subj_sources = {}
    for filepath, subj in file2subj.items():
        if subj in subj_stats:
            source = None
            for idx, path in enumerate(data_paths):
                if path in filepath:
                    source = idx
                    break
            
            if source is not None:
                subj_sources[subj] = source
    
    # 3. Group subjects by SBP-DBP bin and data source
    bins = []
    for i in range(len(sbp_bins)-1):
        for j in range(len(dbp_bins)-1):
            for src in range(len(data_paths)):
                bin_subjects = []
                
                for subj, (sbp, dbp) in subj_stats.items():
                    if not (np.isfinite(sbp) and np.isfinite(dbp)):
                        continue
                        
                    source = subj_sources.get(subj)
                    if source != src:
                        continue
                        
                    if (sbp_bins[i] <= sbp <= sbp_bins[i+1] and 
                        dbp_bins[j] <= dbp <= dbp_bins[j+1]):
                        bin_subjects.append(subj)
                
                if bin_subjects:
                    # Shuffle subjects within each bin
                    rng.shuffle(bin_subjects)
                    bins.append(bin_subjects)
    
    # 4. Distribute subjects in a balanced way
    train, val, test = set(), set(), set()
    pattern = ["train"] * int(train_r * 10) + ["val"] * int(val_r * 10) + ["test"] * int(test_r * 10)
    
    # For each bin, distribute subjects according to the pattern
    for bin_subjects in bins:
        for i, subj in enumerate(bin_subjects):
            category = pattern[i % len(pattern)]
            if category == "train":
                train.add(subj)
            elif category == "val":
                val.add(subj)
            else:
                test.add(subj)
    
    # Handle subjects that weren't categorized (missing source or SBP/DBP values)
    uncategorized = set(subj_stats.keys()) - (train | val | test)
    if uncategorized:
        print(f"[Warning] Found {len(uncategorized)} subjects without proper categorization")
        for subj in uncategorized:
            category = pattern[len(train) % len(pattern)]
            if category == "train":
                train.add(subj)
            elif category == "val":
                val.add(subj)
            else:
                test.add(subj)
    
    print(f"[Info] Split complete: Train={len(train)}, Val={len(val)}, Test={len(test)} subjects")
    return train, val, test

def data_preprocessing():
    rec2subj = build_record_subject_mapping(input_dirs_for_mapping)
    if not rec2subj:
        raise RuntimeError("Unable to build record→subject mapping")

    # Process multiple data sources
    all_csv_files = []
    for data_path in data_paths:
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        if not csv_files:
            print(f"[Warning] No CSV files found in {data_path}")
            continue
        csv_files = csv_files[:int(len(csv_files) * data_usage_ratio)]
        all_csv_files.extend(csv_files)
    
    if not all_csv_files:
        raise FileNotFoundError("No CSV files found in all data sources")
    
    # Build CSV → subject mapping; if not found in rec2subj, parse prefix from filename
    file2subj = {}
    for fp in all_csv_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        subj = rec2subj.get(base)
        if subj is None:
            # fallback: CSV filename prefix is usually the subject id
            subj = base.split('_')[0]
        file2subj[fp] = subj

    # ---------- Phase-A: Calculate subject blood pressure statistics ----------
    subj_stats = compute_subject_bp_stats(all_csv_files, file2subj)

    # ---------- Stratified Split - Use improved version ----------
    train_subj, val_subj, test_subj = improved_stratified_split(
        subj_stats, file2subj, train_ratio, val_ratio, test_ratio)

    # Output average SBP/DBP for each group for validation
    def print_set_stats(subjects, name):
        sbps = [subj_stats[s][0] for s in subjects if s in subj_stats]
        dbps = [subj_stats[s][1] for s in subjects if s in subj_stats]
        print(f"{name} Average SBP: {np.mean(sbps):.2f} ± {np.std(sbps):.2f} mmHg")
        print(f"{name} Average DBP: {np.mean(dbps):.2f} ± {np.std(dbps):.2f} mmHg")
        
    print("\n=== Data split blood pressure statistics ===")
    print_set_stats(train_subj, "Training Set")
    print_set_stats(val_subj, "Validation Set")
    print_set_stats(test_subj, "Test Set")
    
    # Check distribution for each data source
    source_counts = {i: {'train': 0, 'val': 0, 'test': 0} for i in range(len(data_paths))}
    for subj in train_subj | val_subj | test_subj:
        for fp, s in file2subj.items():
            if s == subj:
                for i, path in enumerate(data_paths):
                    if path in fp:
                        if subj in train_subj:
                            source_counts[i]['train'] += 1
                        elif subj in val_subj:
                            source_counts[i]['val'] += 1
                        else:
                            source_counts[i]['test'] += 1
                        break
    
    print("\n=== Distribution of each data source ===")
    for src, counts in source_counts.items():
        total = sum(counts.values())
        if total > 0:
            print(f"Source {src}: Train={counts['train']/total:.1%}, Validation={counts['val']/total:.1%}, Test={counts['test']/total:.1%}")
    
    subj2id = {s: i + 1 for i, s in enumerate(sorted(subj_stats))}
    with open(os.path.join(processed_dir, "subject_id_map.json"), "w") as f:
        json.dump(subj2id, f, ensure_ascii=False, indent=2)

    # ---------- Phase-B: Segment and write npz ----------
    # Clear old files (batch delete instead of processing one by one)
    old_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    if old_files:
        print(f"[Info] Deleting {len(old_files)} old files...")
        for f in old_files:
            os.remove(f)
    
    print(f"[Info] Starting to process {len(all_csv_files)} CSV files using {WORKERS} worker processes...")
    
    # Batch files to reduce management overhead
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    seg_infos = []
    total_files = len(all_csv_files)
    processed_files = 0
    
    # Create a single progress bar to track all processing
    with tqdm(total=total_files, desc="Phase-B: Processing progress") as pbar:
        # Use batch processing
        for chunk in chunks(all_csv_files, IO_CHUNK_SIZE):
            with ProcessPoolExecutor(max_workers=WORKERS) as exe:
                # Submit work for each file
                future_to_file = {exe.submit(process_single_csv, (fp, subj2id[file2subj[fp]])): fp for fp in chunk}
                
                # Process completed results
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
                    except Exception as e:
                        print(f"[Error] Error processing file {fp}: {str(e)}")
                    
                    processed_files += 1
                    pbar.update(1)
    
    if not seg_infos:
        raise RuntimeError("No segments generated")

    # No need to renumber; filenames already include subj_id + record
    train_list = [p for p, t in seg_infos if t == "train"]
    val_list   = [p for p, t in seg_infos if t == "val"]
    test_list  = [p for p, t in seg_infos if t == "test"]

    print(f"\n[Info] Segments: Train={len(train_list)}, Val={len(val_list)}, "
          f"Test={len(test_list)}  | Subjects: "
          f"{len(train_subj)}/{len(val_subj)}/{len(test_subj)}")
    return train_list, val_list, test_list

# ------------------------------------------------------------
# (6a) Load a single segment function
# ------------------------------------------------------------
def load_segment(fp, apply_filter=APPLY_FILTERING):
    """
    Load one .npz segment file and return:
      X: [seg_len, 3] array of (PPG, ECG, dPPG)
      Y: [seg_len] normalized ABP waveform
      mn: original ABP min
      mx: original ABP max
    If apply_filter is True, band‑pass filter is applied to the PPG channel.
    """
    data = np.load(fp)
    X = data['X']
    Y = data['Y']
    mn = float(data['abp_min'])
    mx = float(data['abp_max'])
    if apply_filter:
        # Apply the same band‑pass filter to PPG (channel 0)
        X = X.copy()
        X[:, 0] = filtfilt(b_bp, a_bp, X[:, 0])
    return X, Y, mn, mx

# ------------------------------------------------------------
# (6) Dataset generator (using generator-based approach)
# ------------------------------------------------------------
def gen_data(file_list, model_type="unet", shuffle=False):
    for p in file_list:
        X, Y_or_z, mn, mx = load_segment(p)
        if model_type == "unet":
            yield X, Y_or_z
        else:
            z_min = (mn - TRAIN_MIN_MEAN) / TRAIN_MIN_STD
            z_max = (mx - TRAIN_MAX_MEAN) / TRAIN_MAX_STD
            yield X, [z_min, z_max]

def create_tf_dataset(file_list, model_type="unet", shuffle=False):
    output_signature = (
        tf.TensorSpec(shape=(SEGMENT_LEN, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(SEGMENT_LEN,) if model_type == "unet" else (2,), dtype=tf.float32)
    )
    ds = tf.data.Dataset.from_generator(
        lambda: gen_data(file_list, model_type, shuffle),
        output_signature=output_signature
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size * 10)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------
# (7) Models: ResNet & U‑Net (Enhanced ResNet)
# ------------------------------------------------------------
def simplified_attention(x, ratio=16):
    """Simplified channel attention mechanism"""
    ch = int(x.shape[-1])
    # Use average pooling for channel attention
    avg_pool = layers.GlobalAveragePooling1D()(x)
    avg_pool = layers.Reshape((1, ch))(avg_pool)
    
    # Reduced parameter attention mechanism
    attention = layers.Dense(max(ch // ratio, 4), activation='relu', 
                           kernel_regularizer=regularizers.l2(2e-4))(avg_pool)
    attention = layers.Dense(ch, activation='sigmoid', 
                           kernel_regularizer=regularizers.l2(2e-4))(attention)
    
    # Apply channel attention
    return layers.multiply([x, attention])

def bottleneck_block(x, filters, k=3, dilation_rate=1, downsample=False):
    """Residual block with bottleneck structure, increasing receptive field and efficiency"""
    shortcut = x
    reg = regularizers.l2(2e-4)  # Increase L2 regularization strength
    
    # Bottleneck structure: reduce dimensions -> process -> increase dimensions
    inner_filters = filters // 4
    
    # 1x1 convolution for dimension reduction
    x = layers.Conv1D(inner_filters, 1, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)  # Adjust momentum parameter
    x = layers.ReLU()(x)
    
    # Main convolution - Modify this part to avoid using stride>1 and dilation_rate>1 simultaneously
    stride = 1  # Always use stride=1 for convolution
    x = layers.Conv1D(inner_filters, k, strides=stride, padding="same", 
                    dilation_rate=dilation_rate, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    
    # If downsampling is needed, use MaxPooling1D instead of stride>1
    if downsample:
        x = layers.MaxPooling1D(2)(x)
    
    # 1x1 convolution for dimension increase
    x = layers.Conv1D(filters, 1, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    
    # Process shortcut branch
    if shortcut.shape[-1] != filters or downsample:
        shortcut = layers.Conv1D(filters, 1, padding="same", kernel_regularizer=reg)(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.9)(shortcut)
        if downsample:
            shortcut = layers.MaxPooling1D(2)(shortcut)
    
    # Residual connection
    x = layers.Add()([x, shortcut])
    
    # Use simplified attention mechanism
    x = simplified_attention(x)
    
    x = layers.ReLU()(x)
    return x

def build_resnet_improved(input_shape=(1024, 3), channels=[48, 96, 192, 256],
                         first_k=7, num_blocks_per_stage=[2,2,2,1]):
    """Improved ResNet, simplified to reduce overfitting"""
    inp = layers.Input(shape=input_shape)
    reg = regularizers.l2(2e-4)
    
    # Initial convolution layer
    x = layers.Conv1D(channels[0] // 2, first_k, padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.1)(x)  # Add early dropout
    
    # Save outputs of each stage for multi-scale feature fusion
    stage_outputs = []
    
    # Use fixed dilation rates, simplify design
    dilation_rates = [1, 2, 2, 4]  
    current_channel_idx = 0
    
    # Generate residual blocks for different stages
    for stage_idx, num_blocks in enumerate(num_blocks_per_stage):
        for block_idx in range(num_blocks):
            downsample = (block_idx == 0 and stage_idx > 0)
            dr = dilation_rates[stage_idx]
            x = bottleneck_block(x, channels[current_channel_idx], 
                               dilation_rate=dr, downsample=downsample)
        
        x = layers.Dropout(0.1 + 0.05 * stage_idx)(x)  # Increase dropout rate with depth
        stage_outputs.append(x)
        current_channel_idx = min(current_channel_idx + 1, len(channels) - 1)
    
    # Multi-scale feature fusion - Use attention weights
    pooled_features = []
    common_dim = 128  # Set common dimension to ensure all features have consistent dimensions
    
    # Perform global pooling for each stage output
    for stage_out in stage_outputs:
        # Combine average pooling and max pooling
        avg_pool = layers.GlobalAveragePooling1D()(stage_out)
        max_pool = layers.GlobalMaxPooling1D()(stage_out)
        combined = layers.Add()([avg_pool, max_pool])
        # Project to common dimension
        projected = layers.Dense(common_dim, activation='relu', kernel_regularizer=reg)(combined)
        pooled_features.append(projected)
    
    # Main output
    if len(pooled_features) > 1:
        # Use weighted fusion, now all features have the same dimension
        weights = []
        for i, feat in enumerate(pooled_features):
            w = layers.Dense(1, activation='sigmoid')(feat)
            weights.append(w)
        
        # Normalize weights
        weight_sum = layers.Add()(weights)
        norm_weights = [layers.Lambda(lambda x: x[0]/x[1])([w, weight_sum]) for w in weights]
        
        # Weighted features
        weighted_features = [layers.Multiply()([feat, w]) for feat, w in zip(pooled_features, norm_weights)]
        x = layers.Add()(weighted_features)
    else:
        x = pooled_features[0]
    
    # Fully connected layers - Add gradient clipping
    x = layers.Dropout(0.4)(x)  # Increase dropout rate
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg, 
                   kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg, 
                   kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.Dropout(0.2)(x)
    main_output = layers.Dense(2, activation="linear", 
                             kernel_constraint=tf.keras.constraints.MaxNorm(1),
                             name="main_output")(x)
    
    # Build model
    model = models.Model(inp, main_output, name="ImprovedResNet_for_MinMax")
    return model

def se_module(x, ratio=16):
    """SE module for channel attention"""
    ch = int(x.shape[-1])
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Reshape((1, ch))(se)
    se = layers.Dense(ch // ratio, activation="relu")(se)
    se = layers.Dense(ch, activation="sigmoid")(se)
    return layers.Multiply()([x, se])

def multi_res_block(x, out_filters, alpha=1.0, drop=0.3):
    """MultiRes block for U‑Net modules"""
    W = alpha * out_filters
    reg = regularizers.l2(1e-4)
    c1, c2, c3 = int(W * 0.167), int(W * 0.333), int(W * 0.5)
    sc = layers.Conv1D(c1 + c2 + c3, 1, padding='same', kernel_regularizer=reg)(x)
    sc = layers.BatchNormalization()(sc)
    conv3 = layers.Conv1D(c1, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    conv3 = layers.BatchNormalization()(conv3)
    conv5 = layers.Conv1D(c2, 5, padding='same', activation='relu', kernel_regularizer=reg)(conv3)
    conv5 = layers.BatchNormalization()(conv5)
    conv7 = layers.Conv1D(c3, 7, padding='same', activation='relu', kernel_regularizer=reg)(conv5)
    conv7 = layers.BatchNormalization()(conv7)
    out = layers.Concatenate()([conv3, conv5, conv7])
    out = layers.BatchNormalization()(out)
    out = layers.Add()([sc, out])
    out = layers.ReLU()(out)
    out = layers.Dropout(drop)(out)
    return out

def eca(x, gamma=2, b=1):
    """Efficient Channel Attention (ECA)"""
    ch = int(x.shape[-1])
    k = int(abs((np.log2(ch) / gamma) + b))
    if k % 2 == 0:
        k += 1
    g = layers.GlobalAveragePooling1D()(x)
    g = layers.Reshape((1, ch))(g)
    att = layers.Conv1D(1, k, padding='same', activation='sigmoid', use_bias=False)(g)
    return layers.Multiply()([x, att])

def build_unet(input_shape=(1024, 3)):
    """Build U‑Net model"""
    inp = layers.Input(shape=input_shape)
    x1 = eca(multi_res_block(inp, 16)); p1 = layers.MaxPooling1D(2)(x1)
    x2 = eca(multi_res_block(p1, 32));  p2 = layers.MaxPooling1D(2)(x2)
    x3 = eca(multi_res_block(p2, 64));  p3 = layers.MaxPooling1D(2)(x3)
    x4 = eca(multi_res_block(p3, 128)); p4 = layers.MaxPooling1D(2)(x4)
    bn = eca(multi_res_block(p4, 256))
    u4 = layers.UpSampling1D(2)(bn); c4 = layers.Concatenate()([u4, x4])
    d4 = eca(multi_res_block(c4, 128))
    u3 = layers.UpSampling1D(2)(d4); c3 = layers.Concatenate()([u3, x3])
    d3 = eca(multi_res_block(c3, 64))
    u2 = layers.UpSampling1D(2)(d3); c2 = layers.Concatenate()([u2, x2])
    d2 = eca(multi_res_block(c2, 32))
    u1 = layers.UpSampling1D(2)(d2); c1 = layers.Concatenate()([u1, x1])
    d1 = eca(multi_res_block(c1, 16))
    out = layers.Conv1D(1, 1, padding="same")(d1)
    out = layers.Reshape((input_shape[0],))(out)
    return models.Model(inp, out, name="UNet_ECA")

# ------------------------------------------------------------
# (8) Evaluation functions (unchanged)
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
    return e.mean(), e.std()

def calc_MASE(t, p):
    if len(t) < 2: return np.nan
    mae = np.mean(np.abs(p - t))
    denom = np.mean(np.abs(t[1:] - t[:-1]))
    return mae / denom if denom >= 1e-8 else np.nan

# ------------------------------------------------------------
# (9) Compute training set Z‑score statistics
# ------------------------------------------------------------
def calc_train_minmax(train_list):
    global TRAIN_MIN_MEAN, TRAIN_MIN_STD, TRAIN_MAX_MEAN, TRAIN_MAX_STD
    mins, maxs = [], []
    for p in train_list:
        _, _, mn, mx = load_segment(p, apply_filter=False)
        mins.append(mn); maxs.append(mx)
    mins, maxs = np.array(mins), np.array(maxs)
    TRAIN_MIN_MEAN, TRAIN_MIN_STD = mins.mean(), mins.std() or 1.0
    TRAIN_MAX_MEAN, TRAIN_MAX_STD = maxs.mean(), maxs.std() or 1.0
    print(f"[Info] Training set ABP_min mean/STD = {TRAIN_MIN_MEAN:.4f}/{TRAIN_MIN_STD:.4f}")
    print(f"[Info] Training set ABP_max mean/STD = {TRAIN_MAX_MEAN:.4f}/{TRAIN_MAX_STD:.4f}")

# ------------------------------------------------------------
# (10) Optuna hyperparameter optimization
# ------------------------------------------------------------
def objective(trial):
    """Optuna objective function"""
    global train_list, val_list
    
    # Memory cleanup
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Significantly reduce the amount of data used by Optuna (use only 10-20% of the data for hyperparameter search)
    optuna_data_fraction = 0.15  # Use only 15% of the data for hyperparameter search
    np.random.seed(42)  # Ensure reproducibility
    
    # Random sampling from training and validation sets
    sampled_train = np.random.choice(
        train_list, 
        size=min(int(len(train_list) * optuna_data_fraction), 200),  # Maximum 200 samples
        replace=False
    ).tolist()
    
    sampled_val = np.random.choice(
        val_list, 
        size=min(int(len(val_list) * optuna_data_fraction), 50),  # Maximum 50 samples
        replace=False
    ).tolist()
    
    # Create model
    model = create_model_from_params(trial)
    # Compile the model before training
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    cnorm = trial.suggest_float("clipnorm", 0.5, 2.0, step=0.5)
    delta = trial.suggest_float("huber_delta", 1.0, 3.0, step=0.5)
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=cnorm),
        loss=Huber(delta=delta),
        metrics=["mae"]
    )
    
    # Use smaller batch size
    current_batch_size = trial.suggest_categorical('batch_size', [8, 16, 24])
    
    # Create datasets - Load directly into memory for efficiency
    X_train, Y_train = [], []
    for p in sampled_train:
        X, Y_or_z, mn, mx = load_segment(p)
        z_min = (mn - TRAIN_MIN_MEAN) / TRAIN_MIN_STD
        z_max = (mx - TRAIN_MAX_MEAN) / TRAIN_MAX_STD
        X_train.append(X)
        Y_train.append([z_min, z_max])
    
    X_val, Y_val = [], []
    for p in sampled_val:
        X, Y_or_z, mn, mx = load_segment(p)
        z_min = (mn - TRAIN_MIN_MEAN) / TRAIN_MIN_STD
        z_max = (mx - TRAIN_MAX_MEAN) / TRAIN_MAX_STD
        X_val.append(X)
        Y_val.append([z_min, z_max])
    
    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)
    
    # Create TensorFlow datasets
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    ds_train = ds_train.shuffle(buffer_size=len(X_train)).batch(current_batch_size).prefetch(tf.data.AUTOTUNE)
    
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    ds_val = ds_val.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Reduce Optuna training epochs
    optuna_reduced_epochs = min(5, OPTUNA_EPOCHS)
    
    # Enhanced: Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", 
            factor=trial.suggest_float('lr_reduction_factor', 0.3, 0.8),
            patience=trial.suggest_int('lr_patience', 1, 3),
            verbose=0,
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=trial.suggest_int('early_stop_patience', 2, 5), 
            restore_best_weights=True, 
            verbose=0
        )
    ]
    
    # Train
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=optuna_reduced_epochs,
        callbacks=callbacks,
        verbose=0
    )
    
    # Release memory
    del X_train, Y_train, X_val, Y_val, ds_train, ds_val
    gc.collect()
    
    # Return best validation MAE
    return min(history.history['val_mae'])

def run_optuna_optimization():
    """Run Optuna hyperparameter optimization"""
    global OPTUNA_TRIALS, OPTUNA_EPOCHS, OPTUNA_TIMEOUT
    
    # Reduce Optuna trial count to save memory
    OPTUNA_TRIALS = min(5, OPTUNA_TRIALS)  # Maximum 5 trials
    OPTUNA_EPOCHS = min(5, OPTUNA_EPOCHS)  # Maximum 5 epochs per trial
    OPTUNA_TIMEOUT = min(1800, OPTUNA_TIMEOUT)  # Maximum 30 minutes
    
    # Enhanced: Sampler selection
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # Create Optuna study
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    study_name = f"bp_resnet_optimization_{timestamp}"
    db_path = f"{study_name}.db"
    # Remove existing database file if it exists to avoid stale records.
    if os.path.exists(db_path):
        os.remove(db_path)
    storage_name = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_name,
            sampler=sampler,
            load_if_exists=True
        )
    
    # Optimize simplified parameter space
    initial_params = {
        'use_improved_resnet': True,
        'first_kernel_size': 7,
        'channels_base': 32,  # Reduce base channel count
        'ch_mult_1': 1.5,
        'ch_mult_2': 3.0,
        'ch_mult_3': 5.0,
        'blocks_stage_0': 1,
        'blocks_stage_1': 1,
        'blocks_stage_2': 1,
        'blocks_stage_3': 1,
        'learning_rate': 1e-4,
        'batch_size': 16,
        'clipnorm': 1.0,
        'huber_delta': 2.0,
        'lr_reduction_factor': 0.5,
        'lr_patience': 2
    }
    study.enqueue_trial(initial_params)
    
    # Set pruner to terminate poorly performing trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=1, n_warmup_steps=2, interval_steps=1
    )
    study.pruner = pruner
    
    # Start optimization
    print(f"\n[Info] Starting Optuna hyperparameter optimization (reduced to {OPTUNA_TRIALS} trials to save memory)...")
    try:
        study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    except Exception as e:
        print(f"[Warning] Error during Optuna optimization: {e}")
        if len(study.trials) > 0:
            print(f"Successfully completed {len(study.trials)} trials")
        else:
            raise RuntimeError("Failed to complete any Optuna trials, unable to proceed")
    
    # Save study
    study_file = f"{study_name}.pkl"
    with open(study_file, "wb") as f:
        print(f"Saving study to {study_file}")
        pickle.dump(study, f)
    
    # Enhanced: More detailed result output
    try:
        best_params = study.best_params
    except ValueError:
        print("[Warning] No best trial found. Skipping best trial info.")
        return {}
    
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None:
        print("\n" + "="*50)
        print("[Optuna] Best parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"Best MAE: {best_trial.value:.6f}")
        print("="*50)
    else:
        print("[Warning] No best trial found. Skipping best trial info.")
    
    # Plot optimization history charts (if enough trials)
    if len(study.trials) >= 2:
        try:
            # Plot optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_image(os.path.join(VISUALIZE_DIR, f"{study_name}_history.png"))
            
            # Plot parameter importance
            fig2 = plot_param_importances(study)
            fig2.write_image(os.path.join(VISUALIZE_DIR, f"{study_name}_importance.png"))
        except Exception as e:
            print(f"[Warning] Error plotting charts: {e}")
    
    # Perform final memory cleanup
    gc.collect()
    tf.keras.backend.clear_session()
    
    return best_params

def create_model_from_params(trial, input_shape=(1024, 3)):
    """Create model based on Optuna parameters"""
    # Decide whether to use improved ResNet
    use_improved = trial.suggest_categorical('use_improved_resnet', [True])

    if use_improved:
        # Adjust improved ResNet parameters
        first_k = trial.suggest_int('first_kernel_size', 5, 9, step=2)  # Narrow range
        channels_base = trial.suggest_int('channels_base', 24, 48, step=8)  # Reduce base channel count
        
        # Reduce channel multiplier range
        ch_mult_1 = trial.suggest_float('ch_mult_1', 1.0, 2.0, step=0.25)
        ch_mult_2 = trial.suggest_float('ch_mult_2', 1.5, 3.0, step=0.5)
        ch_mult_3 = trial.suggest_float('ch_mult_3', 2.0, 4.0, step=0.5)
        
        channels = [
            channels_base,
            int(channels_base * ch_mult_1),
            int(channels_base * ch_mult_2),
            int(channels_base * ch_mult_3)
        ]
        
        # Reduce number of blocks per stage
        num_blocks_per_stage = []
        for i in range(4):
            num_blocks_per_stage.append(trial.suggest_int(f'blocks_stage_{i}', 1, 2))
        
        # Build improved ResNet model
        model = build_resnet_improved(
            input_shape=input_shape,
            channels=channels,
            first_k=first_k,
            num_blocks_per_stage=num_blocks_per_stage
        )
    else:
        # If improved ResNet is not selected, extend this part
        raise ValueError("Only improved ResNet is supported in this configuration.")
    
    return model

# ------------------------------------------------------------
# (11) Main program (updated to support new model selection)
# ------------------------------------------------------------
def plot_bp_distributions(train_list, val_list, test_list):
    """Plot the SBP/DBP distribution for train/validation/test sets"""
    print("\n[Info] Calculating SBP/DBP distribution...")
    
    # Function to calculate SBP/DBP for a single segment
    def get_bp_values(file_path):
        X, Y, mn, mx = load_segment(file_path)
        true_abp = Y * (mx - mn) + mn
        sbp, dbp = detect_sbp_dbp(true_abp)
        # Return None if too few points are detected
        if len(sbp) < 3 or len(dbp) < 3:
            return None, None
        # Return averages
        return np.mean(sbp), np.mean(dbp)
    
    # Collect SBP/DBP for all datasets
    datasets = {
        'Train': train_list,
        'Validation': val_list,
        'Test': test_list
    }
    
    # Store SBP/DBP for all datasets
    all_sbp = {}
    all_dbp = {}
    
    # Calculate SBP/DBP for each dataset
    for name, data_list in datasets.items():
        sbp_values = []
        dbp_values = []
        
        # Remove limit of first 500 entries, process all data
        for file_path in tqdm(data_list, desc=f"Calculating {name} SBP/DBP"):
            sbp, dbp = get_bp_values(file_path)
            if sbp is not None and dbp is not None:
                sbp_values.append(sbp)
                dbp_values.append(dbp)
        
        all_sbp[name] = sbp_values
        all_dbp[name] = dbp_values
    
    # Plot distributions
    plt.figure(figsize=(12, 10))
    
    # SBP distribution
    plt.subplot(2, 1, 1)
    for name, values in all_sbp.items():
        plt.hist(values, bins=30, alpha=0.5, label=name)
    plt.title('SBP Distribution Across Datasets')
    plt.xlabel('SBP (mmHg)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # DBP distribution
    plt.subplot(2, 1, 2)
    for name, values in all_dbp.items():
        plt.hist(values, bins=30, alpha=0.5, label=name)
    plt.title('DBP Distribution Across Datasets')
    plt.xlabel('DBP (mmHg)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'bp_distributions.png'))
    plt.show()
    
    # Output statistical information
    print("\n=== SBP/DBP Distribution Stats ===")
    for name in datasets.keys():
        print(f"{name} SBP: {np.mean(all_sbp[name]):.2f} ± {np.std(all_sbp[name]):.2f} mmHg")
        print(f"{name} DBP: {np.mean(all_dbp[name]):.2f} ± {np.std(all_dbp[name]):.2f} mmHg")


def main(skip_preprocessing=False, use_optuna=True):
    global train_list, val_list, test_list
    # 1. Preprocessing
    if not skip_preprocessing:
        train_list, val_list, test_list = data_preprocessing()
        calc_train_minmax(train_list)
    else:
        # Skip preprocessing (assume data already exists)
        processed_files = glob.glob(os.path.join(processed_dir, "*.npz"))
        if not processed_files:
            print("[Error] Failed to find preprocessed data, please run full preprocessing first")
            return
        
        # Read train/val/test lists from existing files
        seg_infos = []
        for p in processed_files:
            data = np.load(p)
            if 'sid' in data:
                subj_id = int(data['sid'])
                if subj_id <= len(processed_files) * train_ratio:
                    seg_infos.append((p, "train"))
                elif subj_id <= len(processed_files) * (train_ratio + val_ratio):
                    seg_infos.append((p, "val"))
                else:
                    seg_infos.append((p, "test"))
        
        train_list = [p for p, t in seg_infos if t == "train"]
        val_list = [p for p, t in seg_infos if t == "val"]
        test_list = [p for p, t in seg_infos if t == "test"]
        calc_train_minmax(train_list)

    # Calculate total number of subjects and segments
    all_segments = train_list + val_list + test_list
    total_segments = len(all_segments)
    
    # Get all unique subjects
    unique_subjects = set()
    for seg_path in all_segments:
        data = np.load(seg_path)
        if 'sid' in data:
            unique_subjects.add(int(data['sid']))
    total_subjects = len(unique_subjects)
    
    print(f"\n[Info] Data statistics: Total segments={total_segments}")

    # Output SBP/DBP distribution
    plot_bp_distributions(train_list, val_list, test_list)

    # 2. Build model
    if use_optuna:
        print("\n[Info] Using Optuna to find the best hyperparameters")
        best_params = run_optuna_optimization()
        
        # Build model using best parameters
        print("\n[Info] Using the best parameters to build improved ResNet")
        resnet = build_resnet_improved(
            input_shape=(1024, 3),
            channels=[
                best_params.get('channels_base', 48),
                best_params.get('channels_base', 48) * 2,
                best_params.get('channels_base', 48) * 4,
                best_params.get('channels_base', 48) * 6
            ],
            first_k=best_params.get('first_kernel_size', 7),
            num_blocks_per_stage=[
                best_params.get('blocks_stage_0', 2),
                best_params.get('blocks_stage_1', 2),
                best_params.get('blocks_stage_2', 2),
                best_params.get('blocks_stage_3', 1)
            ]
        )
        
        # Compile model
        optimizer = Adam(
            learning_rate=best_params.get('learning_rate', learning_rate * 0.5),
            clipnorm=best_params.get('clipnorm', 1.0)
        )
        resnet.compile(
            optimizer=optimizer,
            loss=Huber(delta=best_params.get('huber_delta', 2.0)),
            metrics=["mae"]
        )
    else:
        print("[Error] Optuna is required for model building.")
        return
    
    unet = build_unet()
    unet.compile(Adam(learning_rate), loss="mse", metrics=["mse"])

    # 3. Dataset
    ds_r_tr = create_tf_dataset(train_list, "resnet", shuffle=True)
    ds_r_val = create_tf_dataset(val_list, "resnet")
    ds_u_tr = create_tf_dataset(train_list, "unet", shuffle=True)
    ds_u_val = create_tf_dataset(val_list, "unet")

    # 4. Training
    print("\n[Train] ResNet (Z‑score min/max)")
    cb_r = [
        ReduceLROnPlateau("val_loss", 0.6, 2, verbose=1, min_lr=1e-7),
        ModelCheckpoint("best_resnet_weights.weights.h5", monitor="val_loss", 
                       save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, 
                                       restore_best_weights=True, verbose=1)
    ]
    hist_r = resnet.fit(ds_r_tr, validation_data=ds_r_val,
                        epochs=epochs, callbacks=cb_r)
    # Load best weights
    resnet.load_weights("best_resnet_weights.weights.h5")

    print("\n[Train] U‑Net (0‑1 Waveform)")
    cb_u = [
        ReduceLROnPlateau("val_loss", 0.5, 3, verbose=1, min_lr=1e-7),
        ModelCheckpoint("best_unet_weights.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    ]
    hist_u = unet.fit(ds_u_tr, validation_data=ds_u_val,
                      epochs=epochs, callbacks=cb_u)
    # Load best weights
    unet.load_weights("best_unet_weights.weights.h5")

    # 5. Testing
    print("\n[Test] Inference & evaluation ...")
    wave_mae, wave_me, wave_sd, wave_mase, R_all = [], [], [], [], []
    sbp_mae_list, sbp_me_list, sbp_sd_list = [], [], []
    dbp_mae_list, dbp_me_list, dbp_sd_list = [], [], []
    for p in tqdm(test_list, desc="Testing"):
        X, Y01, mn, mx = load_segment(p)
        z = resnet.predict(X[np.newaxis], verbose=0).ravel()
        p_min = z[0] * TRAIN_MIN_STD + TRAIN_MIN_MEAN
        p_max = z[1] * TRAIN_MAX_STD + TRAIN_MAX_MEAN
        y01 = unet.predict(X[np.newaxis], verbose=0).ravel()
        pred = y01 * (p_max - p_min) + p_min
        true = Y01 * (mx - mn) + mn

        mae = mean_absolute_error(true, pred)
        me, sd = calc_ME_SD(true, pred)
        R = np.corrcoef(true, pred)[0, 1] if (
            np.std(true) >= 1e-8 and np.std(pred) >= 1e-8) else 0.0
        mase = calc_MASE(true, pred)

        # === New: SBP / DBP detection ===
        sbp_t, dbp_t = detect_sbp_dbp(true)
        sbp_p, dbp_p = detect_sbp_dbp(pred)
        if len(sbp_t) > 1 and len(sbp_p) > 1:
            # Use the shorter length of the two sets of peaks to calculate metrics
            min_len_sbp = min(len(sbp_t), len(sbp_p))
            sbp_mae = mean_absolute_error(sbp_t[:min_len_sbp], sbp_p[:min_len_sbp])
            sbp_me, sbp_sd = calc_ME_SD(sbp_t[:min_len_sbp], sbp_p[:min_len_sbp])
            sbp_mae_list.append(sbp_mae); sbp_me_list.append(sbp_me); sbp_sd_list.append(sbp_sd)
        if len(dbp_t) > 1 and len(dbp_p) > 1:
            # Use the shorter length of the two sets of valleys to calculate metrics
            min_len_dbp = min(len(dbp_t), len(dbp_p))
            dbp_mae = mean_absolute_error(dbp_t[:min_len_dbp], dbp_p[:min_len_dbp])
            dbp_me, dbp_sd = calc_ME_SD(dbp_t[:min_len_dbp], dbp_p[:min_len_dbp])
            dbp_mae_list.append(dbp_mae); dbp_me_list.append(dbp_me); dbp_sd_list.append(dbp_sd)

        wave_mae.append(mae); wave_me.append(me); wave_sd.append(sd)
        wave_mase.append(mase); R_all.append(R)

    print("\n=== Waveform‑Level (Test) ===")
    print(f"R        : {np.mean(R_all):.4f}")
    print(f"MAE      : {np.mean(wave_mae):.4f}")
    print(f"ME±SD    : {np.mean(wave_me):.4f} ± {np.mean(wave_sd):.4f}")
    print(f"MASE     : {np.mean(wave_mase):.4f}")

    # === New: Print SBP / DBP results ===
    if sbp_mae_list and dbp_mae_list:
        print("\n=== SBP / DBP (Test) ===")
        print(f"SBP MAE      : {np.mean(sbp_mae_list):.4f}")
        print(f"SBP ME±SD    : {np.mean(sbp_me_list):.4f} ± {np.mean(sbp_sd_list):.4f}")
        print(f"DBP MAE      : {np.mean(dbp_mae_list):.4f}")
        print(f"DBP ME±SD    : {np.mean(dbp_me_list):.4f} ± {np.mean(dbp_sd_list):.4f}")

    # Loss curves
    plt.figure()
    plt.plot(hist_r.history["loss"])
    plt.plot(hist_r.history["val_loss"])
    plt.title("ResNet Loss")
    plt.legend(["Train", "Val"])
    plt.show()  # Display instead of saving

    plt.figure()
    plt.plot(hist_u.history["loss"])
    plt.plot(hist_u.history["val_loss"])
    plt.title("UNet Loss")
    plt.legend(["Train", "Val"])
    plt.show()  # Display instead of saving

    # Save result data
    results_data = {
        "waveform_r": float(np.mean(R_all)),
        "waveform_mae": float(np.mean(wave_mae)),
        "waveform_me": float(np.mean(wave_me)),
        "waveform_sd": float(np.mean(wave_sd)),
        "waveform_mase": float(np.mean(wave_mase)),
        "sbp_mae": float(np.mean(sbp_mae_list)) if sbp_mae_list else None,
        "dbp_mae": float(np.mean(dbp_mae_list)) if dbp_mae_list else None,
        "total_subjects": total_subjects,
        "total_segments": total_segments,
        "train_segments": len(train_list),
        "val_segments": len(val_list),
        "test_segments": len(test_list),
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    
    # Create result directory
    timestamp = time.strftime("%Y%m%d")
    r_value = f"{np.mean(R_all):.4f}".replace(".", "p")
    result_dir = f"results_{timestamp}_R{r_value}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Save results
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n[Info] Results saved in: {result_dir}")
    print("\n[Info] Full pipeline completed.")

# ------------------------------------------------------------
# (12) Main program execution
# ------------------------------------------------------------
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[Info] GPU memory dynamic growth enabled")
    
    # Modify to use global variable to control whether to skip preprocessing
    main(skip_preprocessing=SKIP_PREPROCESSING, use_optuna=True)