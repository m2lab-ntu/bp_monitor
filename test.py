import os
import math

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import clone

import torch

import train as train_mod
from model import ResNetUNet, ResNetUNet3L, _get_skip_mode


def calc_ME_SD(t, p):
    e = p - t
    n = e.size if hasattr(e, 'size') else len(e)
    if n <= 1:
        # Definition: sample SD undefined for n<=1; return 0.0 to avoid NaN in mean
        return float(np.mean(e)), 0.0
    return float(np.mean(e)), float(np.std(e, ddof=1))


def calc_MASE(t, p):
    if len(t) < 2:
        return np.nan
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
    target_keys = tuple(target_keys or train_mod.REGRESSION_TARGETS)
    model.eval()
    stores = {k: {"true": [], "pred": []} for k in target_keys}

    with torch.no_grad():
        for p in tqdm(file_list, desc="Testing (regression)"):
            X, Y, _, _ = train_mod.load_segment(p)
            if X is None:
                continue
            X_norm, stats = train_mod.normalize_ppg_ecg_segment(X)
            sbp_true, dbp_true = train_mod._bp_targets_from_wave(Y)
            target_map = {"sbp": sbp_true, "dbp": dbp_true}

            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)
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
            X, Y, _, _ = train_mod.load_segment(fp)
            if X is None or Y is None:
                skipped += 1
                continue
            X_norm, stats = train_mod.normalize_ppg_ecg_segment(X)
            sbp_m_t, dbp_m_t, _ = train_mod._segment_bp_means_from_wave(Y)
            if not (np.isfinite(sbp_m_t) and np.isfinite(dbp_m_t)):
                skipped += 1
                continue

            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)
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
            tmp_path = os.path.join(train_mod.processed_dir, f"tmp_decoder_{tag}_{len(arr_list)}.dat")
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
            refit_with_train_val=bool(train_mod.BOTTLENECK_RIDGE_USE_TRAIN_VAL)
        )
        results["sbp"] = res_sbp | {"train_N": int(len(y_sbp_tr)), "val_N": int(len(y_sbp_val)), "test_N": int(len(y_sbp_te))}
    else:
        results["sbp"] = {"MAE": float('nan'), "R2": float('nan'), "best_params": {}, "train_N": int(len(y_sbp_tr)), "val_N": int(len(y_sbp_val)), "test_N": int(len(y_sbp_te))}

    if len(y_dbp_tr) >= 4 and len(y_dbp_val) >= 1 and len(y_dbp_te) >= 1:
        res_dbp = _sklearn_regression_probe(
            estimator, param_grid,
            X_train, y_dbp_tr, X_val, y_dbp_val, X_test, y_dbp_te,
            refit_with_train_val=bool(train_mod.BOTTLENECK_RIDGE_USE_TRAIN_VAL)
        )
        results["dbp"] = res_dbp | {"train_N": int(len(y_dbp_tr)), "val_N": int(len(y_dbp_val)), "test_N": int(len(y_dbp_te))}
    else:
        results["dbp"] = {"MAE": float('nan'), "R2": float('nan'), "best_params": {}, "train_N": int(len(y_dbp_tr)), "val_N": int(len(y_dbp_val)), "test_N": int(len(y_dbp_te))}

    return results


# ------------------------------------------------------------
# Segment-level evaluation & plotting
# ------------------------------------------------------------

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
            X, Y, _, _ = train_mod.load_segment(p)
            if X is None:
                continue
            X_norm, stats = train_mod.normalize_ppg_ecg_segment(X)
            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)
            out_all = model(x_tensor, stats_tensor)
            pred_norm, _, _ = train_mod._unpack_model_output(out_all)
            pred_norm = pred_norm.cpu().numpy().ravel()
            pred = train_mod.denormalize_abp(pred_norm)

            s_m, d_m, a_m = train_mod._segment_bp_means_from_wave(Y)
            s_pm, d_pm, a_pm = train_mod._segment_bp_means_from_wave(pred)

            if np.isfinite(s_m) and np.isfinite(s_pm):
                sbp_t.append(s_m)
                sbp_p.append(s_pm)
            if np.isfinite(d_m) and np.isfinite(d_pm):
                dbp_t.append(d_m)
                dbp_p.append(d_pm)
            if np.isfinite(a_m) and np.isfinite(a_pm):
                abp_t.append(a_m)
                abp_p.append(a_pm)

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
    if not (train_mod.ENABLE_PLOT_BA_SCATTER or train_mod.ENABLE_PLOT_TRUEPRED_SCATTER):
        return
    seg = evaluate_segment_level_predictions(model, file_list, device)
    sbp_t, sbp_p = seg['sbp_true'], seg['sbp_pred']
    dbp_t, dbp_p = seg['dbp_true'], seg['dbp_pred']
    abp_t, abp_p = seg['abp_true'], seg['abp_pred']

    sbp_err = np.abs(sbp_p - sbp_t) if sbp_t.size and sbp_p.size else np.array([])
    dbp_err = np.abs(dbp_p - dbp_t) if dbp_t.size and dbp_p.size else np.array([])

    # Figure 1: Bland–Altman plots (per segment)
    if train_mod.ENABLE_PLOT_BA_SCATTER:
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
        train_mod.save_plot("post_train_ba_scatter")

    # Figure 2: True vs Pred distributions for SBP/DBP/ABP
    if train_mod.ENABLE_PLOT_TRUEPRED_SCATTER:
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
        train_mod.save_plot("post_train_true_pred_scatter")


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
    suffix = title_prefix.strip().replace(" ", "_") if title_prefix else "segmean"
    train_mod.save_plot(f"segmean_true_pred_{suffix}")


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
            X, Y, _, _ = train_mod.load_segment(p)
            if X is None:
                continue
            # Standardize inputs; normalize/denormalize handled outside predictions
            X_norm, stats = train_mod.normalize_ppg_ecg_segment(X)
            X_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)

            # Model inference (prediction on normalized scale -> denormalize to mmHg)
            out_all = model(X_tensor, stats_tensor)
            wave_pred_norm, sbp_head_pred, dbp_head_pred = train_mod._unpack_model_output(out_all)
            pred_norm = wave_pred_norm.cpu().numpy().ravel()
            pred = train_mod.denormalize_abp(pred_norm)

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
                sbp_m_t, dbp_m_t, _ = train_mod._segment_bp_means_from_wave(Y)
                sbp_m_p, dbp_m_p, _ = train_mod._segment_bp_means_from_wave(pred)

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

                if train_mod.USE_MULTITASK_SBP_DBP and sbp_head_pred is not None and dbp_head_pred is not None:
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
    # Finalize SBP seg-mean filtered metrics (SBP<=170 and SBP<=160)
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
        # SBP seg-mean filtered groups
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
    sample_indices = np.linspace(0, len(test_list) - 1, num_samples, dtype=int)
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            p = test_list[idx]
            X, Y, _, _ = train_mod.load_segment(p)
            if X is None:
                continue
            X_norm, stats = train_mod.normalize_ppg_ecg_segment(X)
            X_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)
            out_all = model(X_tensor, stats_tensor)
            pred_norm, _, _ = train_mod._unpack_model_output(out_all)
            pred_norm = pred_norm.cpu().numpy().ravel()
            pred_sig = train_mod.denormalize_abp(pred_norm)

            plt.figure(figsize=(15, 5))
            plt.plot(Y, label="True ABP", alpha=0.8, linewidth=1)
            plt.plot(pred_sig, label="Predicted ABP", alpha=0.8, linewidth=1)
            plt.title(f"Sample {i + 1}: True Range {np.min(Y):.1f}-{np.max(Y):.1f}")
            plt.xlabel("Sample Index")
            plt.ylabel("Blood Pressure (mmHg)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            train_mod.save_plot(f"sample_{i + 1}_true_vs_pred")


def main_test_only():
    global train_mod

    train_mod.set_global_seed(train_mod.GLOBAL_SEED, bool(train_mod.SEED_DETERMINISTIC))
    if not bool(train_mod.SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("[Info] Test-only mode: build data lists...")
    train_mod.train_list, train_mod.val_list, train_mod.test_list = train_mod.data_preprocessing()

    stats_loaded = False
    if train_mod.TEST_ONLY_USE_SAVED_STATS:
        stats_loaded = train_mod._load_saved_norm_stats(train_mod.TEST_ONLY_STATS_PATH)
    if not stats_loaded or train_mod.GLOBAL_ABP_MIN is None or train_mod.GLOBAL_ABP_MAX is None:
        print("[Info] Compute training ABP stats...")
        train_mod.compute_global_abp_stats(train_mod.train_list)

    def _build_test_model():
        try:
            depth_mode = int(train_mod.USE_UNET_3L)
        except Exception:
            depth_mode = 0
        if depth_mode not in (0, 1, 2):
            depth_mode = 0

        bn_key = _get_skip_mode()
        if depth_mode == 1:
            return ResNetUNet3L(input_channels=3, base_channels=train_mod.BASE_CHANNELS, bottleneck=bn_key).to(device), f"resnet_unet3l_{bn_key}"
        return ResNetUNet(input_channels=3, base_channels=train_mod.BASE_CHANNELS, bottleneck=bn_key).to(device), f"resnet_unet_{bn_key}"

    def _safe_load(model, path):
        sd = torch.load(path, map_location=device)
        try:
            getattr(model, "load_state_dict")(sd, strict=True)
            return True
        except Exception:
            try:
                cleaned = {(k.replace("_orig_mod.", "") if isinstance(k, str) else k): v for k, v in sd.items()}
                getattr(model, "load_state_dict")(cleaned, strict=False)
                print("[Warn] Weight keys include _orig_mod. prefix; converted and loaded (strict=False)")
                return True
            except Exception as e:
                print(f"[Error] Failed to load weights: {e}")
                return False

    model, run_name = _build_test_model()
    ckpt_path = str(train_mod.TEST_ONLY_CHECKPOINT).strip()
    if not ckpt_path:
        ckpt_path = f"best_{run_name}.pth"
    if not os.path.isfile(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return
    _safe_load(model, ckpt_path)

    # Test-only: evaluate test only
    final_test_results = test_model(model, train_mod.test_list, device, collect_errors=False)
    final_w_mae = float(np.mean(final_test_results['wave_mae'])) if final_test_results['wave_mae'] else float('inf')
    final_w_r = float(np.mean(final_test_results['R_all'])) if final_test_results['R_all'] else float('nan')

    print(
        f"[Test-Only] Waveform: MAE {final_w_mae:.4f}, R {final_w_r:.4f}"
    )

    ridge_res_test = None
    if train_mod.USE_BOTTLENECK_RIDGE_PROBE and train_mod.TEST_ONLY_RUN_RIDGE:
        try:
            ridge_res_test = run_decoder_official_ridge(model, train_mod.train_list, train_mod.val_list, train_mod.test_list, device)
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

    if train_mod.TEST_ONLY_PLOT_TRUEPRED_SCATTER:
        try:
            seg = evaluate_segment_level_predictions(model, train_mod.test_list, device)
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

    if train_mod.TEST_ONLY_VISUALIZE:
        visualize_results(model, train_mod.test_list, device)

    print("\n[Info] Test-only complete!")


if __name__ == "__main__":
    main_test_only()
