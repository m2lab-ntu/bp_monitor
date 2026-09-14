import os
import json
import logging
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import clone

import joblib
import torch

import train as train_mod
import preprocess


# Calibration context loading and forward
def _load_cal_context_tensors(filepath, device):
    window_pairs = train_mod._WINDOW_CAL_PAIRS

    if filepath in window_pairs:
        p = window_pairs[filepath]
        cal_low_path = p["cal_low_path"]
        cal_high_path = p["cal_high_path"]
        sbp1, dbp1 = float(p["cal_low_sbp"]), float(p["cal_low_dbp"])
        sbp2, dbp2 = float(p["cal_high_sbp"]), float(p["cal_high_dbp"])
    else:
        raise RuntimeError(
            f"Missing calibration pair for target window: {filepath}. "
            f"This should never happen after filter_target_lists(). "
            f"If this is the no-calibration variant, the caller must "
            f"branch on use_calibration=False instead of invoking "
            f"_load_cal_context_tensors."
        )

    X_c1, _, _, _ = preprocess.load_segment(cal_low_path)
    X_c1, _ = preprocess.normalize_ppg_ecg_segment(X_c1)
    X_c2, _, _, _ = preprocess.load_segment(cal_high_path)
    X_c2, _ = preprocess.normalize_ppg_ecg_segment(X_c2)

    x_cal1 = torch.FloatTensor(X_c1).transpose(0, 1).unsqueeze(0).to(device)
    x_cal2 = torch.FloatTensor(X_c2).transpose(0, 1).unsqueeze(0).to(device)
    cal_bp = torch.FloatTensor([sbp1, dbp1, sbp2, dbp2]).unsqueeze(0).to(device)
    return x_cal1, x_cal2, cal_bp


def _forward_eval_one(model, X_tensor, stats_tensor, device, fp, *,
                      use_calibration: bool):
    if use_calibration:
        train_mod._maybe_swap_adapter(model, fp, device)
        x_c1, x_c2, c_bp = _load_cal_context_tensors(fp, device)
        return model(X_tensor, x_c1, x_c2, c_bp, stats_tensor)
    return model(X_tensor, stats_tensor)


def _enforce_static_first_n_mode(cal_pairs_path: str):
    try:
        with open(cal_pairs_path, "r") as f:
            d = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Cannot read calibration pair JSON {cal_pairs_path}: {e}"
        )
    mode = d.get("mode", "")
    accepted = {
        train_mod.CALIBRATION_MODE_TAG,
        getattr(train_mod, "CALIBRATION_MODE_TAG_JUST_BEFORE_TARGET", ""),
        # Equivalent algorithm under an older name — accept silently.
        "per_master_record_fixed_initial_window",
    }
    if mode in accepted:
        return
    raise RuntimeError(
        f"Refusing to test-only with calibration pairs from "
        f"{cal_pairs_path}: mode='{mode}' is not one of the accepted "
        f"leakage-free modes ({sorted(m for m in accepted if m)}). "
        f"Re-run training to regenerate calibration pairs."
    )


def calc_ME_SD(t, p):
    e = p - t
    n = e.size if hasattr(e, 'size') else len(e)
    if n <= 1:
        return float(np.mean(e)), 0.0
    return float(np.mean(e)), float(np.std(e, ddof=1))


# Per-segment testing
def test_model(model, test_list, device, *, use_calibration: bool = True):
    model.eval()
    records = []

    # Sample-level (population) accumulators for global ABP residuals
    total_n = 0
    sum_e = 0.0
    sum_abs_e = 0.0
    sum_e2 = 0.0

    with torch.no_grad():
        for fp in tqdm(test_list, desc="Testing"):
            X, Y, _, _ = preprocess.load_segment(fp)
            if X is None or Y is None:
                continue
            X_norm, stats = preprocess.normalize_ppg_ecg_segment(X)
            X_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)

            out_all = _forward_eval_one(
                model, X_tensor, stats_tensor, device, fp,
                use_calibration=use_calibration,
            )
            wave_pred_norm, _, _ = train_mod._unpack_model_output(out_all)
            pred_norm = wave_pred_norm.cpu().numpy().ravel()
            pred = preprocess.denormalize_abp(pred_norm)

            # Waveform per-segment metrics
            mae = float(mean_absolute_error(Y, pred))
            me, sd = calc_ME_SD(Y, pred)
            R = (float(np.corrcoef(Y, pred)[0, 1])
                 if (np.std(Y) >= 1e-8 and np.std(pred) >= 1e-8) else 0.0)

            sbp_t, dbp_t, _ = preprocess.segment_bp_means_from_wave(Y)
            sbp_p, dbp_p, _ = preprocess.segment_bp_means_from_wave(pred)
            sbp_err = (sbp_p - sbp_t) if (np.isfinite(sbp_t) and np.isfinite(sbp_p)) else float('nan')
            dbp_err = (dbp_p - dbp_t) if (np.isfinite(dbp_t) and np.isfinite(dbp_p)) else float('nan')

            meta = (preprocess.meta_for(fp) or {})
            sid = meta.get("sid", preprocess.get_sid_from_path(fp))
            master = meta.get("master_record", "")
            sw_idx = int(meta.get("subject_window_index", -1))
            rec_idx = int(meta.get("record_window_index", -1))
            raw_s = int(meta.get("raw_start_idx", -1))
            raw_e_idx = int(meta.get("raw_end_idx", -1))

            # Calibration distances — record_window_index and raw_start_idx are master-local.
            cal = train_mod._WINDOW_CAL_PAIRS.get(fp, None) if use_calibration else None
            if cal is not None:
                cl_rwi = int(cal.get("cal_low_record_window_index", -1))
                ch_rwi = int(cal.get("cal_high_record_window_index", -1))
                cl_raw = int(cal.get("cal_low_raw_start_idx", -1))
                ch_raw = int(cal.get("cal_high_raw_start_idx", -1))
                # Older JSON variants may not record cal master; fall back to assuming same-master.
                cl_master = cal.get("cal_low_master_record", master)
                ch_master = cal.get("cal_high_master_record", master)
                cl_same = (str(cl_master) == str(master))
                ch_same = (str(ch_master) == str(master))

                d_wins = []
                if rec_idx >= 0:
                    if cl_same and cl_rwi >= 0:
                        d_wins.append(abs(rec_idx - cl_rwi))
                    if ch_same and ch_rwi >= 0:
                        d_wins.append(abs(rec_idx - ch_rwi))
                d_win = float(min(d_wins)) if d_wins else float('nan')
                d_min = ((d_win * train_mod.SEGMENT_LEN / train_mod.FS / 60.0)
                         if math.isfinite(d_win) else float('nan'))

                d_samps = []
                if raw_s >= 0:
                    if cl_same and cl_raw >= 0:
                        d_samps.append(abs(raw_s - cl_raw))
                    if ch_same and ch_raw >= 0:
                        d_samps.append(abs(raw_s - ch_raw))
                if d_samps:
                    d_samp = int(min(d_samps))
                    d_time_min = d_samp / train_mod.FS / 60.0
                else:
                    d_samp = -1
                    d_time_min = float('nan')
            else:
                cl_rwi = ch_rwi = cl_raw = ch_raw = -1
                cl_master = ch_master = ""
                cl_same = ch_same = False
                d_win = float('nan')
                d_min = float('nan')
                d_samp = -1
                d_time_min = float('nan')

            # Time from master start to target; independent of which calibration mode picked the pair.
            target_time_from_master_start_minutes = (
                rec_idx * train_mod.SEGMENT_LEN / train_mod.FS / 60.0
                if rec_idx >= 0 else float('nan')
            )

            rec = {
                "npz_path": fp,
                "sid": sid,
                "master_record": master,
                "subject_window_index": sw_idx,
                "record_window_index": rec_idx,
                "raw_start_idx": raw_s,
                "raw_end_idx": raw_e_idx,
                "cal_low_record_window_index": cl_rwi,
                "cal_high_record_window_index": ch_rwi,
                "cal_low_raw_start_idx": cl_raw,
                "cal_high_raw_start_idx": ch_raw,
                "cal_low_master_record": cl_master,
                "cal_high_master_record": ch_master,
                "cal_low_is_cross_master": (not cl_same) if cal is not None else False,
                "cal_high_is_cross_master": (not ch_same) if cal is not None else False,
                "cal_distance_windows": d_win,
                "cal_distance_minutes": d_min,
                "cal_distance_samples": d_samp,
                "cal_distance_time_minutes": d_time_min,
                "target_time_from_master_start_minutes":
                    target_time_from_master_start_minutes,
                "wave_mae": mae,
                "wave_me": me,
                "wave_sd": sd,
                "wave_R": R,
                "waveform_sbp_true": sbp_t,
                "waveform_sbp_pred": sbp_p,
                "waveform_sbp_error": sbp_err,
                "waveform_dbp_true": dbp_t,
                "waveform_dbp_pred": dbp_p,
                "waveform_dbp_error": dbp_err,
            }
            # Stamp per-master adapter audit fields onto every per-segment record so test_segment_metrics.csv.
            rec.update(train_mod._adapter_record_fields(
                model, fp, use_calibration=use_calibration))
            records.append(rec)

            # Sample-level accumulators
            e = pred - Y
            msk = np.isfinite(e)
            e = e[msk]
            n = e.size
            if n > 0:
                total_n += int(n)
                sum_e += float(np.sum(e))
                sum_abs_e += float(np.sum(np.abs(e)))
                sum_e2 += float(np.sum(e * e))

    # Aggregate metrics
    if total_n > 1:
        me_g = sum_e / total_n
        mae_g = sum_abs_e / total_n
        var_n = sum_e2 - (sum_e * sum_e) / total_n
        sd_g = math.sqrt(max(0.0, var_n / (total_n - 1)))
        rmse_g = math.sqrt(sum_e2 / total_n)
    else:
        me_g = mae_g = sd_g = rmse_g = float('nan')

    def _arr(field):
        vals = [r[field] for r in records if r[field] is not None and (
            isinstance(r[field], str) or np.isfinite(r[field]))]
        return np.asarray(vals, dtype=np.float32) if vals else np.empty((0,), dtype=np.float32)

    wave_mae_arr = _arr("wave_mae")
    wave_R_arr = _arr("wave_R")

    sbp_t_arr = np.asarray([r["waveform_sbp_true"] for r in records
                            if np.isfinite(r["waveform_sbp_true"])
                            and np.isfinite(r["waveform_sbp_pred"])], dtype=np.float32)
    sbp_p_arr = np.asarray([r["waveform_sbp_pred"] for r in records
                            if np.isfinite(r["waveform_sbp_true"])
                            and np.isfinite(r["waveform_sbp_pred"])], dtype=np.float32)
    dbp_t_arr = np.asarray([r["waveform_dbp_true"] for r in records
                            if np.isfinite(r["waveform_dbp_true"])
                            and np.isfinite(r["waveform_dbp_pred"])], dtype=np.float32)
    dbp_p_arr = np.asarray([r["waveform_dbp_pred"] for r in records
                            if np.isfinite(r["waveform_dbp_true"])
                            and np.isfinite(r["waveform_dbp_pred"])], dtype=np.float32)

    def _agg(true_a, pred_a):
        if true_a.size == 0:
            return {"N": 0, "MAE": float('nan'), "ME": float('nan'),
                    "SD": float('nan'), "RMSE": float('nan'), "R": float('nan')}
        err = pred_a - true_a
        mae_ = float(np.mean(np.abs(err)))
        me_ = float(np.mean(err))
        sd_ = float(np.std(err, ddof=1)) if err.size > 1 else 0.0
        rmse_ = float(np.sqrt(np.mean(err * err)))
        if (true_a.size > 1 and np.std(true_a) >= 1e-8 and np.std(pred_a) >= 1e-8):
            r_ = float(np.corrcoef(true_a, pred_a)[0, 1])
        else:
            r_ = 0.0
        return {"N": int(true_a.size), "MAE": mae_, "ME": me_, "SD": sd_,
                "RMSE": rmse_, "R": r_}

    summary = {
        "N_segments": len(records),
        "wave_mae_mean": float(np.mean(wave_mae_arr)) if wave_mae_arr.size else float('nan'),
        "wave_R_mean": float(np.mean(wave_R_arr)) if wave_R_arr.size else float('nan'),
        "sample_global": {"N": total_n, "ME": me_g, "MAE": mae_g, "SD": sd_g, "RMSE": rmse_g},
        "sbp_waveform": _agg(sbp_t_arr, sbp_p_arr),
        "dbp_waveform": _agg(dbp_t_arr, dbp_p_arr),
    }

    return {"records": records, "summary": summary}


# Metric logging
def log_test_summary(test_results, tag="Test-Final"):
    s = test_results["summary"]
    logging.info(
        f"[{tag}] Waveform: MAE {s['wave_mae_mean']:.4f}, R {s['wave_R_mean']:.4f}"
    )


_SEGMENT_METRICS_BASENAME = "test_segment_metrics.csv"


def save_test_segment_metrics(test_results, run_dir):
    df = pd.DataFrame(test_results["records"])
    out_path = os.path.join(run_dir, _SEGMENT_METRICS_BASENAME)
    df.to_csv(out_path, index=False)
    logging.info(f"Saved test_segment_metrics: {out_path} ({len(df)} rows)")
    return out_path


# Decoder embeddings and ridge probe
def _collect_decoder_embeddings(file_list, model, device, desc_tag="train",
                                *, use_calibration: bool = True):
    base_model = getattr(model, "_orig_mod", model)
    if not hasattr(base_model, "extract_decoder_embedding"):
        raise AttributeError("Model lacks extract_decoder_embedding(); cannot run decoder ridge probe")
    base_model.eval()

    X_list = []
    y_sbp = []
    y_dbp = []
    paths_kept = []
    skipped = 0

    with torch.no_grad():
        for fp in tqdm(file_list, desc=f"Extracting decoder embeddings ({desc_tag})"):
            X, Y, _, _ = preprocess.load_segment(fp)
            if X is None or Y is None:
                skipped += 1
                continue
            X_norm, stats = preprocess.normalize_ppg_ecg_segment(X)
            sbp_m_t, dbp_m_t, _ = preprocess.segment_bp_means_from_wave(Y)
            if not (np.isfinite(sbp_m_t) and np.isfinite(dbp_m_t)):
                skipped += 1
                continue

            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)

            # Calibrated path: a missing pinned cal pair is a hard error.
            if use_calibration and fp not in train_mod._WINDOW_CAL_PAIRS:
                raise RuntimeError(
                    f"Missing calibration pair during decoder embedding extraction: {fp}"
                )

            try:
                if use_calibration:
                    # Swap to this segment's per-master adapter (if any) so the extracted decoder embedding reflects.
                    train_mod._maybe_swap_adapter(base_model, fp, device)
                    x_c1, x_c2, c_bp = _load_cal_context_tensors(fp, device)
                    emb = base_model.extract_decoder_embedding(
                        x_tensor, x_c1, x_c2, c_bp, stats_tensor
                    )
                else:
                    emb = base_model.extract_decoder_embedding(x_tensor, stats_tensor)
            except Exception as e:
                if use_calibration:
                    raise RuntimeError(
                        f"Failed to extract calibrated decoder embedding for "
                        f"target window: {fp}"
                    ) from e
                skipped += 1
                continue
            emb_np = emb.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            X_list.append(emb_np)
            y_sbp.append(float(sbp_m_t))
            y_dbp.append(float(dbp_m_t))
            paths_kept.append(fp)

    embed_dim = int(X_list[0].shape[0]) if X_list else 0
    X_arr = (np.stack(X_list, axis=0).astype(np.float32, copy=False)
             if X_list else np.empty((0, embed_dim), dtype=np.float32))
    y_sbp_arr = np.asarray(y_sbp, dtype=np.float32) if y_sbp else np.empty((0,), dtype=np.float32)
    y_dbp_arr = np.asarray(y_dbp, dtype=np.float32) if y_dbp else np.empty((0,), dtype=np.float32)
    return X_arr, y_sbp_arr, y_dbp_arr, paths_kept, skipped


def _fit_ridge_one_target(X_train, y_train, X_val, y_val, X_test, y_test,
                          *, refit_with_train_val: bool):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if (X_val is not None and len(X_val) > 0) else None
    X_test_s = scaler.transform(X_test)

    has_val = X_val_s is not None and len(y_val) > 0
    estimator = Ridge()
    param_grid = {
        'alpha': list(train_mod.MODEL_SPEC.ridge_alpha_grid),
        'solver': ['auto', 'cholesky', 'sparse_cg'],
    }

    best_params = {}
    best_mae = float('inf')

    for params in ParameterGrid(param_grid):
        m = clone(estimator)
        m.set_params(**params)
        m.fit(X_train_s, y_train)
        if not has_val:
            best_params = params
            break
        y_val_pred = m.predict(X_val_s)
        mae = float(mean_absolute_error(y_val, y_val_pred))
        if mae < best_mae:
            best_mae = mae
            best_params = params

    final_model = clone(estimator)
    if best_params:
        final_model.set_params(**best_params)
    if refit_with_train_val and has_val:
        X_fit = np.concatenate([X_train_s, X_val_s], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
    else:
        X_fit, y_fit = X_train_s, y_train
    final_model.fit(X_fit, y_fit)
    y_pred = final_model.predict(X_test_s)

    err = y_pred - y_test
    n = int(err.size)
    me = float(np.mean(err)) if n > 0 else float('nan')
    sd = float(np.std(err, ddof=1)) if n > 1 else 0.0
    if n > 1 and (np.std(y_test) >= 1e-8) and (np.std(y_pred) >= 1e-8):
        r = float(np.corrcoef(y_test, y_pred)[0, 1])
    else:
        r = 0.0

    metrics = {
        "best_params": best_params,
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "ME": me,
        "SD": sd,
        "R": r,
        "R2": float(r2_score(y_test, y_pred)),
        "coef_shape": list(final_model.coef_.shape),
        "intercept": float(np.asarray(final_model.intercept_).reshape(-1)[0]),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    return metrics, scaler, final_model


def fit_and_save_decoder_ridge(model, train_list, val_list, test_list, device,
                               *, lr_sbp_path, lr_dbp_path, summary_path,
                               use_calibration: bool = True):
    X_tr, y_sbp_tr, y_dbp_tr, paths_tr, skip_tr = _collect_decoder_embeddings(
        train_list, model, device, desc_tag="train",
        use_calibration=use_calibration)
    X_va, y_sbp_va, y_dbp_va, paths_va, skip_va = _collect_decoder_embeddings(
        val_list, model, device, desc_tag="val",
        use_calibration=use_calibration)
    X_te, y_sbp_te, y_dbp_te, paths_te, skip_te = _collect_decoder_embeddings(
        test_list, model, device, desc_tag="test",
        use_calibration=use_calibration)

    refit = bool(train_mod.BOTTLENECK_RIDGE_USE_TRAIN_VAL)
    results = {
        "skipped_train": int(skip_tr),
        "skipped_val": int(skip_va),
        "skipped_test": int(skip_te),
        "embed_dim": int(X_tr.shape[1]) if X_tr.ndim == 2 else 0,
        "refit_with_train_val": refit,
        "test_paths": paths_te,
    }

    if len(y_sbp_tr) >= 4 and len(y_sbp_va) >= 1 and len(y_sbp_te) >= 1:
        m_sbp, scaler_sbp, model_sbp = _fit_ridge_one_target(
            X_tr, y_sbp_tr, X_va, y_sbp_va, X_te, y_sbp_te, refit_with_train_val=refit)
        results["sbp"] = m_sbp | {
            "train_N": int(len(y_sbp_tr)),
            "val_N": int(len(y_sbp_va)),
            "test_N": int(len(y_sbp_te)),
        }
        joblib.dump({
            "scaler": scaler_sbp,
            "model": model_sbp,
            "best_params": m_sbp["best_params"],
            "target": "sbp",
            "feature_type": "decoder_embedding",
            "refit_with_train_val": refit,
        }, lr_sbp_path)
        logging.info(f"Saved SBP regression model: {lr_sbp_path}")
    else:
        results["sbp"] = {
            "MAE": float('nan'), "R": float('nan'), "best_params": {},
            "train_N": int(len(y_sbp_tr)), "val_N": int(len(y_sbp_va)),
            "test_N": int(len(y_sbp_te)),
        }

    if len(y_dbp_tr) >= 4 and len(y_dbp_va) >= 1 and len(y_dbp_te) >= 1:
        m_dbp, scaler_dbp, model_dbp = _fit_ridge_one_target(
            X_tr, y_dbp_tr, X_va, y_dbp_va, X_te, y_dbp_te, refit_with_train_val=refit)
        results["dbp"] = m_dbp | {
            "train_N": int(len(y_dbp_tr)),
            "val_N": int(len(y_dbp_va)),
            "test_N": int(len(y_dbp_te)),
        }
        joblib.dump({
            "scaler": scaler_dbp,
            "model": model_dbp,
            "best_params": m_dbp["best_params"],
            "target": "dbp",
            "feature_type": "decoder_embedding",
            "refit_with_train_val": refit,
        }, lr_dbp_path)
        logging.info(f"Saved DBP regression model: {lr_dbp_path}")
    else:
        results["dbp"] = {
            "MAE": float('nan'), "R": float('nan'), "best_params": {},
            "train_N": int(len(y_dbp_tr)), "val_N": int(len(y_dbp_va)),
            "test_N": int(len(y_dbp_te)),
        }

    summary = {
        "embed_dim": results["embed_dim"],
        "refit_with_train_val": refit,
        "sbp": {k: v for k, v in results["sbp"].items() if k not in ("y_test", "y_pred")},
        "dbp": {k: v for k, v in results["dbp"].items() if k not in ("y_test", "y_pred")},
    }
    if "sbp" in results and "best_params" in results["sbp"]:
        summary["sbp"]["best_params"] = results["sbp"]["best_params"]
    if "dbp" in results and "best_params" in results["dbp"]:
        summary["dbp"]["best_params"] = results["dbp"]["best_params"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved linear regression summary: {summary_path}")

    return results


def predict_with_saved_ridge(model, test_list, device, *, lr_sbp_path, lr_dbp_path,
                             use_calibration: bool = True):
    if not (os.path.isfile(lr_sbp_path) and os.path.isfile(lr_dbp_path)):
        raise FileNotFoundError(
            f"Linear regression artifacts missing: {lr_sbp_path} / {lr_dbp_path}"
        )
    obj_sbp = joblib.load(lr_sbp_path)
    obj_dbp = joblib.load(lr_dbp_path)

    X_te, y_sbp_te, y_dbp_te, paths_te, skip_te = _collect_decoder_embeddings(
        test_list, model, device, desc_tag="test",
        use_calibration=use_calibration)

    res = {"skipped_test": int(skip_te), "test_paths": paths_te,
           "embed_dim": int(X_te.shape[1]) if X_te.ndim == 2 else 0,
           "refit_with_train_val": bool(obj_sbp.get("refit_with_train_val", False))}

    def _eval(obj, y_true):
        if X_te.size == 0 or y_true.size == 0:
            return None
        X_s = obj["scaler"].transform(X_te)
        y_pred = obj["model"].predict(X_s)
        err = y_pred - y_true
        n = int(err.size)
        me = float(np.mean(err)) if n > 0 else float('nan')
        sd = float(np.std(err, ddof=1)) if n > 1 else 0.0
        if n > 1 and np.std(y_true) >= 1e-8 and np.std(y_pred) >= 1e-8:
            r = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            r = 0.0
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "ME": me, "SD": sd, "R": r,
            "best_params": obj.get("best_params", {}),
            "y_test": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "test_N": int(n),
        }

    res["sbp"] = _eval(obj_sbp, y_sbp_te) or {"MAE": float('nan'), "R": float('nan')}
    res["dbp"] = _eval(obj_dbp, y_dbp_te) or {"MAE": float('nan'), "R": float('nan')}
    return res


def log_ridge_summary(ridge_res, tag="Test-Final"):
    sbp = ridge_res.get("sbp", {})
    dbp = ridge_res.get("dbp", {})
    logging.info(
        f"[{tag}][LinReg-DEC] "
        f"SBP: MAE {sbp.get('MAE', float('nan')):.4f}, "
        f"ME {sbp.get('ME', float('nan')):.4f}, "
        f"SD {sbp.get('SD', float('nan')):.4f}, "
        f"R {sbp.get('R', float('nan')):.4f} | "
        f"DBP: MAE {dbp.get('MAE', float('nan')):.4f}, "
        f"ME {dbp.get('ME', float('nan')):.4f}, "
        f"SD {dbp.get('SD', float('nan')):.4f}, "
        f"R {dbp.get('R', float('nan')):.4f}"
    )


def augment_segment_csv_with_ridge(*, run_dir, model, test_list, device,
                                   lr_sbp_path, lr_dbp_path,
                                   use_calibration: bool = True):
    csv_path = os.path.join(run_dir, _SEGMENT_METRICS_BASENAME)
    if not os.path.isfile(csv_path):
        logging.warning(f"{csv_path} not found; skipping ridge column augmentation")
        return
    if not (os.path.isfile(lr_sbp_path) and os.path.isfile(lr_dbp_path)):
        logging.warning("Ridge joblibs missing; skipping ridge column augmentation")
        return

    obj_sbp = joblib.load(lr_sbp_path)
    obj_dbp = joblib.load(lr_dbp_path)

    X_te, y_sbp_te, y_dbp_te, paths_te, _ = _collect_decoder_embeddings(
        test_list, model, device, desc_tag="test-augment",
        use_calibration=use_calibration)
    if X_te.size == 0:
        return
    X_s_sbp = obj_sbp["scaler"].transform(X_te)
    sbp_pred = obj_sbp["model"].predict(X_s_sbp)
    X_s_dbp = obj_dbp["scaler"].transform(X_te)
    dbp_pred = obj_dbp["model"].predict(X_s_dbp)

    add_df = pd.DataFrame({
        "npz_path": paths_te,
        "linear_sbp_true": y_sbp_te.tolist(),
        "linear_sbp_pred": sbp_pred.tolist(),
        "linear_sbp_error": (sbp_pred - y_sbp_te).tolist(),
        "linear_dbp_true": y_dbp_te.tolist(),
        "linear_dbp_pred": dbp_pred.tolist(),
        "linear_dbp_error": (dbp_pred - y_dbp_te).tolist(),
    })

    base = pd.read_csv(csv_path)
    # Drop pre-existing linear columns (re-run case)
    drop_cols = [c for c in base.columns if c.startswith("linear_")]
    base = base.drop(columns=drop_cols, errors="ignore")
    merged = base.merge(add_df, how="left", on="npz_path")
    merged.to_csv(csv_path, index=False)
    logging.info(f"Augmented test_segment_metrics with linear regression columns: {csv_path}")


# Calibration-distance analysis
def _make_distance_bins(values: np.ndarray, default_width=25, min_per_bin=20):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return []
    v_min, v_max = float(np.min(v)), float(np.max(v))
    if v_max <= v_min:
        return [(v_min, v_max + 1e-6)]
    # Linear bins
    edges = np.arange(np.floor(v_min / default_width) * default_width,
                      np.ceil(v_max / default_width) * default_width + default_width,
                      default_width, dtype=np.float64)
    if len(edges) < 2:
        edges = np.array([v_min, v_max + 1e-6])
    bins = list(zip(edges[:-1], edges[1:]))

    counts = []
    for lo, hi in bins:
        m = (values >= lo) & (values < hi)
        counts.append(int(np.sum(m)))
    if counts and min(counts) < min_per_bin:
        # Fall back to quantile bins
        n_q = max(2, min(10, v.size // max(1, min_per_bin)))
        qs = np.quantile(v, np.linspace(0, 1, n_q + 1))
        qs = np.unique(qs)
        if len(qs) >= 2:
            bins = list(zip(qs[:-1], qs[1:]))
        # Adjust the last bin upper to be inclusive
        if bins:
            last_lo, last_hi = bins[-1]
            bins[-1] = (last_lo, last_hi + 1e-6)
    return bins


def _bins_from_edges(edges):
    edges = np.asarray(edges, dtype=np.float64)
    edges = np.unique(edges[np.isfinite(edges)])
    if edges.size < 2:
        return []
    bins = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    if bins:
        lo, hi = bins[-1]
        bins[-1] = (lo, hi + 1e-6)
    return bins


def _calibration_distance_analysis_one(df, run_dir, *, distance_col, suffix,
                                       axis_label, plot_title_unit,
                                       default_width, min_per_bin=20,
                                       custom_edges=None, log_x=False,
                                       max_distance=None,
                                       xtick_positions=None,
                                       xtick_labels=None):
    if distance_col not in df.columns:
        logging.warning(f"{distance_col} column missing; skipping {suffix} analysis")
        return

    distances = pd.to_numeric(df[distance_col], errors="coerce").to_numpy()
    if max_distance is not None and np.isfinite(max_distance):
        keep_mask = np.isfinite(distances) & (distances <= float(max_distance))
        df = df.loc[keep_mask].reset_index(drop=True)
        distances = distances[keep_mask]
    if custom_edges is not None:
        bins = _bins_from_edges(custom_edges)
    else:
        bins = _make_distance_bins(distances, default_width=default_width,
                                   min_per_bin=min_per_bin)
    if not bins:
        logging.warning(f"No bins constructed for {distance_col}; skipping")
        return

    rows = []
    for lo, hi in bins:
        m = (distances >= lo) & (distances < hi)
        sub = df[m]
        N = int(len(sub))
        if N == 0:
            continue
        wave_R = sub["wave_R"].astype(float).mean(skipna=True)
        wave_MAE = sub["wave_mae"].astype(float).mean(skipna=True)

        def _stats(col):
            v = pd.to_numeric(sub.get(col, pd.Series([], dtype=float)),
                              errors="coerce").dropna().to_numpy()
            if v.size == 0:
                return float('nan'), float('nan'), float('nan')
            mae_ = float(np.mean(np.abs(v)))
            me_ = float(np.mean(v))
            sd_ = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
            return mae_, me_, sd_

        sbp_mae, sbp_me, sbp_sd = _stats("linear_sbp_error")
        dbp_mae, dbp_me, dbp_sd = _stats("linear_dbp_error")
        rows.append({
            "bin_start": float(lo), "bin_end": float(hi),
            "bin_center": float(0.5 * (lo + hi)),
            "N": N,
            "mean_wave_R": float(wave_R) if pd.notna(wave_R) else float('nan'),
            "mean_wave_MAE": float(wave_MAE) if pd.notna(wave_MAE) else float('nan'),
            "linear_regression_SBP_MAE": sbp_mae,
            "linear_regression_SBP_ME": sbp_me,
            "linear_regression_SBP_SD": sbp_sd,
            "linear_regression_DBP_MAE": dbp_mae,
            "linear_regression_DBP_ME": dbp_me,
            "linear_regression_DBP_SD": dbp_sd,
        })

    out_csv = os.path.join(run_dir, f"calibration_distance_bin_metrics{suffix}.csv")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    logging.info(f"Saved {out_csv} ({len(out_df)} bins)")

    if out_df.empty:
        return

    def _apply_xaxis(ax):
        if log_x:
            try:
                ax.set_xscale("log")
            except Exception:
                pass
        if xtick_positions is not None:
            ax.set_xticks(list(xtick_positions))
            if xtick_labels is not None:
                ax.set_xticklabels(list(xtick_labels))

    # Plot 1: waveform R + MAE
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(out_df["bin_center"], out_df["mean_wave_R"], '-o', color="#4C78A8", label="wave R")
    ax1.set_xlabel(axis_label)
    ax1.set_ylabel("wave R", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(out_df["bin_center"], out_df["mean_wave_MAE"], '-s', color="#E45756", label="wave MAE")
    ax2.set_ylabel("wave MAE (mmHg)", color="#E45756")
    _apply_xaxis(ax1)
    plt.title(f"Waveform metrics vs distance from calibration ({plot_title_unit})")
    for x, n in zip(out_df["bin_center"], out_df["N"]):
        ax1.annotate(f"N={n}", (x, ax1.get_ylim()[0]), fontsize=7, alpha=0.6)
    fig.tight_layout()
    out_png = os.path.join(run_dir, f"distance_vs_waveform_metrics{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved plot: {out_png}")

    # Plot 2: SBP linear regression MAE/ME/SD
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(out_df["bin_center"], out_df["linear_regression_SBP_MAE"], '-o', label="SBP MAE")
    ax.plot(out_df["bin_center"], out_df["linear_regression_SBP_ME"], '-s', label="SBP ME")
    ax.plot(out_df["bin_center"], out_df["linear_regression_SBP_SD"], '-^', label="SBP SD")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("mmHg")
    _apply_xaxis(ax)
    ax.set_title(f"Linear regression SBP error vs distance ({plot_title_unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(run_dir, f"distance_vs_linear_regression_sbp{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved plot: {out_png}")

    # Plot 3: DBP linear regression MAE/ME/SD
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(out_df["bin_center"], out_df["linear_regression_DBP_MAE"], '-o', label="DBP MAE")
    ax.plot(out_df["bin_center"], out_df["linear_regression_DBP_ME"], '-s', label="DBP ME")
    ax.plot(out_df["bin_center"], out_df["linear_regression_DBP_SD"], '-^', label="DBP SD")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("mmHg")
    _apply_xaxis(ax)
    ax.set_title(f"Linear regression DBP error vs distance ({plot_title_unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(run_dir, f"distance_vs_linear_regression_dbp{suffix}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved plot: {out_png}")


def run_calibration_distance_analysis(run_dir):
    csv_path = os.path.join(run_dir, _SEGMENT_METRICS_BASENAME)
    if not os.path.isfile(csv_path):
        logging.warning(f"{csv_path} not found; skipping distance analysis")
        return
    df = pd.read_csv(csv_path)

    # (a) Cumulative window-count distance (existing behavior, file names unchanged)
    _calibration_distance_analysis_one(
        df, run_dir,
        distance_col="cal_distance_windows",
        suffix="",
        axis_label="Distance from calibration (windows)",
        plot_title_unit="windows",
        default_width=25,
    )

    # (b) Real elapsed time distance, derived from raw sample indices
    _calibration_distance_analysis_one(
        df, run_dir,
        distance_col="cal_distance_time_minutes",
        suffix="_time_min",
        axis_label="Distance from calibration (minutes, real time)",
        plot_title_unit="real time",
        default_width=5.0,
    )

    # (c) Fine-grained view up to 1 hour (5-minute bins).
    fine_minute_edges = list(np.arange(0.0, 60.0 + 1e-9, 5.0))
    _calibration_distance_analysis_one(
        df, run_dir,
        distance_col="cal_distance_time_minutes",
        suffix="_fine_minutes",
        axis_label="Distance from calibration (minutes, real time)",
        plot_title_unit="minute resolution (0–60 min)",
        default_width=5.0,
        custom_edges=fine_minute_edges,
        max_distance=60.0,
    )

    # (d) Fine-grained multi-resolution view up to 24 hours.
    multi_edges_min = [
        0, 5, 15, 30,        # 0-30 min
        60, 120, 180,        # 1, 2, 3 h
        240, 360, 480,       # 4, 6, 8 h
        720, 1080, 1440,     # 12, 18, 24 h
    ]
    hour_ticks_min = [60, 180, 360, 720, 1440]
    hour_tick_labels = ["1h", "3h", "6h", "12h", "24h"]
    _calibration_distance_analysis_one(
        df, run_dir,
        distance_col="cal_distance_time_minutes",
        suffix="_up_to_24h",
        axis_label="Distance from calibration (minutes, real time; log scale)",
        plot_title_unit="0–24h",
        default_width=60.0,
        custom_edges=multi_edges_min,
        max_distance=1440.0,
        log_x=True,
        xtick_positions=hour_ticks_min,
        xtick_labels=hour_tick_labels,
    )

    # (e) Hourly bins up to 24h, linear scale — easier to read for "how much does error grow.
    hourly_edges_min = list(np.arange(0.0, 1440.0 + 1e-9, 60.0))  # 0-24h, 1h bins
    _calibration_distance_analysis_one(
        df, run_dir,
        distance_col="cal_distance_time_minutes",
        suffix="_hourly_24h",
        axis_label="Distance from calibration (hours, real time)",
        plot_title_unit="hourly, 0–24h",
        default_width=60.0,
        custom_edges=hourly_edges_min,
        max_distance=1440.0,
        xtick_positions=list(range(60, 1500, 120)),
        xtick_labels=[f"{int(m / 60)}h" for m in range(60, 1500, 120)],
    )

    # (f) Target offset from the master's start, up to 24h.
    if "target_time_from_master_start_minutes" in df.columns:
        _calibration_distance_analysis_one(
            df, run_dir,
            distance_col="target_time_from_master_start_minutes",
            suffix="_target_from_master_start_24h",
            axis_label="Target time from master start (hours, real time)",
            plot_title_unit="target offset, 0–24h",
            default_width=60.0,
            custom_edges=hourly_edges_min,
            max_distance=1440.0,
            xtick_positions=list(range(60, 1500, 120)),
            xtick_labels=[f"{int(m / 60)}h" for m in range(60, 1500, 120)],
        )
        # Finer multi-resolution view (5/15/30 min, 1/2/3/.../24 h) using the same set of mixed edges.
        _calibration_distance_analysis_one(
            df, run_dir,
            distance_col="target_time_from_master_start_minutes",
            suffix="_target_from_master_start_multi",
            axis_label="Target time from master start (minutes, real time; log scale)",
            plot_title_unit="target offset, 0–24h, multi-resolution",
            default_width=60.0,
            custom_edges=[
                0, 5, 15, 30,
                60, 120, 180,
                240, 360, 480,
                720, 1080, 1440,
            ],
            max_distance=1440.0,
            log_x=True,
            xtick_positions=[60, 180, 360, 720, 1440],
            xtick_labels=["1h", "3h", "6h", "12h", "24h"],
        )


# Segment-level evaluation and plots
def evaluate_segment_level_predictions(model, file_list, device,
                                       *, use_calibration: bool = True):
    sbp_t, sbp_p = [], []
    dbp_t, dbp_p = [], []
    abp_t, abp_p = [], []
    model.eval()
    with torch.no_grad():
        for fp in tqdm(file_list, desc="Post-train: collect segment-level SBP/DBP/ABP"):
            X, Y, _, _ = preprocess.load_segment(fp)
            if X is None:
                continue
            X_norm, stats = preprocess.normalize_ppg_ecg_segment(X)
            x_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)

            out_all = _forward_eval_one(
                model, x_tensor, stats_tensor, device, fp,
                use_calibration=use_calibration,
            )
            pred_norm, _, _ = train_mod._unpack_model_output(out_all)
            pred_norm = pred_norm.cpu().numpy().ravel()
            pred = preprocess.denormalize_abp(pred_norm)

            s_m, d_m, a_m = preprocess.segment_bp_means_from_wave(Y)
            s_pm, d_pm, a_pm = preprocess.segment_bp_means_from_wave(pred)
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


def plot_post_training_metrics(model, file_list, device,
                               *, use_calibration: bool = True):
    if not (train_mod.ENABLE_PLOT_BA_SCATTER or train_mod.ENABLE_PLOT_TRUEPRED_SCATTER):
        return
    seg = evaluate_segment_level_predictions(
        model, file_list, device, use_calibration=use_calibration
    )
    sbp_t, sbp_p = seg['sbp_true'], seg['sbp_pred']
    dbp_t, dbp_p = seg['dbp_true'], seg['dbp_pred']
    abp_t, abp_p = seg['abp_true'], seg['abp_pred']

    if train_mod.ENABLE_PLOT_BA_SCATTER:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        if sbp_t.size and sbp_p.size:
            sbp_mean = 0.5 * (sbp_t + sbp_p)
            sbp_diff = sbp_p - sbp_t
            md = float(np.nanmean(sbp_diff))
            sd = float(np.nanstd(sbp_diff, ddof=1)) if sbp_diff.size > 1 else 0.0
            loa_low, loa_high = md - 1.96 * sd, md + 1.96 * sd
            plt.scatter(sbp_mean, sbp_diff, s=8, alpha=0.35, color="#4C78A8", edgecolors='none')
            xmin, xmax = float(np.nanmin(sbp_mean)), float(np.nanmax(sbp_mean))
            plt.hlines([md, loa_low, loa_high], xmin, xmax,
                       colors=['k', 'r', 'r'], linestyles=['-', '--', '--'], linewidths=1)
        plt.title("SBP Bland–Altman (per segment)")
        plt.xlabel("Mean SBP (mmHg)")
        plt.ylabel("Difference (Pred - True) (mmHg)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if dbp_t.size and dbp_p.size:
            dbp_mean = 0.5 * (dbp_t + dbp_p)
            dbp_diff = dbp_p - dbp_t
            md = float(np.nanmean(dbp_diff))
            sd = float(np.nanstd(dbp_diff, ddof=1)) if dbp_diff.size > 1 else 0.0
            loa_low, loa_high = md - 1.96 * sd, md + 1.96 * sd
            plt.scatter(dbp_mean, dbp_diff, s=8, alpha=0.35, color="#E45756", edgecolors='none')
            xmin, xmax = float(np.nanmin(dbp_mean)), float(np.nanmax(dbp_mean))
            plt.hlines([md, loa_low, loa_high], xmin, xmax,
                       colors=['k', 'r', 'r'], linestyles=['-', '--', '--'], linewidths=1)
        plt.title("DBP Bland–Altman (per segment)")
        plt.xlabel("Mean DBP (mmHg)")
        plt.ylabel("Difference (Pred - True) (mmHg)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        train_mod.save_plot("post_train_ba_scatter")

    if train_mod.ENABLE_PLOT_TRUEPRED_SCATTER:
        plt.figure(figsize=(15, 4))

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

        _scatter(1, sbp_t, sbp_p, "SBP: Actual vs. Predicted", "#4C78A8")
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
    _scatter(1, sbp_t, sbp_p, f"{title_prefix}SBP: True vs Pred", "#4C78A8")
    _scatter(2, dbp_t, dbp_p, f"{title_prefix}DBP: True vs Pred", "#F58518")
    plt.tight_layout()
    suffix = title_prefix.strip().replace(" ", "_") if title_prefix else "segmean"
    train_mod.save_plot(f"segmean_true_pred_{suffix}")


def visualize_results(model, test_list, device, num_samples=3,
                      *, use_calibration: bool = True):
    model.eval()
    if len(test_list) == 0:
        return
    sample_indices = np.linspace(0, len(test_list) - 1, num_samples, dtype=int)
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            fp = test_list[int(idx)]
            X, Y, _, _ = preprocess.load_segment(fp)
            if X is None:
                continue
            X_norm, stats = preprocess.normalize_ppg_ecg_segment(X)
            X_tensor = torch.FloatTensor(X_norm).transpose(0, 1).unsqueeze(0).to(device)
            stats_tensor = (torch.FloatTensor(stats).unsqueeze(0).to(device)
                            if train_mod.USE_SIGNAL_STATS else None)
            out_all = _forward_eval_one(
                model, X_tensor, stats_tensor, device, fp,
                use_calibration=use_calibration,
            )
            pred_norm, _, _ = train_mod._unpack_model_output(out_all)
            pred_norm = pred_norm.cpu().numpy().ravel()
            pred_sig = preprocess.denormalize_abp(pred_norm)

            plt.figure(figsize=(15, 5))
            plt.plot(Y, label="True ABP", alpha=0.8, linewidth=1)
            plt.plot(pred_sig, label="Predicted ABP", alpha=0.8, linewidth=1)
            plt.title(f"Sample {i + 1}: True Range {np.min(Y):.1f}-{np.max(Y):.1f}")
            plt.xlabel("Sample Index")
            plt.ylabel("Blood Pressure (mmHg)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            train_mod.save_plot(f"sample_{i + 1}_true_vs_pred")


# Test-only entry point
def main_test_only():
    if not train_mod.TEST_ONLY_RUN_DIR:
        raise RuntimeError("TEST_ONLY_RUN_DIR is empty; set it to a training run directory.")
    run_dir = str(train_mod.TEST_ONLY_RUN_DIR).rstrip("/")
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"TEST_ONLY_RUN_DIR not found: {run_dir}")

    # Set RUN_DIR so save_plot writes plots into the same run directory
    train_mod.RUN_DIR = run_dir

    train_mod.set_global_seed(train_mod.GLOBAL_SEED, bool(train_mod.SEED_DETERMINISTIC))
    if not bool(train_mod.SEED_DETERMINISTIC):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mod.setup_logging(prefix="test")
    logging.info(f"Using device: {device}")
    logging.info(f"Test-only run directory: {run_dir}")
    logging.info(
        f"Calibration window count N = {train_mod.CALIBRATION_WINDOW_COUNT}"
    )

    # Read variant from run_manifest.json (default to calibrated for legacy runs)
    run_manifest_path_loaded = os.path.join(run_dir, "run_manifest.json")
    use_calibration = True
    variant = "calibrated"
    model_name = "CalibrationUNet"
    if os.path.isfile(run_manifest_path_loaded):
        try:
            with open(run_manifest_path_loaded, "r") as _f:
                _rm = json.load(_f)
            use_calibration = bool(_rm.get("USE_CALIBRATION", True))
            variant = str(_rm.get("variant", "calibrated"))
            model_name = str(_rm.get("model_name", "CalibrationUNet"))
        except Exception as e:
            logging.warning(f"Failed to read run_manifest.json; assuming calibrated: {e}")
    logging.info(
        f"Test-only variant: {variant} (USE_CALIBRATION={use_calibration}, model={model_name})"
    )

    # Load processed manifest (mandatory)
    manifest_path = os.path.join(run_dir, "processed_manifest.csv")
    df_manifest = preprocess.load_processed_manifest(manifest_path)
    if df_manifest is None:
        raise RuntimeError(f"processed_manifest.csv not found in {run_dir}")
    preprocess.install_manifest(df_manifest)

    # Load data lists (exact paths/order)
    data_lists_path = os.path.join(run_dir, "data_lists.json")
    train_l, val_l, test_l = train_mod.load_data_lists(data_lists_path)
    train_mod.train_list = train_l
    train_mod.val_list = val_l
    train_mod.test_list = test_l
    logging.info(
        f"Loaded data_lists: train={len(train_l)}, val={len(val_l)}, test={len(test_l)}"
    )

    # Load calibration pairs only when this run is a calibrated variant.
    if use_calibration:
        fixed_path = os.path.join(run_dir, "calibration_fixed_pairs.json")
        # Fallback to parent run dir's shared cal pairs JSON.
        parent_dir = os.path.dirname(os.path.abspath(run_dir))
        parent_fixed = os.path.join(parent_dir, "calibration_fixed_pairs.json")
        legacy_path = os.path.join(run_dir, "calibration_update_pairs.json")

        if os.path.isfile(fixed_path):
            cal_pairs_path = fixed_path
        elif os.path.isfile(parent_fixed):
            cal_pairs_path = parent_fixed
            logging.info(
                f"Using shared calibration pairs from parent run dir: {parent_fixed}"
            )
        elif os.path.isfile(legacy_path):
            raise RuntimeError(
                f"Refusing to load legacy 'calibration_update_pairs.json' "
                f"in {run_dir}: that file was produced by the "
                f"'per_master_record_sequential_update' mode, which "
                f"updated cal_low/cal_high from target-window ABP and "
                f"therefore contains ground-truth leakage. Re-run "
                f"training to regenerate per-master static-first-N pairs."
            )
        else:
            raise RuntimeError(
                f"No calibration pair JSON found in {run_dir} (or parent): "
                f"expected 'calibration_fixed_pairs.json' "
                f"(per_master_record_static_first_n)."
            )
        train_mod.load_calibration_pairs_json(cal_pairs_path)
        # Defense-in-depth: even if a stray non-static JSON ever lands at one of the accepted paths.
        _enforce_static_first_n_mode(cal_pairs_path)
    else:
        # Hard-clear so no stale pairs from a previous run leak into the no-calibration test path.
        train_mod._WINDOW_CAL_PAIRS = {}

    # Load normalization stats (mandatory)
    abp_stats_path = os.path.join(run_dir, "abp_normalization_stats.json")
    if not preprocess.load_saved_norm_stats(abp_stats_path):
        raise RuntimeError(f"Cannot load ABP normalization stats from {abp_stats_path}")

    # Build the model the run manifest names and load its checkpoint.
    model_variant = train_mod._variant_for_model_name(
        model_name, use_calibration=use_calibration
    )
    model = train_mod._new_model(model_variant, device)
    ckpt_path = os.path.join(
        run_dir, train_mod._VARIANT_MODELS[model_variant]["checkpoint_basename"]
    )
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    train_mod._safe_load(model, ckpt_path, device)

    # Load per-master Residual Feature Adapter cache (calibrated only).
    if use_calibration:
        adapter_path = os.path.join(run_dir, train_mod.ADAPTERS_BASENAME)
        parent_dir = os.path.dirname(os.path.abspath(run_dir))
        parent_adapter_path = os.path.join(parent_dir, train_mod.ADAPTERS_BASENAME)
        if os.path.isfile(adapter_path):
            adapter_cache, adapter_diag_by_key = \
                train_mod.load_per_master_adapters(adapter_path)
        elif os.path.isfile(parent_adapter_path):
            adapter_cache, adapter_diag_by_key = \
                train_mod.load_per_master_adapters(parent_adapter_path)
            logging.info(
                f"Using shared per-master adapters from parent run dir: "
                f"{parent_adapter_path}"
            )
        else:
            adapter_cache = {}
            adapter_diag_by_key = {}
            logging.info(
                "No per_master_adapters.pt found; running with Identity adapter "
                "(pre-fine-tune behaviour)."
            )
        train_mod._attach_adapter_cache(model, adapter_cache,
                                        diag_by_key=adapter_diag_by_key)
    else:
        train_mod._attach_adapter_cache(model, {})

    # Run test (same algorithm as final test in train.py)
    test_results = test_model(model, train_mod.test_list, device,
                              use_calibration=use_calibration)
    log_test_summary(test_results, tag=f"Test-Only[{variant}]")
    save_test_segment_metrics(test_results, run_dir)

    # Linear/Ridge regression: load saved scaler+model, predict only (no fit)
    lr_sbp_path = os.path.join(run_dir, "linear_regression_sbp.joblib")
    lr_dbp_path = os.path.join(run_dir, "linear_regression_dbp.joblib")
    ridge_res_test = None
    if train_mod.USE_BOTTLENECK_RIDGE_PROBE and train_mod.TEST_ONLY_RUN_RIDGE \
            and os.path.isfile(lr_sbp_path) and os.path.isfile(lr_dbp_path):
        try:
            ridge_res_test = predict_with_saved_ridge(
                model, train_mod.test_list, device,
                lr_sbp_path=lr_sbp_path, lr_dbp_path=lr_dbp_path,
                use_calibration=use_calibration,
            )
            log_ridge_summary(ridge_res_test, tag=f"Test-Only[{variant}]")
            augment_segment_csv_with_ridge(
                run_dir=run_dir, model=model, test_list=train_mod.test_list,
                device=device, lr_sbp_path=lr_sbp_path, lr_dbp_path=lr_dbp_path,
                use_calibration=use_calibration,
            )
        except Exception as e:
            logging.warning(f"Test-only ridge failed: {e}")

    # Distance analysis (only meaningful for calibrated runs)
    if use_calibration:
        try:
            run_calibration_distance_analysis(run_dir)
        except Exception as e:
            logging.warning(f"Test-only distance analysis failed: {e}")

    if train_mod.TEST_ONLY_PLOT_TRUEPRED_SCATTER:
        try:
            seg = evaluate_segment_level_predictions(
                model, train_mod.test_list, device,
                use_calibration=use_calibration,
            )
            plot_segmean_truepred_scatter(
                seg['sbp_true'], seg['sbp_pred'],
                seg['dbp_true'], seg['dbp_pred'],
                title_prefix="Waveform"
            )
        except Exception as e:
            logging.warning(f"Test-only waveform scatter failed: {e}")

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
                        sbp_t, sbp_p, dbp_t, dbp_p, title_prefix="LinReg-DEC"
                    )
            except Exception as e:
                logging.warning(f"Test-only LinReg-DEC scatter failed: {e}")

    if train_mod.TEST_ONLY_VISUALIZE:
        try:
            visualize_results(model, train_mod.test_list, device,
                              use_calibration=use_calibration)
        except Exception as e:
            logging.warning(f"Test-only visualize failed: {e}")

    # Leave the model with an Identity adapter so it's deterministic for any downstream re-use.
    train_mod._reset_adapter(model)
    logging.info("Test-only complete!")


if __name__ == "__main__":
    main_test_only()
