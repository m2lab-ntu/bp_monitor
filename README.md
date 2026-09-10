# bp_monitor

**Blood Pressure Estimation**

A two-stage, cuff-less blood-pressure estimation pipeline using PPG and ECG signals. The workflow converts raw waveforms into fixed-length segments, performs subject-wise stratified data splitting, and trains a U-Net model to reconstruct the ABP waveform.

---

## 1. Repository Structure

| Path / File            | Purpose |
|------------------------|---------|
| `pre_train.py`         | Converts raw *WFDB* `.dat` files to cleaned CSV (PPG, ECG, dPPG, ABP). |
| `models/`              | Model architectures and the registry. |
| `train.py`             | End-to-end preprocessing, k-fold split, per-master adapter fine-tune, Optuna LR search, model training. |
| `test.py`              | Evaluation. |
| `environment.yml`      | Conda environment definition. |

---

## 2. Environment Setup

1. Create conda environment from `environment.yml`.
2. Activate the environment before running any script.
3. CUDA GPU is recommended for training.

Example:
```bash
conda env create -f environment.yml
conda activate mamba_6_op
```

---

## 3. Data Preparation

### Raw-to-CSV
If your raw data is in WFDB `.dat` format, use `pre_train.py` to generate CSV files containing:
- `PLETH` (PPG)
- `II` (ECG Lead II)
- `ABP`

Run:
```bash
python pre_train.py
```

---

## 4. Training

Run full preprocessing + training:
```bash
python train.py
```

Which architecture runs is set by `MODEL` at the top of `train.py`.

### Choosing a model

No calibration:
- `unet`
- `unet_mamba_skip`
- `unet_mamba_decoder`
- `unet_transformer_skip`
- `unet_transformer_decoder`

Calibrated:
- `unet_cal`
- `unet_cal_mamba`
- `unet_cal_transformer`
- `unet_cal_abp_ft`
- `unet_cal_bp_ft`

Key configuration options in `train.py`:
- `REUSE_PROCESSED`: reuse existing `.npz` segments.
- `USE_KFOLD`: 5 subject-wise folds.
- `USE_OPTUNA_LR_SEARCH`: Optuna search before training.
- `batch_size`, `epochs`, `learning_rate`.

---

## 5. Test / Evaluation

`train.py` runs the evaluation itself after each variant finishes training.

Evaluation includes:
- Waveform MAE, ME, SD, RMSE
- Ridge regression
