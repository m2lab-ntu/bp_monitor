# bp_monitor

**Blood Pressure Estimation**

A two-stage, cuff-less blood-pressure estimation pipeline using **PPG** and **ECG (Lead II)** signals. The workflow converts raw waveforms into fixed-length segments, performs subject-wise stratified splitting, and trains a ResNet-UNet model for ABP waveform reconstruction with optional Mamba/Transformer skip refinement.

---

## 1. Repository Structure

| Path / File            | Purpose |
|------------------------|---------|
| `pre_train.py`         | Converts raw *WFDB* `.dat` files to cleaned CSV (PPG, ECG, dPPG, ABP). |
| `model.py`             | Model architectures, skip blocks, and loss functions. |
| `train.py`             | End-to-end preprocessing, stratified split, Optuna optimisation, model training. |
| `test.py`              | Evaluation utilities and test-only execution. |
| `processed_minmax/`    | Auto-generated `.npz` segments (features, labels, metadata). |
| `plots/`               | Saved figures (timestamped filenames). |
| `environment.yml`      | Conda environment definition. |

---

## 2. Environment Setup

1. Create conda environment from `environment.yml`.
2. Activate the environment before running any script.
3. CUDA GPU is recommended for training.

Run:
```bash
conda env create -f environment.yml
conda activate bp_monitor
```

---

## 3. Data Preparation

If your raw data is in WFDB `.dat` format, use `pre_train.py` to generate CSV files containing:
- `PLETH` (PPG)
- `II` (ECG Lead II)
- `dPPG`
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

Key configuration options in `train.py`:
- `REUSE_PROCESSED`: reuse existing `.npz` segments.
- `SPLIT_MODE`: `subject` or `segment`.


---

## 5. Test / Evaluation

Run evaluation without training:
```bash
python test.py
```

The test-only flow reads:
- `TEST_ONLY_CHECKPOINT`
- `TEST_ONLY_STATS_PATH`

These are configured in `train.py`. You can also run test-only by setting:
- `TEST_ONLY_MODE=1` in `train.py`

---

## 6. Plots

All plots are saved to `plots/` with timestamps in the filename. There is no inline display. Plot toggles:
- `ENABLE_PLOT_PRETRAIN_DISTRIBUTIONS`
- `ENABLE_PLOT_BA_SCATTER`
- `ENABLE_PLOT_TRUEPRED_SCATTER`

You can change the output directory or timestamp format in `train.py`:
- `PLOT_OUTPUT_DIR`
- `PLOT_TIMESTAMP_FMT`

---

## 7. Notes

- Ensure `data_paths`, `processed_dir`, and `SUBJECT_FILE_MAP_PATHS` are updated for your machine.
- If `REUSE_PROCESSED=True`, training will skip CSV processing and use existing `.npz` files.

---
