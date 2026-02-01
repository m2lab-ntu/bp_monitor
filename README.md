# bp_monitor

**Blood Pressure Estimation**

A concise and professional implementation of a **two-stage, cuff-less blood-pressure estimation pipeline** based on **photoplethysmography (PPG)** and **electrocardiography (ECG Lead II)** signals. The project converts raw waveform data into machine-learning friendly segments, conducts subject-wise stratified splits, and trains an Improved ResNet (min–max prediction) in parallel with a U-Net (waveform reconstruction). In addition, the U-Net incorporates sequence modeling via Mamba/Transformer blocks. 

---

## 1. Repository Structure

| Path / File            | Purpose |
|------------------------|---------|
| `pre_train.py`         | Converts raw *WFDB* `.dat` files to cleaned CSV (PPG, ECG (Lead II), dPPG, ABP). |
| `train.py`             | End-to-end preprocessing, stratified split, Optuna optimisation, model training & evaluation. |
| `processed_minmax/`    | Auto-generated .npz segments (features, labels, metadata). |

---

## 2. Requirements

* **Python 3.8+**
* GPU with CUDA 11+ (optional but recommended)
* `pip install -r requirements.txt`
  * core: `numpy pandas scipy matplotlib tqdm scikit-learn`
  * deep learning: `tensorflow keras`
  * optimisation / I/O: `optuna wfdb`

---
