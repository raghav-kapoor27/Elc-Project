# Noise-Aware Zero-Shot OMI Detection on PTB-XL

TensorFlow/Keras research scaffold for **Noise-Aware Zero-Shot Detection of Occlusion Myocardial Infarction using PTB-XL**.

This project uses **only PTB-XL**, loads ECGs with **WFDB**, supports **100 Hz and 500 Hz**, and does **not** include TinyML, ESP32 deployment, TFLite, or ONNX.

## Lead Selection

Change the ECG lead configuration in one place only:

```python
# =====================================
# CHANGE ECG LEADS HERE
# =====================================
SELECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# =====================================
```

Examples:

```python
SELECTED_LEADS = ["I"]
SELECTED_LEADS = ["I", "II", "III"]
SELECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
SELECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
```

The model input shape, WFDB lead indexing, preprocessing, reconstruction head, prediction path, and explainability outputs automatically adapt to this list.

## Dataset

Download and extract PTB-XL from PhysioNet, then set this path in `config.py`:

```python
PTBXL_ROOT = Path("data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
```

The loader expects:

- `ptbxl_database.csv`
- `scp_statements.csv`
- `records100/...`
- `records500/...`

PTB-XL `strat_fold` is used for splits:

- Train: folds 1-8
- Validation: fold 9
- Test: fold 10

## Labels

The classification head predicts three PTB-XL diagnostic groups:

- Normal
- Myocardial Infarction
- ST/T Abnormality

PTB-XL samples with multiple diagnostic superclasses are converted to one softmax label with this priority:

```python
MI > STTC > NORM
```

Because PTB-XL does not provide direct OMI labels, OMI is treated as a **zero-shot anomaly/risk signal** using:

- MI and ST/T probabilities
- Autoencoder reconstruction error
- Mahalanobis distance in the latent embedding space

## Architecture

```text
Input ECG
  -> 1D CNN encoder
  -> Transformer encoder
  -> Latent embedding
       -> Classification head: Normal / MI / ST-T abnormality
       -> Noise head: signal quality score / noise score
       -> Reconstruction head: denoising autoencoder output
```

Mahalanobis statistics are fitted after training using normal PTB-XL training embeddings.

## Preprocessing

Each selected-lead signal is processed with:

- Baseline wander removal
- Butterworth bandpass filtering
- Per-record, per-lead z-score normalization
- Optional training-time noise augmentation

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Train at 100 Hz:

```bash
python train.py --sampling-rate 100
```

Train at 500 Hz:

```bash
python train.py --sampling-rate 500 --batch-size 8
```

Training runs in two phases by design:

1. CNN+Transformer classifier and noise head.
2. Denoising reconstruction head plus Mahalanobis anomaly statistics.

Outputs are written to `outputs/`:

- `outputs/checkpoints/classifier_best.weights.h5`
- `outputs/checkpoints/research_model_final.weights.h5`
- `outputs/mahalanobis_stats.npz`
- TensorBoard logs in `outputs/logs/`

## Evaluate

```bash
python evaluate.py --split test --sampling-rate 100
```

Metrics include:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Macro ROC-AUC one-vs-rest
- Noise MAE
- Reconstruction error summary
- Mahalanobis distance summary

## Predict

Predict by PTB-XL ECG ID:

```bash
python predict.py --ecg-id 1 --sampling-rate 100
```

Predict from a WFDB record path without extension:

```bash
python predict.py --record data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr --sampling-rate 100
```

Save explainability maps:

```bash
python predict.py --ecg-id 1 --sampling-rate 100 --save-explainability
```

This writes saliency and transformer attention maps to `outputs/explainability/` as `.npy` and `.png` files.
