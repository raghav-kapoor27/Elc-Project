# Noise-Aware Zero-Shot OMI Suspicion Detection

Research-grade Python 3.11 system for single-lead Lead I ECG foundation learning, noise-aware inference, and zero-shot OMI suspicion scoring.

The project does **not** require manual dataset downloads. PhysioNet datasets are lazily downloaded into `.cache/ecg_data/` on first use and reused afterward.

## Scope

This is not a binary OMI classifier. It learns normal ECG embedding structure and returns anomaly-style OMI suspicion scores:

```json
{
  "signal_quality": 0.91,
  "noise_score": 0.09,
  "anomaly_score": 2.14,
  "omi_suspicion_score": 0.72,
  "confidence": 0.83
}
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional credentials are read from environment variables:

- `KAGGLE_USERNAME`, `KAGGLE_KEY` if a future Kaggle dataset is added.
- `HF_TOKEN` if a future private HuggingFace dataset is added.

The required datasets in this scaffold are acquired from PhysioNet through WFDB.

## Datasets

Lazy cache root: `.cache/ecg_data/`

| Dataset | PhysioNet slug | Role |
| --- | --- | --- |
| PTB-XL | `ptb-xl` | Lead I pathology adaptation |
| Icentia11k | `icentia11k-continuous-ecg` | self-supervised pretraining |
| STAFF III | `staffiii` | ischemic/ST evaluation and stress testing |
| MIT-BIH | `mitdb` | rhythm/noise robustness evaluation |

Acquire explicitly:

```bash
python scripts/acquire_datasets.py --dataset all
```

Or let training/inference scripts trigger lazy acquisition automatically.

## Training

Self-supervised pretraining, primarily on Icentia11k:

```bash
python scripts/train_self_supervised.py --dataset icentia11k --max-records 100 --epochs 10
```

Pathology adaptation on PTB-XL Lead I:

```bash
python scripts/train_pathology.py --max-records 2000 --epochs 10
```

Zero-shot anomaly detector from normal ECG embeddings:

```bash
python scripts/train_zero_shot.py --checkpoint checkpoints/pathology.ckpt --max-records 2000
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/pathology.ckpt --detector checkpoints/zero_shot_detector.npz
```

## Real-Time Inference

For a CSV stream/file containing one voltage sample per line:

```bash
python scripts/infer_stream.py --checkpoint checkpoints/pathology.ckpt --detector checkpoints/zero_shot_detector.npz --input samples.csv
```

## Export

```bash
python scripts/export_models.py --checkpoint checkpoints/pathology.ckpt --detector checkpoints/zero_shot_detector.npz
```

Exports are written to `exports/`:

- `foundation.onnx`
- `foundation.ts`
- `foundation_saved_model/`
- `foundation.tflite`

TensorFlow Lite conversion uses ONNX as the interchange boundary when available; see [ecg_omi/export.py](ecg_omi/export.py).

## ESP32

See [deployment/esp32_streaming_example/esp32_streaming_example.ino](deployment/esp32_streaming_example/esp32_streaming_example.ino). The ESP32 streams Lead I ADC samples over serial/Wi-Fi to the Python inference service. The full transformer is intended for edge-server or phone-class inference; ESP32 deployment normally uses a distilled/TFLite Micro model or streaming feature extraction.

## Architecture Diagrams

- [diagrams/architecture.mmd](diagrams/architecture.mmd)
- [diagrams/training_flow.mmd](diagrams/training_flow.mmd)

Render with Mermaid-compatible tooling.

