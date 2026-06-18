import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import config
from model import build_embedding_model, build_model, get_output
from ptbxl_data import PTBXLSequence, load_ptbxl_metadata, split_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the PTB-XL research model.")
    parser.add_argument("--data-dir", type=Path, default=config.PTBXL_ROOT)
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=config.SAMPLING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--weights", type=Path, default=config.FINAL_WEIGHTS_PATH)
    parser.add_argument("--mahalanobis-stats", type=Path, default=config.MAHALANOBIS_STATS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    config.validate_config()
    config.make_output_dirs()

    metadata = load_ptbxl_metadata(args.data_dir, sampling_rate=args.sampling_rate)
    split_map = dict(zip(["train", "val", "test"], split_metadata(metadata)))
    eval_meta = split_map[args.split]

    model = build_model(
        input_shape=(config.signal_length(args.sampling_rate), len(config.SELECTED_LEADS))
    )
    model.load_weights(args.weights)

    seq = PTBXLSequence(
        eval_meta,
        root=args.data_dir,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
    )
    results = predict_sequence(model, seq, args.mahalanobis_stats)
    metrics = compute_metrics(results)

    metrics_path = config.METRICS_DIR / f"{args.split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")


def predict_sequence(model: tf.keras.Model, seq: PTBXLSequence, stats_path: Path):
    embedding_model = build_embedding_model(model)
    y_true = []
    y_score = []
    y_pred = []
    noise_true = []
    noise_pred = []
    reconstruction_error = []
    embeddings = []

    for batch_x, batch_y in seq:
        outputs = model.predict(batch_x, verbose=0)
        class_probs = get_output(outputs, "classification")
        noise_scores = get_output(outputs, "noise")
        reconstruction = get_output(outputs, "reconstruction")
        embedding = embedding_model.predict(batch_x, verbose=0)

        y_true.append(batch_y["classification"])
        y_score.append(class_probs)
        y_pred.append(np.argmax(class_probs, axis=1))
        noise_true.append(batch_y["noise"])
        noise_pred.append(noise_scores)
        reconstruction_error.append(np.mean(np.square(batch_y["reconstruction"] - reconstruction), axis=(1, 2)))
        embeddings.append(embedding)

    results = {
        "y_true": np.concatenate(y_true, axis=0),
        "y_score": np.concatenate(y_score, axis=0),
        "y_pred": np.concatenate(y_pred, axis=0),
        "noise_true": np.concatenate(noise_true, axis=0),
        "noise_pred": np.concatenate(noise_pred, axis=0),
        "reconstruction_error": np.concatenate(reconstruction_error, axis=0),
        "embeddings": np.concatenate(embeddings, axis=0),
    }

    if Path(stats_path).exists():
        results["mahalanobis_distance"] = mahalanobis_distance(results["embeddings"], stats_path)
    else:
        results["mahalanobis_distance"] = np.full((results["embeddings"].shape[0],), np.nan)

    return results


def compute_metrics(results):
    y_true_onehot = results["y_true"]
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = results["y_pred"]
    y_score = results["y_score"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    try:
        roc_auc = roc_auc_score(
            y_true_onehot,
            y_score,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_macro_ovr": None if roc_auc is None else float(roc_auc),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=config.TARGET_CLASSES,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "noise_mae": float(np.mean(np.abs(results["noise_true"] - results["noise_pred"]))),
        "reconstruction_error_mean": float(np.mean(results["reconstruction_error"])),
        "reconstruction_error_std": float(np.std(results["reconstruction_error"])),
        "mahalanobis_distance_mean": float(np.nanmean(results["mahalanobis_distance"])),
        "mahalanobis_distance_std": float(np.nanstd(results["mahalanobis_distance"])),
    }


def mahalanobis_distance(embeddings: np.ndarray, stats_path: Path) -> np.ndarray:
    stats = np.load(stats_path, allow_pickle=True)
    mean = stats["mean"]
    precision = stats["precision"]
    delta = embeddings - mean[None, :]
    squared = np.sum((delta @ precision) * delta, axis=1)
    return np.sqrt(np.maximum(squared, 0.0))


if __name__ == "__main__":
    main()
