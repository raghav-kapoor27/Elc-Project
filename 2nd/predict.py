import argparse
import json
from pathlib import Path

import numpy as np

import config
from evaluate import mahalanobis_distance
from explainability import (
    compute_saliency,
    extract_attention_maps,
    save_attention_plot,
    save_saliency_plot,
)
from model import build_embedding_model, build_model, get_output
from ptbxl_data import load_and_preprocess_record, load_ptbxl_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction on one PTB-XL ECG record.")
    parser.add_argument("--record", type=Path, help="WFDB record path without extension.")
    parser.add_argument("--ecg-id", type=int, help="PTB-XL ecg_id from ptbxl_database.csv.")
    parser.add_argument("--data-dir", type=Path, default=config.PTBXL_ROOT)
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=config.SAMPLING_RATE)
    parser.add_argument("--weights", type=Path, default=config.FINAL_WEIGHTS_PATH)
    parser.add_argument("--mahalanobis-stats", type=Path, default=config.MAHALANOBIS_STATS_PATH)
    parser.add_argument("--save-explainability", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config.validate_config()
    config.make_output_dirs()

    if args.record is None and args.ecg_id is None:
        raise ValueError("Provide either --record or --ecg-id.")

    record_path = args.record or record_path_from_ecg_id(args.ecg_id, args.data_dir, args.sampling_rate)
    signal = load_and_preprocess_record(
        record_path,
        sampling_rate=args.sampling_rate,
        selected_leads=config.SELECTED_LEADS,
    )
    batch = signal[None, ...]

    model = build_model(
        input_shape=(config.signal_length(args.sampling_rate), len(config.SELECTED_LEADS))
    )
    model.load_weights(args.weights)
    embedding_model = build_embedding_model(model)

    outputs = model.predict(batch, verbose=0)
    embedding = embedding_model.predict(batch, verbose=0)
    class_output = get_output(outputs, "classification")
    noise_output = get_output(outputs, "noise")
    reconstruction_output = get_output(outputs, "reconstruction")
    reconstruction_error = float(np.mean(np.square(batch - reconstruction_output)))

    if Path(args.mahalanobis_stats).exists():
        mahalanobis = float(mahalanobis_distance(embedding, args.mahalanobis_stats)[0])
    else:
        mahalanobis = None

    class_probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(config.TARGET_CLASSES, class_output[0])
    }
    prediction = {
        "record": str(record_path),
        "sampling_rate": args.sampling_rate,
        "selected_leads": config.SELECTED_LEADS,
        "predicted_class": config.TARGET_CLASSES[int(np.argmax(class_output[0]))],
        "class_probabilities": class_probabilities,
        "signal_quality_score": float(noise_output[0, 0]),
        "noise_score": float(noise_output[0, 1]),
        "reconstruction_error": reconstruction_error,
        "mahalanobis_distance": mahalanobis,
        "zero_shot_omi_risk_note": (
            "PTB-XL has no OMI label here; use the MI/ST-T probabilities together with "
            "reconstruction error and Mahalanobis distance as zero-shot anomaly evidence."
        ),
    }

    print(json.dumps(prediction, indent=2))

    if args.save_explainability:
        save_explainability_outputs(model, batch, args.sampling_rate)


def record_path_from_ecg_id(ecg_id: int, data_dir: Path, sampling_rate: int) -> Path:
    metadata = load_ptbxl_metadata(data_dir, sampling_rate=sampling_rate)
    if ecg_id not in metadata.index:
        raise ValueError(f"ecg_id={ecg_id} not found in PTB-XL metadata or has no target label.")
    return Path(data_dir) / metadata.loc[ecg_id, "filename"]


def save_explainability_outputs(model, batch: np.ndarray, sampling_rate: int) -> None:
    output_dir = config.EXPLAINABILITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saliency = compute_saliency(model, batch)[0]
    np.save(output_dir / "saliency.npy", saliency)
    save_saliency_plot(
        saliency,
        output_dir / "saliency.png",
        lead_names=config.SELECTED_LEADS,
        sampling_rate=sampling_rate,
    )

    attention_maps = extract_attention_maps(model, batch)
    for layer_name, scores in attention_maps.items():
        np.save(output_dir / f"{layer_name}_attention.npy", scores)
        save_attention_plot(scores, output_dir / f"{layer_name}_attention.png")

    print(f"Saved explainability outputs to: {output_dir}")


if __name__ == "__main__":
    main()
