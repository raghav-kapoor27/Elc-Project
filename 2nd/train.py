import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.covariance import LedoitWolf

import config
from model import build_embedding_model, build_model, compile_model
from ptbxl_data import PTBXLSequence, load_ptbxl_metadata, split_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PTB-XL CNN+Transformer research model.")
    parser.add_argument("--data-dir", type=Path, default=config.PTBXL_ROOT)
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=config.SAMPLING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--classifier-epochs", type=int, default=config.CLASSIFIER_EPOCHS)
    parser.add_argument("--anomaly-epochs", type=int, default=config.ANOMALY_EPOCHS)
    parser.add_argument("--no-augmentation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config.validate_config()
    config.make_output_dirs()
    set_reproducibility(config.RANDOM_SEED)
    configure_precision()

    metadata = load_ptbxl_metadata(args.data_dir, sampling_rate=args.sampling_rate)
    train_meta, val_meta, test_meta = split_metadata(metadata)
    print_dataset_summary(train_meta, val_meta, test_meta, args.sampling_rate)

    train_seq = PTBXLSequence(
        train_meta,
        root=args.data_dir,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        augment=not args.no_augmentation,
        shuffle=True,
    )
    val_seq = PTBXLSequence(
        val_meta,
        root=args.data_dir,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
    )

    input_shape = (config.signal_length(args.sampling_rate), len(config.SELECTED_LEADS))
    model = build_model(input_shape=input_shape)

    # Phase 1: make the CNN+Transformer classifier work first.
    compile_model(
        model,
        learning_rate=config.LEARNING_RATE,
        loss_weights=config.CLASSIFIER_PHASE_LOSS_WEIGHTS,
    )
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.classifier_epochs,
        callbacks=callbacks(
            checkpoint_path=config.CLASSIFIER_WEIGHTS_PATH,
            monitor="val_classification_accuracy",
            mode="max",
            phase_name="classifier",
        ),
    )

    if config.CLASSIFIER_WEIGHTS_PATH.exists():
        model.load_weights(config.CLASSIFIER_WEIGHTS_PATH)

    # Phase 2: add the anomaly machinery through denoising reconstruction.
    if args.anomaly_epochs > 0:
        compile_model(
            model,
            learning_rate=config.ANOMALY_LEARNING_RATE,
            loss_weights=config.MULTITASK_PHASE_LOSS_WEIGHTS,
        )
        model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=args.anomaly_epochs,
            callbacks=callbacks(
                checkpoint_path=config.FINAL_WEIGHTS_PATH,
                monitor="val_loss",
                mode="min",
                phase_name="multitask_anomaly",
            ),
        )
        if config.FINAL_WEIGHTS_PATH.exists():
            model.load_weights(config.FINAL_WEIGHTS_PATH)
    else:
        model.save_weights(config.FINAL_WEIGHTS_PATH)

    fit_and_save_mahalanobis(model, train_meta, args.data_dir, args.sampling_rate, args.batch_size)
    save_run_summary(args, train_meta, val_meta, test_meta)
    print(f"Saved final weights to: {config.FINAL_WEIGHTS_PATH}")
    print(f"Saved Mahalanobis statistics to: {config.MAHALANOBIS_STATS_PATH}")


def callbacks(checkpoint_path: Path, monitor: str, mode: str, phase_name: str):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.LOG_DIR / f"{phase_name}_{timestamp}"),
            histogram_freq=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def fit_and_save_mahalanobis(
    model: tf.keras.Model,
    train_meta,
    data_dir: Path,
    sampling_rate: int,
    batch_size: int,
) -> None:
    normal_meta = train_meta[train_meta["target_id"] == config.SUPERCLASS_TO_TARGET["NORM"]].copy()
    if normal_meta.empty:
        normal_meta = train_meta.copy()

    seq = PTBXLSequence(
        normal_meta,
        root=data_dir,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )
    embedding_model = build_embedding_model(model)
    embeddings = []
    for batch_x, _ in seq:
        embeddings.append(embedding_model.predict(batch_x, verbose=0))

    embeddings = np.concatenate(embeddings, axis=0)
    covariance = LedoitWolf().fit(embeddings)
    np.savez(
        config.MAHALANOBIS_STATS_PATH,
        mean=covariance.location_.astype(np.float32),
        precision=covariance.precision_.astype(np.float32),
        reference_class="Normal",
        sampling_rate=sampling_rate,
        selected_leads=np.array(config.SELECTED_LEADS),
    )


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_precision() -> None:
    if config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")


def print_dataset_summary(train_meta, val_meta, test_meta, sampling_rate: int) -> None:
    print("PTB-XL training setup")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Selected leads ({len(config.SELECTED_LEADS)}): {config.SELECTED_LEADS}")
    print(f"Input shape: ({config.signal_length(sampling_rate)}, {len(config.SELECTED_LEADS)})")
    print(f"Train/val/test records: {len(train_meta)}/{len(val_meta)}/{len(test_meta)}")
    print("Class counts:")
    for class_idx, class_name in enumerate(config.TARGET_CLASSES):
        count = int((train_meta["target_id"] == class_idx).sum())
        print(f"  {class_name}: {count}")


def save_run_summary(args, train_meta, val_meta, test_meta) -> None:
    summary = {
        "sampling_rate": args.sampling_rate,
        "selected_leads": config.SELECTED_LEADS,
        "target_classes": config.TARGET_CLASSES,
        "train_records": int(len(train_meta)),
        "val_records": int(len(val_meta)),
        "test_records": int(len(test_meta)),
        "classifier_epochs": int(args.classifier_epochs),
        "anomaly_epochs": int(args.anomaly_epochs),
    }
    with open(config.OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
