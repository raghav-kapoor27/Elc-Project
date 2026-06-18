from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

import config
from model import get_output


def compute_saliency(
    model: Model,
    batch: np.ndarray,
    class_index: Optional[int] = None,
) -> np.ndarray:
    inputs = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs, training=False)
        probabilities = get_output(outputs, "classification")
        if class_index is None:
            indices = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
        else:
            indices = tf.fill([tf.shape(probabilities)[0]], tf.cast(class_index, tf.int32))
        gather_indices = tf.stack([tf.range(tf.shape(probabilities)[0]), indices], axis=1)
        class_scores = tf.gather_nd(probabilities, gather_indices)

    gradients = tape.gradient(class_scores, inputs)
    return tf.abs(gradients).numpy()


def extract_attention_maps(model: Model, batch: np.ndarray) -> Dict[str, np.ndarray]:
    attention_maps = {}
    for layer in model.layers:
        if not layer.name.startswith("transformer_") or not layer.name.endswith("_mha"):
            continue

        block_name = layer.name.replace("_mha", "")
        norm_layer_name = f"{block_name}_attn_norm"
        feature_model = Model(inputs=model.input, outputs=model.get_layer(norm_layer_name).output)
        transformer_input = feature_model.predict(batch, verbose=0)
        _, scores = layer(
            transformer_input,
            transformer_input,
            return_attention_scores=True,
            training=False,
        )
        attention_maps[layer.name] = scores.numpy()

    return attention_maps


def save_saliency_plot(
    saliency: np.ndarray,
    output_path: Path,
    lead_names=None,
    sampling_rate: int = config.SAMPLING_RATE,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lead_names = list(lead_names or config.SELECTED_LEADS)
    time_axis = np.arange(saliency.shape[0]) / float(sampling_rate)

    plt.figure(figsize=(12, max(3, 0.35 * len(lead_names))))
    plt.imshow(
        saliency.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[time_axis[0], time_axis[-1], 0, len(lead_names)],
    )
    plt.yticks(np.arange(len(lead_names)) + 0.5, lead_names)
    plt.xlabel("Time (s)")
    plt.ylabel("Lead")
    plt.colorbar(label="Absolute gradient")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_attention_plot(attention_scores: np.ndarray, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = attention_scores[0].mean(axis=0)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.xlabel("Key timestep")
    plt.ylabel("Query timestep")
    plt.colorbar(label="Attention")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
