import math
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras import Model, layers, losses, metrics, optimizers

import config


OUTPUT_ORDER = ("classification", "noise", "reconstruction")


class PositionalEmbedding(layers.Layer):
    def __init__(self, max_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(input_dim=max_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_length": self.max_length, "embed_dim": self.embed_dim})
        return cfg


def build_model(
    input_shape: Optional[tuple] = None,
    num_classes: int = len(config.TARGET_CLASSES),
) -> Model:
    if input_shape is None:
        input_shape = (config.signal_length(config.SAMPLING_RATE), len(config.SELECTED_LEADS))

    inputs = layers.Input(shape=input_shape, name="ecg")
    x = inputs

    for idx, filters in enumerate(config.CNN_FILTERS, start=1):
        x = _conv_block(x, filters=filters, block_idx=idx)

    encoded_steps = math.ceil(input_shape[0] / (2 ** len(config.CNN_FILTERS)))
    x = PositionalEmbedding(
        max_length=encoded_steps,
        embed_dim=config.CNN_FILTERS[-1],
        name="positional_embedding",
    )(x)

    for idx in range(1, config.TRANSFORMER_LAYERS + 1):
        x = _transformer_encoder(x, block_idx=idx)

    sequence_embedding = x
    pooled = layers.GlobalAveragePooling1D(name="global_context_pool")(sequence_embedding)
    latent = layers.Dense(config.EMBEDDING_DIM, name="latent_embedding")(pooled)
    latent = layers.LayerNormalization(name="latent_embedding_norm")(latent)
    latent = layers.Dropout(config.DROPOUT, name="latent_dropout")(latent)

    classification = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification",
    )(latent)
    noise = layers.Dense(
        2,
        activation="sigmoid",
        dtype="float32",
        name="noise",
    )(latent)
    reconstruction = _decoder(sequence_embedding, input_length=input_shape[0], num_leads=input_shape[1])

    return Model(
        inputs=inputs,
        outputs={
            "classification": classification,
            "noise": noise,
            "reconstruction": reconstruction,
        },
        name="noise_aware_zero_shot_ptbxl_model",
    )


def compile_model(
    model: Model,
    learning_rate: float,
    loss_weights: Dict[str, float],
) -> Model:
    try:
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.WEIGHT_DECAY)
    except AttributeError:
        optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            "classification": losses.CategoricalCrossentropy(
                label_smoothing=config.LABEL_SMOOTHING
            ),
            "noise": losses.MeanSquaredError(),
            "reconstruction": losses.MeanAbsoluteError(),
        },
        loss_weights=loss_weights,
        metrics={
            "classification": [
                metrics.CategoricalAccuracy(name="accuracy"),
                metrics.AUC(name="auc", multi_label=True, num_labels=len(config.TARGET_CLASSES)),
            ],
            "noise": [metrics.MeanAbsoluteError(name="mae")],
            "reconstruction": [metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model


def build_embedding_model(model: Model) -> Model:
    return Model(
        inputs=model.input,
        outputs=model.get_layer("latent_embedding_norm").output,
        name="embedding_model",
    )


def get_output(outputs, name: str):
    if isinstance(outputs, dict):
        return outputs[name]
    return outputs[OUTPUT_ORDER.index(name)]


def _conv_block(inputs, filters: int, block_idx: int):
    x = layers.Conv1D(
        filters,
        kernel_size=config.CNN_KERNEL_SIZE,
        strides=2,
        padding="same",
        use_bias=False,
        name=f"cnn_{block_idx}_conv",
    )(inputs)
    x = layers.BatchNormalization(name=f"cnn_{block_idx}_bn")(x)
    x = layers.Activation("gelu", name=f"cnn_{block_idx}_gelu")(x)
    x = layers.SpatialDropout1D(config.DROPOUT, name=f"cnn_{block_idx}_dropout")(x)
    return x


def _transformer_encoder(inputs, block_idx: int):
    attn_input = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"transformer_{block_idx}_attn_norm",
    )(inputs)
    attn_output = layers.MultiHeadAttention(
        num_heads=config.TRANSFORMER_HEADS,
        key_dim=config.TRANSFORMER_KEY_DIM,
        dropout=config.DROPOUT,
        name=f"transformer_{block_idx}_mha",
    )(attn_input, attn_input)
    x = layers.Add(name=f"transformer_{block_idx}_attn_add")([inputs, attn_output])

    ffn_input = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"transformer_{block_idx}_ffn_norm",
    )(x)
    ffn = layers.Dense(config.TRANSFORMER_MLP_DIM, activation="gelu", name=f"transformer_{block_idx}_ffn_1")(
        ffn_input
    )
    ffn = layers.Dropout(config.DROPOUT, name=f"transformer_{block_idx}_ffn_dropout")(ffn)
    ffn = layers.Dense(inputs.shape[-1], name=f"transformer_{block_idx}_ffn_2")(ffn)
    return layers.Add(name=f"transformer_{block_idx}_ffn_add")([x, ffn])


def _decoder(sequence_embedding, input_length: int, num_leads: int):
    x = sequence_embedding
    decoder_filters = list(reversed(config.CNN_FILTERS[:-1])) + [64]
    for idx, filters in enumerate(decoder_filters, start=1):
        x = layers.UpSampling1D(size=2, name=f"decoder_{idx}_upsample")(x)
        x = layers.Conv1D(
            filters,
            kernel_size=5,
            padding="same",
            use_bias=False,
            name=f"decoder_{idx}_conv",
        )(x)
        x = layers.BatchNormalization(name=f"decoder_{idx}_bn")(x)
        x = layers.Activation("gelu", name=f"decoder_{idx}_gelu")(x)

    x = layers.Conv1D(
        num_leads,
        kernel_size=7,
        padding="same",
        dtype="float32",
        name="reconstruction_conv",
    )(x)
    return layers.Lambda(lambda tensor: tensor[:, :input_length, :], name="reconstruction")(x)
