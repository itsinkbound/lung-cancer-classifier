"""
Dual-branch feature-fusion model: ResNet50 + EfficientNetB3.
See README/train.py docstring for the architecture diagram and design notes.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB3
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from src.config import config


def build_model() -> tf.keras.Model:
    inputs = layers.Input(shape=(*config.input_image_size, 3), name="ct_scan_input")

    # --- Branch A: ResNet50 ---
    resnet_input = layers.Resizing(*config.resnet_branch_size, name="resize_for_resnet")(inputs)
    resnet_input = layers.Lambda(resnet_preprocess, name="resnet_preprocess")(resnet_input)
    resnet_base = ResNet50(
        include_top=False, weights="imagenet",
        input_shape=(*config.resnet_branch_size, 3), pooling=None,
    )
    resnet_base._name = "resnet50_backbone"
    resnet_base.trainable = not config.freeze_backbones
    resnet_features = resnet_base(resnet_input)
    resnet_features = layers.GlobalAveragePooling2D(name="gap_resnet")(resnet_features)

    # --- Branch B: EfficientNetB3 ---
    # No manual preprocessing — EfficientNet's Keras implementation includes
    # its own Rescaling/Normalization as its first layers, expects raw [0,255].
    efficientnet_base = EfficientNetB3(
        include_top=False, weights="imagenet",
        input_shape=(*config.input_image_size, 3), pooling=None,
    )
    efficientnet_base._name = "efficientnetb3_backbone"
    efficientnet_base.trainable = not config.freeze_backbones
    efficientnet_features = efficientnet_base(inputs)
    efficientnet_features = layers.GlobalAveragePooling2D(name="gap_efficientnet")(efficientnet_features)

    # --- Fusion ---
    fused = layers.Concatenate(name="fuse_features")([resnet_features, efficientnet_features])
    x = layers.Dense(config.dense_units, activation="relu", name="dense_128")(fused)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(config.num_classes, activation="softmax", name="classification_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_efficientnetb3_fusion")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float | None = None) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate or config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
        # Disables XLA step-fusion. This is the direct fix for the
        # "tf2xla conversion failed" / shape-mismatch crash on GPU: fused
        # multi-step execution requires every batch to be an identical
        # shape, which breaks on the dataset's final (smaller) batch.
        # Costs a small amount of speed; correctness > that speed here.
        jit_compile=False,
    )
    return model


if __name__ == "__main__":
    from src.config import configure_gpu
    configure_gpu()
    model = build_model()
    model = compile_model(model)
    model.summary()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    total = sum(tf.size(w).numpy() for w in model.weights)
    print(f"\nTrainable params: {trainable:,} / {total:,} total "
          f"(backbones frozen: {config.freeze_backbones})")