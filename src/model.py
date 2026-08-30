# Phase 1: dual-branch ResNet50 + EfficientNetB3 fusion model definition
"""
Dual-branch feature-fusion model: ResNet50 + EfficientNetB3.

    Input image (300x300x3, raw pixels in [0,255])
            |
      +-----+------------------+
      |                          |
  Resize to 224x224        (used as-is, 300x300)
      |                          |
  preprocess_input          (EfficientNet has its own
  (ResNet50-specific:        built-in Rescaling/Normalization
   BGR + mean subtraction)   layer, so raw [0,255] pixels are
      |                       fed in directly - do NOT call
   ResNet50 backbone          efficientnet.preprocess_input,
  (frozen, ImageNet          it's a no-op anyway and calling
   weights, no top)          it explicitly just adds confusion)
      |                          |
                          EfficientNetB3 backbone
                          (frozen, ImageNet weights, no top)
      |                          |
   GlobalAveragePooling2D    GlobalAveragePooling2D
   (2048-dim vector)         (1536-dim vector)
      |                          |
      +----------+---------------+
             Concatenate (3,584-dim)
                 |
          Dense(128, ReLU)
                 |
            Dropout(0.3)      <- not in the original paper; added for
                 |               regularization since we have relatively
                 |               little data (1,097 images) for two large
                 |               backbones' combined feature space
            Dense(3)
                 |
             Softmax

NOTE ON DEVIATING FROM THE PAPER: Fig. 1 in the paper literally says
"Flatten Layer". We use GlobalAveragePooling2D instead — Flatten on raw
conv feature maps produces a ~254,000-dim fused vector here, which alone
would make the next Dense(128) layer ~32.5M parameters against only ~600
training images: a near-guaranteed overfitting setup. GlobalAveragePooling2D
is the standard transfer-learning choice for exactly this situation and
reduces the fused vector to 3,584 dims. This is a deliberate, documented
deviation from the paper's exact diagram, not an oversight.

Both backbones start frozen (transfer learning, matching the paper).
Phase 3 covers optionally unfreezing top layers for fine-tuning if
accuracy needs a boost after the first training run.
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
        include_top=False,
        weights="imagenet",
        input_shape=(*config.resnet_branch_size, 3),
        pooling=None,
    )
    resnet_base._name = "resnet50_backbone"
    resnet_base.trainable = not config.freeze_backbones
    resnet_features = resnet_base(resnet_input)
    resnet_features = layers.GlobalAveragePooling2D(name="gap_resnet")(resnet_features)

    # --- Branch B: EfficientNetB3 ---
    # No manual preprocessing here on purpose - EfficientNet's Keras
    # implementation includes its own Rescaling/Normalization as the first
    # layers of the model, so it expects raw [0, 255] pixels directly.
    efficientnet_base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*config.input_image_size, 3),
        pooling=None,
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


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",  # labels are plain ints, not one-hot
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    # Quick manual check: run `python -m src.model` from the repo root.
    # Confirms both backbones load their ImageNet weights correctly and
    # the fused model builds/compiles without shape errors, before we
    # spend time wiring up the full training loop.
    model = build_model()
    model = compile_model(model)
    model.summary()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    total_params = sum(tf.size(w).numpy() for w in model.weights)
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} total "
          f"(backbones frozen: {config.freeze_backbones})")