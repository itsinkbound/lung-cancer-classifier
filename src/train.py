"""
Training loop for the ResNet50 + EfficientNetB3 fusion model.
Runs identically locally or on Kaggle — see config.py for path resolution
via environment variables, and configure_gpu() for the GPU setup that
fixes the XLA step-fusion crash seen on Kaggle's T4 GPUs.
"""

import argparse
import json
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import config, configure_gpu
from src.data_loader import load_datasets
from src.model import build_model, compile_model


def get_callbacks(checkpoint_path, patience: int):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(2, patience // 2), min_lr=1e-7, verbose=1,
        ),
    ]


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int):
    for backbone_name in ("resnet50_backbone", "efficientnetb3_backbone"):
        backbone = model.get_layer(backbone_name)
        backbone.trainable = True
        for layer in backbone.layers[:-n_layers]:
            layer.trainable = False
    return model


def evaluate_and_report(model, test_ds, run_dir):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.argmax(axis=1).tolist())

    report = classification_report(
        y_true, y_pred, target_names=list(config.class_names), output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("\n=== Test set evaluation ===")
    print(classification_report(y_true, y_pred, target_names=list(config.class_names)))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    with open(run_dir / "test_report.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm}, f, indent=2)

    return report


def main(run_fine_tuning: bool = True):
    configure_gpu()  # must run before any TF op touches a device

    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.model_save_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    print("\nLoading data...")
    train_ds, val_ds, test_ds, class_weights = load_datasets()
    print(f"Class weights: {class_weights}")

    print("\nBuilding model (Stage 1: frozen backbones)...")
    model = build_model()
    model = compile_model(model, learning_rate=config.learning_rate)

    stage1_checkpoint = run_dir / "stage1_best.keras"
    history_stage1 = model.fit(
        train_ds, validation_data=val_ds, epochs=config.epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(stage1_checkpoint, config.early_stopping_patience),
    )
    with open(run_dir / "history_stage1.json", "w") as f:
        json.dump(history_stage1.history, f, indent=2)

    model = tf.keras.models.load_model(stage1_checkpoint)

    val_losses = history_stage1.history["val_loss"]
    still_improving = len(val_losses) < 3 or val_losses[-1] <= min(val_losses[:-2])

    if run_fine_tuning and still_improving:
        print("\nStage 1 was still improving — proceeding to Stage 2 fine-tuning...")
        model = unfreeze_top_layers(model, config.fine_tune_unfreeze_layers)
        model = compile_model(model, learning_rate=config.fine_tune_learning_rate)

        stage2_checkpoint = run_dir / "stage2_best.keras"
        history_stage2 = model.fit(
            train_ds, validation_data=val_ds, epochs=config.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=get_callbacks(stage2_checkpoint, config.early_stopping_patience),
        )
        with open(run_dir / "history_stage2.json", "w") as f:
            json.dump(history_stage2.history, f, indent=2)
        model = tf.keras.models.load_model(stage2_checkpoint)
    elif run_fine_tuning:
        print("\nStage 1 val_loss was already rising — skipping Stage 2 fine-tuning.")
    else:
        print("\nSkipping Stage 2 (run_fine_tuning=False).")

    final_path = run_dir / "final_model.keras"
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    evaluate_and_report(model, test_ds, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fine-tune", action="store_true")
    args = parser.parse_args()
    main(run_fine_tuning=not args.no_fine_tune)