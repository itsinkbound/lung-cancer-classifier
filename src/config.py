"""
Central configuration for the lung cancer classification project.
Every tunable value lives here — no magic numbers in model/train/inference code.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = Path(os.environ.get(
        "LUNG_CANCER_RAW_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "raw"),
    ))
    processed_data_dir: Path = project_root / "data" / "processed"
    model_save_dir: Path = Path(os.environ.get(
        "LUNG_CANCER_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent / "models"),
    ))

    # --- Classes ---
    class_names: tuple = ("normal", "benign", "malignant")
    num_classes: int = 3

    # --- Image sizes (dual-branch architecture) ---
    input_image_size: tuple = (300, 300)
    resnet_branch_size: tuple = (224, 224)

    # --- Training ---
    # 32 assumes a real GPU (Kaggle T4 or better). Drop to 16 if you ever
    # run this on CPU again — larger batches take proportionally longer
    # per step with no GPU to parallelize across.
    batch_size: int = 32
    epochs: int = 30
    fine_tune_epochs: int = 15
    learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5
    val_split: float = 0.15
    test_split: float = 0.30
    random_seed: int = 42
    early_stopping_patience: int = 5
    fine_tune_unfreeze_layers: int = 20

    # --- Model ---
    dense_units: int = 128
    freeze_backbones: bool = True


def configure_gpu():
    """
    Kaggle-specific GPU setup. Call once, before building/training the model.

    - Memory growth: TF defaults to grabbing ALL GPU memory upfront on the
      first op. On a shared Kaggle session this is wasteful and occasionally
      causes confusing allocation errors; growth mode allocates incrementally
      as actually needed.
    - Pin to GPU:0 only: Kaggle's "GPU T4 x2" option exposes two GPUs, but
      this model isn't written for distributed training, and letting
      TensorFlow implicitly touch both devices is part of what caused the
      step-fusion shape-mismatch crash. Explicitly restricting to one GPU
      keeps execution simple and avoids that whole bug class.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected — running on CPU.")
        return

    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using single GPU: {gpus[0].name} (memory growth enabled, "
              f"{len(gpus)} total GPU(s) visible on this machine)")
    except RuntimeError as e:
        # Only raised if GPUs were already initialized before this ran —
        # harmless if it happens, just means this was called too late.
        print(f"GPU config warning (non-fatal): {e}")


config = Config()