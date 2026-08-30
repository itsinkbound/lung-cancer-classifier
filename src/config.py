"""
Central configuration for the lung cancer classification project.
Every tunable value lives here — no magic numbers in model/train/inference code.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    model_save_dir: Path = project_root / "models"

    # --- Classes ---
    class_names: tuple = ("normal", "benign", "malignant")
    num_classes: int = 3

    # --- Image sizes (dual-branch architecture) ---
    # EfficientNetB3's native resolution — this is the size we read images in at.
    input_image_size: tuple = (300, 300)
    # ResNet50's native resolution — a resized copy is fed to this branch internally.
    resnet_branch_size: tuple = (224, 224)

    # --- Training ---
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-4
    val_split: float = 0.15   # carved out of the training portion
    test_split: float = 0.30  # matches the paper's 70-30 split
    random_seed: int = 42

    # --- Model ---
    dense_units: int = 128
    freeze_backbones: bool = True  # set False in Phase-3+ fine-tuning experiments


config = Config()
