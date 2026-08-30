"""
Data loading for the IQ-OTH/NCCD lung CT dataset.

Handles:
  - Locating class folders under data/raw regardless of nesting depth
    (the Kaggle zip double-nests: archive/.../<dataset>/<dataset>/<class> cases/)
  - Normalizing the dataset's own inconsistent naming ("Bengin" -> "benign")
  - Building tf.data.Dataset pipelines for train/val/test.

KNOWN LIMITATION — patient-level leakage:
This dataset's filenames (e.g. "Bengin case (7).jpg") use a single
incrementing counter per class, NOT a repeated patient/case ID — confirmed
by inspecting the actual files. There is no way to recover which slices
belong to the same patient from what's distributed. This means our
train/val/test split is done at the IMAGE level, so slices from the same
patient can land in different splits, which can inflate reported accuracy
somewhat. This matches what the original paper and other public
implementations of this dataset do — it's a documented limitation of the
dataset as distributed, not a shortcut we're taking. Worth stating plainly
if this project's numbers are ever scrutinized.

NOTE: images are returned as float32 in [0, 255], NOT normalized here.
Backbone-specific preprocessing (ResNet50 vs EfficientNetB3 each expect
different input scaling) is applied inside the model itself (see model.py),
so this loader stays backbone-agnostic.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import config

# The dataset's own folder names -> our normalized class names.
# "Bengin" is the dataset's actual (misspelled) folder name, not a typo here.
_CLASS_DIR_ALIASES = {
    "bengin": "benign",
    "benign": "benign",
    "malignant": "malignant",
    "normal": "normal",
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    path: Path
    label: int  # index into config.class_names


def _normalize_class_name(dir_name: str) -> str | None:
    lowered = dir_name.lower()
    for alias, normalized in _CLASS_DIR_ALIASES.items():
        if alias in lowered:
            return normalized
    return None


def scan_dataset(raw_dir: Path | None = None) -> list[Sample]:
    """
    Recursively walk raw_dir, find class folders (Bengin/Malignant/Normal
    cases, at any nesting depth), and collect every image inside them.
    """
    raw_dir = raw_dir or config.raw_data_dir
    samples: list[Sample] = []

    class_dirs_found = []
    for path in raw_dir.rglob("*"):
        if path.is_dir():
            normalized = _normalize_class_name(path.name)
            if normalized is not None:
                class_dirs_found.append((path, normalized))

    if not class_dirs_found:
        raise FileNotFoundError(
            f"No class folders (Bengin/Malignant/Normal) found under {raw_dir}. "
            f"Did you extract the dataset zip into data/raw/?"
        )

    for class_dir, normalized_class in class_dirs_found:
        label_idx = config.class_names.index(normalized_class)
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]
        for img_path in images:
            samples.append(Sample(path=img_path, label=label_idx))

    print(f"Found {len(samples)} images across {len(class_dirs_found)} class folder(s)")
    for class_name in config.class_names:
        count = sum(1 for s in samples if s.label == config.class_names.index(class_name))
        print(f"  {class_name}: {count} images")

    return samples


def _split_stratified(samples: list[Sample], val_split: float, test_split: float, seed: int):
    """
    Stratified split at the IMAGE level (see module docstring for why —
    this dataset's filenames don't encode patient identity, so case-level
    grouping isn't possible with the data as distributed).
    """
    labels = [s.label for s in samples]

    train, temp, train_labels, temp_labels = train_test_split(
        samples, labels,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=seed,
    )
    relative_test_size = test_split / (val_split + test_split)
    val, test = train_test_split(
        temp,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=seed,
    )

    print(f"Stratified image-level split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def _load_image(path: str, label: int):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.input_image_size)
    image = tf.cast(image, tf.float32)  # stays in [0, 255]; backbones preprocess internally
    return image, label


def _build_tf_dataset(samples: list[Sample], batch_size: int, shuffle: bool) -> tf.data.Dataset:
    paths = [str(s.path) for s in samples]
    labels = [s.label for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples), seed=config.random_seed)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(samples: list[Sample]) -> dict[int, float]:
    """
    Inverse-frequency class weights, to pass to model.fit(class_weight=...)
    since benign (15 cases) is heavily underrepresented vs normal (55 cases).
    """
    labels = np.array([s.label for s in samples])
    counts = np.bincount(labels, minlength=config.num_classes)
    total = len(labels)
    weights = {
        i: total / (config.num_classes * count) if count > 0 else 0.0
        for i, count in enumerate(counts)
    }
    return weights


def load_datasets(raw_dir: Path | None = None):
    """
    Main entry point. Returns (train_ds, val_ds, test_ds, class_weights).
    """
    samples = scan_dataset(raw_dir)
    train_samples, val_samples, test_samples = _split_stratified(
        samples, config.val_split, config.test_split, config.random_seed
    )
    class_weights = compute_class_weights(train_samples)

    train_ds = _build_tf_dataset(train_samples, config.batch_size, shuffle=True)
    val_ds = _build_tf_dataset(val_samples, config.batch_size, shuffle=False)
    test_ds = _build_tf_dataset(test_samples, config.batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, class_weights


if __name__ == "__main__":
    # Quick manual check: run `python -m src.data_loader` from the repo root
    # after placing the extracted dataset under data/raw/
    train_ds, val_ds, test_ds, weights = load_datasets()
    print("\nClass weights (for handling benign under-representation):", weights)
    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape, "| Label sample:", labels.numpy())