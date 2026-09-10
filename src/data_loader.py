"""
Data loading for the IQ-OTH/NCCD lung CT dataset.
See earlier version's docstring for full context on folder scanning and
the documented image-level-split limitation (dataset filenames don't
encode patient case IDs).
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import config

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
    label: int


def _normalize_class_name(dir_name: str) -> str | None:
    lowered = dir_name.lower()
    for alias, normalized in _CLASS_DIR_ALIASES.items():
        if alias in lowered:
            return normalized
    return None


def scan_dataset(raw_dir: Path | None = None) -> list[Sample]:
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
            f"Did you extract/attach the dataset correctly?"
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
    labels = [s.label for s in samples]
    train, temp, train_labels, temp_labels = train_test_split(
        samples, labels, test_size=(val_split + test_split), stratify=labels, random_state=seed,
    )
    relative_test_size = test_split / (val_split + test_split)
    val, test = train_test_split(
        temp, test_size=relative_test_size, stratify=temp_labels, random_state=seed,
    )
    print(f"Stratified image-level split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def _load_image(path: str, label: int):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.input_image_size)
    image = tf.cast(image, tf.float32)
    return image, label


def _build_tf_dataset(samples: list[Sample], batch_size: int, shuffle: bool,
                       drop_remainder: bool) -> tf.data.Dataset:
    """
    drop_remainder=True discards the final, smaller-than-batch_size batch
    of each epoch. Required for the training set on GPU: TensorFlow's XLA
    step-fusion assumes every batch has an identical shape, and the uneven
    last batch is exactly what caused the "Incompatible shapes" crash.
    We keep drop_remainder=False for the test set so every image still
    gets evaluated at least once when reporting final metrics.
    """
    paths = [str(s.path) for s in samples]
    labels = [s.label for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples), seed=config.random_seed)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(samples: list[Sample]) -> dict[int, float]:
    labels = np.array([s.label for s in samples])
    counts = np.bincount(labels, minlength=config.num_classes)
    total = len(labels)
    return {
        i: total / (config.num_classes * count) if count > 0 else 0.0
        for i, count in enumerate(counts)
    }


def load_datasets(raw_dir: Path | None = None):
    samples = scan_dataset(raw_dir)
    train_samples, val_samples, test_samples = _split_stratified(
        samples, config.val_split, config.test_split, config.random_seed
    )
    class_weights = compute_class_weights(train_samples)

    # Train AND val both flow through model.fit's fused-execution path each
    # epoch, so both need drop_remainder=True. Test is evaluated manually
    # batch-by-batch in train.py's evaluate_and_report (not via model.fit),
    # so it's safe to keep every image there.
    train_ds = _build_tf_dataset(train_samples, config.batch_size, shuffle=True, drop_remainder=True)
    val_ds = _build_tf_dataset(val_samples, config.batch_size, shuffle=False, drop_remainder=True)
    test_ds = _build_tf_dataset(test_samples, config.batch_size, shuffle=False, drop_remainder=False)

    return train_ds, val_ds, test_ds, class_weights


if __name__ == "__main__":
    train_ds, val_ds, test_ds, weights = load_datasets()
    print("\nClass weights:", weights)
    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape, "| Label sample:", labels.numpy())