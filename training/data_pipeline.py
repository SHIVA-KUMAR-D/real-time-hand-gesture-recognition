"""
training/data_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Builds efficient tf.data pipelines with aggressive augmentation for training
and clean pipelines for validation / test / inference.

Architecture rationale
───────────────────────
We use MobileNetV2 (ImageNet pre-trained) as our backbone.
Input size  : 224 × 224 × 3  (MobileNetV2 canonical input)
Augmentation: random flip, rotation, zoom, brightness, contrast.
              This is applied ONLY during training to prevent overfitting.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "dataset" / "processed"
LABEL_MAP = ROOT / "dataset" / "label_map.json"

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)           # MobileNetV2 input
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE


# ══════════════════════════════════════════════════════════════════════════
# Augmentation layers (TF-native, GPU-accelerated)
# ══════════════════════════════════════════════════════════════════════════
def build_augmentation_layer() -> tf.keras.Sequential:
    """Returns a Keras augmentation pipeline applied on-the-fly during training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),          # ±15 °
            tf.keras.layers.RandomZoom((-0.2, 0.2)),       # zoom in/out
            tf.keras.layers.RandomTranslation(0.1, 0.1),   # ±10 % shift
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )


# ══════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════
def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor
                         ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load an image file and apply MobileNetV2 preprocessing."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # → [-1, 1]
    return image, label


def _get_paths_and_labels(split: str) -> Tuple[list[str], list[int]]:
    """Collect all image paths and integer labels for a given split."""
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Run  python dataset/download_dataset.py  first."
        )

    # Derive class→index mapping (sorted alphabetically for reproducibility)
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    paths, labels = [], []
    for cls, idx in class_to_idx.items():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img in (split_dir / cls).glob(ext):
                paths.append(str(img))
                labels.append(idx)

    return paths, labels, class_to_idx


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════
def build_dataset(split: str,
                  batch_size: int = BATCH_SIZE,
                  augment: bool = False,
                  shuffle: bool = True) -> Tuple[tf.data.Dataset, dict]:
    """
    Parameters
    ──────────
    split      : 'train' | 'val' | 'test'
    batch_size : mini-batch size
    augment    : apply data augmentation (train only)
    shuffle    : shuffle dataset (train only)

    Returns
    ───────
    (dataset, class_to_idx)  where dataset yields (images, labels) batches.
    """
    paths, labels, class_to_idx = _get_paths_and_labels(split)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        aug_layer = build_augmentation_layer()
        ds = ds.map(
            lambda img, lbl: (aug_layer(img, training=True), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    return ds, class_to_idx


def get_class_names(split: str = "train") -> list[str]:
    """Return sorted list of class names."""
    split_dir = DATA_DIR / split
    return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])


def get_num_classes(split: str = "train") -> int:
    return len(get_class_names(split))


def get_dataset_info() -> dict:
    """Return a summary dict with per-split counts."""
    info = {}
    for split in ("train", "val", "test"):
        try:
            paths, labels, class_to_idx = _get_paths_and_labels(split)
            info[split] = {
                "total"          : len(paths),
                "num_classes"    : len(class_to_idx),
                "class_to_idx"   : class_to_idx,
            }
        except FileNotFoundError:
            info[split] = "not found"
    return info


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    info = get_dataset_info()
    for split, data in info.items():
        print(f"\n[{split}]")
        if isinstance(data, str):
            print(f"  {data}")
        else:
            print(f"  Total images : {data['total']}")
            print(f"  Classes      : {data['num_classes']}")
            print(f"  Class map    : {data['class_to_idx']}")

    print("\nBuilding train dataset …")
    ds_train, c2i = build_dataset("train", augment=True)
    for batch_imgs, batch_labels in ds_train.take(1):
        print(f"  Batch shape  : {batch_imgs.shape}")
        print(f"  Label dtype  : {batch_labels.dtype}")
        print(f"  Pixel range  : [{batch_imgs.numpy().min():.2f}, {batch_imgs.numpy().max():.2f}]")
