"""
inference/predict.py
─────────────────────────────────────────────────────────────────────────────
Single-image and batch inference utilities.
Loads a trained model and returns gesture class + confidence.

Usage:
  python inference/predict.py --model models/saved/gesture_model_final_*.keras
                               --image path/to/image.jpg
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf

IMG_SIZE = (224, 224)


# ══════════════════════════════════════════════════════════════════════════
#  Inference engine
# ══════════════════════════════════════════════════════════════════════════
class GesturePredictor:
    """
    Wraps a trained Keras model for efficient single / batch inference.

    Parameters
    ──────────
    model_path   : path to saved .keras model
    label_map    : dict {int_index → class_name}.
                   If None, searches ROOT/dataset/label_map.json.
    """

    def __init__(self, model_path: str | Path, label_map: dict | None = None):
        self.model = tf.keras.models.load_model(str(model_path))

        if label_map is None:
            map_file = ROOT / "dataset" / "label_map.json"
            if map_file.exists():
                raw = json.loads(map_file.read_text())
                label_map = {int(k): v for k, v in raw.items()}
            else:
                # Fall back to numeric labels
                num_classes = self.model.output_shape[-1]
                label_map = {i: str(i) for i in range(num_classes)}

        self.label_map = label_map
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy forward pass to initialise GPU kernels."""
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype="float32")
        self.model(dummy, training=False)

    # ── Core preprocessing ─────────────────────────────────────────────
    @staticmethod
    def preprocess_array(img_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR uint8 numpy array (from OpenCV) to model input.
        Returns shape (1, 224, 224, 3) float32 in [-1, 1].
        """
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype("float32")
        # MobileNetV2 preprocessing: [0,255] → [-1,1]
        img_norm = tf.keras.applications.mobilenet_v2.preprocess_input(img_float)
        return np.expand_dims(img_norm, axis=0)

    @staticmethod
    def preprocess_path(image_path: str | Path) -> np.ndarray:
        """
        Preprocess an image file path to model input.
        Returns shape (1, 224, 224, 3) float32 in [-1, 1].
        """
        raw   = tf.io.read_file(str(image_path))
        img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img   = tf.image.resize(img, IMG_SIZE)
        img   = tf.keras.applications.mobilenet_v2.preprocess_input(
                    tf.cast(img, tf.float32))
        return img.numpy()[np.newaxis, ...]

    # ── Prediction ─────────────────────────────────────────────────────
    def predict(self, image: np.ndarray | str | Path,
                top_k: int = 3) -> Tuple[str, float, list[dict]]:
        """
        Parameters
        ──────────
        image  : BGR np.ndarray (OpenCV frame) OR path to image file
        top_k  : number of top predictions to return

        Returns
        ───────
        (top_class, top_confidence, top_k_list)
        top_k_list is a list of {"class": str, "confidence": float} dicts.
        """
        if isinstance(image, (str, Path)):
            inp = self.preprocess_path(image)
        else:
            inp = self.preprocess_array(image)

        probs: np.ndarray = self.model(inp, training=False).numpy()[0]

        top_indices  = np.argsort(probs)[::-1][:top_k]
        top_k_preds  = [
            {"class": self.label_map.get(i, str(i)),
             "confidence": float(probs[i])}
            for i in top_indices
        ]

        best_cls  = top_k_preds[0]["class"]
        best_conf = top_k_preds[0]["confidence"]

        return best_cls, best_conf, top_k_preds

    def predict_batch(self, images: list) -> list[Tuple[str, float]]:
        """
        Batch inference for a list of BGR arrays or file paths.
        Returns list of (class, confidence) tuples.
        """
        batch = np.vstack([
            self.preprocess_path(img) if isinstance(img, (str, Path))
            else self.preprocess_array(img)
            for img in images
        ])
        probs = self.model(batch, training=False).numpy()
        results = []
        for p in probs:
            idx = int(np.argmax(p))
            results.append((self.label_map.get(idx, str(idx)), float(p[idx])))
        return results


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
def _find_latest_model() -> Path | None:
    saved = ROOT / "models" / "saved"
    models = sorted(saved.glob("gesture_model_final_*.keras"), reverse=True)
    return models[0] if models else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gesture prediction on an image")
    parser.add_argument("--model", default=None,
                        help="Path to .keras model (default: latest in models/saved/)")
    parser.add_argument("--image", required=True,
                        help="Path to input image")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    model_path = args.model or _find_latest_model()
    if model_path is None:
        sys.exit("[ERROR] No trained model found. Run training/train.py first.")

    print(f"\n[Model] {model_path}")
    predictor = GesturePredictor(model_path)

    top_class, confidence, top_k = predictor.predict(args.image, top_k=args.top_k)

    print(f"\n[Result] Predicted Gesture : {top_class.upper()}")
    print(f"         Confidence        : {confidence * 100:.1f} %")
    print(f"\n[Top-{args.top_k}]")
    for i, p in enumerate(top_k, 1):
        bar = "█" * int(p["confidence"] * 30)
        print(f"  {i}. {p['class']:<15} {p['confidence']*100:5.1f}%  {bar}")
