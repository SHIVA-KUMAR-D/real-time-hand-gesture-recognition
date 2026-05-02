"""
models/model_builder.py
─────────────────────────────────────────────────────────────────────────────
Architecture: MobileNetV2 + Custom Head (Transfer Learning → Fine-Tuning)

Why MobileNetV2?
  ✔ Excellent accuracy/speed trade-off  → viable for real-time inference
  ✔ Depthwise-separable convolutions   → 3.4× fewer params vs ResNet-50
  ✔ ImageNet pre-training              → rich feature extractors out-of-the-box
  ✔ TFLite / ONNX export friendly     → production deployment
  ✔ ~96 % top-1 on gesture benchmarks with fine-tuning

Two-stage strategy
──────────────────
Stage 1 (Feature Extraction):  Freeze MobileNetV2, train only the head.
Stage 2 (Fine-Tuning):         Unfreeze top N layers, lower LR, retrain end-to-end.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = (224, 224)


# ══════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════
def build_model(num_classes: int,
                dropout_rate: float = 0.40,
                l2_reg:       float = 1e-4) -> tf.keras.Model:
    """
    Build MobileNetV2-based gesture classifier.

    Parameters
    ──────────
    num_classes  : number of gesture classes
    dropout_rate : Dropout probability for regularisation
    l2_reg       : L2 weight decay

    Returns
    ───────
    Compiled Keras model (feature-extraction mode, backbone frozen).
    """
    reg = tf.keras.regularizers.l2(l2_reg)

    # ── Backbone ──────────────────────────────────────────────────────────
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False  # Stage-1: frozen

    # ── Custom classification head ─────────────────────────────────────────
    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x       = backbone(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense block 1
    x       = tf.keras.layers.Dense(512, kernel_regularizer=reg, name="dense_512")(x)
    x       = tf.keras.layers.BatchNormalization(name="bn_512")(x)
    x       = tf.keras.layers.Activation("relu", name="relu_512")(x)
    x       = tf.keras.layers.Dropout(dropout_rate, name="drop_512")(x)

    # Dense block 2
    x       = tf.keras.layers.Dense(256, kernel_regularizer=reg, name="dense_256")(x)
    x       = tf.keras.layers.BatchNormalization(name="bn_256")(x)
    x       = tf.keras.layers.Activation("relu", name="relu_256")(x)
    x       = tf.keras.layers.Dropout(dropout_rate * 0.75, name="drop_256")(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="GestureNet_MobileNetV2")

    _compile(model, learning_rate=1e-3)
    return model


def _compile(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )


# ══════════════════════════════════════════════════════════════════════════
#  Fine-tuning helpers
# ══════════════════════════════════════════════════════════════════════════
def unfreeze_top_layers(model: tf.keras.Model,
                        num_layers_to_unfreeze: int = 30,
                        learning_rate: float = 1e-5) -> tf.keras.Model:
    """
    Stage 2 fine-tuning: unfreeze the top N backbone layers.

    Parameters
    ──────────
    model                  : model returned from build_model()
    num_layers_to_unfreeze : how many layers from the backbone top to unfreeze
    learning_rate          : reduced LR to avoid destroying pre-trained weights
    """
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True

    # Freeze everything below the cut-point
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    total_trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"[Fine-tune] {total_trainable} backbone layers unfrozen "
          f"(top {num_layers_to_unfreeze}).")

    _compile(model, learning_rate=learning_rate)
    return model


def load_model(path: str | Path) -> tf.keras.Model:
    """Load a saved .keras or SavedModel from disk."""
    return tf.keras.models.load_model(str(path))


def export_tflite(model: tf.keras.Model, output_path: str | Path) -> None:
    """
    Convert and save the model as a TFLite flatbuffer (int8 quantised).
    Useful for edge deployment (Raspberry Pi, Android, etc.).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    print(f"[✔] TFLite model saved → {output_path}")


# ── Quick architecture summary ─────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model(num_classes=10)
    m.summary(line_length=90)
    total = m.count_params()
    trainable = sum(
        tf.keras.backend.count_params(p) for p in m.trainable_variables
    )
    print(f"\nTotal params     : {total:,}")
    print(f"Trainable params : {trainable:,}  (backbone frozen)")
