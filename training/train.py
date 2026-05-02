"""
training/train.py
─────────────────────────────────────────────────────────────────────────────
Two-stage training pipeline:
  Stage 1 – Feature Extraction   (backbone frozen,  LR = 1e-3, 20 epochs)
  Stage 2 – Fine-Tuning          (top-30 unfrozen,  LR = 1e-5, 30 epochs)

Usage:
  python training/train.py [--epochs_stage1 20] [--epochs_stage2 30]
                           [--batch_size 32]     [--skip_stage2]
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf

from models.model_builder import build_model, unfreeze_top_layers
from training.data_pipeline import build_dataset, get_num_classes, get_class_names

# ── Reproducibility ────────────────────────────────────────────────────────
tf.random.set_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR   = ROOT / "models" / "saved"
LOGS_DIR     = ROOT / "models" / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════
def build_callbacks(stage: int, run_id: str) -> list:
    ckpt_path = str(MODELS_DIR / f"best_stage{stage}_{run_id}.keras")

    callbacks = [
        # Save best model by val_accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Stop early when val_accuracy plateaus
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7 if stage == 1 else 10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR on plateau (useful in stage 2)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        # TensorBoard logs
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / f"stage{stage}_{run_id}"),
            histogram_freq=1,
        ),
        # CSV history log
        tf.keras.callbacks.CSVLogger(
            filename=str(LOGS_DIR / f"history_stage{stage}_{run_id}.csv"),
            append=False,
        ),
    ]
    return callbacks, ckpt_path


# ══════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════
def train(epochs_stage1: int = 20,
          epochs_stage2: int = 30,
          batch_size:    int = 32,
          skip_stage2:   bool = False) -> None:

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'═'*60}")
    print(f"  Hand Gesture Recognition – Training Run {run_id}")
    print(f"{'═'*60}\n")

    # ── GPU / CPU info ─────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[Device] {'GPU: ' + str(gpus) if gpus else 'CPU (no GPU detected)'}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # ── Data ───────────────────────────────────────────────────────────
    print("\n[Data] Building tf.data pipelines …")
    ds_train, class_to_idx = build_dataset("train", batch_size, augment=True,  shuffle=True)
    ds_val,   _            = build_dataset("val",   batch_size, augment=False, shuffle=False)

    num_classes  = get_num_classes()
    class_names  = get_class_names()
    print(f"       Classes ({num_classes}): {class_names}")

    # ── Class weights (handle imbalance) ───────────────────────────────
    from collections import Counter
    label_counts: Counter = Counter()
    for _, lbls in ds_train.unbatch():
        label_counts[int(lbls.numpy())] += 1
    total = sum(label_counts.values())
    class_weight = {
        k: total / (num_classes * v)
        for k, v in label_counts.items()
    }
    print(f"[Data] Class weights: {class_weight}")

    # ── Build model ────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes)
    model.summary(line_length=80, print_fn=lambda x: print("  " + x))

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1 – Feature Extraction
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  STAGE 1 – Feature Extraction  ({epochs_stage1} epochs max)")
    print(f"{'─'*60}")

    cbs_s1, ckpt_s1 = build_callbacks(stage=1, run_id=run_id)
    t0 = time.time()
    history_s1 = model.fit(
        ds_train,
        epochs=epochs_stage1,
        validation_data=ds_val,
        callbacks=cbs_s1,
        class_weight=class_weight,
        verbose=1,
    )
    elapsed_s1 = time.time() - t0

    best_val_acc_s1 = max(history_s1.history.get("val_accuracy", [0]))
    print(f"\n[Stage 1] Best val_accuracy : {best_val_acc_s1:.4f}")
    print(f"[Stage 1] Time elapsed      : {elapsed_s1/60:.1f} min")

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 2 – Fine-Tuning
    # ══════════════════════════════════════════════════════════════════
    if not skip_stage2:
        print(f"\n{'─'*60}")
        print(f"  STAGE 2 – Fine-Tuning  ({epochs_stage2} epochs max)")
        print(f"{'─'*60}")

        model = unfreeze_top_layers(model, num_layers_to_unfreeze=30, learning_rate=1e-5)
        cbs_s2, ckpt_s2 = build_callbacks(stage=2, run_id=run_id)

        t0 = time.time()
        history_s2 = model.fit(
            ds_train,
            epochs=epochs_stage2,
            validation_data=ds_val,
            callbacks=cbs_s2,
            class_weight=class_weight,
            verbose=1,
        )
        elapsed_s2 = time.time() - t0

        best_val_acc_s2 = max(history_s2.history.get("val_accuracy", [0]))
        print(f"\n[Stage 2] Best val_accuracy : {best_val_acc_s2:.4f}")
        print(f"[Stage 2] Time elapsed      : {elapsed_s2/60:.1f} min")

        # Reload best Stage-2 weights
        model.load_weights(ckpt_s2)
        final_ckpt = ckpt_s2
    else:
        model.load_weights(ckpt_s1)
        final_ckpt = ckpt_s1

    # ── Save final model ───────────────────────────────────────────────
    final_path = MODELS_DIR / f"gesture_model_final_{run_id}.keras"
    model.save(str(final_path))
    print(f"\n[✔] Final model saved → {final_path}")

    # ── Save metadata ──────────────────────────────────────────────────
    meta = {
        "run_id"          : run_id,
        "num_classes"     : num_classes,
        "class_names"     : class_names,
        "class_to_idx"    : class_to_idx,
        "img_size"        : [224, 224],
        "batch_size"      : batch_size,
        "best_stage1_acc" : best_val_acc_s1,
        "model_path"      : str(final_path),
        "checkpoint_path" : str(final_ckpt),
    }
    meta_path = MODELS_DIR / f"meta_{run_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[✔] Metadata saved    → {meta_path}\n")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hand Gesture Classifier")
    parser.add_argument("--epochs_stage1", type=int, default=20,
                        help="Epochs for feature-extraction stage")
    parser.add_argument("--epochs_stage2", type=int, default=30,
                        help="Epochs for fine-tuning stage")
    parser.add_argument("--batch_size",    type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("--skip_stage2",   action="store_true",
                        help="Skip fine-tuning (faster, lower accuracy)")
    args = parser.parse_args()

    train(
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        skip_stage2=args.skip_stage2,
    )
