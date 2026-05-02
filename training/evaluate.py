"""
training/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive model evaluation on the held-out test set.

Outputs:
  • Accuracy, precision, recall, F1 (per-class + macro/weighted)
  • Confusion matrix (saved as PNG)
  • Training curves (loss & accuracy, saved as PNG)
  • Misclassified samples grid
─────────────────────────────────────────────────────────────────────────────

Usage:
  python training/evaluate.py --model models/saved/gesture_model_final_<run_id>.keras
                               [--log_csv models/logs/history_stage1_<id>.csv]
                               [--log_csv2 models/logs/history_stage2_<id>.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")                         # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import tensorflow as tf

from models.model_builder import load_model
from training.data_pipeline import build_dataset, get_class_names

RESULTS_DIR = ROOT / "models" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  Prediction helpers
# ══════════════════════════════════════════════════════════════════════════
def predict_all(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Run inference over the entire dataset, return (y_true, y_pred, y_prob)."""
    y_true_list, y_prob_list = [], []
    for imgs, labels in dataset:
        probs = model(imgs, training=False).numpy()
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, class_names: list[str],
                          save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Confusion Matrix – Test Set", fontsize=16, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm,       cm_norm],
        ["d",      ".2f"],
        ["Counts", "Normalised"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✔] Confusion matrix saved → {save_path}")


def plot_training_curves(csv_paths: list[str], save_path: Path) -> None:
    """Plot loss and accuracy curves from CSVLogger output(s)."""
    import pandas as pd

    dfs = []
    for p in csv_paths:
        if p and Path(p).exists():
            df = pd.read_csv(p)
            dfs.append(df)

    if not dfs:
        print("[WARN] No CSV log files found – skipping training curves.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined["epoch_global"] = range(len(combined))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    ax_loss.plot(combined["epoch_global"], combined["loss"],
                 label="Train loss", color="#E63946", linewidth=2)
    ax_loss.plot(combined["epoch_global"], combined["val_loss"],
                 label="Val loss",   color="#457B9D", linewidth=2, linestyle="--")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    # Accuracy
    ax_acc.plot(combined["epoch_global"], combined["accuracy"],
                label="Train acc", color="#2A9D8F", linewidth=2)
    ax_acc.plot(combined["epoch_global"], combined["val_accuracy"],
                label="Val acc",   color="#E9C46A", linewidth=2, linestyle="--")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(); ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✔] Training curves saved → {save_path}")


def plot_misclassified(dataset: tf.data.Dataset,
                       model:   tf.keras.Model,
                       class_names: list[str],
                       save_path:   Path,
                       max_samples: int = 16) -> None:
    """Save a grid of misclassified test images."""
    wrong_imgs, wrong_true, wrong_pred = [], [], []

    for imgs, labels in dataset.unbatch().batch(1):
        prob  = model(imgs, training=False).numpy()[0]
        pred  = int(np.argmax(prob))
        true  = int(labels.numpy()[0])
        if pred != true and len(wrong_imgs) < max_samples:
            # Denormalise MobileNetV2 input: [-1,1] → [0,1]
            img_np = (imgs.numpy()[0] + 1.0) / 2.0
            img_np = np.clip(img_np, 0, 1)
            wrong_imgs.append(img_np)
            wrong_true.append(class_names[true])
            wrong_pred.append(class_names[pred])

    n = len(wrong_imgs)
    if n == 0:
        print("[INFO] No misclassified samples found – perfect test set!")
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Misclassified Samples", fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for i in range(n):
        axes[i].imshow(wrong_imgs[i])
        axes[i].set_title(
            f"True: {wrong_true[i]}\nPred: {wrong_pred[i]}",
            fontsize=8, color="#E63946",
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✔] Misclassified grid saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model_path: str,
             log_csv:  str = "",
             log_csv2: str = "") -> dict:

    print(f"\n{'═'*60}")
    print("  Hand Gesture Recognition – Model Evaluation")
    print(f"{'═'*60}\n")

    # ── Load model & data ──────────────────────────────────────────────
    print(f"[Load] Model: {model_path}")
    model = load_model(model_path)
    class_names = get_class_names()
    print(f"[Data] Classes: {class_names}\n")

    ds_test, _ = build_dataset("test", batch_size=32, augment=False, shuffle=False)

    # ── Predictions ────────────────────────────────────────────────────
    print("[Eval] Running inference on test set …")
    y_true, y_pred, y_prob = predict_all(model, ds_test)

    # ── Metrics ────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[Result] Test Accuracy : {acc * 100:.2f} %\n")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # ── Plots ──────────────────────────────────────────────────────────
    ts = Path(model_path).stem
    plot_confusion_matrix(y_true, y_pred, class_names,
                          RESULTS_DIR / f"confusion_matrix_{ts}.png")
    plot_training_curves([log_csv, log_csv2],
                         RESULTS_DIR / f"training_curves_{ts}.png")
    plot_misclassified(ds_test, model, class_names,
                       RESULTS_DIR / f"misclassified_{ts}.png")

    print(f"\n[✔] All results saved to {RESULTS_DIR}\n")
    return {"accuracy": acc, "report": report}


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gesture model")
    parser.add_argument("--model",    required=True,
                        help="Path to saved .keras model")
    parser.add_argument("--log_csv",  default="",
                        help="Stage-1 training CSV log")
    parser.add_argument("--log_csv2", default="",
                        help="Stage-2 training CSV log")
    args = parser.parse_args()

    evaluate(args.model, args.log_csv, args.log_csv2)
