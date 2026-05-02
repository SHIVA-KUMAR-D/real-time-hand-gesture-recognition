"""
dataset/download_dataset.py
─────────────────────────────────────────────────────────────────────────────
Downloads the LeapGestRecog dataset from Kaggle and organises it for training.

Dataset  : LeapGestRecog  (gti-upm/leapgestrecog)
Classes  : 10 hand gestures
Images   : ~20 000 near-infrared frames (160×120 px)
Source   : https://www.kaggle.com/datasets/gti-upm/leapgestrecog

Why this dataset?
  ✔ 10 distinct, HCI-relevant gesture classes
  ✔ Pre-segmented, consistent background → clean labels
  ✔ ~20 k images – large enough for transfer learning, small enough to iterate fast
  ✔ Grayscale IR images reduce colour-bias and generalise well
  ✔ Widely cited in gesture-recognition literature (benchmark quality)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import zipfile
import shutil
import json
from pathlib import Path

# ── Project root (one level up from this file) ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "dataset" / "raw"
DATA_DIR = ROOT / "dataset" / "processed"

# ── Gesture class mapping (LeapGestRecog folder names → friendly labels) ──
GESTURE_MAP = {
    "01_palm"      : "palm",
    "02_l"         : "l",
    "03_fist"      : "fist",
    "04_fist_moved": "fist_moved",
    "05_thumb"     : "thumb",
    "06_index"     : "index",
    "07_ok"        : "ok",
    "08_palm_moved": "palm_moved",
    "09_c"         : "c",
    "10_down"      : "down",
}


def check_kaggle_credentials() -> bool:
    """Verify kaggle.json exists and is readable."""
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        print("\n[ERROR] kaggle.json not found at ~/.kaggle/kaggle.json")
        print("  1. Go to https://www.kaggle.com → Account → 'Create New API Token'")
        print("  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  3. Run:  chmod 600 ~/.kaggle/kaggle.json")
        return False
    try:
        creds = json.loads(kaggle_cfg.read_text())
        if "username" not in creds or "key" not in creds:
            raise ValueError("Malformed kaggle.json")
        print(f"[✔] Kaggle credentials found for user: {creds['username']}")
        return True
    except Exception as exc:
        print(f"[ERROR] Could not read kaggle.json: {exc}")
        return False


def download_dataset() -> None:
    """Download and extract the LeapGestRecog dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Downloading LeapGestRecog from Kaggle …")
    exit_code = os.system(
        f"kaggle datasets download -d gti-upm/leapgestrecog --path {RAW_DIR} --unzip"
    )
    if exit_code != 0:
        sys.exit("[ERROR] Kaggle download failed. Check credentials and internet access.")
    print("[✔] Download complete.")


def organise_dataset(train_ratio: float = 0.70,
                     val_ratio:   float = 0.15) -> None:
    """
    Re-organise raw files into:
        dataset/processed/
            train/<class>/
            val/<class>/
            test/<class>/
    """
    import random, glob
    random.seed(42)

    # Locate gesture folders – they live inside leapGestRecog/leapGestRecog/
    raw_root = RAW_DIR / "leapGestRecog" / "leapGestRecog"
    if not raw_root.exists():
        # Fallback: search anywhere under RAW_DIR
        candidates = list(RAW_DIR.rglob("01_palm"))
        if not candidates:
            sys.exit("[ERROR] Raw dataset structure not found. Re-run download.")
        raw_root = candidates[0].parent

    print(f"\n[2/3] Organising dataset from: {raw_root}")

    for split in ("train", "val", "test"):
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    for folder_name, label in GESTURE_MAP.items():
        gesture_dirs = sorted(raw_root.glob(f"**/{folder_name}"))
        all_images: list[Path] = []
        for gd in gesture_dirs:
            all_images.extend(gd.glob("*.png"))
            all_images.extend(gd.glob("*.jpg"))
            all_images.extend(gd.glob("*.jpeg"))

        if not all_images:
            print(f"  [WARN] No images found for '{folder_name}' – skipping.")
            continue

        random.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": all_images[:n_train],
            "val"  : all_images[n_train : n_train + n_val],
            "test" : all_images[n_train + n_val:],
        }

        for split, imgs in splits.items():
            dest = DATA_DIR / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, dest / img.name)

        stats[label] = {s: len(imgs) for s, imgs in splits.items()}

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[3/3] Dataset split summary:")
    print(f"  {'Class':<15} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("  " + "─" * 35)
    totals = {"train": 0, "val": 0, "test": 0}
    for label, counts in stats.items():
        print(f"  {label:<15} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6}")
        for s in totals:
            totals[s] += counts.get(s, 0)
    print("  " + "─" * 35)
    print(f"  {'TOTAL':<15} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6}")

    # Save label map for inference
    label_map = {i: lbl for i, lbl in enumerate(sorted(GESTURE_MAP.values()))}
    label_map_path = ROOT / "dataset" / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2))
    print(f"\n[✔] Label map saved → {label_map_path}")
    print(f"[✔] Processed dataset ready → {DATA_DIR}\n")


if __name__ == "__main__":
    if not check_kaggle_credentials():
        sys.exit(1)
    download_dataset()
    organise_dataset()
