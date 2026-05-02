"""
inference/realtime_inference.py
─────────────────────────────────────────────────────────────────────────────
Real-time hand gesture recognition using webcam.

Features:
  • Smooth 30 fps inference via frame skipping (predict every N frames)
  • Rolling confidence average over last K predictions (reduce flicker)
  • ROI (Region of Interest) box for focused hand detection
  • Overlay: gesture name, confidence bar, FPS counter, top-3 predictions
  • Press 'q' to quit, 's' to screenshot, 'r' to reset smoothing buffer

Usage:
  python inference/realtime_inference.py
         [--model models/saved/gesture_model_final_*.keras]
         [--camera 0]
         [--predict_every 2]
         [--smooth 5]
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.predict import GesturePredictor


# ══════════════════════════════════════════════════════════════════════════
#  Colour palette
# ══════════════════════════════════════════════════════════════════════════
PALETTE = {
    "bg"        : (15,  15,  15),
    "primary"   : (0,  200, 100),   # green accent
    "secondary" : (255, 165,  0),   # orange accent
    "white"     : (240, 240, 240),
    "red"       : (80,  60, 220),
    "roi_box"   : (0,  200, 100),
}


# ══════════════════════════════════════════════════════════════════════════
#  UI helpers
# ══════════════════════════════════════════════════════════════════════════
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12):
    x1, y1 = pt1; x2, y2 = pt2
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius),   0, 0, 90,  color, thickness)


def draw_overlay(frame: np.ndarray,
                 gesture: str,
                 confidence: float,
                 top3: list[dict],
                 fps: float,
                 roi: tuple) -> np.ndarray:
    """Render all HUD elements onto the frame."""
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ───────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Gesture name ───────────────────────────────────────────────────
    label = f"{gesture.upper()}"
    cv2.putText(frame, label, (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.6,
                PALETTE["primary"], 2, cv2.LINE_AA)

    # ── Confidence bar ─────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 20, 70, 300, 18
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled = int(bar_w * confidence)
    bar_color = (PALETTE["primary"] if confidence > 0.75
                 else PALETTE["secondary"] if confidence > 0.5
                 else PALETTE["red"])
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (bar_x + bar_w + 8, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, PALETTE["white"], 1, cv2.LINE_AA)

    # ── FPS ────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, PALETTE["white"], 1, cv2.LINE_AA)

    # ── Top-3 sidebar ──────────────────────────────────────────────────
    panel_x = w - 220
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x - 10, 110), (w, 110 + 30*3 + 20),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    for i, pred in enumerate(top3):
        y_off = 130 + i * 30
        cls_short = pred["class"][:12]
        bar_len   = int(190 * pred["confidence"])
        clr       = PALETTE["primary"] if i == 0 else (80, 80, 80)
        cv2.rectangle(frame, (panel_x, y_off - 12),
                      (panel_x + bar_len, y_off + 2), clr, -1)
        cv2.putText(frame, f"{cls_short}: {pred['confidence']*100:.0f}%",
                    (panel_x, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE["white"], 1, cv2.LINE_AA)

    # ── ROI box ────────────────────────────────────────────────────────
    rx, ry, rw, rh = roi
    draw_rounded_rect(frame, (rx, ry), (rx+rw, ry+rh),
                      PALETTE["roi_box"], thickness=2)
    cv2.putText(frame, "Place hand here", (rx + 4, ry - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["roi_box"], 1, cv2.LINE_AA)

    # ── Controls hint ──────────────────────────────────────────────────
    cv2.putText(frame, "Q: Quit  S: Screenshot  R: Reset",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════
def run_realtime(model_path: str | Path,
                 camera_idx:      int   = 0,
                 predict_every:   int   = 2,
                 smooth_window:   int   = 5) -> None:

    print(f"\n[Load] Model: {model_path}")
    predictor = GesturePredictor(model_path)
    print("[✔] Model loaded. Starting webcam …\n")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {camera_idx}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──────────────────────────────────────────────────────────
    frame_idx     = 0
    last_gesture  = "…"
    last_conf     = 0.0
    last_top3     = []
    smooth_buf: collections.deque = collections.deque(maxlen=smooth_window)
    fps_times     = collections.deque(maxlen=30)
    screenshots   = ROOT / "models" / "results" / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)

    print("Controls: Q – quit | S – screenshot | R – reset smoothing buffer\n")

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed.")
            continue

        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)           # mirror

        # ── Define ROI (centre of frame) ──────────────────────────────
        roi_size  = min(h, w) // 2
        roi_x     = (w - roi_size) // 2
        roi_y     = (h - roi_size) // 2 + 20
        roi       = (roi_x, roi_y, roi_size, roi_size)

        # ── Predict (every N frames) ──────────────────────────────────
        if frame_idx % predict_every == 0:
            roi_crop = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            if roi_crop.size > 0:
                gesture, conf, top3 = predictor.predict(roi_crop, top_k=3)
                smooth_buf.append((gesture, conf))
                last_top3 = top3

        # ── Smoothing: majority vote over buffer ──────────────────────
        if smooth_buf:
            gestures  = [g for g, _ in smooth_buf]
            from collections import Counter
            dominant  = Counter(gestures).most_common(1)[0][0]
            avg_conf  = np.mean([c for g, c in smooth_buf if g == dominant])
            last_gesture = dominant
            last_conf    = float(avg_conf)

        # ── FPS ────────────────────────────────────────────────────────
        fps_times.append(time.perf_counter())
        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-6) \
              if len(fps_times) > 1 else 0.0

        # ── Draw HUD ───────────────────────────────────────────────────
        frame = draw_overlay(frame, last_gesture, last_conf,
                             last_top3, fps, roi)

        cv2.imshow("Hand Gesture Recognition  –  Real-Time", frame)

        # ── Key handler ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = screenshots / f"capture_{ts}.png"
            cv2.imwrite(str(path), frame)
            print(f"[Screenshot] {path}")
        elif key == ord("r"):
            smooth_buf.clear()
            print("[Reset] Smoothing buffer cleared.")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✔] Webcam session ended.")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time gesture recognition")
    parser.add_argument("--model",          default=None,
                        help="Path to saved .keras model")
    parser.add_argument("--camera",         type=int, default=0,
                        help="Webcam device index")
    parser.add_argument("--predict_every",  type=int, default=2,
                        help="Run model every N frames (lower=faster but more CPU)")
    parser.add_argument("--smooth",         type=int, default=5,
                        help="Smoothing window size (frames)")
    args = parser.parse_args()

    # Find latest model if none specified
    model_path = args.model
    if model_path is None:
        saved = ROOT / "models" / "saved"
        candidates = sorted(saved.glob("gesture_model_final_*.keras"), reverse=True)
        if not candidates:
            sys.exit("[ERROR] No trained model found. Run training/train.py first.")
        model_path = str(candidates[0])

    run_realtime(
        model_path=model_path,
        camera_idx=args.camera,
        predict_every=args.predict_every,
        smooth_window=args.smooth,
    )
