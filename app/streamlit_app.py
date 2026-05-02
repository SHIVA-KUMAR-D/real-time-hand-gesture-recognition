"""
app/streamlit_app.py
─────────────────────────────────────────────────────────────────────────────
Interactive Streamlit demo for the Hand Gesture Recognition system.

Tabs:
  1. 📷 Upload Image   – upload any image and see real-time prediction
  2. 🎥 Webcam Demo    – live gesture recognition (WebRTC or snapshot)
  3. 📊 Model Info     – architecture details, dataset stats, class examples
  4. 📈 Training Plots – loss/accuracy curves + confusion matrix

Run:
  streamlit run app/streamlit_app.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

  html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }

  .hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00c896 0%, #00a3ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .hero-sub {
    color: #888;
    font-size: 1rem;
    margin-top: 0.2rem;
  }
  .gesture-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }
  .gesture-name {
    font-size: 2rem;
    font-weight: 700;
    color: #00c896;
  }
  .confidence-text {
    font-size: 1.1rem;
    color: #ccc;
  }
  .metric-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.4rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_predictor(model_path: str):
    """Load predictor once and cache across reruns."""
    from inference.predict import GesturePredictor
    return GesturePredictor(model_path)


def find_latest_model() -> Path | None:
    saved = ROOT / "models" / "saved"
    models = sorted(saved.glob("gesture_model_final_*.keras"), reverse=True)
    return models[0] if models else None


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    import cv2
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def confidence_bar_html(label: str, value: float, is_top: bool = False) -> str:
    pct = value * 100
    color = "#00c896" if is_top else "#444"
    return f"""
    <div style="margin: 6px 0;">
      <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
        <span style="color:#ccc; font-size:0.85rem;">{label}</span>
        <span style="color:{'#00c896' if is_top else '#888'}; font-size:0.85rem; font-weight:{'700' if is_top else '400'};">{pct:.1f}%</span>
      </div>
      <div style="background:#1e1e2e; border-radius:4px; height:10px; overflow:hidden;">
        <div style="background:{color}; width:{pct:.1f}%; height:100%; border-radius:4px; transition:width 0.3s;"></div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    saved_dir = ROOT / "models" / "saved"
    model_files = sorted(saved_dir.glob("*.keras"), reverse=True)

    if not model_files:
        st.warning("No trained models found.\nRun `python training/train.py` first.")
        selected_model = None
    else:
        model_names = [m.name for m in model_files]
        selected_name = st.selectbox("Select Model", model_names, index=0)
        selected_model = str(saved_dir / selected_name)
        st.success(f"Model: **{selected_name[:30]}…**" if len(selected_name) > 30 else f"Model: **{selected_name}**")

    st.markdown("---")
    top_k = st.slider("Show Top-K Predictions", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")

    label_map_path = ROOT / "dataset" / "label_map.json"
    if label_map_path.exists():
        lmap = json.loads(label_map_path.read_text())
        classes = list(lmap.values())
        st.markdown(f"**Classes ({len(classes)}):**")
        for cls in sorted(classes):
            st.markdown(f"  - {cls}")

    st.markdown("---")
    st.markdown("**Project Links**")
    st.markdown("📦 [Dataset on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)")
    st.markdown("🧠 [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)")


# ══════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🤚 Hand Gesture Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Production-grade deep learning · MobileNetV2 · Real-time capable</div>', unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════════════
tab_upload, tab_webcam, tab_info, tab_results = st.tabs([
    "📷 Upload Image",
    "🎥 Webcam Demo",
    "📊 Model Info",
    "📈 Training Results",
])


# ─── TAB 1: Upload Image ──────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload a hand gesture image")
    st.markdown("Supports JPG, PNG, WEBP. The model will predict the gesture class.")

    uploaded = st.file_uploader("Choose an image …",
                                type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        col_img, col_pred = st.columns([1, 1], gap="large")

        pil_img = Image.open(BytesIO(uploaded.read()))

        with col_img:
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        with col_pred:
            if selected_model is None:
                st.error("No model loaded. Train one first.")
            else:
                with st.spinner("Running inference …"):
                    predictor = load_predictor(selected_model)
                    bgr = pil_to_bgr(pil_img)
                    gesture, conf, top_k_preds = predictor.predict(bgr, top_k=top_k)

                st.markdown(f"""
                <div class="gesture-card">
                  <div style="font-size:3rem;">🤚</div>
                  <div class="gesture-name">{gesture.upper()}</div>
                  <div class="confidence-text">{conf*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Top Predictions")
                bars_html = "".join(
                    confidence_bar_html(p["class"], p["confidence"], i == 0)
                    for i, p in enumerate(top_k_preds)
                )
                st.markdown(bars_html, unsafe_allow_html=True)

                if conf < 0.5:
                    st.warning("⚠️ Low confidence – ensure the hand is clearly visible with good lighting.")
                elif conf > 0.9:
                    st.success("✅ High-confidence prediction!")


# ─── TAB 2: Webcam Demo ───────────────────────────────────────────────────
with tab_webcam:
    st.markdown("### Live Webcam Gesture Recognition")
    st.info("📌 For **real-time webcam inference**, run the dedicated script:\n"
            "```bash\npython inference/realtime_inference.py\n```\n"
            "Or capture a single frame below for in-app prediction.")

    st.markdown("#### Capture & Predict")

    cam_img = st.camera_input("📸 Take a photo of your hand gesture")
    if cam_img:
        if selected_model is None:
            st.error("No model loaded.")
        else:
            pil_cam = Image.open(BytesIO(cam_img.read()))
            with st.spinner("Predicting …"):
                predictor = load_predictor(selected_model)
                bgr = pil_to_bgr(pil_cam)
                gesture, conf, top_k_preds = predictor.predict(bgr, top_k=top_k)

            c1, c2 = st.columns(2)
            c1.image(pil_cam, caption="Captured Frame", use_container_width=True)
            with c2:
                st.markdown(f"""
                <div class="gesture-card">
                  <div style="font-size:3rem;">🤚</div>
                  <div class="gesture-name">{gesture.upper()}</div>
                  <div class="confidence-text">{conf*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                bars = "".join(
                    confidence_bar_html(p["class"], p["confidence"], i == 0)
                    for i, p in enumerate(top_k_preds)
                )
                st.markdown(bars, unsafe_allow_html=True)


# ─── TAB 3: Model Info ────────────────────────────────────────────────────
with tab_info:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
| Component | Detail |
|-----------|--------|
| **Backbone** | MobileNetV2 (ImageNet pre-trained) |
| **Input** | 224 × 224 × 3 (RGB) |
| **Head** | GAP → Dense(512) → BN → Dropout → Dense(256) → BN → Softmax |
| **Params** | ~2.4 M trainable (fine-tune stage) |
| **Regularisation** | L2(1e-4) + Dropout(0.40) |
| **Optimiser** | Adam · LR schedule |
| **Loss** | Sparse Categorical Cross-Entropy |

**Training Strategy:**
1. **Stage 1** – Backbone frozen, train head only (LR=1e-3, 20 epochs)
2. **Stage 2** – Unfreeze top-30 layers, fine-tune end-to-end (LR=1e-5, 30 epochs)
        """)

    with col_b:
        st.markdown("### 📦 Dataset – LeapGestRecog")
        st.markdown("""
| Property | Value |
|----------|-------|
| **Source** | [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) |
| **Classes** | 10 distinct hand gestures |
| **Images** | ~20,000 near-IR frames |
| **Resolution** | 160 × 120 → resized to 224 × 224 |
| **Background** | Controlled (dark) |
| **Split** | 70 % train / 15 % val / 15 % test |

**Why this dataset?**
- ✅ Clean, well-labelled, HCI-relevant gestures
- ✅ Small enough for rapid iteration
- ✅ IR images reduce colour/lighting bias
- ✅ Widely used benchmark
        """)

    st.markdown("### 🧩 Gesture Classes")
    gesture_emojis = {
        "palm": "🖐️", "l": "👆", "fist": "✊", "fist_moved": "🤜",
        "thumb": "👍", "index": "☝️", "ok": "👌",
        "palm_moved": "🙌", "c": "🤏", "down": "👇",
    }

    cols = st.columns(5)
    for i, (name, emoji) in enumerate(gesture_emojis.items()):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="metric-box" style="text-align:center;">
              <div style="font-size:1.8rem;">{emoji}</div>
              <div style="font-weight:600; color:#00c896;">{name}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 🚀 Performance Targets")
    p1, p2, p3 = st.columns(3)
    p1.metric("Target Val Accuracy",  ">= 95 %",  "Stage-2 fine-tuning")
    p2.metric("Inference Speed",      "< 20 ms",   "MobileNetV2 on CPU")
    p3.metric("Real-time FPS",        ">= 25 fps", "Webcam stream")


# ─── TAB 4: Training Results ──────────────────────────────────────────────
with tab_results:
    st.markdown("### 📈 Training Curves & Evaluation")

    results_dir = ROOT / "models" / "results"
    plots = list(results_dir.glob("*.png"))

    if not plots:
        st.info("No result plots found yet.\n\n"
                "Run:\n```bash\npython training/evaluate.py --model <path>\n```")
    else:
        curve_plots = [p for p in plots if "training_curves" in p.name]
        cm_plots    = [p for p in plots if "confusion_matrix" in p.name]
        misc_plots  = [p for p in plots if "misclassified" in p.name]

        if curve_plots:
            st.markdown("#### Training Curves (Loss & Accuracy)")
            st.image(str(sorted(curve_plots, reverse=True)[0]),
                     use_container_width=True)

        if cm_plots:
            st.markdown("#### Confusion Matrix")
            st.image(str(sorted(cm_plots, reverse=True)[0]),
                     use_container_width=True)

        if misc_plots:
            st.markdown("#### Misclassified Samples")
            st.image(str(sorted(misc_plots, reverse=True)[0]),
                     use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Hand Gesture Recognition · MobileNetV2 + Transfer Learning · Built with TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
