# 🤚 Hand Gesture Recognition System
### Production-grade · MobileNetV2 + Transfer Learning · Real-time Capable

---

## 🎯 Project Overview

A complete, end-to-end **Hand Gesture Recognition** pipeline built with:
- **MobileNetV2** backbone (ImageNet pre-trained) + custom classification head
- **Two-stage training**: Feature Extraction → Fine-Tuning
- **Real-time inference** at >25 FPS via OpenCV
- **Streamlit demo app** for interactive testing
- **Target accuracy**: ≥95% on test set

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Name** | LeapGestRecog |
| **Source** | [Kaggle – gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) |
| **Classes** | 10 hand gestures |
| **Images** | ~20,000 near-infrared frames |
| **Split** | 70% train / 15% val / 15% test |

**Gesture Classes:**
`palm` · `l` · `fist` · `fist_moved` · `thumb` · `index` · `ok` · `palm_moved` · `c` · `down`

---

## 🏗️ Architecture

```
Input (224×224×3)
       │
  MobileNetV2 (frozen → fine-tuned)
       │  ImageNet pre-trained backbone
       │  1280-dim feature maps
       │
  GlobalAveragePooling2D
       │
  Dense(512) → BatchNorm → ReLU → Dropout(0.4)
       │
  Dense(256) → BatchNorm → ReLU → Dropout(0.3)
       │
  Dense(10) → Softmax
       │
  Gesture Class
```

**Why MobileNetV2?**
- ✅ Depthwise-separable convolutions → 9× fewer FLOPs vs ResNet-50
- ✅ Real-time on CPU (< 20ms inference)
- ✅ Rich ImageNet features transferable to gesture domain
- ✅ TFLite-compatible for edge deployment

---

## 📁 Project Structure

```
hand_gesture_recognition/
│
├── dataset/
│   ├── download_dataset.py     ← Kaggle API download + organise splits
│   ├── raw/                    ← Raw Kaggle download (auto-created)
│   ├── processed/              ← train/ val/ test/ splits (auto-created)
│   │   ├── train/<class>/
│   │   ├── val/<class>/
│   │   └── test/<class>/
│   └── label_map.json          ← Index → class name mapping
│
├── models/
│   ├── model_builder.py        ← MobileNetV2 architecture + fine-tune helpers
│   ├── saved/                  ← Saved .keras checkpoints (auto-created)
│   ├── logs/                   ← TensorBoard + CSV training logs (auto-created)
│   └── results/                ← Evaluation plots + screenshots (auto-created)
│
├── training/
│   ├── data_pipeline.py        ← tf.data pipeline + augmentation
│   ├── train.py                ← Two-stage training script
│   └── evaluate.py             ← Test-set evaluation + plots
│
├── inference/
│   ├── predict.py              ← GesturePredictor class + CLI
│   └── realtime_inference.py   ← OpenCV webcam real-time demo
│
├── app/
│   └── streamlit_app.py        ← Interactive web demo
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/SHIVA-KUMAR-D/real-time-hand-gesture-recognition
cd real-time-hand-gesture-recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Kaggle API

```bash
# 1. Log in to https://www.kaggle.com
# 2. Profile → Settings → Account → "Create New API Token"
# 3. Move the downloaded file:

mkdir $HOME\.kaggle
notepad $HOME\.kaggle\kaggle.json

Setup Kaggle API

Create file:

C:\Users\<your-username>\.kaggle\kaggle.json
{
  "username": "your_kaggle_username",
  "key": "your_api_key"
}

```

### 3. Download & Prepare Dataset

```bash
python dataset/download_dataset.py
```

Expected output:
```
[✔] Kaggle credentials found for user: <your_username>
[1/3] Downloading LeapGestRecog from Kaggle …
[2/3] Organising dataset …
[3/3] Dataset split summary:
  Class           Train    Val   Test
  ───────────────────────────────────
  c                980    210    210
  down             980    210    210
  fist             980    210    210
  ...
[✔] Label map saved → dataset/label_map.json
[✔] Processed dataset ready → dataset/processed
```

### 4. Train the Model

```bash
# Full two-stage training (recommended)
python training/train.py

# Custom epochs
python training/train.py --epochs_stage1 25 --epochs_stage2 40

# Fast run (stage 1 only, skip fine-tuning)
python training/train.py --skip_stage2 --epochs_stage1 15

# GPU training (TF auto-detects)
python training/train.py --batch_size 64
```

### 5. Evaluate on Test Set

```bash
# Replace <run_id> with your actual run timestamp (e.g. 20240101_120000)
python training/evaluate.py \
  --model models/saved/gesture_model_final_<run_id>.keras \
  --log_csv models/logs/history_stage1_<run_id>.csv \
  --log_csv2 models/logs/history_stage2_<run_id>.csv
```

This generates:
- `models/results/confusion_matrix_*.png`
- `models/results/training_curves_*.png`
- `models/results/misclassified_*.png`

### 6. Real-Time Webcam Demo

```bash
# Default webcam (index 0)
python inference/realtime_inference.py

# External webcam
python inference/realtime_inference.py --camera 1

# Controls: Q=quit  S=screenshot  R=reset smoothing
```

### 7. Streamlit App

```bash
streamlit run app/streamlit_app.py
# Opens: http://localhost:8501
```

### 8. Single Image Prediction (CLI)

```bash
python inference/predict.py \
  --model models/saved/gesture_model_final_<run_id>.keras \
  --image path/to/hand_gesture.jpg
```

---

## 📊 Expected Results

| Metric | Stage 1 | Stage 2 (Fine-tuned) |
|--------|---------|----------------------|
| Val Accuracy | ~88–90% | **~95–98%** |
| Test Accuracy | ~87–89% | **~94–97%** |
| Inference Time | < 15ms | < 15ms |
| Params (trainable) | ~800K | ~2.4M |

---

## 🔧 Optimization Techniques Applied

| Technique | Where | Purpose |
|-----------|-------|---------|
| Transfer Learning | MobileNetV2 backbone | Leverage ImageNet features |
| Fine-Tuning | Stage 2 top-30 layers | Domain adaptation |
| Data Augmentation | Training pipeline | Reduce overfitting |
| Batch Normalisation | Dense head | Training stability |
| Dropout (0.4) | Dense head | Regularisation |
| L2 Weight Decay | Dense head | Regularisation |
| Class Weights | Training | Handle class imbalance |
| Early Stopping | Both stages | Prevent overfitting |
| ReduceLROnPlateau | Stage 2 | Escape local minima |
| Frame smoothing | Real-time | Reduce prediction flicker |

---

## 💡 Further Improvements

1. **MediaPipe Hands** – Use hand landmark detection to crop the hand ROI precisely before classification → removes background noise.
2. **EfficientNetV2-S** – Swap backbone for potentially +2–3% accuracy at the cost of ~2× inference time.
3. **CNN + LSTM** – For video sequences, add a temporal LSTM layer over CNN features extracted per-frame.
4. **TFLite Quantisation** – Export to int8 TFLite for deployment on Raspberry Pi / Android / iOS.
5. **More Data** – Combine with HaGRID (552K images, 18 classes) for a more robust, generalisable model.
6. **YOLOv8 Hand Detector** – Add a real-time hand detector upstream to handle multiple hands and cluttered backgrounds.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `kaggle: command not found` | Run `pip install kaggle` and check `~/.kaggle/kaggle.json` |
| `CUDA out of memory` | Reduce `--batch_size` (try 16 or 8) |
| Low accuracy on webcam | Use the ROI box, ensure good lighting, plain background |
| `FileNotFoundError: processed/` | Run `python dataset/download_dataset.py` first |
| OpenCV window not opening | Set `DISPLAY` env var on headless Linux; use Streamlit app instead |

---

## 📄 License

MIT License. Dataset is subject to Kaggle dataset terms.
