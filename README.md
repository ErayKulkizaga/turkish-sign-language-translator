# 🤟 Turkish Sign Language — Recognition & Animation

<p align="center">
  <img src="screenshots/animation.png" alt="Animation" width="100%"/>
</p>

<p align="center">
  <b>Real-time Turkish Sign Language (TİD) recognition from webcam + text-to-sign 3D stick figure animation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/FastAPI-green?style=flat-square&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Three.js-black?style=flat-square&logo=threedotjs"/>
  <img src="https://img.shields.io/badge/MediaPipe-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-AUTSL-purple?style=flat-square"/>
</p>

---

## Overview

This project consists of two complementary systems for Turkish Sign Language (TİD):

- **Recognition** — Webcam input → real-time sign word prediction using a BiLSTM model
- **Animation** — Text input → 3D stick figure performs the corresponding sign(s)

Both systems run in the browser backed by a FastAPI server. No GPU required for inference.

---

## Demo

### Sign Recognition

<p align="center">
  <img src="screenshots/cam2.png" width="48%"/>
  <img src="screenshots/cam1.png" width="48%"/>
</p>

MediaPipe Holistic runs entirely in the browser (WASM) and extracts hand + pose landmarks in real time. The 252-dimensional landmark vectors are streamed to the backend via WebSocket where the BiLSTM model performs inference.

<p align="center">
  <img src="screenshots/recognation.png" width="70%"/>
</p>

Top-3 predictions with confidence bars, session statistics (total predictions, average confidence, latency ms), and a scrollable prediction history.

---

### Sign Animation

<p align="center">
  <img src="screenshots/animation.png" width="48%"/>
  <img src="screenshots/signs.png" width="48%"/>
</p>

Type any word or sentence — the stick figure performs each sign sequentially with smooth frame interpolation between words. 184 signs available, driven by landmark data extracted directly from AUTSL videos.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **76.14%** |
| Top-3 Accuracy | **89.28%** |
| Top-5 Accuracy | **91.88%** |

Evaluated on a **cross-subject** validation split — training and validation sets contain entirely different signers (31 train / 6 val), reflecting real-world generalization performance.

---

## Dataset

[AUTSL](https://cvml.ankara.edu.tr/datasets/) — Ankara University Turkish Sign Language Dataset

| Property | Value |
|----------|-------|
| Total classes | 226 words |
| Total videos | ~38,000 |
| Format | RGB + Depth, 512×512, 30fps |
| Total signers | 43 |
| Train signers | 31 (~28k videos) |
| Validation signers | 6 (~4.4k videos) |

---

## Pipeline

### Step 1 — Landmark Extraction (`01_landmark_extraction.ipynb`)

MediaPipe Holistic (`model_complexity=2`, `static_image_mode=True`) is run on every video. For each video, 16 frames are sampled at equal intervals and the following landmarks are extracted:

| Source | Landmarks | Dimensions |
|--------|-----------|------------|
| Left hand | 21 keypoints × (x, y, z) | 63 |
| Right hand | 21 keypoints × (x, y, z) | 63 |
| Pose | 33 keypoints × (x, y, z) | 99 |
| **Total per frame** | | **225** |

Both **color** and **depth** video streams are processed independently and concatenated:

```
color_landmarks (225) + depth_landmarks (225) = 450 per frame
Final shape per sample: (16 frames, 450 features)
```

Missing landmarks (hand not detected) are filled with zeros. A resume mechanism ensures extraction can be interrupted and restarted without reprocessing completed samples.

---

### Step 2 — Transformer Model (`02_model_training.ipynb`)

The first model was trained on all 226 classes using a Transformer Encoder architecture:

**Architecture:**
- Input projection → Dense(256) + LayerNorm
- Sinusoidal Positional Encoding
- 4× Transformer Encoder blocks:
  - Multi-Head Attention (8 heads, key_dim=32)
  - Feed-Forward Network (GELU, dim=512)
  - LayerNorm + Dropout(0.3)
- Global Average Pool + Global Max Pool → Concatenate
- Dense(512, GELU) → Dropout(0.4) → Dense(256, GELU) → Dropout(0.3)
- Output: Dense(226, softmax)

**Training details:**
- Optimizer: AdamW (weight_decay=1e-4)
- LR schedule: Cosine Decay with 5-epoch warmup (1e-3 → 1e-5)
- Loss: Sparse Categorical Crossentropy with **label smoothing=0.1**
- Batch size: 64
- Max epochs: 100 (EarlyStopping patience=15)
- Mixed precision: float16
- Class weights: balanced, clipped to [0.1, 10.0]

**Data augmentation (applied during training only):**
- Gaussian noise (std=0.02, p=0.5)
- Horizontal flip — x coordinates negated (p=0.5)
- Time masking — 1-2 random frames zeroed (p=0.3)
- Scale jitter (0.9–1.1, p=0.5)

---

### Step 3 — 184-Class BiLSTM Model (`03_model_184class.ipynb`)

Per-class accuracy analysis on the 226-class model revealed that 42 classes achieved less than 50% validation accuracy. These were removed, and a new model was trained on the remaining **184 classes**.

**Feature dimension change:** The 226-class model used `feat_dim=450` (full pose). The 184-class model uses `feat_dim=252` — only hand landmarks (color + depth, 126 each), dropping the pose stream for a leaner input.

**Architecture (BiLSTM):**

```
Input (16, 252)
→ Bidirectional LSTM(256, return_sequences=True) + Dropout(0.3)
→ Bidirectional LSTM(128, return_sequences=False) + Dropout(0.3)
→ Dense(512, ReLU) + BatchNorm + Dropout(0.4)
→ Dense(256, ReLU) + BatchNorm + Dropout(0.3)
→ Dense(184, softmax)
```

**Training details:**
- Optimizer: Adam (lr=1e-3)
- LR schedule: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Max epochs: 100 (EarlyStopping patience=15)
- Mixed precision: float16
- Class weights: balanced, clipped to [0.1, 10.0]
- Data augmentation: Gaussian noise + time masking + scale jitter

**Class filtering threshold:** Validation accuracy ≥ 50% (from 226-class per_class_accuracy.csv)

Labels are remapped 0–183 using `sklearn.LabelEncoder`. The encoder's class mapping is saved to `label_encoder_classes.npy` for use during inference.

---

## Architecture Overview

```
Browser                              FastAPI Backend
────────────────────────────         ──────────────────────────────────
Webcam
  → MediaPipe Holistic (WASM)   →    WebSocket
    (hand + pose landmarks)           → BiLSTM Model (TensorFlow)
    16 frames × 252 features          → Top-3 predictions + confidence

Text Input
  → fetch /landmark/{word}      →    Read landmark JSON from dataset/landmarks/
  → Three.js stick figure              (30 frames × Pose + Hands keypoints)
    (pose + hand bones rendered
     as colored cylinders)
```

---

## Project Structure

```
├── backend.py                    # FastAPI — WebSocket, ML inference, landmark API
├── index.html                    # Frontend — recognition UI + 3D animation viewer
├── extract_landmarks.py          # Extracts MediaPipe landmarks from AUTSL videos
├── model_assets/
│   ├── model.keras               # Trained BiLSTM model (download separately)
│   ├── label_map.json            # Class ID → {TR, EN} word mapping
│   ├── norm_stats.json           # Per-feature normalization mean/std
│   ├── demo_config.json          # Model config (seq_len, feat_dim, thresholds)
│   └── label_encoder_classes.npy # LabelEncoder mapping for 184-class remapping
└── dataset/
    ├── landmarks/                # 226 × 30-frame landmark JSON files (for animation)
    ├── SignList_ClassId_TR_EN.csv
    ├── train_labels.csv
    └── validation_labels.csv
```

---

## Setup

### Requirements

```bash
pip install fastapi uvicorn tensorflow mediapipe opencv-python numpy pandas
```

### 1. Download the trained model

The `model.keras` file is not included in this repo due to file size.

> **[Download model.keras from Google Drive](https://drive.google.com/file/d/1nSiWfa8YZYXqZeG2xm_YOfqBefw6IuVZ/view?usp=sharing)**

Place it in `model_assets/`.

### 2. (Optional) Re-extract landmarks for animation

If you have the AUTSL dataset and want to regenerate the animation landmark files:

```bash
python extract_landmarks.py --dataset "path/to/dataset"
```

Output: `dataset/landmarks/` — one JSON file per word, 30 frames each, containing MediaPipe Pose (upper body) and Hands (21 keypoints each) data. Takes ~12 minutes on a standard CPU (i5, no GPU).

### 3. Run the demo

**Option A — Direct**
```bash
python backend.py
```

**Option B — Docker**
```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop
docker-compose down
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

> **Note:** `model.keras` and `dataset/landmarks/` are mounted as volumes — you do not need to rebuild the image when updating the model.

---

## Usage

**Recognition tab (Tanıma — Kameradan)**
1. Allow camera access when prompted
2. Show a sign to the camera — the system collects 16 frames automatically
3. Top-3 predictions appear with confidence scores and session statistics

**Animation tab (Animasyon — Kelimeden)**
1. Type a word (e.g. `merhaba`) or a sentence (e.g. `merhaba tesekkur`)
2. Click **Animasyonu Göster** or press Enter
3. The stick figure performs the sign(s) in sequence with smooth transitions
4. Browse all 184 available signs using the word grid

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| ML Model | TensorFlow / Keras (BiLSTM) |
| Landmark extraction (training) | MediaPipe Holistic (Python, model_complexity=2) |
| Landmark extraction (live) | MediaPipe Holistic (WASM, browser-side) |
| 3D Animation | Three.js (stick figure renderer) |
| Real-time communication | WebSocket |
| Training environment | Google Colab (T4 GPU) |

---

## Notes

- Depth camera is **not required** for the live demo — depth landmarks default to zero
- The animation system uses real landmark sequences extracted from AUTSL, not synthesized motion
- Word names in the animation system use ASCII-normalized labels (ç→c, ş→s, ğ→g, etc.) matching dataset filenames
- The 184-class model uses only hand landmarks (feat_dim=252); the pose stream used in the original 226-class Transformer model was dropped for the final BiLSTM to reduce input noise
