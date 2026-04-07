# Signatürk 🤟

**Signatürk** — Turkish Sign Language (TİD) **recognition**, **animation**, and **text-to-speech** in the browser, backed by FastAPI and PostgreSQL.

<p align="center">
  <img src="screenshots/animation.png" alt="Animation" width="100%"/>
</p>

<p align="center">
  <b>Real-time sign recognition from webcam + text-to-sign 3D stick figure animation + voice output</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/FastAPI-green?style=flat-square&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Three.js-black?style=flat-square&logo=threedotjs"/>
  <img src="https://img.shields.io/badge/MediaPipe-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql"/>
  <img src="https://img.shields.io/badge/Dataset-AUTSL-purple?style=flat-square"/>
</p>

---

## Overview

Signatürk consists of three complementary systems for Turkish Sign Language (TİD):

- **Recognition** — Webcam input → real-time sign word prediction using a BiLSTM model
- **Animation** — Text input → 3D stick figure performs the corresponding sign(s)
- **Text-to-Speech** — Recognized signs are read aloud via gTTS for audio feedback
- **Database** — All predictions and session statistics are logged to a PostgreSQL database

Both recognition and animation run in the browser backed by a FastAPI server. No GPU required for inference.

### First launch (splash)

On load, a **welcome screen** runs for at least **10 seconds** while MediaPipe scripts, the Holistic WASM runtime, and the webcam pipeline start in the background. The GIF is served as **`/static/a.gif`** — place your file at **`a.gif`** in the project root (same folder as `backend.py`). A short **startup chime** plays when the splash closes (browsers may require a click first for audio).

Heavy libraries load **progressively** (MediaPipe first; **Three.js** loads only when you open the **Animation** tab) to reduce main-thread stalls on refresh.

---

## Demo

### Sign Recognition

<p align="center">
  <img src="screenshots/cam2.png" width="48%"/>
  <img src="screenshots/cam1.png" width="48%"/>
</p>

MediaPipe Holistic runs entirely in the browser (WASM) and extracts hand landmarks in real time. The 156-dimensional feature vectors are streamed to the backend via **WebSocket** (`ws://` or `wss://`) where the BiLSTM model performs inference.

<p align="center">
  <img src="screenshots/recognation.png" width="70%"/>
</p>

Top-3 predictions with confidence bars, session statistics (total predictions, average confidence, latency ms), a scrollable prediction history, and **voice readout of the top prediction via gTTS**.

---

### Sign Animation

<p align="center">
  <img src="screenshots/animation.png" width="48%"/>
  <img src="screenshots/signs.png" width="48%"/>
</p>

Type any word or sentence — the stick figure performs each sign sequentially with smooth frame interpolation between words. 179 signs available, driven by landmark data extracted directly from AUTSL videos.

---

## Model Performance

| Metric | Baseline | **Final Model** | Gain |
|--------|----------|-----------------|------|
| Top-1 Accuracy | 76.14% | **85.65%** | +9.51% |
| Top-3 Accuracy | 89.28% | **93.96%** | +4.68% |
| Top-5 Accuracy | 91.88% | **95.59%** | +3.71% |

Evaluated on a **cross-subject** validation split — training and validation sets contain entirely different signers (31 train / 6 val), reflecting real-world generalization performance.

---

## What Changed From Baseline to Final

Four targeted improvements drove the +9.51 point gain:

**1. Depth stream removed**
The original model used `feat_dim=252` (color + depth landmarks). Since the live demo uses a standard webcam with no depth sensor, depth slots were always zero — creating a train/inference distribution mismatch. Dropping depth reduced input to `feat_dim=126`, eliminating the noise.

**2. Relative coordinate normalization**
Raw landmark coordinates are screen-relative and vary with hand position and distance from the camera. Each hand is re-centered on its wrist (landmark 0) and scaled by the wrist-to-middle-MCP distance, making features position- and scale-invariant.

**3. Finger angle features**
15 joint angles per hand (5 fingers × 3 joints) are appended to the coordinate features, bringing `feat_dim` from 126 to 156. Angles are rotation-invariant and directly encode hand shape — the key discriminator for sign language.

**4. Temporal attention**
A self-attention layer over the 16 time steps replaces the final `return_sequences=False` LSTM. The model learns which frames carry the most discriminative information rather than treating all frames equally.

**5. Problematic class removal**
Per-class accuracy analysis identified 5 classes with below 50% accuracy. Cosine similarity analysis revealed these classes had near-identical average landmark profiles (similarity > 0.96) — caused by signs that differ only in motion continuation beyond what 16 frames can capture (e.g. `gulmek` vs `cuma`). Removing these classes produced a cleaner 179-class model.

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

MediaPipe Hands (`model_complexity=2`) is run on every video. For each video, 16 frames are sampled at equal intervals and the following landmarks are extracted:

| Source | Landmarks | Dimensions |
|--------|-----------|------------|
| Left hand | 21 keypoints × (x, y, z) | 63 |
| Right hand | 21 keypoints × (x, y, z) | 63 |
| **Total per frame (color only)** | | **126** |

Only the **color** stream is used. Depth is discarded to match the webcam-only inference environment.

```
color_landmarks (126) per frame
Final shape per sample: (16 frames, 126 features)
```

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
- Loss: Sparse Categorical Crossentropy with label smoothing=0.1
- Batch size: 64, Max epochs: 100, EarlyStopping patience=15

---

### Step 3 — 184-Class BiLSTM (`03_model_184class.ipynb`)

Classes with validation accuracy below 50% on the Transformer model were removed, leaving 184 classes. A BiLSTM was trained on these with `feat_dim=252` (color+depth). Top-1: **76.14%**.

---

### Step 4 — Final 179-Class Model (`04_improved_model.ipynb`)

Full improvement pipeline applied on top of Step 3:

**Preprocessing pipeline:**
```
Raw (N, 16, 252)
→ Drop depth: (N, 16, 126)
→ Relative coordinate normalization: (N, 16, 126)
→ Finger angle features: (N, 16, 156)
→ Z-score normalization: (N, 16, 156)
```

**Architecture:**
```
Input (16, 156)
→ Bidirectional LSTM(256, return_sequences=True) + Dropout(0.20)
→ Bidirectional LSTM(256, return_sequences=True) + Dropout(0.20)
→ Temporal Attention (attn_hidden=512)
→ Dense(512, ReLU) + BatchNorm + Dropout(0.20)
→ Dense(256, ReLU) + BatchNorm + Dropout(0.15)
→ Dense(179, softmax)
```

**Training details:**
- Optimizer: Adam (lr=1e-3)
- Loss: CategoricalCrossentropy (label_smoothing=0.05)
- LR schedule: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
- Batch size: 32, Max epochs: 60, EarlyStopping patience=10
- Data augmentation: Gaussian noise + time masking + scale jitter

---

## Architecture Overview

```
Browser                                  FastAPI Backend
──────────────────────────────           ──────────────────────────────────────
Webcam
  → MediaPipe Holistic (WASM)       →    WebSocket
    (hand landmarks, 16 frames)           → Preprocessing:
                                              • Drop depth → color only (126)
                                              • Relative coord normalization
                                              • Finger angle features (156)
                                              • Z-score normalization
                                          → BiLSTM + Temporal Attention (179 classes)
                                          → Top-3 predictions + confidence
                                          → PostgreSQL: log prediction + session
                                          → gTTS: audio for top-1 prediction

Text Input
  → fetch /landmark/{word}          →    Read landmark JSON from dataset/landmarks/
  → Three.js stick figure                 (30 frames × Pose + Hands keypoints)
    (pose + hand bones rendered
     as colored cylinders)
```

---

## Project Structure

```
├── backend.py                    # FastAPI — WebSocket, ML inference, landmark API, TTS, DB logging
├── index.html                    # Frontend — Signatürk UI, splash, recognition + 3D animation
├── a.gif                         # Splash animation (optional; served as /static/a.gif)
├── extract_landmarks.py          # Extracts MediaPipe landmarks from AUTSL videos
├── model_assets/
│   ├── model.keras               # Trained BiLSTM model (download separately)
│   ├── label_map.json            # Class ID → {TR, EN} word mapping (179 classes)
│   ├── norm_stats.json           # Per-feature normalization mean/std
│   ├── demo_config.json          # Model config (seq_len=16, feat_dim=156, num_classes=179)
│   └── label_encoder_classes.npy # LabelEncoder mapping for 179-class remapping
└── dataset/
    ├── landmarks/                # 179 × 30-frame landmark JSON files (for animation)
    ├── SignList_ClassId_TR_EN.csv
    ├── train_labels.csv
    └── validation_labels.csv
```

---

## Setup

### Requirements

```bash
pip install fastapi uvicorn tensorflow mediapipe opencv-python numpy pandas gtts psycopg2-binary sqlalchemy
```

### 1. PostgreSQL Setup

```bash
# Create database
createdb signatürk

# Tables are auto-created on first backend startup
```

Update the connection string in `backend.py`:
```python
DATABASE_URL = "postgresql://user:password@localhost/signatürk"
```

### 2. Download the trained model

The `model.keras` file is not included in this repo due to file size.

> **[Download model.keras from Google Drive](https://drive.google.com/file/d/1nSiWfa8YZYXqZeG2xm_YOfqBefw6IuVZ/view?usp=sharing)**

Place it in `model_assets/`.

### 3. (Optional) Re-extract landmarks for animation

```bash
python extract_landmarks.py --dataset "path/to/dataset"
```

Output: `dataset/landmarks/` — one JSON file per word, 30 frames each. Takes ~12 minutes on a standard CPU.

### 4. Run the demo

**Option A — Direct**
```bash
python backend.py
```

**Option B — Docker**
```bash
docker-compose up --build
docker-compose up -d --build  # background
docker-compose down
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

### Health & diagnostics

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Backend up; `num_classes`, `seq_len`, `feat_dim` |
| `GET /startup-check` | Config/model files on disk, landmark index count |

---

## Usage

**Recognition tab (Tanıma — Kameradan)**
1. Allow camera access when prompted
2. Show a sign to the camera — the system collects 16 frames automatically
3. Top-3 predictions appear with confidence scores and session statistics
4. Top-1 prediction is read aloud via text-to-speech

**Animation tab (Animasyon — Kelimeden)**
1. Type a word (e.g. `merhaba`) or a sentence (e.g. `merhaba tesekkur`)
2. Click **Animasyonu Göster** or press Enter
3. The stick figure performs the sign(s) in sequence with smooth transitions
4. Browse all 179 available signs using the word grid

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| ML Model | TensorFlow / Keras (BiLSTM + Temporal Attention) |
| Landmark extraction (training) | MediaPipe Holistic (Python, model_complexity=2) |
| Landmark extraction (live) | MediaPipe Holistic (WASM, browser-side) |
| 3D Animation | Three.js (stick figure renderer) |
| Real-time communication | WebSocket |
| Text-to-Speech | gTTS |
| Database | PostgreSQL + SQLAlchemy |
| Training environment | Google Colab (T4 GPU) |

---

## Notes

- Depth camera is **not required** — the final model was intentionally trained without depth to match webcam-only inference
- The animation system uses real landmark sequences extracted from AUTSL, not synthesized motion
- Word names use ASCII-normalized labels (ç→c, ş→s, ğ→g, ü→u, ö→o, ı→i) matching dataset filenames
- 5 classes were removed from the final model (`gulmek`, `ilgilenmemek`, `siz`, `fil`, `oruc`) due to near-identical landmark profiles with other classes (cosine similarity > 0.96), caused by signs that differ only in motion continuation beyond 16 frames
- `feat_dim` changed from 252 (baseline) → 156 (final): depth dropped, finger angles added
