# Türk İşaret Dili — Tanıma ve Animasyon Sistemi

Real-time Turkish Sign Language (TİD) recognition from webcam, plus text-to-sign animation with a 3D stick figure renderer.

---

## Features

**Sign Recognition (Webcam → Word)**
- Live hand landmark extraction via MediaPipe Holistic (runs in-browser, no GPU needed)
- BiLSTM model trained on AUTSL dataset — 226 word classes
- Top-1: **76.14%** | Top-3: **89.28%** | Top-5: **91.88%** (cross-subject split)
- WebSocket streaming between browser and FastAPI backend
- Idle detection — no output when hands are not visible

**Sign Animation (Word → Animation)**
- Type any word or sentence to see its sign language animation
- 3D stick figure rendered with Three.js — no external 3D model required
- Pose (upper body) + both hands (21 landmarks each) animated per frame
- Sentence support — multiple words play sequentially with smooth transitions
- Autocomplete suggestions as you type

---

## Dataset

[AUTSL](https://cvml.ankara.edu.tr/datasets/) — Ankara University Turkish Sign Language Dataset
- 226 word classes
- ~38,000 videos (RGB + Depth, 512×512, 30fps)
- 43 signers, cross-subject train/validation split

---

## Model

- **Architecture:** Bidirectional LSTM + Dense head
- **Input:** 16 frames × 252 features (color landmarks + depth landmarks, both hands)
- **Training:** 184 classes (42 low-accuracy classes removed)
- **Landmark extraction:** MediaPipe Hands, color and depth streams concatenated

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ML Model | TensorFlow / Keras |
| Landmark extraction | MediaPipe Holistic (WASM, browser-side) |
| 3D Animation | Three.js |
| Communication | WebSocket |

---

## Project Structure

```
demo/
├── backend.py              # FastAPI backend — WebSocket + ML inference + animation endpoints
├── index.html              # Frontend — webcam recognition + 3D animation viewer
└── model_assets/
    ├── model.keras         # Trained BiLSTM model
    ├── label_map.json      # Class ID → TR/EN word mapping
    ├── norm_stats.json     # Normalization mean/std
    ├── demo_config.json    # Model configuration
    └── label_encoder_classes.npy

dataset/
├── train/                  # AUTSL training videos
├── validation/             # AUTSL validation videos
├── SignList_ClassId_TR_EN.csv
├── train_labels.csv
└── landmarks/              # Extracted landmark JSON files (226 × 30 frames)

extract_landmarks.py        # Extracts Pose + Hands landmarks from dataset videos
```

---

## Setup

### Requirements

```bash
pip install fastapi uvicorn tensorflow mediapipe opencv-python numpy pandas
```

### 1. Extract Landmarks

```bash
python extract_landmarks.py --dataset "path/to/dataset"
```

This processes the AUTSL dataset and saves one JSON file per word in `dataset/landmarks/`. Each JSON contains 30 frames of MediaPipe Pose (upper body, 9 keypoints) and Hands (21 keypoints each) landmarks.

### 2. Run the Demo

```bash
cd demo
python backend.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Usage

**Tanıma (Recognition)**
1. Click the **Tanıma — Kameradan** tab
2. Allow camera access
3. Show a sign to the camera — the system collects 16 frames and predicts the word
4. Top-3 predictions are shown with confidence scores

**Animasyon (Animation)**
1. Click the **Animasyon — Kelimeden** tab
2. Type a word (e.g. `merhaba`) or a sentence (e.g. `merhaba tesekkur`)
3. Click **Animasyonu Göster** or press Enter
4. The stick figure performs the sign(s) in sequence

---

## Notes

- Depth camera is not required — depth landmarks are set to zero for webcam demos
- The animation system uses raw MediaPipe coordinates extracted from AUTSL videos, not synthesized motion
- Words in the animation system use ASCII-normalized names (ç→c, ş→s, etc.) matching the dataset labels
