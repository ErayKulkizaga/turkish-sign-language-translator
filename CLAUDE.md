# CLAUDE.md — Proje Bağlamı

Bu dosya Claude'un projeyi hızlıca anlaması için yazılmıştır.
Her konuşmada baştan sormadan devam edebilmek için buraya bak.

---

## Proje Özeti

**Türk İşaret Dili (TİD) — Tanıma ve Animasyon Sistemi**

İki modül:
1. **Tanıma** — Kameradan el göster → BiLSTM model kelimeyi tahmin eder
2. **Animasyon** — Kelime/cümle gir → 3D stick figure işareti gösterir

---

## Dosya Yapısı

```
demo/                          ← Ana uygulama klasörü
├── backend.py                 ← FastAPI sunucu
├── index.html                 ← Tek sayfa frontend
└── model_assets/
    ├── model.keras            ← BiLSTM modeli (Drive'da)
    ├── label_map.json         ← class_id → {TR, EN}
    ├── norm_stats.json        ← normalizasyon mean/std
    ├── demo_config.json       ← seq_len:16, feat_dim:252, num_classes:184
    ├── label_encoder_classes.npy
    └── animations/            ← (artık kullanılmıyor, BVH kaldırıldı)

dataset/
├── landmarks/                 ← 226 JSON, her biri 30 frame Pose+Hands verisi
├── train/                     ← AUTSL videoları (yüklenmedi, çok büyük)
├── validation/
├── SignList_ClassId_TR_EN.csv
├── train_labels.csv
└── validation_labels.csv

extract_landmarks.py           ← MediaPipe ile landmark çıkarma scripti
json_to_bvh.py                 ← (artık kullanılmıyor)
fbx_to_glb.py                  ← (artık kullanılmıyor)
```

---

## Backend (backend.py)

**Framework:** FastAPI + Uvicorn, port 8000

**Endpoints:**
- `GET /` → index.html
- `GET /health` → model durumu
- `GET /signs` → 226 kelime listesi
- `GET /landmark/{word}` → JSON landmark verisi (animasyon için)
- `WS /ws` → WebSocket (tanıma için)
- `GET /static/*` → statik dosyalar

**WebSocket mesaj tipleri:**
- Client → Server: `{type: "frame", color_landmarks: [...], depth_landmarks: [...], hand_detected: bool}`
- Server → Client: `{type: "idle"}` | `{type: "collecting", progress, frames}` | `{type: "prediction", top_k, status, latency_ms}`

**Model:** BiLSTM, input (1, 16, 252), output (1, 184)
- 252 = color_landmarks(126) + depth_landmarks(126)
- 126 = sol_el(63) + sag_el(63)
- 184 sınıf (226'dan %50 altı accuracy olanlar çıkarıldı)

**Landmark dizini:** `dataset/landmarks/` — her JSON:
```json
{
  "class_id": 0,
  "word_tr": "abla",
  "frame_count": 30,
  "frames": [{"pose": {"0": [x,y,z], "11": [...], ...}, "left_hand": [[...], ...], "right_hand": [[...], ...]}, ...]
}
```

---

## Frontend (index.html)

**Kütüphaneler:**
- MediaPipe Holistic 0.5 (CDN, WASM)
- Three.js 0.128.0 (CDN)
- Vanilla JS, sıfır framework

**İki sekme:**
1. **Tanıma** — kamera → MediaPipe → WebSocket → tahmin
2. **Animasyon** — kelime gir → /landmark/{word} → Three.js stick figure

**Animasyon sistemi (stick figure):**
- GLB model YOK — tamamen Three.js ile çizilmiş silindir+küre
- `drawFrame(frameData)` — her frame'de tüm kemikleri yeniden çizer
- `poseToV3(arr)` — MediaPipe pose koordinatı → Three.js vektör: `((x-0.5)*2, -(y-0.5)*2, -z*2)`
- `h2v(arr)` — el koordinatı → Three.js vektör: `(x, -y, -z)`
- El noktaları bilek pozisyonuyla hizalanıyor: `offset = poseWrist - handWrist`
- Kamera: `position(0.2, 0.1, 2.6)`, `lookAt(0, 0, 0)`, FOV=90
- Baş: pose nokta "0", radius=0.13 (büyük küre)
- Diğer eklemler: radius=0.012

**Cümle desteği:**
- `playSentence(text)` → kelimeleri parse eder → parallel fetch → `ANIM.queue`'ya ekler
- `TRANSITION_FRAMES=6` frame'de yumuşak geçiş
- Tekrar yok — son frame'de bekler

**Renk kodlaması:**
- Mavi `0x4f8ef7` → gövde/kol
- Yeşil `0x22c55e` → sol el
- Turuncu `0xf59e0b` → sağ el

---

## Önemli Notlar

- **Depth kamera yok** — depth_landmarks her zaman sıfır gönderilir
- **BVH sistemi kaldırıldı** — eski json_to_bvh.py, fbx_to_glb.py artık kullanılmıyor
- **GLB model kaldırıldı** — bot.glb artık kullanılmıyor, stick figure tercih edildi
- **Kelime isimleri ASCII normalize** — ç→c, ş→s, ğ→g, ü→u, ö→o, ı→i
- **MediaPipe yavaş yükleniyor** — ilk açılışta ~20-30 saniye bekleme normal (CDN'den ~50MB)
- **F5 yavaş** — tanıma sekmesinde MediaPipe her seferinde yeniden yükleniyor

---

## Dataset

**AUTSL** — Ankara Üniversitesi Türk İşaret Dili
- 226 kelime, ~38k video, RGB+Depth, 512×512, 30fps
- 43 signer, cross-subject split (31 train / 6 val)
- GitHub'a yüklenmedi (çok büyük)

---

## Model Eğitimi

**3 notebook:**
1. `01_landmark_extraction.ipynb` — MediaPipe Holistic ile landmark çıkarma
   - Input: (16, 450) = color(225) + depth(225), sol_el+sag_el+pose
2. `02_model_training.ipynb` — 226 sınıf Transformer (ilk deneme)
   - 4x Multi-Head Attention, d_model=256, dropout=0.3
3. `03_model_184class.ipynb` — 184 sınıf BiLSTM (final model)
   - BiLSTM(256) + BiLSTM(128) + Dense(512) + Dense(256) + Dense(184)
   - Input: (16, 252) — sadece el landmark'ları (pose çıkarıldı)
   - Adam lr=1e-3, ReduceLROnPlateau, EarlyStopping patience=15
   - Augmentation: Gaussian noise, time masking, scale jitter

**Sonuçlar (184 sınıf):**
- Top-1: 76.14%
- Top-3: 89.28%
- Top-5: 91.88%

---

## GitHub

**Repo:** https://github.com/ErayKulkizaga/turkish-sign-language-translator
**Model (Drive):** https://drive.google.com/file/d/1nSiWfa8YZYXqZeG2xm_YOfqBefw6IuVZ/view?usp=sharing

---

## Çalıştırma

```bash
cd demo
python backend.py
# http://localhost:8000 aç
```

Docker:
```bash
docker-compose up --build
```
