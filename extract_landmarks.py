"""
AUTSL → Landmark JSON Extractor (Pose + Hands)
===============================================
Her kelime için train setindeki en kaliteli videoyu seçer.
MediaPipe Pose (üst vücut) + Hands (her el 21 nokta) birleştirir.
Normalize eder, 30 frame'e örnekler, JSON olarak kaydeder.

Kullanım:
    python extract_landmarks.py --dataset C:/Users/Eray/Desktop/dataset

Çıktı:
    landmarks/ klasörü altında 226 JSON dosyası
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
import json
import os
import urllib.request
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Ayarlar ──────────────────────────────────────────────────────────────────
MAX_FRAMES  = 30    # Animasyon için hedef frame sayısı
MIN_FRAMES  = 8     # Bu kadar frame altındaki video geçersiz
CANDIDATES  = 3     # Her kelime için denenecek max aday video
N_WORKERS   = 4     # Paralel işçi sayısı
# ─────────────────────────────────────────────────────────────────────────────

# Pose landmark indeksleri — sadece üst vücut
# 0=burun, 11=sol omuz, 12=sağ omuz,
# 13=sol dirsek, 14=sağ dirsek,
# 15=sol bilek, 16=sağ bilek,
# 23=sol kalça, 24=sağ kalça
POSE_UPPER_BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Model URL ve yolları
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"


def ensure_models():
    """Model dosyalarını indir (yoksa)."""
    for url, path in [(HAND_MODEL_URL, HAND_MODEL_PATH), (POSE_MODEL_URL, POSE_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"İndiriliyor: {path} ...")
            urllib.request.urlretrieve(url, path)
            print(f"İndirildi: {path}")


def normalize_pose(raw_pts):
    # Normalizasyon yok — ham koordinatları string key ile döndür
    return {str(k): v for k, v in raw_pts.items()}


def normalize_hand(pts):
    if pts is None:
        return None
    return pts  # Ham koordinatlar


def resample_frames(frames, target=MAX_FRAMES):
    """
    Frame dizisini hedef uzunluğa lineer interpolasyonla örnekler.
    """
    n = len(frames)
    if n == 0:
        return frames
    if n == target:
        return frames

    indices   = np.linspace(0, n - 1, target)
    resampled = []

    for idx in indices:
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        t  = idx - lo
        frame = {}

        # Eller
        for hand in ("left_hand", "right_hand"):
            a = frames[lo].get(hand)
            b = frames[hi].get(hand)
            if a is None and b is None:
                frame[hand] = None
            elif a is None:
                frame[hand] = b
            elif b is None:
                frame[hand] = a
            else:
                frame[hand] = (np.array(a) * (1 - t) + np.array(b) * t).tolist()

        # Pose
        pa       = frames[lo].get("pose", {})
        pb       = frames[hi].get("pose", {})
        all_keys = set(pa.keys()) | set(pb.keys())
        pose_out = {}
        for k in all_keys:
            va = pa.get(k)
            vb = pb.get(k)
            if va is None and vb is None:
                continue
            elif va is None:
                pose_out[k] = vb
            elif vb is None:
                pose_out[k] = va
            else:
                pose_out[k] = (np.array(va) * (1 - t) + np.array(vb) * t).tolist()
        frame["pose"] = pose_out

        resampled.append(frame)

    return resampled


def process_video(video_path):
    """
    Tek bir videodan Pose + Hands landmark dizisi çıkarır.
    Dönüş: (frames_list, hand_detected_count) ya da (None, 0)
    """
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose_det = mp_vision.PoseLandmarker.create_from_options(pose_options)
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose_det.close()
        hand_det.close()
        return None, 0

    fps           = cap.get(cv2.CAP_PROP_FPS) or 30
    frames        = []
    hand_detected = 0
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(frame_idx * 1000 / fps)

        # Pose
        pose_res  = pose_det.detect_for_video(mp_img, ts_ms)
        pose_data = {}
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            raw = {idx: [lms[idx].x, lms[idx].y, lms[idx].z] for idx in POSE_UPPER_BODY}
            pose_data = normalize_pose(raw)

        # Hands
        hand_res   = hand_det.detect_for_video(mp_img, ts_ms)
        left_hand  = None
        right_hand = None
        if hand_res.hand_landmarks and hand_res.handedness:
            hand_detected += 1
            for lm_list, hd_list in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = hd_list[0].category_name
                pts   = [[p.x, p.y, p.z] for p in lm_list]
                if label == "Left":
                    left_hand  = normalize_hand(pts)
                else:
                    right_hand = normalize_hand(pts)

        frames.append({
            "pose":       pose_data,
            "left_hand":  left_hand,
            "right_hand": right_hand,
        })
        frame_idx += 1

    cap.release()
    pose_det.close()
    hand_det.close()
    return frames, hand_detected


def best_video_for_class(class_id, sample_names, dataset_path):
    """En çok el algılanan adayı döner."""
    best_frames = None
    best_score  = -1

    for sample in sample_names[:CANDIDATES]:
        color_path = Path(dataset_path) / "train" / (sample + "_color.mp4")
        if not color_path.exists():
            continue
        frames, score = process_video(color_path)
        if frames is None or len(frames) < MIN_FRAMES:
            continue
        if score > best_score:
            best_score  = score
            best_frames = frames

    return best_frames, best_score


def extract_worker(args):
    class_id, tr_word, sample_names, dataset_path, output_dir = args
    try:
        frames, score = best_video_for_class(class_id, sample_names, dataset_path)
        if frames is None:
            return class_id, tr_word, False, "video bulunamadı"

        frames_out = resample_frames(frames, MAX_FRAMES)

        out = {
            "class_id":      class_id,
            "word_tr":       tr_word,
            "frame_count":   len(frames_out),
            "hand_detected": score,
            "pose_indices":  POSE_UPPER_BODY,
            "frames":        frames_out,
        }

        out_path = Path(output_dir) / f"{tr_word}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

        return class_id, tr_word, True, f"score={score}"
    except Exception as e:
        return class_id, tr_word, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="AUTSL Pose+Hands Landmark Extractor")
    parser.add_argument("--dataset", default=r"C:\Users\Eray\Desktop\dataset")
    parser.add_argument("--output",  default="landmarks")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir   = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    sign_list    = pd.read_csv(dataset_path / "SignList_ClassId_TR_EN.csv")
    train_labels = pd.read_csv(
        dataset_path / "train_labels.csv",
        header=None, names=["sample", "class_id"]
    )

    class_to_samples = train_labels.groupby("class_id")["sample"].apply(list).to_dict()
    class_to_tr      = dict(zip(sign_list["ClassId"], sign_list["TR"]))

    tasks = []
    for class_id, tr_word in class_to_tr.items():
        samples = class_to_samples.get(class_id, [])
        if not samples:
            continue
        tasks.append((class_id, tr_word, samples, str(dataset_path), str(output_dir)))

    ensure_models()

    print(f"\n{'─'*52}")
    print(f"  Toplam kelime : {len(tasks)}")
    print(f"  Paralel işçi  : {args.workers}")
    print(f"  Veri          : Pose (9 nokta) + Eller (21×2 nokta)")
    print(f"  Çıktı         : {output_dir.resolve()}")
    print(f"{'─'*52}\n")

    start   = time.time()
    success = 0
    fail    = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_worker, t): t[1] for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            class_id, word, ok, msg = future.result()
            status = "✓" if ok else "✗"
            print(f"[{i:3}/{len(tasks)}] {status} {word:<22} {msg}")
            if ok:
                success += 1
            else:
                fail += 1

    elapsed = time.time() - start
    print(f"\n{'─'*52}")
    print(f"  Başarılı : {success}  |  Başarısız : {fail}")
    print(f"  Süre     : {elapsed:.0f}s ({elapsed/60:.1f} dk)")
    print(f"  JSON'lar : {output_dir.resolve()}")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()
