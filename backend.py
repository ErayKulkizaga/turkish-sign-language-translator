"""
backend.py — TSL Demo Backend (184 sinif + Animasyon)
"""

import os, json, asyncio, time, traceback
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import uvicorn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

ASSETS_DIR  = os.path.join(os.path.dirname(__file__), "model_assets")
CONFIG_FILE = os.path.join(ASSETS_DIR, "demo_config.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

SEQ_LEN     = CONFIG["seq_len"]
NUM_CLASSES = CONFIG["num_classes"]
CONF_THRESH = CONFIG["confidence_threshold"]
TOP_K       = CONFIG["top_k_display"]
SINGLE_DIM  = 126
MODEL_PATH = os.path.join(ASSETS_DIR, CONFIG["model_file"])
NORM_STATS_PATH = os.path.join(ASSETS_DIR, CONFIG["norm_stats_file"])
LABEL_MAP_PATH = os.path.join(ASSETS_DIR, CONFIG["label_map_file"])

print("Model yukleniyor...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print(f"Model yuklendi: input={MODEL.input_shape}, output={MODEL.output_shape}")

with open(NORM_STATS_PATH) as f:
    norm_data = json.load(f)
NORM_MEAN = np.array(norm_data["mean"], dtype=np.float32)
NORM_STD  = np.array(norm_data["std"],  dtype=np.float32)

with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    raw_map = json.load(f)
LABEL_MAP = {int(k): v for k, v in raw_map.items()}
print(f"{len(LABEL_MAP)} sinif yuklendi")

# Landmark dizini
LANDMARKS_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "landmarks")
if not os.path.isdir(LANDMARKS_DIR):
    LANDMARKS_DIR = os.path.join(os.path.dirname(__file__), "landmarks")

LANDMARK_INDEX = {}
if os.path.isdir(LANDMARKS_DIR):
    for fname in os.listdir(LANDMARKS_DIR):
        if fname.endswith(".json"):
            word = fname[:-5].lower()
            LANDMARK_INDEX[word] = os.path.join(LANDMARKS_DIR, fname)
    print(f"{len(LANDMARK_INDEX)} landmark yuklendi")
else:
    print(f"UYARI: Landmark klasoru bulunamadi: {LANDMARKS_DIR}")


def build_feature_vector(color_seq, depth_seq):
    color_norm = (color_seq - NORM_MEAN[:126]) / NORM_STD[:126]
    depth_norm = (depth_seq - NORM_MEAN[126:]) / NORM_STD[126:]
    return np.concatenate([color_norm, depth_norm], axis=-1)[np.newaxis, :, :]


def run_inference(feature_input):
    t0    = time.perf_counter()
    preds = MODEL.predict(feature_input, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    top_indices = np.argsort(preds)[::-1][:TOP_K]
    top_results = []
    for idx in top_indices:
        info = LABEL_MAP.get(int(idx), {"TR": "?", "EN": "?"})
        top_results.append({
            "class_id":   int(idx),
            "TR":         info.get("TR", "?"),
            "EN":         info.get("EN", "?"),
            "confidence": float(preds[idx])
        })
    status = "ok" if top_results[0]["confidence"] >= CONF_THRESH else "low_confidence"
    return {"status": status, "top_k": top_results, "latency_ms": round(latency_ms, 1)}


class SessionState:
    def __init__(self):
        self.color_buffer = []
        self.depth_buffer = []
        self.collecting   = False

    def reset(self):
        self.color_buffer = []
        self.depth_buffer = []
        self.collecting   = False

    def is_ready(self):
        return len(self.color_buffer) >= SEQ_LEN


app = FastAPI(title="TSL Demo")
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")

@app.get("/")
async def root():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )

@app.get("/health")
async def health():
    return {"status": "ok", "num_classes": NUM_CLASSES, "seq_len": SEQ_LEN}

@app.get("/startup-check")
async def startup_check():
    required_files = {
        "config": CONFIG_FILE,
        "model": MODEL_PATH,
        "norm_stats": NORM_STATS_PATH,
        "label_map": LABEL_MAP_PATH,
    }
    file_status = {
        key: {"path": path, "exists": os.path.isfile(path)}
        for key, path in required_files.items()
    }
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "num_classes": NUM_CLASSES,
        "seq_len": SEQ_LEN,
        "landmarks_dir": LANDMARKS_DIR,
        "landmark_count": len(LANDMARK_INDEX),
        "required_files": file_status,
    }

@app.get("/signs")
async def list_signs():
    return {"words": sorted(LANDMARK_INDEX.keys()), "count": len(LANDMARK_INDEX)}

@app.get("/landmark/{word}")
async def get_landmark(word: str):
    key = word.lower().strip()
    path = LANDMARK_INDEX.get(key)
    if path is None:
        return Response(status_code=404, content=f"'{word}' bulunamadi")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="application/json")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()
    print("[WS] Yeni baglanti")
    try:
        async for message in websocket.iter_text():
            data     = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "frame":
                hand_detected = data.get("hand_detected", False)
                color_lm = np.array(data.get("color_landmarks", [0.0]*SINGLE_DIM), dtype=np.float32)
                depth_lm = np.array(data.get("depth_landmarks", [0.0]*SINGLE_DIM), dtype=np.float32)

                if not hand_detected:
                    if state.collecting: state.reset()
                    await websocket.send_text(json.dumps({"type": "idle"}))
                    continue

                if not state.collecting:
                    state.collecting = True

                state.color_buffer.append(color_lm)
                state.depth_buffer.append(depth_lm)

                await websocket.send_text(json.dumps({
                    "type":     "collecting",
                    "progress": round(len(state.color_buffer) / SEQ_LEN, 2),
                    "frames":   len(state.color_buffer)
                }))

                if state.is_ready():
                    color_seq = np.array(state.color_buffer[:SEQ_LEN])
                    depth_seq = np.array(state.depth_buffer[:SEQ_LEN])
                    feature_input = build_feature_vector(color_seq, depth_seq)
                    loop   = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, run_inference, feature_input)
                    result["type"] = "prediction"
                    await websocket.send_text(json.dumps(result))
                    state.reset()
                continue

            if msg_type == "reset":
                state.reset()
                await websocket.send_text(json.dumps({"type": "reset_ack"}))

    except WebSocketDisconnect:
        print("[WS] Baglanti kesildi")
    except Exception as e:
        print(f"[WS] Hata: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*50)
    print("TSL Demo Backend - 184 Sinif + Animasyon")
    print("="*50)
    print("Tarayicida ac: http://localhost:8000")
    print("Durdurmak icin: Ctrl+C")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
