"""
SafeSense AI inference — ONNX Runtime version (no TensorFlow, no PyTorch).

Same pipeline and same output as the original ai_inference.py:

  1. keras_model.onnx  (Teachable Machine classifier, 6 classes). If it is
     >60% sure the photo is "normal", we stop and return "safe".
  2. flood / hazard / fire_building YOLOv8 models (.onnx). All three are run
     and the single most confident detection wins (the Teachable Machine
     result also competes if it is a non-normal class).

Memory: free hosts have ~512 MB. So the three YOLO models are loaded one at
a time, used, then released (set KEEP_MODELS_LOADED=1 to keep them in RAM
if your host has more memory).

Run a quick local test:   python ai_inference.py some_photo.jpg
"""

import ast
import gc
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).parent / "ai_model"
KEEP_LOADED = os.getenv("KEEP_MODELS_LOADED", "0") == "1"
YOLO_CONF = 0.4

YOLO_FILES = {
    "flood": "flood_best.onnx",
    "hazard": "hazard_best.onnx",
    "fire_building": "hazard_fire_building_best.onnx",
}
TM_FILE = "keras_model.onnx"

CLASS_INFO = {
    "flood": {"hazard_level": "danger", "road_status": "Blocked"},
    "hazard": {"hazard_level": "danger", "road_status": "Structural damage — avoid area"},
    "fire": {"hazard_level": "danger", "road_status": "Blocked"},
    "collapsed_building": {"hazard_level": "danger", "road_status": "Blocked"},
    "earthquake": {"hazard_level": "danger", "road_status": "Structural damage — avoid area"},
    "smoke": {"hazard_level": "moderate", "road_status": "Reduced visibility — proceed with caution"},
    "landslide": {"hazard_level": "danger", "road_status": "Blocked"},
}


def _session(path):
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------
# Teachable Machine classifier (tiny — always kept in memory)
# ---------------------------------------------------------------------
TM_SESSION = None
TM_CLASS_NAMES = []
try:
    TM_SESSION = _session(MODEL_DIR / TM_FILE)
    with open(MODEL_DIR / "labels.txt", "r") as f:
        TM_CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✓ Teachable Machine classifier loaded: {TM_CLASS_NAMES}")
except Exception as e:  # noqa: BLE001
    print(f"⚠ Teachable Machine model not loaded: {e}")


def models_status():
    return {
        "teachable_machine": TM_SESSION is not None,
        "yolo_files_present": {k: (MODEL_DIR / v).exists() for k, v in YOLO_FILES.items()},
    }


def classify_with_teachable_machine(image_path):
    """Returns (label_lowercase, confidence) or (None, 0.0)."""
    if TM_SESSION is None:
        return None, 0.0
    try:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        arr = (np.asarray(image).astype(np.float32) / 127.5) - 1.0
        data = arr[np.newaxis, ...]  # NHWC (1, 224, 224, 3)

        in_name = TM_SESSION.get_inputs()[0].name
        prediction = TM_SESSION.run(None, {in_name: data})[0]
        index = int(np.argmax(prediction[0]))
        raw_label = TM_CLASS_NAMES[index]
        label = raw_label.split(" ", 1)[-1].strip().lower() if " " in raw_label else raw_label.strip().lower()
        return label, float(prediction[0][index])
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Teachable Machine classify error: {e}")
        traceback.print_exc()
        return None, 0.0


# ---------------------------------------------------------------------
# YOLO (ONNX) helpers
# ---------------------------------------------------------------------
_yolo_cache = {}


def _load_yolo(name):
    if name in _yolo_cache:
        return _yolo_cache[name]
    path = MODEL_DIR / YOLO_FILES[name]
    if not path.exists():
        return None
    sess = _session(path)
    meta = sess.get_modelmeta().custom_metadata_map
    names = {}
    try:
        names = ast.literal_eval(meta.get("names", "{}"))
    except Exception:  # noqa: BLE001
        pass
    entry = {"session": sess, "names": names, "task": meta.get("task", "detect")}
    if KEEP_LOADED:
        _yolo_cache[name] = entry
    return entry


def _letterbox(image, size_hw):
    """Resize keeping aspect ratio, pad with grey (114) to (h, w)."""
    h, w = size_hw
    iw, ih = image.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (w, h), (114, 114, 114))
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _yolo_best(entry, image):
    """Highest-confidence class from one YOLO model -> (label, conf) or (None, 0).

    The original code only kept the single most confident box, so boxes and
    NMS are not needed: the best score over all candidates is the same value.
    """
    sess = entry["session"]
    names = entry["names"]
    inp = sess.get_inputs()[0]
    shape = inp.shape
    h = shape[2] if isinstance(shape[2], int) else 640
    w = shape[3] if isinstance(shape[3], int) else 640

    arr = np.asarray(_letterbox(image, (h, w))).astype(np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    out = sess.run(None, {inp.name: tensor})[0]

    if entry["task"] == "classify":
        scores = out[0]
    else:
        # detect / segment: (1, 4 + nc [+ masks], N)
        nc = len(names) if names else out.shape[1] - 4
        scores = out[0][4:4 + nc, :].max(axis=1)
    idx = int(np.argmax(scores))
    conf = float(scores[idx])
    if conf < YOLO_CONF:
        return None, 0.0
    return str(names.get(idx, idx)).lower(), conf


def analyze_damage_with_yolo(image_path):
    """Returns {damage_type, hazard_level, confidence, road_status}."""
    tm_label, tm_conf = classify_with_teachable_machine(image_path)

    if tm_label == "normal" and tm_conf > 0.6:
        return {"damage_type": "none", "hazard_level": "safe",
                "confidence": round(tm_conf, 2), "road_status": "Clear"}

    try:
        image = Image.open(image_path).convert("RGB")
        best_label, best_conf = None, 0.0

        for name in YOLO_FILES:
            entry = _load_yolo(name)
            if entry is None:
                continue
            try:
                label, conf = _yolo_best(entry, image)
                if label is not None and conf > best_conf:
                    best_label, best_conf = label, conf
            finally:
                del entry
                if not KEEP_LOADED:
                    gc.collect()

        if tm_label and tm_label != "normal" and tm_conf > best_conf:
            best_label, best_conf = tm_label, tm_conf

        if best_label is None:
            return {"damage_type": "none", "hazard_level": "safe",
                    "confidence": 0.0, "road_status": "Clear"}

        info = CLASS_INFO.get(best_label, {"hazard_level": "moderate", "road_status": "Unknown"})
        return {"damage_type": best_label, "hazard_level": info["hazard_level"],
                "confidence": round(best_conf, 2), "road_status": info["road_status"]}

    except Exception as e:  # noqa: BLE001
        print(f"⚠ YOLO error: {e}")
        traceback.print_exc()
        return {"damage_type": "unknown", "hazard_level": "moderate",
                "confidence": 0.5, "road_status": "Unknown"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_inference.py <image.jpg>")
        sys.exit(1)
    print(analyze_damage_with_yolo(sys.argv[1]))