"""
SafeSense AI inference layer.

Everything in this file is pure model-loading and inference — no Flask, no
MySQL, no Firebase. app.py imports analyze_damage_with_yolo() from here and
that's the only thing it needs to know about.

Four models are used, applied in two stages:

  1. keras_model.h5 (a Google Teachable Machine export) — fast 6-class
     classifier (Earthquake, Smoke, Normal, Landslide, Flood, Fire). Runs
     first on every image. If it confidently says "normal" (>60%), the
     pipeline stops there and returns "safe" — this is the common case
     (most uploaded photos aren't actually hazards) and skipping YOLO for
     it is a meaningful speed win.

  2. flood_best.pt, hazard_best.pt, hazard_fire_building_best.pt (three
     separately trained YOLOv8 models) — only run if step 1 didn't
     confidently clear the image. Each model detects a different class of
     hazard (see CLASS_INFO below for exactly which). All three run, and
     whichever single detection (from any of the three, or the Teachable
     Machine result if it beat all three) has the highest confidence wins.

See AI_SETUP.md for how to run this, required packages, and deployment
notes.
"""

import traceback
from pathlib import Path

MODEL_DIR = Path(__file__).parent / 'ai_model'

# What each detectable class means for the app's UI — which model
# produces which label is noted alongside it.
CLASS_INFO = {
    # flood_best.pt
    'flood':               {'hazard_level': 'danger',   'road_status': 'Blocked'},

    # hazard_best.pt — trained specifically on earthquake damage
    'hazard':               {'hazard_level': 'danger',   'road_status': 'Structural damage — avoid area'},

    # hazard_fire_building_best.pt
    'fire':                 {'hazard_level': 'danger',   'road_status': 'Blocked'},
    'collapsed_building':   {'hazard_level': 'danger',   'road_status': 'Blocked'},

    # keras_model.h5 (Teachable Machine) — categories not covered by any YOLO model
    'earthquake':           {'hazard_level': 'danger',   'road_status': 'Structural damage — avoid area'},
    'smoke':                {'hazard_level': 'moderate', 'road_status': 'Reduced visibility — proceed with caution'},
    'landslide':             {'hazard_level': 'danger',  'road_status': 'Blocked'},
}

# ---------------------------------------------------------------------
# Model loading — happens once at import time, not per-request.
# ---------------------------------------------------------------------

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_MODELS = {
        'flood': YOLO(str(MODEL_DIR / 'flood_best.pt')),
        'hazard': YOLO(str(MODEL_DIR / 'hazard_best.pt')),
        'fire_building': YOLO(str(MODEL_DIR / 'hazard_fire_building_best.pt')),
    }
    print(f"✓ YOLO models loaded: {list(YOLO_MODELS.keys())}")
    for name, m in YOLO_MODELS.items():
        print(f"  - {name}: classes = {m.names}")
except ImportError:
    YOLO_AVAILABLE = False
    YOLO_MODELS = {}
    print("⚠ YOLOv8 not installed. Install with: pip install ultralytics opencv-python")
except Exception as e:
    YOLO_AVAILABLE = False
    YOLO_MODELS = {}
    print(f"⚠ YOLO warning: {e}")

try:
    from tensorflow.keras.models import load_model
    from PIL import Image, ImageOps
    import numpy as np

    TM_MODEL = load_model(str(MODEL_DIR / 'keras_model.h5'), compile=False)
    with open(MODEL_DIR / 'labels.txt', 'r') as f:
        TM_CLASS_NAMES = [line.strip() for line in f.readlines()]
    TM_AVAILABLE = True
    print(f"✓ Teachable Machine classifier loaded: {TM_CLASS_NAMES}")
except Exception as e:
    TM_AVAILABLE = False
    print(f"⚠ Teachable Machine model not loaded: {e}")


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def classify_with_teachable_machine(image_path):
    """
    Runs the Teachable Machine Keras classifier on the image.
    Returns (label_lowercase, confidence) or (None, 0.0) if unavailable.
    Labels file lines look like '0 Earthquake', so we strip the index prefix.
    """
    if not TM_AVAILABLE:
        return None, 0.0
    try:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = TM_MODEL.predict(data, verbose=0)
        index = np.argmax(prediction)
        raw_label = TM_CLASS_NAMES[index]
        label = raw_label.split(' ', 1)[-1].strip().lower() if ' ' in raw_label else raw_label.strip().lower()
        confidence = float(prediction[0][index])
        return label, confidence
    except Exception as e:
        print(f"⚠ Teachable Machine classify error: {e}")
        traceback.print_exc()
        return None, 0.0


def analyze_damage_with_yolo(image_path):
    """
    Fast first pass: Teachable Machine classifier.
    If it confidently says "normal", skip YOLO entirely and return safe
    (big speed win for the common case where nothing's actually wrong).
    Otherwise, run all trained YOLO models too and keep whichever single
    result (YOLO or Teachable Machine) is more confident.

    Returns a dict: {damage_type, hazard_level, confidence, road_status}
    — this exact shape is what Flutter's ResultPage and FirestoreService
    expect; don't rename these keys without updating both.
    """
    tm_label, tm_conf = classify_with_teachable_machine(image_path)

    if tm_label == 'normal' and tm_conf > 0.6:
        return {
            'damage_type': 'none',
            'hazard_level': 'safe',
            'confidence': round(tm_conf, 2),
            'road_status': 'Clear'
        }

    if not YOLO_AVAILABLE:
        if tm_label:
            info = CLASS_INFO.get(tm_label, {'hazard_level': 'moderate', 'road_status': 'Unknown'})
            return {
                'damage_type': tm_label,
                'hazard_level': info['hazard_level'],
                'confidence': round(tm_conf, 2),
                'road_status': info['road_status']
            }
        return {
            'damage_type': 'flood',
            'hazard_level': 'moderate',
            'confidence': 0.65,
            'road_status': 'Partially blocked'
        }

    try:
        best_label = None
        best_conf = 0.0

        for model_name, model in YOLO_MODELS.items():
            results = model.predict(image_path, conf=0.4, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    if conf > best_conf:
                        best_conf = conf
                        best_label = label.lower()

        if tm_label and tm_label != 'normal' and tm_conf > best_conf:
            best_conf = tm_conf
            best_label = tm_label

        if best_label is None:
            return {
                'damage_type': 'none',
                'hazard_level': 'safe',
                'confidence': 0.0,
                'road_status': 'Clear'
            }

        info = CLASS_INFO.get(best_label, {'hazard_level': 'moderate', 'road_status': 'Unknown'})

        return {
            'damage_type': best_label,
            'hazard_level': info['hazard_level'],
            'confidence': round(best_conf, 2),
            'road_status': info['road_status']
        }

    except Exception as e:
        print(f"⚠ YOLO error: {e}")
        traceback.print_exc()
        return {
            'damage_type': 'unknown',
            'hazard_level': 'moderate',
            'confidence': 0.5,
            'road_status': 'Unknown'
        }
