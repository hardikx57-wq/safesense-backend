"""
SafeSense AI inference — ONNX Runtime version (no TensorFlow, no PyTorch).

Pipeline:
  1. keras_model.onnx
     Teachable Machine classifier with 6 classes.
     If it is >60% sure the photo is "normal", return "safe".

  2. YOLOv8 ONNX models:
       - flood
       - hazard
       - fire_building

     All available models are run and the most confident
     detection wins.

Memory:
  Free hosts have ~512 MB, so YOLO models are loaded one at
  a time and released afterward.

Set KEEP_MODELS_LOADED=1 if the host has enough RAM.
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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(__file__).parent / "ai_model"

KEEP_LOADED = os.getenv("KEEP_MODELS_LOADED", "0") == "1"

YOLO_CONF = 0.4

YOLO_FILES = {
    "flood": "flood_best.onnx",
    "hazard": "hazard_best.onnx",
    "fire_building": "hazard_fire_building_best.onnx",
}

TM_FILE = "keras_model.onnx"


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_INFO = {
    "flood": {
        "hazard_level": "danger",
        "road_status": "Blocked"
    },

    "hazard": {
        "hazard_level": "danger",
        "road_status": "Structural damage — avoid area"
    },

    "fire": {
        "hazard_level": "danger",
        "road_status": "Blocked"
    },

    "collapsed_building": {
        "hazard_level": "danger",
        "road_status": "Blocked"
    },

    "earthquake": {
        "hazard_level": "danger",
        "road_status": "Structural damage — avoid area"
    },

    "smoke": {
        "hazard_level": "moderate",
        "road_status": "Reduced visibility — proceed with caution"
    },

    "landslide": {
        "hazard_level": "danger",
        "road_status": "Blocked"
    },
}


# ============================================================
# ONNX SESSION
# ============================================================

def _session(path):
    """
    Create a lightweight ONNX Runtime CPU session.
    """

    so = ort.SessionOptions()

    # Reduce memory usage on free/small hosts
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False

    # Render free instances have limited CPU
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(path),
        so,
        providers=["CPUExecutionProvider"]
    )


# ============================================================
# TEACHABLE MACHINE CLASSIFIER
# ============================================================

TM_SESSION = None
TM_CLASS_NAMES = []

try:

    TM_SESSION = _session(MODEL_DIR / TM_FILE)

    with open(MODEL_DIR / "labels.txt", "r") as f:
        TM_CLASS_NAMES = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    print(
        f"✓ Teachable Machine classifier loaded: "
        f"{TM_CLASS_NAMES}"
    )

except Exception as e:

    print(
        f"⚠ Teachable Machine model not loaded: {e}"
    )


# ============================================================
# MODEL AVAILABILITY FLAGS
# ============================================================
#
# IMPORTANT:
# app.py imports these two variables:
#
# from ai_inference import (
#     analyze_damage_with_yolo,
#     YOLO_AVAILABLE,
#     TM_AVAILABLE
# )
#
# They were missing previously and caused the Render
# ImportError.
# ============================================================

TM_AVAILABLE = TM_SESSION is not None

YOLO_AVAILABLE = any(
    (MODEL_DIR / filename).exists()
    for filename in YOLO_FILES.values()
)

print(f"✓ YOLO files available: {YOLO_AVAILABLE}")
print(f"✓ Teachable Machine available: {TM_AVAILABLE}")


# ============================================================
# MODEL STATUS
# ============================================================

def models_status():
    """
    Return the availability of the AI models.
    """

    return {
        "teachable_machine": TM_SESSION is not None,

        "yolo_files_present": {
            name: (MODEL_DIR / filename).exists()
            for name, filename in YOLO_FILES.items()
        },
    }


# ============================================================
# TEACHABLE MACHINE CLASSIFICATION
# ============================================================

def classify_with_teachable_machine(image_path):
    """
    Returns:

        (label, confidence)

    or:

        (None, 0.0)
    """

    if TM_SESSION is None:
        return None, 0.0

    try:

        image = Image.open(image_path).convert("RGB")

        image = ImageOps.fit(
            image,
            (224, 224),
            Image.Resampling.LANCZOS
        )

        # Teachable Machine expects values from -1 to 1
        arr = (
            np.asarray(image).astype(np.float32) / 127.5
        ) - 1.0

        # NHWC
        data = arr[np.newaxis, ...]

        in_name = TM_SESSION.get_inputs()[0].name

        prediction = TM_SESSION.run(
            None,
            {
                in_name: data
            }
        )[0]

        index = int(
            np.argmax(prediction[0])
        )

        raw_label = TM_CLASS_NAMES[index]

        if " " in raw_label:
            label = raw_label.split(
                " ",
                1
            )[-1].strip().lower()
        else:
            label = raw_label.strip().lower()

        confidence = float(
            prediction[0][index]
        )

        return label, confidence

    except Exception as e:

        print(
            f"⚠ Teachable Machine classify error: {e}"
        )

        traceback.print_exc()

        return None, 0.0


# ============================================================
# YOLO
# ============================================================

_yolo_cache = {}


def _load_yolo(name):
    """
    Load a YOLO ONNX model.

    If KEEP_LOADED is false, the model is not kept
    permanently in the cache.
    """

    if name in _yolo_cache:
        return _yolo_cache[name]

    path = MODEL_DIR / YOLO_FILES[name]

    if not path.exists():
        print(
            f"⚠ YOLO model not found: {path}"
        )
        return None

    try:

        sess = _session(path)

        meta = sess.get_modelmeta().custom_metadata_map

        names = {}

        try:

            names = ast.literal_eval(
                meta.get("names", "{}")
            )

        except Exception:
            pass

        entry = {
            "session": sess,
            "names": names,
            "task": meta.get(
                "task",
                "detect"
            )
        }

        if KEEP_LOADED:
            _yolo_cache[name] = entry

        return entry

    except Exception as e:

        print(
            f"⚠ Could not load YOLO model "
            f"{name}: {e}"
        )

        return None


# ============================================================
# LETTERBOX
# ============================================================

def _letterbox(image, size_hw):
    """
    Resize image while maintaining aspect ratio
    and pad with grey pixels.
    """

    h, w = size_hw

    iw, ih = image.size

    scale = min(
        w / iw,
        h / ih
    )

    nw = max(
        1,
        int(round(iw * scale))
    )

    nh = max(
        1,
        int(round(ih * scale))
    )

    resized = image.resize(
        (nw, nh),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (w, h),
        (114, 114, 114)
    )

    canvas.paste(
        resized,
        (
            (w - nw) // 2,
            (h - nh) // 2
        )
    )

    return canvas


# ============================================================
# YOLO BEST DETECTION
# ============================================================

def _yolo_best(entry, image):
    """
    Get the highest-confidence class from one YOLO model.

    Returns:

        (label, confidence)

    or:

        (None, 0.0)
    """

    sess = entry["session"]

    names = entry["names"]

    inp = sess.get_inputs()[0]

    shape = inp.shape

    # Default YOLO input size
    h = (
        shape[2]
        if isinstance(shape[2], int)
        else 640
    )

    w = (
        shape[3]
        if isinstance(shape[3], int)
        else 640
    )

    # Resize and normalize
    arr = np.asarray(
        _letterbox(
            image,
            (h, w)
        )
    ).astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    tensor = np.transpose(
        arr,
        (2, 0, 1)
    )[np.newaxis, ...]

    # Run inference
    out = sess.run(
        None,
        {
            inp.name: tensor
        }
    )[0]

    # Classification model
    if entry["task"] == "classify":

        scores = out[0]

    else:

        # Detection / segmentation
        nc = (
            len(names)
            if names
            else out.shape[1] - 4
        )

        scores = out[0][
            4:4 + nc,
            :
        ].max(axis=1)

    idx = int(
        np.argmax(scores)
    )

    conf = float(
        scores[idx]
    )

    if conf < YOLO_CONF:
        return None, 0.0

    return (
        str(
            names.get(
                idx,
                idx
            )
        ).lower(),
        conf
    )


# ============================================================
# MAIN AI ANALYSIS
# ============================================================

def analyze_damage_with_yolo(image_path):
    """
    Run the complete SafeSense AI pipeline.

    Returns:

    {
        "damage_type": "...",
        "hazard_level": "...",
        "confidence": 0.0,
        "road_status": "..."
    }
    """

    # --------------------------------------------------------
    # STEP 1: Teachable Machine
    # --------------------------------------------------------

    tm_label, tm_conf = (
        classify_with_teachable_machine(
            image_path
        )
    )

    # If the image is confidently normal,
    # immediately classify it as safe.
    if (
        tm_label == "normal"
        and tm_conf > 0.6
    ):

        return {
            "damage_type": "none",
            "hazard_level": "safe",
            "confidence": round(
                tm_conf,
                2
            ),
            "road_status": "Clear"
        }

    # --------------------------------------------------------
    # STEP 2: YOLO MODELS
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        best_label = None
        best_conf = 0.0

        # Run all YOLO models
        for name in YOLO_FILES:

            entry = _load_yolo(name)

            if entry is None:
                continue

            try:

                label, conf = _yolo_best(
                    entry,
                    image
                )

                if (
                    label is not None
                    and conf > best_conf
                ):

                    best_label = label
                    best_conf = conf

            finally:

                del entry

                # Free memory on small hosts
                if not KEEP_LOADED:
                    gc.collect()

        # ----------------------------------------------------
        # STEP 3: Compare Teachable Machine result
        # ----------------------------------------------------

        if (
            tm_label
            and tm_label != "normal"
            and tm_conf > best_conf
        ):

            best_label = tm_label
            best_conf = tm_conf

        # ----------------------------------------------------
        # No detection
        # ----------------------------------------------------

        if best_label is None:

            return {
                "damage_type": "none",
                "hazard_level": "safe",
                "confidence": 0.0,
                "road_status": "Clear"
            }

        # ----------------------------------------------------
        # Get hazard information
        # ----------------------------------------------------

        info = CLASS_INFO.get(
            best_label,
            {
                "hazard_level": "moderate",
                "road_status": "Unknown"
            }
        )

        return {
            "damage_type": best_label,

            "hazard_level": info[
                "hazard_level"
            ],

            "confidence": round(
                best_conf,
                2
            ),

            "road_status": info[
                "road_status"
            ]
        }

    except Exception as e:

        print(
            f"⚠ YOLO error: {e}"
        )

        traceback.print_exc()

        return {
            "damage_type": "unknown",
            "hazard_level": "moderate",
            "confidence": 0.5,
            "road_status": "Unknown"
        }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python ai_inference.py <image.jpg>"
        )

        sys.exit(1)

    result = analyze_damage_with_yolo(
        sys.argv[1]
    )

    print(result)
