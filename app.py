"""
SafeSense AI service (slim build for small hosts such as Render free).

Only one job: receive a photo, run the hazard-detection models and return
the analysis JSON. Login, database and reports all live in Firebase now,
so MySQL / JWT / bcrypt were removed.

Endpoints
  GET  /                -> health check
  POST /api/ai/analyze  -> multipart form field "image" -> {status, analysis}
"""

import os
import tempfile
import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS

from ai_inference import analyze_damage_with_yolo, models_status

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Reject absurdly large uploads (12 MB).
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


@app.route("/")
def home():
    return jsonify({"message": "SafeSense AI is running", "status": "OK",
                    "models": models_status()})


@app.route("/api/ai/analyze", methods=["POST"])
def analyze_only():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        image_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                file.save(tmp.name)
                image_path = tmp.name
            analysis = analyze_damage_with_yolo(image_path)
            return jsonify({"status": "success", "analysis": analysis}), 200
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] /api/ai/analyze failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)