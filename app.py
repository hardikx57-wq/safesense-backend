"""
SafeSense AI service & Firebase-to-Aiven MySQL Sync.

Endpoints:
  GET  /                     -> Health check & AI model status
  POST /api/ai/analyze       -> Multipart form field "image" -> {status, analysis}
  GET/POST /api/sync-all-users?key=<ADMIN_SYNC_KEY> -> Syncs Firebase users to Aiven MySQL
"""

import os
import tempfile
import traceback

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql

import json

from ai_inference import analyze_damage_with_yolo, models_status



# ==============================================================================
# 1. APPLICATION & CONFIGURATION SETUP
# ==============================================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Reject uploads larger than 12 MB
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# Security key required to trigger sync endpoint
ADMIN_SYNC_KEY = os.getenv("ADMIN_SYNC_KEY", "mySecret123")

# Aiven MySQL database connection settings
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql-1485c151-hardik-a95f.i.aivencloud.com")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 27721))
MYSQL_USER = os.getenv("MYSQL_USER", "avnadmin")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "AVNS_ArvR_18f3fxCZ6LWLRM")
MYSQL_DB = os.getenv("MYSQL_DB", "defaultdb")


# ==============================================================================
# 2. FIREBASE ADMIN SDK INITIALIZATION
# ==============================================================================
def init_firebase():
    if not firebase_admin._apps:
        # Check for JSON content directly in environment variable
        firebase_json_env = os.getenv("FIREBASE_CREDENTIALS_JSON")
        
        if firebase_json_env:
            cred_dict = json.loads(firebase_json_env)
            cred = credentials.Certificate(cred_dict)
        else:
            # Fallback to local JSON file if present
            json_files = [f for f in os.listdir(".") if f.endswith(".json") and "firebase" in f]
            if json_files:
                cred = credentials.Certificate(json_files[0])
            else:
                cred = credentials.ApplicationDefault()
                
        firebase_admin.initialize_app(cred)

try:
    init_firebase()
except Exception as firebase_err:
    print(f"[WARNING] Firebase SDK initialization failed: {firebase_err}")


# Helper to acquire an SSL-secured MySQL connection to Aiven
def get_mysql_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        ssl={"ssl": True},  # Enables SSL required by Aiven
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )


# ==============================================================================
# 3. ROUTE DEFINITIONS
# ==============================================================================

@app.route("/")
def home():
    """Health check endpoint showing AI model statuses."""
    return jsonify({
        "message": "SafeSense AI and Sync Service is running",
        "status": "OK",
        "models": models_status()
    })


@app.route("/api/sync-all-users", methods=["GET", "POST"])
def sync_all_users():
    """Syncs all users from Firebase Firestore to Aiven MySQL."""
    # Step 1: Key Validation
    provided_key = request.args.get("key")
    if provided_key != ADMIN_SYNC_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing ADMIN_SYNC_KEY"}), 401

    try:
        # Step 2: Extract data from Firestore 'users' collection
        db_firestore = firestore.client()
        users_ref = db_firestore.collection("users")
        docs = users_ref.stream()

        synced_count = 0
        connection = get_mysql_connection()

        with connection.cursor() as cursor:
            # Ensure the target table exists in Aiven MySQL
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(128) PRIMARY KEY,
                    name VARCHAR(255),
                    email VARCHAR(255),
                    role VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                );
            """)

            # Step 3: Loop through Firestore documents and Upsert into MySQL
            for doc in docs:
                data = doc.to_dict()
                user_id = doc.id
                name = data.get("name") or data.get("displayName") or ""
                email = data.get("email") or ""
                role = data.get("role") or "user"

                sql_upsert = """
                    INSERT INTO users (id, name, email, role)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        email = VALUES(email),
                        role = VALUES(role);
                """
                cursor.execute(sql_upsert, (user_id, name, email, role))
                synced_count += 1

        connection.close()

        return jsonify({
            "status": "success",
            "message": f"Successfully synced {synced_count} user(s) from Firebase to Aiven MySQL.",
            "total_synced": synced_count
        }), 200

    except Exception as e:
        print(f"[FATAL] Sync failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/analyze", methods=["POST"])
def analyze_only():
    """Runs hazard analysis on an uploaded image."""
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

    except Exception as e:
        print(f"[FATAL] /api/ai/analyze failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# 4. SERVER RUNNER
# ==============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
