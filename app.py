from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tempfile
import traceback
import bcrypt
import jwt
import functools
import base64
import json

# Load environment variables — find .env next to this script
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"✓ Loaded .env from {dotenv_path}")
else:
    load_dotenv()
    print(f"⚠ .env not found at {dotenv_path}, using system env vars")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Authorization", "Content-Type"], "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]}})

# ========== CONFIGURATION ==========
DB_HOST = os.getenv('DB_HOST', 'localhost').strip()
DB_PORT = int(os.getenv('DB_PORT', '3306').strip() or '3306')
DB_USER = os.getenv('DB_USER', 'root').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD', '').strip()
DB_NAME = os.getenv('DB_NAME', 'defaultdb').strip()
JWT_SECRET = os.getenv('JWT_SECRET', os.urandom(32).hex())
JWT_EXPIRY_HOURS = int(os.getenv('JWT_EXPIRY_HOURS', '24'))

# NOTE: ai_inference (TensorFlow + YOLO) is imported lazily inside the AI
# routes below. Loading it at startup is slow and memory-heavy, which made
# Render's free instance time out ("Port scan timeout"). Now the server
# opens its port immediately and models load on the first AI request.

# ========== FIREBASE ADMIN ==========
import firebase_admin
from firebase_admin import credentials, firestore as admin_firestore

# Accept either variable name, and either raw JSON or base64-encoded JSON.
_firebase_raw = (
    os.getenv('FIREBASE_SERVICE_ACCOUNT_B64')
    or os.getenv('FIREBASE_CREDENTIALS')
    or ''
).strip()

firestore_client = None
if _firebase_raw:
    try:
        if _firebase_raw.startswith('{'):
            service_account_info = json.loads(_firebase_raw, strict=False)
        else:
            service_account_info = json.loads(base64.b64decode(_firebase_raw))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        firestore_client = admin_firestore.client()
        print("✓ Firebase Admin initialized")
    except Exception as e:
        print(f"❌ Firebase Admin init failed: {e}")
else:
    print("⚠ Firebase credentials not set — /api/sync-all-users will be unavailable")

# ========== DATABASE HELPERS ==========
# Aiven requires SSL. The CA cert may be given raw (-----BEGIN CERTIFICATE-----)
# or base64-encoded in DB_SSL_CA_B64. It's written to a temp file at startup.
DB_SSL_CA_B64 = os.getenv('DB_SSL_CA_B64', '').strip()
DB_SSL_CA_PATH = None
if DB_SSL_CA_B64:
    try:
        DB_SSL_CA_PATH = '/tmp/aiven-ca.pem'
        if DB_SSL_CA_B64.startswith('-----BEGIN'):
            ca_bytes = DB_SSL_CA_B64.encode('utf-8')
        else:
            ca_bytes = base64.b64decode(DB_SSL_CA_B64)
        with open(DB_SSL_CA_PATH, 'wb') as f:
            f.write(ca_bytes)
        print("✓ DB SSL CA certificate written for MySQL connection")
    except Exception as e:
        DB_SSL_CA_PATH = None
        print(f"❌ Failed to write DB SSL CA certificate: {e}")

LAST_DB_ERROR = None


def get_db_connection():
    """Get a fresh database connection (SSL-enabled if a CA cert is set)"""
    global LAST_DB_ERROR
    try:
        connect_kwargs = dict(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            autocommit=True,
            connection_timeout=15
        )
        if DB_SSL_CA_PATH:
            connect_kwargs['ssl_ca'] = DB_SSL_CA_PATH
            connect_kwargs['ssl_verify_cert'] = True
        connection = mysql.connector.connect(**connect_kwargs)
        LAST_DB_ERROR = None
        return connection
    except Error as e:
        LAST_DB_ERROR = str(e)
        print(f"❌ Database connection error: {e}")
        return None


def query_db(sql, params=None, fetch=True):
    """Execute a query and return results"""
    conn = get_db_connection()
    if not conn:
        return None
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params or ())
        if fetch:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = cursor.rowcount
        return result
    except Error as e:
        print(f"❌ Database error: {e}")
        traceback.print_exc()
        return None
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


# ========== JWT AUTH ==========
def generate_token(user_id, email):
    """Generate a JWT token for a user"""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def require_auth(f):
    """Decorator to require a valid JWT token"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]

        if not token:
            return jsonify({'error': 'Authentication required'}), 401

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user_id = payload['user_id']
            request.user_email = payload['email']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorated


# ========== ROUTES ==========

@app.route('/')
def home():
    return jsonify({
        "message": "SafeSense Backend is running!",
        "status": "OK",
        "version": "1.0.1",
        "firebase_ready": firestore_client is not None
    })


@app.route('/test-db')
def test_db():
    """Test database connection"""
    connection = get_db_connection()
    if connection and connection.is_connected():
        connection.close()
        return jsonify({
            "message": "Database connected successfully!",
            "status": "OK"
        }), 200
    else:
        return jsonify({
            "message": "Failed to connect to database",
            "status": "ERROR",
            "error": LAST_DB_ERROR,
            "host": DB_HOST,
            "port": DB_PORT,
            "user": DB_USER,
            "database": DB_NAME,
            "ssl_ca_loaded": DB_SSL_CA_PATH is not None
        }), 500


# ========== USER AUTHENTICATION ==========

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login with JWT token"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        user = query_db(
            "SELECT user_id, email, name, password_hash FROM user WHERE email = %s",
            (email,)
        )

        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        stored_hash = user[0]['password_hash']
        is_bcrypt = isinstance(stored_hash, str) and stored_hash.startswith('$2')

        if is_bcrypt:
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode('utf-8')
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return jsonify({"error": "Invalid credentials"}), 401
        else:
            # Legacy plaintext password
            if stored_hash != password:
                return jsonify({"error": "Invalid credentials"}), 401
            new_hash = bcrypt.hashpw(
                password.encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')
            query_db(
                "UPDATE user SET password_hash = %s WHERE user_id = %s",
                (new_hash, user[0]['user_id']),
                fetch=False
            )
            print(f"✓ Auto-hashed password for user {user[0]['email']}")

        token = generate_token(user[0]['user_id'], user[0]['email'])

        return jsonify({
            "status": "success",
            "user": {
                "id": user[0]['user_id'],
                "email": user[0]['email'],
                "name": user[0]['name']
            },
            "token": token
        }), 200
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration with hashed password"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        phone = data.get('phone')

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400

        existing = query_db(
            "SELECT user_id FROM user WHERE email = %s",
            (email,)
        )
        if existing:
            return jsonify({"error": "Email already registered"}), 409

        password_hash = bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        result = query_db(
            "INSERT INTO user (email, password_hash, name, phone) VALUES (%s, %s, %s, %s)",
            (email, password_hash, name, phone or ''),
            fetch=False
        )

        if result and result > 0:
            return jsonify({"status": "success", "message": "User registered"}), 201
        else:
            return jsonify({"error": "Registration failed"}), 400
    except Exception as e:
        print(f"Register error: {e}")
        return jsonify({"error": str(e)}), 500


# ========== FIREBASE -> MYSQL SYNC ==========

@app.route('/api/sync-all-users', methods=['GET', 'POST'])
def sync_all_users():
    """
    Pulls EVERY user from the Firestore `users` collection and upserts them
    into a MySQL `firebase_users` table.

    Protected by a shared secret: /api/sync-all-users?key=YOUR_SECRET
    Set ADMIN_SYNC_KEY in Render's environment.
    """
    if firestore_client is None:
        return jsonify({'error': 'Firebase Admin not configured on server'}), 500

    expected_key = os.getenv('ADMIN_SYNC_KEY', '').strip()
    if not expected_key:
        return jsonify({'error': 'ADMIN_SYNC_KEY is not set on the server'}), 500
    if request.args.get('key', '') != expected_key:
        return jsonify({'error': 'Missing or wrong ?key= parameter'}), 401

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed', 'detail': LAST_DB_ERROR}), 500

    synced = []
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS firebase_users (
                uid VARCHAR(128) PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255),
                phone VARCHAR(50),
                role VARCHAR(50),
                created_at DATETIME NULL,
                last_synced_at DATETIME NOT NULL
            )
        """)

        for doc in firestore_client.collection('users').stream():
            uid = doc.id
            profile = doc.to_dict() or {}

            name = profile.get('name', '')
            email = profile.get('email', '')
            phone = profile.get('phone', '')
            role = profile.get('role', 'user')
            created_at = profile.get('createdAt')
            created_at_sql = (
                created_at.strftime('%Y-%m-%d %H:%M:%S')
                if hasattr(created_at, 'strftime') else None
            )

            cursor.execute("""
                INSERT INTO users (uid, name, email, phone, role, created_at, last_synced_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    email = VALUES(email),
                    phone = VALUES(phone),
                    role = VALUES(role),
                    last_synced_at = NOW()
            """, (uid, name, email, phone, role, created_at_sql))
            synced.append({'uid': uid, 'name': name, 'email': email})

        conn.commit()
        cursor.close()
    except Error as e:
        return jsonify({'error': f'Database error: {e}'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Sync failed: {e}'}), 500
    finally:
        conn.close()

    return jsonify({'synced_count': len(synced), 'users': synced}), 200


# ========== AI ANALYSIS ==========

@app.route('/api/ai/analyze', methods=['POST'])
def analyze_only():
    """
    Runs the detection pipeline and returns the analysis JSON without
    touching MySQL. Flutter uploads the image to Firebase Storage itself,
    calls this for the AI result, and writes the report to Firestore.
    """
    try:
        # Lazy import: models load on first call, not at server startup.
        from ai_inference import analyze_damage_with_yolo

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        image_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                file.save(tmp.name)
                image_path = tmp.name

            analysis = analyze_damage_with_yolo(image_path)
            return jsonify({"status": "success", "analysis": analysis}), 200
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    except Exception as e:
        print(f"🔴 [FATAL] /api/ai/analyze failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========== HAZARD REPORTS (legacy MySQL path) ==========

@app.route('/api/reports/upload', methods=['POST'])
@require_auth
def upload_report():
    """
    DEPRECATED — kept for the old Flask/MySQL path. New code should use
    /api/ai/analyze plus a Firestore write from the client.
    """
    try:
        from ai_inference import analyze_damage_with_yolo

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        latitude = float(request.form.get('latitude', 19.2456))
        longitude = float(request.form.get('longitude', 73.1300))
        user_id = request.user_id

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            file.save(tmp.name)
            image_path = tmp.name

        analysis = analyze_damage_with_yolo(image_path)

        user_description = request.form.get('description', '')
        description = user_description if user_description else f"Auto-detected: {analysis['damage_type']} - {analysis['road_status']}"

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                """
                INSERT INTO disaster (type, severity, description, start_time, status)
                VALUES (%s, %s, %s, NOW(), 'active')
                """,
                (analysis['damage_type'], analysis['hazard_level'], description),
            )
            disaster_id = cursor.lastrowid

            cursor.execute(
                """
                INSERT INTO uploaded_image (user_id, disaster_id, image_url, latitude, longitude, status)
                VALUES (%s, %s, %s, %s, %s, 'approved')
                """,
                (user_id, disaster_id, image_path, latitude, longitude),
            )
            image_id = cursor.lastrowid

            cursor.execute(
                """
                INSERT INTO detection_result (image_id, damage_type, severity, confidence, road_status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (image_id, analysis['damage_type'], analysis['hazard_level'],
                 analysis['confidence'], analysis['road_status']),
            )

            conn.commit()
            img_result = 1
        except Error as e:
            conn.rollback()
            print(f"🔴 [ERROR] Transaction failed: {e}")
            traceback.print_exc()
            img_result = None
        finally:
            if cursor:
                cursor.close()
            conn.close()

        if img_result:
            return jsonify({
                "status": "success",
                "message": "Report submitted and analyzed",
                "analysis": analysis,
                "location": {"lat": latitude, "lng": longitude}
            }), 201
        else:
            return jsonify({"error": "Failed to store report"}), 400

    except Exception as e:
        print(f"🔴 [FATAL] Unhandled exception: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports/hazards', methods=['GET'])
def get_hazards():
    """Get all active hazard reports for the map"""
    try:
        hazards = query_db(
            """
            SELECT dr.detection_id AS id, dr.damage_type, dr.severity AS hazard_level,
                   dr.confidence, dr.road_status, ui.latitude, ui.longitude, dr.detected_at AS created_at
            FROM detection_result dr
            JOIN uploaded_image ui ON dr.image_id = ui.image_id
            WHERE ui.status = 'approved'
            ORDER BY dr.detected_at DESC
            LIMIT 100
            """
        )

        return jsonify({
            "status": "success",
            "hazards": hazards or []
        }), 200
    except Exception as e:
        print(f"Get hazards error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports/user/<int:user_id>', methods=['GET'])
@require_auth
def get_user_reports(user_id):
    """Get reports submitted by a specific user"""
    try:
        reports = query_db(
            """
            SELECT dr.detection_id AS id, dr.damage_type, dr.severity AS hazard_level,
                   dr.confidence, dr.road_status, ui.latitude, ui.longitude, dr.detected_at AS created_at
            FROM detection_result dr
            JOIN uploaded_image ui ON dr.image_id = ui.image_id
            WHERE ui.user_id = %s
            ORDER BY dr.detected_at DESC
            """,
            (user_id,)
        )

        return jsonify({
            "status": "success",
            "reports": reports or []
        }), 200
    except Exception as e:
        print(f"Get user reports error: {e}")
        return jsonify({"error": str(e)}), 500


# ---- If your original app.py had more routes AFTER get_user_reports,
# ---- paste them here (your upload was cut off at that point). ----


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)
