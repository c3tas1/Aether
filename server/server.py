from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import base64
import datetime
import zipfile
import json
import threading
import time

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
UPLOAD_FOLDER = "uploads/"
ANNOTATIONS_DIR = "annotations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DECOMPRESSION_PROGRESS = {} # Global dict to track progress

# ---------- YOLO HELPERS (No changes) ----------
def load_yolo_annotations(filename, image_width, image_height):
    # ... (code from previous step, no changes needed)
    pass
def save_yolo_annotations(filename, boxes, image_width, image_height):
    # ... (code from previous step, no changes needed)
    pass

# ---------- SQLite Helpers ----------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False) # Allow multi-thread access
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, type TEXT,
                description TEXT, status TEXT DEFAULT 'pending', file_tree TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, original_name TEXT NOT NULL,
                path TEXT NOT NULL, status TEXT DEFAULT '', annotations TEXT DEFAULT '',
                dataset_id INTEGER REFERENCES datasets(id)
            );""")
        # Add columns if they don't exist (safe to re-run)
        cursor = conn.execute("PRAGMA table_info(datasets)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'status' not in columns:
            conn.execute("ALTER TABLE datasets ADD COLUMN status TEXT DEFAULT 'pending'")
        if 'file_tree' not in columns:
            conn.execute("ALTER TABLE datasets ADD COLUMN file_tree TEXT")
        conn.commit()

# ---------- Decompression Worker ----------
def decompress_zip_and_build_tree(zip_path, extract_dir, dataset_id):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.infolist()
            total_files = len(file_list)
            DECOMPRESSION_PROGRESS[dataset_id] = {'progress': 0, 'status': 'decompressing'}

            for i, member in enumerate(file_list):
                zip_ref.extract(member, extract_dir)
                progress = int(((i + 1) / total_files) * 100)
                DECOMPRESSION_PROGRESS[dataset_id]['progress'] = progress

        # Build file tree
        def path_to_dict(path):
            d = {'name': os.path.basename(path)}
            if os.path.isdir(path):
                d['type'] = 'folder'
                d['children'] = [path_to_dict(os.path.join(path, x)) for x in os.listdir(path)]
            else:
                d['type'] = 'file'
            return d
        
        file_tree = path_to_dict(extract_dir)
        
        # Update database
        with get_db_connection() as conn:
            conn.execute("UPDATE datasets SET status = ?, file_tree = ? WHERE id = ?",
                         ('complete', json.dumps(file_tree), dataset_id))
            conn.commit()
            
    except Exception as e:
        print(f"Decompression failed for dataset {dataset_id}: {e}")
        with get_db_connection() as conn:
            conn.execute("UPDATE datasets SET status = ? WHERE id = ?", ('failed', dataset_id))
            conn.commit()
    finally:
        # Clean up progress tracking and the original zip file
        if dataset_id in DECOMPRESSION_PROGRESS:
            del DECOMPRESSION_PROGRESS[dataset_id]
        os.remove(zip_path)


# ---------- FLASK SETUP ----------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- API Endpoints ----------
@app.route("/api/upload", methods=["POST"])
def upload_images():
    if "images" not in request.files:
        return jsonify({"error": "No image files uploaded"}), 400
    
    file_obj = request.files.getlist("images")[0] # Assuming one zip file
    if not file_obj.filename.lower().endswith(".zip"):
        return jsonify({"error": "Only zip files are supported for this endpoint"}), 400

    dataset_name = request.form.get("datasetName")
    dataset_type = request.form.get("datasetType", "")
    dataset_description = request.form.get("datasetDescription", "")
    
    with get_db_connection() as conn:
        try:
            cur = conn.execute("INSERT INTO datasets (name, type, description, status) VALUES (?, ?, ?, ?)",
                               (dataset_name, dataset_type, dataset_description, 'decompressing'))
            dataset_id = cur.lastrowid
            conn.commit()

            # Save the zip file temporarily
            timestamp = int(datetime.datetime.now().timestamp())
            zip_filename = f"{timestamp}-{dataset_id}.zip"
            zip_path = os.path.join(app.config["UPLOAD_FOLDER"], zip_filename)
            file_obj.save(zip_path)

            # Define extract directory
            extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(dataset_id))
            os.makedirs(extract_dir, exist_ok=True)

            # Start decompression in a background thread
            thread = threading.Thread(target=decompress_zip_and_build_tree, args=(zip_path, extract_dir, dataset_id))
            thread.start()

            return jsonify({"message": "Upload received, starting decompression.", "dataset_id": dataset_id})
        except Exception as e:
            conn.rollback()
            return jsonify({"error": str(e)}), 500

@app.route("/api/datasets/<int:dataset_id>/status", methods=["GET"])
def get_dataset_status(dataset_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT status FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not row:
            return jsonify({"error": "Dataset not found"}), 404
        
        status = row['status']
        progress = 0
        if status == 'decompressing':
            progress = DECOMPRESSION_PROGRESS.get(dataset_id, {}).get('progress', 0)
        
        return jsonify({"status": status, "progress": progress})

@app.route("/api/datasets/<int:dataset_id>/file_tree", methods=["GET"])
def get_dataset_file_tree(dataset_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT file_tree FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not row or not row['file_tree']:
            return jsonify({"error": "File tree not available"}), 404
        return jsonify(json.loads(row['file_tree']))

# ... (All other endpoints like /api/images/*, /api/annotations/* remain)

# ---------- Init & Run ----------
if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)