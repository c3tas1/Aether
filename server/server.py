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
import re

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
UPLOAD_FOLDER = "uploads/"
ANNOTATIONS_DIR = "annotations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DECOMPRESSION_PROGRESS = {} 

# ---------- YOLO HELPERS ----------
def load_yolo_annotations(filename, image_width, image_height):
    txt_path = os.path.join(ANNOTATIONS_DIR, f"{filename}.txt")
    if not os.path.exists(txt_path):
        return []
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 5: continue
            class_id, x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts)
            x_center, y_center = x_center_norm * image_width, y_center_norm * image_height
            w, h = w_norm * image_width, h_norm * image_height
            x, y = x_center - w/2, y_center - h/2
            boxes.append({"classId": int(class_id), "x": x, "y": y, "w": w, "h": h})
    return boxes

def save_yolo_annotations(filename, boxes, image_width, image_height):
    txt_path = os.path.join(ANNOTATIONS_DIR, f"{filename}.txt")
    lines = []
    for b in boxes:
        x_center, y_center = b["x"] + b["w"]/2, b["y"] + b["h"]/2
        x_center_n, y_center_n = x_center / image_width, y_center / image_height
        w_n, h_n = b["w"] / image_width, b["h"] / image_height
        lines.append(f"{b['classId']} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

# ---------- SQLite Helpers ----------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, type TEXT,
                description TEXT, status TEXT DEFAULT 'pending', file_tree TEXT,
                thumbnail_path TEXT, storage_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, original_name TEXT NOT NULL,
                path TEXT NOT NULL, status TEXT DEFAULT '', annotations TEXT DEFAULT '',
                dataset_id INTEGER REFERENCES datasets(id)
            );""")
        cursor = conn.execute("PRAGMA table_info(datasets)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'status' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN status TEXT DEFAULT 'pending'")
        if 'file_tree' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN file_tree TEXT")
        if 'thumbnail_path' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN thumbnail_path TEXT")
        if 'storage_path' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN storage_path TEXT")
        conn.commit()

# ---------- Decompression Worker ----------
def slugify(text):
    text = text.lower()
    text = re.sub(r'[\s\W-]+', '-', text)
    return text.strip('-')

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
        
        def path_to_dict(root_path, current_path):
            d = {'name': os.path.basename(current_path)}
            relative_path = os.path.relpath(current_path, root_path)
            d['path'] = '' if relative_path == '.' else relative_path.replace('\\', '/')
            if os.path.isdir(current_path):
                d['type'] = 'folder'
                d['children'] = sorted([path_to_dict(root_path, os.path.join(current_path, x)) for x in os.listdir(current_path)], key=lambda x: x['name'])
            else:
                d['type'] = 'file'
            return d
        
        file_tree = path_to_dict(extract_dir, extract_dir)
        
        with get_db_connection() as conn:
            conn.execute("UPDATE datasets SET status = ?, file_tree = ? WHERE id = ?", ('complete', json.dumps(file_tree), dataset_id))
            conn.commit()
            
    except Exception as e:
        print(f"Decompression failed for dataset {dataset_id}: {e}")
        with get_db_connection() as conn:
            conn.execute("UPDATE datasets SET status = ? WHERE id = ?", ('failed', dataset_id))
            conn.commit()
    finally:
        if dataset_id in DECOMPRESSION_PROGRESS: del DECOMPRESSION_PROGRESS[dataset_id]
        if os.path.exists(zip_path): os.remove(zip_path)

# ---------- FLASK SETUP ----------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- API Endpoints ----------
@app.route("/api/upload", methods=["POST"])
def upload_images():
    if "images" not in request.files: return jsonify({"error": "No image files uploaded"}), 400
    file_obj = request.files.getlist("images")[0]
    if not file_obj.filename.lower().endswith(".zip"): return jsonify({"error": "Only zip files are supported"}), 400
    dataset_name = request.form.get("datasetName")
    if not dataset_name: return jsonify({"error": "Dataset Name is required"}), 400
    dataset_type = request.form.get("datasetType", "")
    dataset_description = request.form.get("datasetDescription", "")
    thumbnail_file = request.files.get("thumbnail")

    base_slug = slugify(dataset_name)
    storage_name = base_slug
    counter = 1
    while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], storage_name)):
        storage_name = f"{base_slug}-{counter}"
        counter += 1
    extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], storage_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    thumbnail_path_db = None
    if thumbnail_file and allowed_file(thumbnail_file.filename):
        ext = os.path.splitext(thumbnail_file.filename)[1]
        thumbnail_filename = f"_thumbnail{ext}"
        thumbnail_save_path = os.path.join(extract_dir, thumbnail_filename)
        thumbnail_file.save(thumbnail_save_path)
        thumbnail_path_db = os.path.join(storage_name, thumbnail_filename)

    with get_db_connection() as conn:
        try:
            cur = conn.execute("INSERT INTO datasets (name, type, description, status, thumbnail_path, storage_path) VALUES (?, ?, ?, ?, ?, ?)",
                               (dataset_name, dataset_type, dataset_description, 'decompressing', thumbnail_path_db, extract_dir))
            dataset_id = cur.lastrowid
            conn.commit()
            zip_filename = f"{dataset_id}-{slugify(file_obj.filename)}.zip"
            zip_path = os.path.join(extract_dir, zip_filename)
            file_obj.save(zip_path)
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
        if not row: return jsonify({"error": "Dataset not found"}), 404
        status = row['status']
        progress = DECOMPRESSION_PROGRESS.get(dataset_id, {}).get('progress', 0) if status == 'decompressing' else 0
        return jsonify({"status": status, "progress": progress})

@app.route("/api/datasets/<int:dataset_id>/file_tree", methods=["GET"])
def get_dataset_file_tree(dataset_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT file_tree FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not row or not row['file_tree']: return jsonify({"error": "File tree not available"}), 404
        return jsonify(json.loads(row['file_tree']))

@app.route("/api/datasets/<int:dataset_id>/metadata", methods=["PUT"])
def save_dataset_metadata(dataset_id):
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No data provided"}), 400
        dataset_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(dataset_id))
        if not os.path.isdir(dataset_dir): return jsonify({"error": "Dataset directory not found"}), 404
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=4)
        return jsonify({"message": "Metadata saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/datasets/<int:dataset_id>/preview", methods=["GET"])
def preview_dataset_file(dataset_id):
    file_path_relative = request.args.get('path')
    if not file_path_relative: return jsonify({"error": "File path is required"}), 400
    try:
        base_dir = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], str(dataset_id)))
        full_path = os.path.abspath(os.path.join(base_dir, file_path_relative))
        if not full_path.startswith(base_dir): return jsonify({"error": "Access denied"}), 403
        if not os.path.isfile(full_path): return jsonify({"error": "File not found"}), 404
        file_ext = os.path.splitext(full_path)[1].lower()
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
            with open(full_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            return jsonify({'type': 'image', 'content': b64_str})
        elif file_ext == '.txt':
            with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({'type': 'text', 'content': content})
        else: return jsonify({"error": f"Preview not supported for {file_ext} files"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ... (Other endpoints like /api/images/* and /api/annotations/* remain)

if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)