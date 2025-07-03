from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import base64
import datetime
import zipfile
import json
import threading
import re

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
DECOMPRESSION_PROGRESS = {} 

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
                description TEXT, class_names TEXT, status TEXT DEFAULT 'pending', file_tree TEXT,
                thumbnail_path TEXT, storage_path TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER NOT NULL,
                image_path TEXT NOT NULL, annotation_path TEXT, image_type TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            );""")
        cursor = conn.execute("PRAGMA table_info(datasets)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'class_names' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN class_names TEXT")
        if 'thumbnail_path' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN thumbnail_path TEXT")
        if 'storage_path' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN storage_path TEXT UNIQUE")
        conn.commit()

# ---------- Decompression Worker & Helpers ----------
def slugify(text):
    text = text.lower()
    text = re.sub(r'[\s\W-]+', '-', text)
    return text.strip('-')

def decompress_zip_and_build_tree(zip_path, extract_dir, dataset_id):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = [f for f in zip_ref.infolist() if not f.is_dir()]
            total_files = len(file_list)
            DECOMPRESSION_PROGRESS[dataset_id] = {'progress': 0, 'status': 'decompressing'}
            for i, member in enumerate(file_list):
                zip_ref.extract(member, extract_dir)
                progress = int(((i + 1) / total_files) * 100) if total_files > 0 else 100
                DECOMPRESSION_PROGRESS[dataset_id]['progress'] = progress
        
        extracted_items = os.listdir(extract_dir)
        effective_root = extract_dir
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
            effective_root = os.path.join(extract_dir, extracted_items[0])

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
        
        file_tree = path_to_dict(effective_root, effective_root)
        
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
def allowed_file(filename, extensions): return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

# ---------- API Endpoints ----------
@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    if "images" not in request.files: return jsonify({"error": "No file part"}), 400
    file_obj = request.files.get("images")
    if not file_obj or file_obj.filename == '': return jsonify({"error": "No selected file"}), 400
    if not file_obj.filename.lower().endswith(".zip"): return jsonify({"error": "Only zip files are supported"}), 400
    dataset_name = request.form.get("datasetName")
    if not dataset_name: return jsonify({"error": "Dataset Name is required"}), 400
    
    dataset_type, dataset_description, class_names = request.form.get("datasetType", ""), request.form.get("datasetDescription", ""), request.form.get("classNames", "")
    thumbnail_file = request.files.get("thumbnail")

    base_slug = slugify(dataset_name)
    storage_name, counter = base_slug, 1
    while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], storage_name)):
        storage_name = f"{base_slug}-{counter}"; counter += 1
    
    extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], storage_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    thumbnail_path_db = None
    if thumbnail_file and allowed_file(thumbnail_file.filename, ALLOWED_IMG_EXTENSIONS):
        ext = os.path.splitext(thumbnail_file.filename)[1]
        thumbnail_filename = f"_thumbnail{ext}"
        thumbnail_file.save(os.path.join(extract_dir, thumbnail_filename))
        thumbnail_path_db = os.path.join(storage_name, thumbnail_filename).replace('\\', '/')

    with get_db_connection() as conn:
        try:
            cur = conn.execute("INSERT INTO datasets (name, type, description, class_names, status, thumbnail_path, storage_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (dataset_name, dataset_type, dataset_description, class_names, 'decompressing', thumbnail_path_db, storage_name))
            dataset_id = cur.lastrowid
            conn.commit()
            zip_path = os.path.join(extract_dir, f"{dataset_id}.zip")
            file_obj.save(zip_path)
            thread = threading.Thread(target=decompress_zip_and_build_tree, args=(zip_path, extract_dir, dataset_id))
            thread.start()
            return jsonify({"message": "Upload received, starting decompression.", "dataset_id": dataset_id, "storage_path": storage_name})
        except Exception as e:
            conn.rollback()
            return jsonify({"error": str(e)}), 500

# MODIFIED: Route now uses the unique storage_path string
@app.route("/api/datasets/<string:storage_path>/status", methods=["GET"])
def get_dataset_status(storage_path):
    with get_db_connection() as conn:
        row = conn.execute("SELECT id, status FROM datasets WHERE storage_path = ?", (storage_path,)).fetchone()
        if not row: return jsonify({"error": "Dataset not found"}), 404
        dataset_id = row['id']
        status = row['status']
        progress = DECOMPRESSION_PROGRESS.get(dataset_id, {}).get('progress', 0) if status == 'decompressing' else 0
        return jsonify({"status": status, "progress": progress})

@app.route("/api/datasets/<string:storage_path>/file_tree", methods=["GET"])
def get_dataset_file_tree(storage_path):
    with get_db_connection() as conn:
        row = conn.execute("SELECT file_tree FROM datasets WHERE storage_path = ?", (storage_path,)).fetchone()
        if not row or not row['file_tree']: return jsonify({"error": "File tree not available"}), 404
        return jsonify(json.loads(row['file_tree']))

@app.route("/api/datasets/<string:storage_path>/metadata", methods=["PUT"])
def save_dataset_metadata(storage_path):
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No data provided"}), 400
        dataset_dir = os.path.join(app.config["UPLOAD_FOLDER"], storage_path)
        if not os.path.isdir(dataset_dir): return jsonify({"error": "Dataset directory not found"}), 404
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=4)
        # Future logic to populate the images table can go here
        return jsonify({"message": "Metadata saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/datasets/<string:storage_path>/preview", methods=["GET"])
def preview_dataset_file(storage_path):
    file_path_relative = request.args.get('path')
    if not file_path_relative: return jsonify({"error": "File path is required"}), 400
    try:
        base_dir = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], storage_path))
        extracted_items = os.listdir(base_dir)
        effective_base_dir = base_dir
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(base_dir, extracted_items[0])):
            effective_base_dir = os.path.join(base_dir, extracted_items[0])
        
        full_path = os.path.abspath(os.path.join(effective_base_dir, file_path_relative))
        print(full_path)
        
        if not full_path.startswith(effective_base_dir): return jsonify({"error": "Access denied"}), 403
        if not os.path.isfile(full_path): return jsonify({"error": "File not found"}), 404
        
        file_ext = os.path.splitext(full_path)[1].lower()
        print(file_ext)
        if file_ext in ALLOWED_IMG_EXTENSIONS:
            with open(full_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            return jsonify({'type': 'image', 'content': b64_str})
        elif file_ext in ['.txt', '.xml']:
            with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({'type': 'text', 'content': content})
        else: return jsonify({"error": f"Preview not supported for {file_ext} files"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/images", methods=["GET"])
def get_images():
    # This endpoint remains for general image Browse, separate from dataset-specific logic
    # You can enhance its search capabilities as needed
    with get_db_connection() as conn:
        rows = conn.execute("SELECT i.*, d.name as dataset_name FROM images i JOIN datasets d ON i.dataset_id = d.id ORDER BY i.id").fetchall()
        response = []
        for row in rows:
            full_image_path = os.path.join(app.config["UPLOAD_FOLDER"], row['image_path'])
            if os.path.exists(full_image_path):
                 with open(full_image_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
                    response.append({ "id": row["id"], "filename": os.path.basename(row['image_path']), "base64": b64_str, "status": "" })
    return jsonify(response)


if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)