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
import subprocess

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
# Global dictionary to track decompression progress
DECOMPRESSION_PROGRESS = {} 

# ---------- SQLite Helpers ----------
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Creates database tables if they don't already exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                class_names TEXT,
                status TEXT DEFAULT 'pending',
                file_tree TEXT,
                thumbnail_path TEXT,
                storage_path TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                annotation_path TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            );""")
        # Compatibility check for older schema versions
        cursor = conn.execute("PRAGMA table_info(datasets)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'class_names' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN class_names TEXT")
        if 'thumbnail_path' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN thumbnail_path TEXT")
        if 'storage_path' not in columns: conn.execute("ALTER TABLE datasets ADD COLUMN storage_path TEXT UNIQUE")
        
        cursor = conn.execute("PRAGMA table_info(images)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'image_type' in columns: # This column is no longer needed
            # A more robust migration would be needed for production data
            print("Note: 'image_type' column found and can be removed.")

        conn.commit()

# ---------- Worker Functions ----------
def slugify(text):
    """Converts text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[\s\W-]+', '-', text)
    return text.strip('-')

def decompress_zip_and_build_tree(zip_path, extract_dir, dataset_id):
    """Decompresses a zip file and creates a JSON representation of its file tree."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = [f for f in zip_ref.infolist() if not f.is_dir()]
            total_files = len(file_list)
            DECOMPRESSION_PROGRESS[dataset_id] = {'progress': 0, 'status': 'decompressing'}
            for i, member in enumerate(file_list):
                zip_ref.extract(member, extract_dir)
                progress = int(((i + 1) / total_files) * 100) if total_files > 0 else 100
                DECOMPRESSION_PROGRESS[dataset_id]['progress'] = progress
        
        # Determine the effective root directory (handles zips with a single top-level folder)
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

def run_script_and_index_db(dataset_dir, script_name, dataset_id):
    """Runs the user's Python script in a sandbox to generate a manifest and populates the DB."""
    try:
        script_path_in_container = f"/data/{script_name}"
        command = [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{os.path.abspath(dataset_dir)}:/data",
            "python-runner",  # Assumes a Docker image named 'python-runner' is available
            "python", script_path_in_container
        ]
        
        # This executes the script. It will raise an exception on non-zero exit code.
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        
        manifest_file = os.path.join(dataset_dir, "manifest.json")
        if not os.path.exists(manifest_file):
            raise Exception("Script did not produce a manifest.json file.")
        
        with open(manifest_file, 'r') as f:
            mapping_data = json.load(f)
            
        with get_db_connection() as conn:
            conn.execute("DELETE FROM images WHERE dataset_id = ?", (dataset_id,))
            for item in mapping_data:
                conn.execute(
                    "INSERT INTO images (dataset_id, image_path, annotation_path) VALUES (?, ?, ?)",
                    (dataset_id, item.get('image'), item.get('annotation'))
                )
            conn.execute("UPDATE datasets SET status = ? WHERE id = ?", ('indexed', dataset_id))
            conn.commit()

    except Exception as e:
        print(f"Script processing error for dataset {dataset_id}: {e}")
        with get_db_connection() as conn:
            conn.execute("UPDATE datasets SET status = ? WHERE id = ?", ('script_failed', dataset_id))
            conn.commit()

# ---------- FLASK SETUP ----------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
def allowed_file(filename, extensions): return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

# ---------- API Endpoints ----------
@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Handles the initial dataset file upload."""
    if "images" not in request.files: return jsonify({"error": "No file part"}), 400
    file_obj = request.files.get("images")
    if not file_obj or file_obj.filename == '': return jsonify({"error": "No selected file"}), 400
    if not file_obj.filename.lower().endswith(".zip"): return jsonify({"error": "Only zip files are supported"}), 400
    dataset_name = request.form.get("datasetName")
    if not dataset_name: return jsonify({"error": "Dataset Name is required"}), 400
    
    dataset_type = request.form.get("datasetType", "")
    dataset_description = request.form.get("datasetDescription", "")
    class_names = request.form.get("classNames", "")
    thumbnail_file = request.files.get("thumbnail")

    # Create a unique directory name for the dataset
    base_slug = slugify(dataset_name)
    storage_name, counter = base_slug, 1
    while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], storage_name)):
        storage_name = f"{base_slug}-{counter}"; counter += 1
    
    extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], storage_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Save thumbnail if provided
    thumbnail_path_db = None
    if thumbnail_file and allowed_file(thumbnail_file.filename, ALLOWED_IMG_EXTENSIONS):
        ext = os.path.splitext(thumbnail_file.filename)[1]
        thumbnail_filename = f"_thumbnail{ext}"
        thumbnail_file.save(os.path.join(extract_dir, thumbnail_filename))
        thumbnail_path_db = os.path.join(storage_name, thumbnail_filename).replace('\\', '/')

    with get_db_connection() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO datasets (name, type, description, class_names, status, thumbnail_path, storage_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dataset_name, dataset_type, dataset_description, class_names, 'decompressing', thumbnail_path_db, storage_name)
            )
            dataset_id = cur.lastrowid
            conn.commit()
            
            # Save zip and start decompression in a background thread
            zip_path = os.path.join(extract_dir, f"{dataset_id}.zip")
            file_obj.save(zip_path)
            thread = threading.Thread(target=decompress_zip_and_build_tree, args=(zip_path, extract_dir, dataset_id))
            thread.start()
            
            return jsonify({"message": "Upload received, starting decompression.", "dataset_id": dataset_id, "storage_path": storage_name})
        except Exception as e:
            conn.rollback()
            return jsonify({"error": str(e)}), 500

@app.route("/api/datasets/<string:storage_path>/status", methods=["GET"])
def get_dataset_status(storage_path):
    """Checks the current status of a dataset (decompressing, complete, etc.)."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT id, status FROM datasets WHERE storage_path = ?", (storage_path,)).fetchone()
        if not row: return jsonify({"error": "Dataset not found"}), 404
        
        dataset_id, status = row['id'], row['status']
        progress = DECOMPRESSION_PROGRESS.get(dataset_id, {}).get('progress', 0) if status == 'decompressing' else 0
        return jsonify({"status": status, "progress": progress})

@app.route("/api/datasets/<string:storage_path>/file_tree", methods=["GET"])
def get_dataset_file_tree(storage_path):
    """Retrieves the JSON file tree for a dataset."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT file_tree FROM datasets WHERE storage_path = ?", (storage_path,)).fetchone()
        if not row or not row['file_tree']: return jsonify({"error": "File tree not available"}), 404
        return jsonify(json.loads(row['file_tree']))

@app.route("/api/datasets/<string:storage_path>/execute-script", methods=["POST"])
def execute_user_script(storage_path):
    """Saves and executes the user-provided Python script to index the dataset."""
    dataset_dir = os.path.join(app.config["UPLOAD_FOLDER"], storage_path)
    if not os.path.isdir(dataset_dir): return jsonify({"error": "Dataset directory not found"}), 404
    
    script_code = request.get_json().get('code')
    if not script_code: return jsonify({"error": "No code provided"}), 400
    
    script_filename = "user_script.py"
    script_path = os.path.join(dataset_dir, script_filename)
    with open(script_path, "w") as f:
        f.write(script_code)

    with get_db_connection() as conn:
        row = conn.execute("SELECT id FROM datasets WHERE storage_path = ?", (storage_path,)).fetchone()
        if not row: return jsonify({"error": "Dataset not found in DB"}), 404
        
        dataset_id = row['id']
        conn.execute("UPDATE datasets SET status = ? WHERE id = ?", ('processing_script', dataset_id))
        conn.commit()
        
        # Run the script in a background thread
        thread = threading.Thread(target=run_script_and_index_db, args=(dataset_dir, script_filename, dataset_id))
        thread.start()
        
    return jsonify({"message": "Script execution started."}), 202
    
@app.route("/api/datasets/<string:storage_path>/preview", methods=["GET"])
def preview_dataset_file(storage_path):
    """Provides a preview of a single file within a dataset."""
    file_path_relative = request.args.get('path')
    if not file_path_relative:
        return jsonify({"error": "File path is required"}), 400

    base_dir = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], storage_path))
    full_path = None

    # NEW: Special handling for manifest.json to ensure it's always found at the root
    if file_path_relative == 'manifest.json':
        full_path = os.path.join(base_dir, 'manifest.json')
        if not os.path.isfile(full_path):
            return jsonify({"error": "'manifest.json' not found in the root dataset directory"}), 404
    else:
        # Standard logic for all other files (images, txt, etc.)
        effective_base_dir = base_dir
        # This handles zip files that extract into a single sub-folder
        if os.path.isdir(base_dir):
            extracted_items = os.listdir(base_dir)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(base_dir, extracted_items[0])):
                effective_base_dir = os.path.join(base_dir, extracted_items[0])
        
        full_path = os.path.abspath(os.path.join(effective_base_dir, file_path_relative))
        
        # Security and existence checks
        if not full_path.startswith(effective_base_dir):
            return jsonify({"error": "Access denied"}), 403
        if not os.path.isfile(full_path):
            return jsonify({"error": "File not found"}), 404

    # The rest of the function reads the file content based on its extension
    try:
        file_ext = os.path.splitext(full_path)[1].lower()
        if file_ext in ALLOWED_IMG_EXTENSIONS:
            with open(full_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            return jsonify({'type': 'image', 'content': b64_str, 'name': os.path.basename(full_path)})
        elif file_ext in ['.txt', '.xml', '.json']:
            with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return jsonify({'type': 'text', 'content': content, 'name': os.path.basename(full_path)})
        else:
            return jsonify({"error": f"Preview not supported for {file_ext} files"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# NEW: Endpoint to get all images for a specific dataset
@app.route("/api/datasets/<string:storage_path>/images", methods=["GET"])
def get_dataset_images(storage_path):
    """Fetches all indexed image records for a specific dataset."""
    response = []
    with get_db_connection() as conn:
        dataset_row = conn.execute("SELECT id FROM datasets WHERE storage_path = ?", (storage_path,)).fetchone()
        if not dataset_row: return jsonify({"error": "Dataset not found"}), 404
        
        image_rows = conn.execute("SELECT * FROM images WHERE dataset_id = ?", (dataset_row['id'],)).fetchall()
        
        # This logic must be consistent with the preview endpoint
        base_dir = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], storage_path))
        extracted_items = os.listdir(base_dir)
        effective_base_dir = base_dir
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(base_dir, extracted_items[0])):
            effective_base_dir = os.path.join(base_dir, extracted_items[0])

        for row in image_rows:
            full_image_path = os.path.join(effective_base_dir, row['image_path'])
            if os.path.exists(full_image_path):
                with open(full_image_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
                response.append({ 
                    "id": row["id"], 
                    "filename": os.path.basename(row['image_path']), 
                    "base64": b64_str, 
                    "status": "", # Frontend expects this
                    "boxes": [] # Frontend expects this
                })
    return jsonify(response)

# NEW: Endpoint to save annotations for a specific image
@app.route("/api/images/<int:image_id>/annotations", methods=["PUT"])
def save_image_annotations(image_id):
    """Saves annotation data for a single image."""
    with get_db_connection() as conn:
        image_info = conn.execute(
            "SELECT i.image_path, i.annotation_path, d.storage_path FROM images i JOIN datasets d ON i.dataset_id = d.id WHERE i.id = ?", 
            (image_id,)
        ).fetchone()

        if not image_info: return jsonify({"error": "Image not found"}), 404

        data = request.get_json()
        if not data or 'boxes' not in data: return jsonify({"error": "Invalid annotation data"}), 400

        # Convert pixel coordinates from frontend to normalized YOLO format
        image_width = data.get('imageWidth', 1)
        image_height = data.get('imageHeight', 1)
        annotation_lines = []
        for box in data['boxes']:
            x_center = (box['x'] + box['w'] / 2) / image_width
            y_center = (box['y'] + box['h'] / 2) / image_height
            width = box['w'] / image_width
            height = box['h'] / image_height
            annotation_lines.append(f"{box['classId']} {x_center} {y_center} {width} {height}")
        
        annotation_content = "\n".join(annotation_lines)
        
        annotation_path_relative = image_info['annotation_path']
        if not annotation_path_relative:
            base, _ = os.path.splitext(image_info['image_path'])
            annotation_path_relative = f"{base}.txt"
        
        dataset_dir = os.path.join(app.config["UPLOAD_FOLDER"], image_info['storage_path'])
        full_annotation_path = os.path.join(dataset_dir, annotation_path_relative)
        
        os.makedirs(os.path.dirname(full_annotation_path), exist_ok=True)
        with open(full_annotation_path, "w") as f:
            f.write(annotation_content)
        
        if not image_info['annotation_path']:
            conn.execute("UPDATE images SET annotation_path = ? WHERE id = ?", (annotation_path_relative, image_id))
        
        conn.commit()

    return jsonify({"message": "Annotations saved successfully"}), 200

# IMPROVED: General image search with fallback
@app.route("/api/images", methods=["GET"])
def get_images():
    """Fetches all images, with an optional search query."""
    search_query = request.args.get('search', '')
    with get_db_connection() as conn:
        sql = "SELECT i.*, d.name as dataset_name, d.storage_path FROM datasets d JOIN images i ON d.id = i.dataset_id"
        params = []
        if search_query:
            sql += " WHERE i.image_path LIKE ? OR d.name LIKE ?"
            params.extend([f"%{search_query}%", f"%{search_query}%"])
        sql += " ORDER BY i.id DESC"
        
        rows = conn.execute(sql, params).fetchall()
        response = []
        for row in rows:
            # Note: This is inefficient for large-scale searching. Caching would be needed.
            # This logic also needs to account for the effective root directory.
            # For simplicity in this context, we'll assume a flat structure for search results.
            full_image_path = os.path.join(app.config["UPLOAD_FOLDER"], row['storage_path'], row['image_path'])
            if os.path.exists(full_image_path):
                   with open(full_image_path, "rb") as f:
                       b64_str = base64.b64encode(f.read()).decode("utf-8")
                       response.append({ "id": row["id"], "filename": os.path.basename(row['image_path']), "base64": b64_str, "status": "" })
    return jsonify(response)
@app.route("/api/datasets/<string:storage_path>/image-batch", methods=["POST"])
def get_dataset_image_batch(storage_path):
    """Fetches a batch of images and their data in a single request."""
    image_paths = request.get_json()
    if not isinstance(image_paths, list):
        return jsonify({"error": "A list of image paths is required"}), 400

    # This logic correctly determines the directory where the images are stored
    base_dir = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], storage_path))
    effective_base_dir = base_dir
    if os.path.isdir(base_dir):
        extracted_items = os.listdir(base_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(base_dir, extracted_items[0])):
            effective_base_dir = os.path.join(base_dir, extracted_items[0])

    response = []
    for relative_path in image_paths:
        full_image_path = os.path.join(effective_base_dir, relative_path)
        if os.path.exists(full_image_path):
            with open(full_image_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            response.append({
                "id": relative_path,  # Use path as ID
                "filename": os.path.basename(relative_path),
                "base64": b64_str,
                "status": "",
                "boxes": []
            })
    return jsonify(response)
if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)