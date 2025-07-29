from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import base64
import datetime
import zipfile
import json
import shutil
from PIL import Image

# --- MOCK YOLOv7 INFERENCE ---
# In a real application, you would import torch and your YOLOv7 model here.
# For this example, we'll simulate the model's output.
YOLO_MODELS = {
    "v1.0": {"weights": "/path/to/yolov7_v1.pt", "confidence": 0.4},
    "v2.1": {"weights": "/path/to/yolov7_v2.1.pt", "confidence": 0.5}
}

def run_yolo_inference(model_version, image_path):
    """
    This is a mock function. In a real implementation, this function would:
    1. Load the specified YOLOv7 model weights.
    2. Preprocess the image from `image_path`.
    3. Run inference on the image.
    4. Post-process the results to get bounding boxes.
    5. Return the bounding boxes in the same format as load_yolo_annotations.
    """
    print(f"--- MOCK INFERENCE: Running YOLOv7 {model_version} on {os.path.basename(image_path)} ---")
    # Simulate finding a different number of objects with different models
    if model_version == "v1.0":
        # Simulate finding two objects
        return [
            {"classId": 0, "x": 50, "y": 60, "w": 120, "h": 180},
            {"classId": 1, "x": 200, "y": 100, "w": 80, "h": 90}
        ]
    elif model_version == "v2.1":
        # Simulate finding one larger object
        return [
            {"classId": 0, "x": 40, "y": 50, "w": 250, "h": 300}
        ]
    return []

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
BASE_DATA_DIR = "uploads/"
os.makedirs(BASE_DATA_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------- ANNOTATION HELPERS ----------
def load_yolo_annotations(annotation_path, original_image_width, original_image_height):
    if not os.path.exists(annotation_path) or not original_image_width or not original_image_height:
        return []
    boxes = []
    try:
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                x = (x_center - w_norm / 2) * original_image_width
                y = (y_center - h_norm / 2) * original_image_height
                w = w_norm * original_image_width
                h = h_norm * original_image_height
                boxes.append({"classId": int(class_id), "x": x, "y": y, "w": w, "h": h})
    except (ValueError, IndexError, IOError) as e:
        print(f"Warning: Could not process annotation file {annotation_path}: {e}")
    return boxes

def save_yolo_annotations(annotation_path, boxes, original_image_width, original_image_height):
    if not original_image_width or not original_image_height: return
    lines = []
    for b in boxes:
        x_center_n = (b['x'] + b['w'] / 2) / original_image_width
        y_center_n = (b['y'] + b['h'] / 2) / original_image_height
        w_n = b['w'] / original_image_width
        h_n = b['h'] / original_image_height
        lines.append(f"{b['classId']} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}")
    try:
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        with open(annotation_path, "w") as f:
            f.write("\n".join(lines))
    except IOError as e:
        print(f"ERROR: Failed to write to {annotation_path}: {e}")

# ---------- SQLITE HELPERS ----------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    with get_db_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                name TEXT PRIMARY KEY NOT NULL,
                class_names TEXT DEFAULT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                path TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                FOREIGN KEY (dataset_name) REFERENCES datasets(name) ON DELETE CASCADE,
                UNIQUE(dataset_name, filename)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotation_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                name TEXT NOT NULL, -- e.g., "default", "yolov7_v1.0"
                path TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                UNIQUE(image_id, name)
            );
        """)
        conn.commit()
    print("Database initialized or verified successfully.")

# ---------- FLASK SETUP & API ENDPOINTS ----------
app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/upload", methods=["POST"])
def upload_images():
    if "images" not in request.files: return jsonify({"error": "No image files part in the request"}), 400
    dataset_name = request.form.get("datasetName", "").strip()
    if not dataset_name: return jsonify({"error": "Dataset name is required"}), 400

    dataset_dir = os.path.join(BASE_DATA_DIR, dataset_name)
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)
    
    saved_images_info = []
    zip_class_names = None
    timestamp = int(datetime.datetime.now().timestamp())
    temp_dir = os.path.join(BASE_DATA_DIR, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        all_temp_image_sources = {}
        all_temp_annotation_sources = {}
        for file_obj in request.files.getlist("images"):
            if not file_obj.filename: continue
            if file_obj.filename.lower().endswith(".zip"):
                zip_path = os.path.join(temp_dir, file_obj.filename)
                file_obj.save(zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    class_file_path = next((name for name in zip_ref.namelist() if os.path.basename(name).lower() == 'classes.txt'), None)
                    if class_file_path:
                        with zip_ref.open(class_file_path) as class_file:
                            zip_class_names = [line.decode('utf-8').strip() for line in class_file if line.strip()]
                    for member in zip_ref.infolist():
                        if member.is_dir(): continue
                        member_basename = os.path.basename(member.filename).lower()
                        base_name = os.path.splitext(member_basename)[0]
                        if allowed_file(member.filename):
                            temp_path = zip_ref.extract(member, temp_dir)
                            all_temp_image_sources[base_name] = {'original_name': os.path.basename(member.filename), 'temp_path': temp_path}
                        elif member_basename.endswith('.txt') and member_basename != 'classes.txt':
                            temp_path = zip_ref.extract(member, temp_dir)
                            all_temp_annotation_sources[base_name] = {'original_name': os.path.basename(member.filename), 'temp_path': temp_path}
            elif allowed_file(file_obj.filename):
                base_name = os.path.splitext(file_obj.filename)[0].lower()
                temp_path = os.path.join(temp_dir, file_obj.filename)
                file_obj.save(temp_path)
                all_temp_image_sources[base_name] = {'original_name': file_obj.filename, 'temp_path': temp_path}

        with get_db_connection() as conn:
            for base_name, img_info in all_temp_image_sources.items():
                original_filename = img_info['original_name']
                unique_stem = f"{timestamp}_{os.path.splitext(original_filename)[0]}"
                final_filename = f"{unique_stem}{os.path.splitext(original_filename)[1]}"
                final_img_path = os.path.join(dataset_dir, "images", final_filename)
                shutil.move(img_info['temp_path'], final_img_path)
                
                width, height = Image.open(final_img_path).size
                
                cur = conn.cursor()
                cur.execute("INSERT INTO images (filename, original_name, path, dataset_name, width, height) VALUES (?, ?, ?, ?, ?, ?)",
                            (final_filename, original_filename, final_img_path, dataset_name, width, height))
                image_id = cur.lastrowid
                
                if base_name in all_temp_annotation_sources:
                    ann_info = all_temp_annotation_sources[base_name]
                    final_ann_path = os.path.join(dataset_dir, "annotations", f"{unique_stem}_default.txt")
                    shutil.move(ann_info['temp_path'], final_ann_path)
                    cur.execute("INSERT INTO annotation_sets (image_id, name, path) VALUES (?, ?, ?)",
                                (image_id, "default", final_ann_path))
                
                saved_images_info.append({"id": image_id, "filename": final_filename})
            
            if zip_class_names is not None:
                conn.execute("INSERT INTO datasets (name, class_names) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET class_names=excluded.class_names",
                             (dataset_name, json.dumps(zip_class_names)))
            else:
                conn.execute("INSERT INTO datasets (name) VALUES (?) ON CONFLICT(name) DO NOTHING", (dataset_name,))
            conn.commit()
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    return jsonify({"message": "Upload successful", "data": saved_images_info})

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify(list(YOLO_MODELS.keys()))

@app.route("/api/images/<int:image_id>/generate_annotations", methods=["POST"])
def generate_annotations(image_id):
    model_version = request.json.get("model_version")
    if not model_version or model_version not in YOLO_MODELS:
        return jsonify({"error": "Invalid model version"}), 400

    with get_db_connection() as conn:
        image_row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if not image_row: return jsonify({"error": "Image not found"}), 404

        boxes = run_yolo_inference(model_version, image_row['path'])
        
        unique_stem = os.path.splitext(image_row['filename'])[0]
        ann_set_name = f"yolov7_{model_version}"
        ann_path = os.path.join(BASE_DATA_DIR, image_row['dataset_name'], "annotations", f"{unique_stem}_{ann_set_name}.txt")
        
        save_yolo_annotations(ann_path, boxes, image_row['width'], image_row['height'])
        
        conn.execute("INSERT INTO annotation_sets (image_id, name, path) VALUES (?, ?, ?) ON CONFLICT(image_id, name) DO UPDATE SET path=excluded.path",
                     (image_id, ann_set_name, ann_path))
        conn.commit()

        new_set_row = conn.execute("SELECT id, name FROM annotation_sets WHERE image_id = ? AND name = ?", (image_id, ann_set_name)).fetchone()

    return jsonify({
        "message": "Annotations generated",
        "annotation_set": {"id": new_set_row['id'], "name": new_set_row['name'], "boxes": boxes}
    }), 201

@app.route("/api/annotations/<int:set_id>", methods=["PUT"])
def update_annotations(set_id):
    data = request.get_json()
    with get_db_connection() as conn:
        set_row = conn.execute("SELECT * FROM annotation_sets WHERE id = ?", (set_id,)).fetchone()
        if not set_row: return jsonify({"error": "Annotation set not found"}), 404
        image_row = conn.execute("SELECT width, height FROM images WHERE id = ?", (set_row['image_id'],)).fetchone()
        
        save_yolo_annotations(set_row['path'], data['boxes'], image_row['width'], image_row['height'])
        conn.commit()
    return jsonify({"message": "Annotations updated successfully"})

def fetch_images_from_db(view_mode):
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("limit" if view_mode == "multiple" else "per_page", 10))
    dataset_name = request.args.get("dataset", "").strip()

    if not dataset_name:
        return jsonify({"images": [], "total_count": 0})

    base_query = "FROM images i WHERE i.dataset_name = ?"
    params = [dataset_name]
    
    with get_db_connection() as conn:
        total_count = conn.execute(f"SELECT COUNT(*) {base_query}", tuple(params)).fetchone()[0]
        
        image_rows = conn.execute(
            f"SELECT * {base_query} ORDER BY i.id LIMIT ? OFFSET ?",
            tuple(params) + (per_page, (page - 1) * per_page)
        ).fetchall()

        response_images = []
        for row in image_rows:
            image_dict = dict(row)
            try:
                with open(image_dict["path"], "rb") as f:
                    image_dict["base64"] = base64.b64encode(f.read()).decode("utf-8")
            except FileNotFoundError:
                continue
            
            ann_sets = conn.execute("SELECT * FROM annotation_sets WHERE image_id = ?", (image_dict['id'],)).fetchall()
            
            annotation_sets_data = []
            for ann_set in ann_sets:
                boxes = load_yolo_annotations(ann_set['path'], image_dict['width'], image_dict['height'])
                annotation_sets_data.append({"id": ann_set['id'], "name": ann_set['name'], "boxes": boxes})
            
            # Ensure there's always at least one "default" set for annotation
            if not any(s['name'] == 'default' for s in annotation_sets_data):
                annotation_sets_data.insert(0, {"id": None, "name": "default", "boxes": []})

            image_dict["annotationSets"] = annotation_sets_data
            response_images.append(image_dict)
            
    return jsonify({"images": response_images, "total_count": total_count})

@app.route("/api/images/single", methods=["GET"])
def get_images_single():
    return fetch_images_from_db("single")

@app.route("/api/images/multiple", methods=["GET"])
def get_images_multiple():
    return fetch_images_from_db("multiple")

@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT name FROM datasets ORDER BY name ASC").fetchall()
        return jsonify([row['name'] for row in rows])

@app.route("/api/classes/<dataset_name>", methods=["GET"])
def get_dataset_classes(dataset_name):
    if not dataset_name: return jsonify([])
    with get_db_connection() as conn:
        row = conn.execute("SELECT class_names FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
        return jsonify(json.loads(row["class_names"]) if row and row["class_names"] else [])

# ---------- INIT & RUN ----------
if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
