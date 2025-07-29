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

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
BASE_DATA_DIR = "uploads/"
os.makedirs(BASE_DATA_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------- YOLO HELPERS ----------
def load_yolo_annotations(image_filename_in_db, original_image_width, original_image_height, dataset_name):
    """Loads YOLO annotations from a text file corresponding to an image."""
    base_name_without_ext = os.path.splitext(image_filename_in_db)[0]
    txt_filename = f"{base_name_without_ext}.txt"
    txt_path = os.path.join(BASE_DATA_DIR, dataset_name, "annotations", txt_filename)
    
    if not os.path.exists(txt_path):
        return []
    if not original_image_width or not original_image_height:
        return []

    boxes = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:])
                
                x = (x_center_norm - w_norm / 2) * original_image_width
                y = (y_center_norm - h_norm / 2) * original_image_height
                w = w_norm * original_image_width
                h = h_norm * original_image_height
                
                boxes.append({"classId": class_id, "x": x, "y": y, "w": w, "h": h})
    except (ValueError, IndexError, IOError) as e:
        print(f"Warning (load_yolo): Could not process annotation file {txt_path}: {e}")
    return boxes

def save_yolo_annotations(image_filename_in_db, boxes, original_image_width, original_image_height, dataset_name):
    """Saves YOLO annotations to a text file."""
    if not original_image_width or not original_image_height:
        print(f"Warning (save_yolo): Invalid dimensions for {image_filename_in_db}. Cannot save.")
        return

    base_name_without_ext = os.path.splitext(image_filename_in_db)[0]
    txt_filename = f"{base_name_without_ext}.txt"
    dataset_ann_dir = os.path.join(BASE_DATA_DIR, dataset_name, "annotations")
    os.makedirs(dataset_ann_dir, exist_ok=True)
    txt_path = os.path.join(dataset_ann_dir, txt_filename)
    
    lines = []
    for b in boxes:
        x_center_n = (b['x'] + b['w'] / 2) / original_image_width
        y_center_n = (b['y'] + b['h'] / 2) / original_image_height
        w_n = b['w'] / original_image_width
        h_n = b['h'] / original_image_height
        lines.append(f"{b['classId']} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}")

    try:
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")
    except IOError as e:
        print(f"ERROR (save_yolo): Failed to write to {txt_path}: {e}")

# ---------- SQLITE HELPERS ----------
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initializes the database with the required schema and indexes."""
    with get_db_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # Datasets table stores the name and associated class list (if any).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                name TEXT PRIMARY KEY NOT NULL,
                class_names TEXT DEFAULT NULL
            );
        """)
        
        # Images table stores metadata for each image file.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT DEFAULT '',
                dataset_name TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                FOREIGN KEY (dataset_name) REFERENCES datasets(name) ON DELETE CASCADE,
                UNIQUE(dataset_name, filename)
            );
        """)
        
        # Many-to-many relationship table for images and their annotated classes.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_classes (
                image_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                PRIMARY KEY (image_id, class_id),
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );
        """)
        
        # Indexes for faster query performance.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_dataset_name ON images(dataset_name);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_image_classes_class_id ON image_classes(class_id);")
        conn.commit()
    print("Database initialized or verified successfully.")

# ---------- FLASK SETUP ----------
app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    """Checks if a file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- API ENDPOINTS ----------

@app.route("/api/upload", methods=["POST"])
def upload_images():
    """Handles file uploads, including individual images and ZIP archives."""
    if "images" not in request.files:
        return jsonify({"error": "No image files part in the request"}), 400
    
    dataset_name = request.form.get("datasetName", "").strip()
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    dataset_images_dir = os.path.join(BASE_DATA_DIR, dataset_name, "Images")
    dataset_annotations_dir = os.path.join(BASE_DATA_DIR, dataset_name, "annotations")
    os.makedirs(dataset_images_dir, exist_ok=True)
    os.makedirs(dataset_annotations_dir, exist_ok=True)
    
    saved_images = []
    zip_class_names = None
    timestamp_upload_session = int(datetime.datetime.now().timestamp())
    
    all_temp_image_sources = {}
    all_temp_annotation_sources = {}
    temp_processing_dir = os.path.join(BASE_DATA_DIR, f"temp_processing_{timestamp_upload_session}")
    os.makedirs(temp_processing_dir, exist_ok=True)

    try:
        for file_obj in request.files.getlist("images"):
            if file_obj.filename == "":
                continue

            if file_obj.filename.lower().endswith(".zip"):
                zip_path = os.path.join(temp_processing_dir, file_obj.filename)
                file_obj.save(zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    # FIX: Robustly find classes.txt anywhere in the zip archive.
                    class_file_path = None
                    for name in zip_ref.namelist():
                        if os.path.basename(name).lower() == 'classes.txt':
                            class_file_path = name
                            break 
                    
                    if class_file_path:
                        with zip_ref.open(class_file_path) as class_file:
                            zip_class_names = [line.decode('utf-8').strip() for line in class_file if line.strip()]
                    
                    for member in zip_ref.infolist():
                        if member.is_dir(): continue
                        
                        member_basename = os.path.basename(member.filename).lower()
                        base_name = os.path.splitext(member_basename)[0]

                        if allowed_file(member.filename):
                            temp_path = zip_ref.extract(member, temp_processing_dir)
                            all_temp_image_sources[base_name] = {'original_name': os.path.basename(member.filename), 'temp_path': temp_path}
                        # FIX: Ensure classes.txt is not treated as an annotation file.
                        elif member_basename.endswith('.txt') and member_basename != 'classes.txt':
                            temp_path = zip_ref.extract(member, temp_processing_dir)
                            all_temp_annotation_sources[base_name] = {'original_name': os.path.basename(member.filename), 'temp_path': temp_path}
            elif allowed_file(file_obj.filename):
                base_name = os.path.splitext(file_obj.filename)[0].lower()
                temp_path = os.path.join(temp_processing_dir, file_obj.filename)
                file_obj.save(temp_path)
                all_temp_image_sources[base_name] = {'original_name': file_obj.filename, 'temp_path': temp_path}

        # Unified processing after extraction/saving
        for base_name, img_info in all_temp_image_sources.items():
            original_filename = img_info['original_name']
            temp_img_path = img_info['temp_path']
            
            unique_stem = f"{timestamp_upload_session}_{os.path.splitext(original_filename)[0]}"
            final_filename = f"{unique_stem}{os.path.splitext(original_filename)[1]}"
            final_path = os.path.join(dataset_images_dir, final_filename)

            shutil.move(temp_img_path, final_path)
            
            width, height = 0, 0
            try:
                with Image.open(final_path) as img_pil:
                    width, height = img_pil.size
            except Exception as e:
                print(f"Warning: Could not get dimensions for {final_filename}: {e}")

            class_ids = []
            if base_name in all_temp_annotation_sources:
                ann_info = all_temp_annotation_sources[base_name]
                final_ann_path = os.path.join(dataset_annotations_dir, f"{unique_stem}.txt")
                shutil.move(ann_info['temp_path'], final_ann_path)
                boxes = load_yolo_annotations(final_filename, width, height, dataset_name)
                class_ids = list(set(box['classId'] for box in boxes))

            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO images (filename, original_name, path, dataset_name, width, height) VALUES (?, ?, ?, ?, ?, ?)", 
                            (final_filename, original_filename, final_path, dataset_name, width, height))
                new_id = cur.lastrowid
                if class_ids:
                    cur.executemany("INSERT INTO image_classes (image_id, class_id) VALUES (?, ?)", [(new_id, cid) for cid in class_ids])
                conn.commit()
                saved_images.append({ "id": new_id, "filename": final_filename, "original_name": original_filename })

    except zipfile.BadZipFile:
        return jsonify({"error": "An invalid ZIP file was uploaded."}), 400
    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_processing_dir):
            shutil.rmtree(temp_processing_dir)

    # Save dataset and its classes
    with get_db_connection() as conn:
        if zip_class_names is not None:
            conn.execute("INSERT INTO datasets (name, class_names) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET class_names=excluded.class_names", 
                         (dataset_name, json.dumps(zip_class_names)))
        else:
            conn.execute("INSERT INTO datasets (name) VALUES (?) ON CONFLICT(name) DO NOTHING", (dataset_name,))
        conn.commit()

    return jsonify({"message": "Upload successful", "data": saved_images})

@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """NEW: Returns a list of all unique dataset names."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT name FROM datasets ORDER BY name ASC").fetchall()
            dataset_names = [row['name'] for row in rows]
            return jsonify(dataset_names)
    except Exception as e:
        print(f"ERROR: Failed to fetch dataset list: {e}")
        return jsonify([]), 500

@app.route("/api/classes/<dataset_name>", methods=["GET"])
def get_dataset_classes(dataset_name):
    """Returns the list of class names for a given dataset."""
    if not dataset_name:
        return jsonify([])
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT class_names FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
            if row and row["class_names"]:
                return jsonify(json.loads(row["class_names"]))
            else:
                return jsonify([])
    except Exception as e:
        print(f"ERROR: Failed to fetch classes for '{dataset_name}': {e}")
        return jsonify([])

def fetch_images_from_db(view_mode):
    """A common function to fetch and filter images for both single and multiple views."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("limit" if view_mode == "multiple" else "per_page", 10))
        search_text = request.args.get("search", "")
        fee = request.args.get("fee", "")
        dataset_name = request.args.get("dataset", "").strip()
        search_class_name = request.args.get("class", "").strip()

        sql_conditions = []
        params = []

        if search_text:
            sql_conditions.append("(i.filename LIKE ? OR i.original_name LIKE ?)")
            params.extend([f"%{search_text}%", f"%{search_text}%"])
        if dataset_name:
            sql_conditions.append("i.dataset_name = ?")
            params.append(dataset_name)
        
        if search_class_name and dataset_name:
            with get_db_connection() as conn_inner:
                class_row = conn_inner.execute("SELECT class_names FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
                if class_row and class_row['class_names']:
                    dataset_class_names = json.loads(class_row["class_names"])
                    try:
                        target_class_id = [c.lower() for c in dataset_class_names].index(search_class_name.lower())
                        sql_conditions.append("EXISTS (SELECT 1 FROM image_classes ic WHERE ic.image_id = i.id AND ic.class_id = ?)")
                        params.append(target_class_id)
                    except ValueError:
                        return jsonify({"images": [], "total_count": 0})
                else:
                    return jsonify({"images": [], "total_count": 0})

        where_clause = f"WHERE {' AND '.join(sql_conditions)}" if sql_conditions else ""
        
        with get_db_connection() as conn:
            count_query_sql = f"SELECT COUNT(DISTINCT i.id) FROM images i {where_clause}"
            total_count = conn.execute(count_query_sql, tuple(params)).fetchone()[0]

            query_sql = f"""
                SELECT i.*, (SELECT json_group_array(ic.class_id) FROM image_classes ic WHERE ic.image_id = i.id) as stored_class_ids 
                FROM images i {where_clause} ORDER BY i.id LIMIT ? OFFSET ?
            """
            rows = conn.execute(query_sql, tuple(params) + (per_page, (page - 1) * per_page)).fetchall()

        response_images = []
        for row in rows:
            try:
                with open(row["path"], "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
            except FileNotFoundError:
                continue
            
            boxes = load_yolo_annotations(row["filename"], row["width"], row["height"], row["dataset_name"]) if fee == "Obj Det" else []
            
            response_images.append({
                "id": row["id"], "filename": row["filename"], "original_name": row["original_name"], "base64": b64_str,
                "status": row["status"] or "", "boxes": boxes, "dataset_name": row["dataset_name"],
                "original_width": row["width"], "original_height": row["height"],
                "stored_class_ids": json.loads(row["stored_class_ids"]) if row["stored_class_ids"] else []
            })
        
        return jsonify({"images": response_images, "total_count": total_count})
    except Exception as e:
        print(f"ERROR (API Fetch): {e}")
        return jsonify({"error": "An internal error occurred while fetching images."}), 500

@app.route("/api/images/single", methods=["GET"])
def get_images_single():
    """Endpoint for fetching images in single-view mode."""
    return fetch_images_from_db("single")

@app.route("/api/images/multiple", methods=["GET"])
def get_images_multiple():
    """Endpoint for fetching images in multiple-view mode."""
    return fetch_images_from_db("multiple")

@app.route("/api/images/<int:image_id>/discard", methods=["PUT"])
def discard_image(image_id):
    """Marks an image as 'discarded'."""
    try:
        with get_db_connection() as conn:
            cur = conn.execute("UPDATE images SET status = 'discarded' WHERE id = ?", (image_id,))
            if cur.rowcount == 0:
                return jsonify({"error": "Image not found"}), 404
            conn.commit()
            return jsonify({"message": "Image discarded", "image_id": image_id, "status": "discarded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/annotations/<dataset_name>/<filename>", methods=["PUT"])
def put_yolo_annotations(dataset_name, filename):
    """Saves or updates annotations for a specific image."""
    try:
        data = request.get_json()
        boxes = data.get("boxes", [])
        width = data.get("imageWidth")
        height = data.get("imageHeight")

        if not all([width, height]):
            return jsonify({"error": "Original image dimensions are required"}), 400

        save_yolo_annotations(filename, boxes, width, height, dataset_name)
        
        with get_db_connection() as conn:
            cur = conn.cursor()
            image_row = cur.execute("SELECT id FROM images WHERE filename = ? AND dataset_name = ?", (filename, dataset_name)).fetchone()
            if not image_row:
                return jsonify({"error": "Image not found in database"}), 404
            image_id = image_row['id']

            cur.execute("DELETE FROM image_classes WHERE image_id = ?", (image_id,))
            
            unique_class_ids = list(set(b["classId"] for b in boxes))
            if unique_class_ids:
                cur.executemany("INSERT INTO image_classes (image_id, class_id) VALUES (?, ?)", [(image_id, cid) for cid in unique_class_ids])
            
            conn.commit()

        return jsonify({"message": "Annotations saved"}), 200
    except Exception as e:
        print(f"Error saving YOLO annotations: {e}")
        return jsonify({"error": "An internal error occurred while saving annotations."}), 500

# ---------- INIT & RUN ----------
if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
