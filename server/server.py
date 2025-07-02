from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import base64
import datetime
import zipfile
import json

# ---------- CONFIGURATION ----------
DB_PATH = "images.db"
UPLOAD_FOLDER = "uploads/"
ANNOTATIONS_DIR = "annotations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

SINGLE_MODE_WIDTH = 400
SINGLE_MODE_HEIGHT = 400
MULTIPLE_MODE_WIDTH = 160
MULTIPLE_MODE_HEIGHT = 160

# ---------- YOLO HELPERS (No changes) ----------
def load_yolo_annotations(filename, image_width, image_height):
    """
    Reads <filename>.txt from ANNOTATIONS_DIR, returning a list of
    {classId, x, y, w, h} in *pixel coords* for the given image_width/height.
    If file not found, return [].
    """
    txt_path = os.path.join(ANNOTATIONS_DIR, f"{filename}.txt")
    if not os.path.exists(txt_path):
        return []

    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
            # Convert normalized => absolute
            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            w = w_norm * image_width
            h = h_norm * image_height
            x = x_center - w/2
            y = y_center - h/2
            boxes.append({
                "classId": class_id,
                "x": x, "y": y, "w": w, "h": h
            })
    return boxes

def save_yolo_annotations(filename, boxes, image_width, image_height):
    """
    boxes is a list of {classId, x, y, w, h} in absolute pixel coords.
    Convert to YOLO lines => "class_id x_center_norm y_center_norm w_norm h_norm"
    """
    txt_path = os.path.join(ANNOTATIONS_DIR, f"{filename}.txt")
    lines = []
    for b in boxes:
        class_id = b["classId"]
        x_center = b["x"] + b["w"]/2
        y_center = b["y"] + b["h"]/2
        x_center_n = x_center / image_width
        y_center_n = y_center / image_height
        w_n = b["w"] / image_width
        h_n = b["h"] / image_height
        line = f"{class_id} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}"
        lines.append(line)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

# ---------- SQLite Helpers ----------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """
    Initializes the database, creating tables and adding columns if they don't exist.
    """
    with get_db_connection() as conn:
        # Create datasets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create images table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT DEFAULT '',
                annotations TEXT DEFAULT ''
            );
        """)

        # Add dataset_id to images table if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(images)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'dataset_id' not in columns:
            conn.execute("ALTER TABLE images ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)")

        conn.commit()

# ---------- FLASK SETUP ----------
app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- /api/upload (MODIFIED) -------------
@app.route("/api/upload", methods=["POST"])
def upload_images():
    """
    Receives dataset metadata and image files.
    1. Creates a new dataset record.
    2. Saves images and associates them with the new dataset.
    """
    if "images" not in request.files:
        return jsonify({"error": "No image files uploaded"}), 400

    # Get dataset metadata from the form
    dataset_name = request.form.get("datasetName", f"dataset-{int(datetime.datetime.now().timestamp())}")
    dataset_type = request.form.get("datasetType", "")
    dataset_description = request.form.get("datasetDescription", "")

    files = request.files.getlist("images")
    saved_images = []
    
    with get_db_connection() as conn:
        try:
            # 1. Create the dataset record
            cur = conn.execute("""
                INSERT INTO datasets (name, type, description) VALUES (?, ?, ?)
            """, (dataset_name, dataset_type, dataset_description))
            dataset_id = cur.lastrowid
            
            # 2. Process and save each image, linking it to the dataset
            for file_obj in files:
                if file_obj.filename == "":
                    continue

                if file_obj.filename.lower().endswith(".zip"):
                    # Handle zip files
                    timestamp = int(datetime.datetime.now().timestamp())
                    base_name = f"{timestamp}-{os.path.splitext(file_obj.filename)[0]}"
                    extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
                    os.makedirs(extract_dir, exist_ok=True)

                    with zipfile.ZipFile(file_obj, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)

                    for root, _, extracted_files in os.walk(extract_dir):
                        for extracted_filename in extracted_files:
                            if allowed_file(extracted_filename):
                                new_filename = f"{timestamp}-{extracted_filename}"
                                new_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
                                os.rename(os.path.join(root, extracted_filename), new_path)
                                
                                cur = conn.execute("""
                                    INSERT INTO images (filename, original_name, path, dataset_id)
                                    VALUES (?, ?, ?, ?)
                                """, (new_filename, extracted_filename, new_path, dataset_id))
                                saved_images.append({"id": cur.lastrowid, "filename": new_filename})
                
                elif allowed_file(file_obj.filename):
                    # Handle single image files
                    filename = f"{int(datetime.datetime.now().timestamp())}-{file_obj.filename}"
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file_obj.save(image_path)
                    
                    cur = conn.execute("""
                        INSERT INTO images (filename, original_name, path, dataset_id)
                        VALUES (?, ?, ?, ?)
                    """, (filename, file_obj.filename, image_path, dataset_id))
                    saved_images.append({"id": cur.lastrowid, "filename": filename})

            conn.commit()
            return jsonify({"message": f"Dataset '{dataset_name}' created and {len(saved_images)} images uploaded.", "data": saved_images})

        except sqlite3.Error as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {e}"}), 500
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"An error occurred: {e}"}), 500

# ---------- Query Builder Helper (NEW) ----------
def build_image_query(args):
    """
    Builds the SELECT, JOIN, and WHERE clauses for fetching images.
    Returns the query string and a tuple of parameters.
    """
    search_text = args.get("search", "")
    
    # Base query with a join
    query_sql = """
        SELECT i.*, d.name as dataset_name
        FROM images i
        LEFT JOIN datasets d ON i.dataset_id = d.id
    """
    
    sql_conditions = []
    params = []

    if search_text:
        # Search across multiple fields in both tables
        sql_conditions.append("""
            (i.filename LIKE ? OR 
             i.original_name LIKE ? OR 
             d.name LIKE ? OR 
             d.type LIKE ? OR 
             d.description LIKE ?)
        """)
        like_pattern = f"%{search_text}%"
        for _ in range(5):
            params.append(like_pattern)
            
    # You can add more specific filters here later, e.g.:
    # if args.get("datasetType"):
    #     sql_conditions.append("d.type = ?")
    #     params.append(args.get("datasetType"))

    if sql_conditions:
        query_sql += " WHERE " + " AND ".join(sql_conditions)
        
    return query_sql, tuple(params)


# ---------- /api/images/single (MODIFIED) -------------
@app.route("/api/images/single", methods=["GET"])
def get_images_single():
    """Single-mode: returns images, optionally with YOLO boxes."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        fee = request.args.get("fee", "")
        skip = (page - 1) * per_page
        
        query_sql, params = build_image_query(request.args)
        
        # Add ordering and pagination
        query_sql += " ORDER BY i.id DESC LIMIT ? OFFSET ?"
        params += (per_page, skip)

        response = []
        with get_db_connection() as conn:
            rows = conn.execute(query_sql, params).fetchall()
            for row in rows:
                with open(row["path"], "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")

                boxes = []
                if fee == "Obj Det":
                    boxes = load_yolo_annotations(row["filename"], SINGLE_MODE_WIDTH, SINGLE_MODE_HEIGHT)

                response.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "original_name": row["original_name"],
                    "base64": b64_str,
                    "status": row["status"] or "",
                    "boxes": boxes,
                    "dataset_id": row["dataset_id"],
                    "dataset_name": row["dataset_name"]
                })

        return jsonify(response)
    except Exception as e:
        print("Error fetching single images:", e)
        return jsonify({"error": str(e)}), 500

# ---------- /api/images/multiple (MODIFIED) -------------
@app.route("/api/images/multiple", methods=["GET"])
def get_images_multiple():
    """Multiple-mode: returns images, optionally with YOLO boxes."""
    try:
        fee = request.args.get("fee", "")
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
        skip = (page - 1) * limit

        query_sql, params = build_image_query(request.args)
        
        # Add ordering and pagination
        query_sql += " ORDER BY i.id DESC LIMIT ? OFFSET ?"
        params += (limit, skip)
        
        response = []
        with get_db_connection() as conn:
            rows = conn.execute(query_sql, params).fetchall()
            for row in rows:
                with open(row["path"], "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")

                boxes = []
                if fee == "Obj Det":
                    boxes = load_yolo_annotations(row["filename"], MULTIPLE_MODE_WIDTH, MULTIPLE_MODE_HEIGHT)

                response.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "original_name": row["original_name"],
                    "base64": b64_str,
                    "status": row["status"] or "",
                    "boxes": boxes,
                    "dataset_id": row["dataset_id"],
                    "dataset_name": row["dataset_name"]
                })

        return jsonify(response)
    except Exception as e:
        print("Error fetching multiple images:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Discard (No changes) -------------
@app.route("/api/images/<int:image_id>/discard", methods=["PUT"])
def discard_image(image_id):
    try:
        with get_db_connection() as conn:
            cur = conn.execute("UPDATE images SET status = 'discarded' WHERE id = ?", (image_id,))
            if cur.rowcount == 0:
                return jsonify({"error": "Image not found"}), 404
            
            conn.commit()
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row:
                return jsonify({ "message": "Image discarded", "image_id": row["id"], "status": row["status"] })
            else:
                return jsonify({"error": "Image not found after update"}), 404
    except Exception as e:
        print("Error discarding image:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Save YOLO Annotations (No changes) -------------
@app.route("/api/annotations/<filename>", methods=["PUT"])
def put_yolo_annotations(filename):
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No annotation data"}), 400

        boxes = data.get("boxes", [])
        image_width = data.get("imageWidth", 400)
        image_height = data.get("imageHeight", 400)

        save_yolo_annotations(filename, boxes, image_width, image_height)
        return jsonify({"message": "Annotations saved"}), 200
    except Exception as e:
        print("Error saving YOLO annotations:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Init & Run ----------
if __name__ == "__main__":
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)