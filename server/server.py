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

# YOLO bounding boxes assume these display sizes.
# Single mode = 400x400, multiple mode = 160x160
SINGLE_MODE_WIDTH = 400
SINGLE_MODE_HEIGHT = 400
MULTIPLE_MODE_WIDTH = 160
MULTIPLE_MODE_HEIGHT = 160

# ---------- YOLO HELPERS ----------
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
    with get_db_connection() as conn:
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
        conn.commit()

# ---------- FLASK SETUP ----------
app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- /api/upload -------------
@app.route("/api/upload", methods=["POST"])
def upload_images():
    """
    Upload images or a zip. If a zip, extract images.
    Insert records into SQLite. Return JSON of saved images.
    """
    if "images" not in request.files:
        return jsonify({"error": "No image files uploaded"}), 400

    files = request.files.getlist("images")
    saved_images = []

    for file_obj in files:
        if file_obj.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if file_obj.filename.lower().endswith(".zip"):
            # Extract the zip
            timestamp = int(datetime.datetime.now().timestamp())
            base_name = f"{timestamp}-{os.path.splitext(file_obj.filename)[0]}"
            extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(file_obj, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            for root, dirs, extracted_files in os.walk(extract_dir):
                for extracted_filename in extracted_files:
                    extracted_path = os.path.join(root, extracted_filename)
                    if allowed_file(extracted_filename):
                        new_filename = f"{int(datetime.datetime.now().timestamp())}-{extracted_filename}"
                        new_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
                        os.rename(extracted_path, new_path)

                        with get_db_connection() as conn:
                            cur = conn.execute("""
                                INSERT INTO images (filename, original_name, path, status, annotations)
                                VALUES (?, ?, ?, ?, ?)
                            """, (new_filename, extracted_filename, new_path, "", ""))
                            new_id = cur.lastrowid
                        image_data = {
                            "id": new_id,
                            "filename": new_filename,
                            "original_name": extracted_filename,
                            "path": new_path,
                            "status": ""
                        }
                        saved_images.append(image_data)
                    else:
                        os.remove(extracted_path)
        else:
            # Normal image
            if allowed_file(file_obj.filename):
                filename = f"{int(datetime.datetime.now().timestamp())}-{file_obj.filename}"
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file_obj.save(image_path)

                with get_db_connection() as conn:
                    cur = conn.execute("""
                        INSERT INTO images (filename, original_name, path, status, annotations)
                        VALUES (?, ?, ?, ?, ?)
                    """, (filename, file_obj.filename, image_path, "", ""))
                    new_id = cur.lastrowid

                image_data = {
                    "id": new_id,
                    "filename": filename,
                    "original_name": file_obj.filename,
                    "path": image_path,
                    "status": ""
                }
                saved_images.append(image_data)
            else:
                return jsonify({"error": "Unsupported file extension"}), 400

    return jsonify({"message": "Upload successful", "data": saved_images})

# ---------- /api/images/single -------------
@app.route("/api/images/single", methods=["GET"])
def get_images_single():
    """
    Single-mode: returns images, optionally with YOLO boxes if fee=Obj Det.
    We'll assume images are displayed at 400x400 in the frontend.
    """
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        search_text = request.args.get("search", "")
        fee = request.args.get("fee", "")

        skip = (page - 1) * per_page
        sql_conditions = []
        params = []

        if search_text:
            sql_conditions.append("(filename LIKE ? OR original_name LIKE ?)")
            like_pattern = f"%{search_text}%"
            params.append(like_pattern)
            params.append(like_pattern)

        # Optionally exclude discarded
        # sql_conditions.append("status != 'discarded'")

        where_clause = ""
        if sql_conditions:
            where_clause = "WHERE " + " AND ".join(sql_conditions)

        query_sql = f"""
            SELECT *
            FROM images
            {where_clause}
            LIMIT ? OFFSET ?
        """
        params.append(per_page)
        params.append(skip)

        response = []
        with get_db_connection() as conn:
            rows = conn.execute(query_sql, tuple(params)).fetchall()
            for row in rows:
                image_id = row["id"]
                filename = row["filename"]
                image_path = row["path"]

                with open(image_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")

                # Load YOLO boxes if fee=Obj Det
                boxes = []
                if fee == "Obj Det":
                    boxes = load_yolo_annotations(filename, SINGLE_MODE_WIDTH, SINGLE_MODE_HEIGHT)

                response.append({
                    "id": image_id,
                    "filename": filename,
                    "original_name": row["original_name"],
                    "base64": b64_str,
                    "status": row["status"] or "",
                    "boxes": boxes
                })

        return jsonify(response)
    except Exception as e:
        print("Error fetching single images:", e)
        return jsonify({"error": str(e)}), 500

# ---------- /api/images/multiple -------------
@app.route("/api/images/multiple", methods=["GET"])
def get_images_multiple():
    """
    Multiple-mode: returns images, optionally with YOLO boxes if fee=Obj Det.
    We'll assume images are displayed at 160x160 in the frontend.
    """
    try:
        fee = request.args.get("fee", "")
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
        search_text = request.args.get("search", "")

        skip = (page - 1) * limit

        sql_conditions = []
        params = []

        if search_text:
            sql_conditions.append("(filename LIKE ? OR original_name LIKE ?)")
            like_pattern = f"%{search_text}%"
            params.append(like_pattern)
            params.append(like_pattern)

        where_clause = ""
        if sql_conditions:
            where_clause = "WHERE " + " AND ".join(sql_conditions)

        query_sql = f"""
            SELECT *
            FROM images
            {where_clause}
            LIMIT ? OFFSET ?
        """
        params.append(limit)
        params.append(skip)

        response = []
        with get_db_connection() as conn:
            rows = conn.execute(query_sql, tuple(params)).fetchall()
            for row in rows:
                image_id = row["id"]
                filename = row["filename"]
                image_path = row["path"]

                with open(image_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")

                boxes = []
                if fee == "Obj Det":
                    boxes = load_yolo_annotations(filename, MULTIPLE_MODE_WIDTH, MULTIPLE_MODE_HEIGHT)

                response.append({
                    "id": image_id,
                    "filename": filename,
                    "original_name": row["original_name"],
                    "base64": b64_str,
                    "status": row["status"] or "",
                    "boxes": boxes
                })

        return jsonify(response)
    except Exception as e:
        print("Error fetching multiple images:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Discard -------------
@app.route("/api/images/<int:image_id>/discard", methods=["PUT"])
def discard_image(image_id):
    try:
        with get_db_connection() as conn:
            cur = conn.execute("""
                UPDATE images
                SET status = 'discarded'
                WHERE id = ?
            """, (image_id,))
            if cur.rowcount == 0:
                return jsonify({"error": "Image not found"}), 404

            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row:
                return jsonify({
                    "message": "Image discarded",
                    "image_id": row["id"],
                    "status": row["status"]
                })
            else:
                return jsonify({"error": "Image not found after update"}), 404
    except Exception as e:
        print("Error discarding image:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Save YOLO Annotations -------------
@app.route("/api/annotations/<filename>", methods=["PUT"])
def put_yolo_annotations(filename):
    """
    Expects JSON:
      {
        "boxes": [ { "classId":..., "x":..., "y":..., "w":..., "h":... }, ...],
        "imageWidth": 400 or 160,
        "imageHeight": 400 or 160
      }
    Writes YOLO lines to <filename>.txt in ANNOTATIONS_DIR.
    """
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
    if not os.path.exists(DB_PATH):
        open(DB_PATH, 'a').close()
    initialize_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
