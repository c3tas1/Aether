from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from PIL import Image
import os
import base64
import datetime
import zipfile

# ------------- SQLite Helpers -------------
DB_PATH = "images.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)



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

# ------------- Flask Setup -------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
allowed_extensions = {"png", "jpg", "jpeg"}



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


# ------------- /api/images/<image_id>/annotations-------------
@app.route("/api/images/<int:image_id>/annotations", methods=["GET"])
def get_annotations(image_id):
    """
    Returns JSON array of YOLO lines, e.g.:
      [
        {"classId": 0, "x_center": 0.5, "y_center": 0.4, "w": 0.1, "h": 0.2},
        ...
      ]
    plus an optional "labelMap" if you store classId->label.
    """
    import os
    from flask import jsonify

    # Suppose we store annotations in "annotations/<image_id>.txt"
    # or "<filename>.txt". For brevity, let's guess:
    txt_path = os.path.join("annotations", f"{image_id}.txt")
    if not os.path.exists(txt_path):
        # No annotations
        return jsonify({
            "boxes": [],
            "labelMap": { "0": "Unknown" }
        })

    lines = []
    with open(txt_path, "r") as f:
        lines = f.read().strip().split("\n")

    # Example: each line looks like:
    # class x_center y_center w h
    # e.g.: "0 0.500000 0.400000 0.100000 0.200000"
    results = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            classId = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            results.append({
                "classId": classId,
                "x_center": x_center,
                "y_center": y_center,
                "w": w,
                "h": h
            })

    # For demonstration, let's say classId=0 => "Person", 1 => "Car", etc.
    # In reality, store a separate dictionary or read from config:
    label_map = {
        "0": "Person",
        "1": "Car",
        "2": "Tree"
    }

    return jsonify({
        "boxes": results,
        "labelMap": label_map
    })




# ------------- /api/upload -------------
@app.route("/api/upload", methods=["POST"])
def upload_images():
    """
    Handle either:
    - Multiple image files (PNG/JPG/JPEG), or
    - A .zip file containing images
    """
    if "images" not in request.files:
        return jsonify({"error": "No image files uploaded"}), 400

    files = request.files.getlist("images")
    saved_images = []

    for file_obj in files:
        # If the file has an empty filename
        if file_obj.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Check if it's a ZIP file
        if file_obj.filename.lower().endswith(".zip"):
            # Create a unique directory name to extract this ZIP
            timestamp = int(datetime.datetime.now().timestamp())
            base_name = f"{timestamp}-{os.path.splitext(file_obj.filename)[0]}"
            extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
            os.makedirs(extract_dir, exist_ok=True)

            # Extract ZIP contents
            with zipfile.ZipFile(file_obj, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Walk through extracted files
            for root, dirs, extracted_files in os.walk(extract_dir):
                for extracted_filename in extracted_files:
                    extracted_path = os.path.join(root, extracted_filename)

                    if allowed_file(extracted_filename):
                        # Create a unique filename in the main uploads folder
                        new_filename = f"{int(datetime.datetime.now().timestamp())}-{extracted_filename}"
                        new_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
                        os.rename(extracted_path, new_path)

                        # Insert into SQLite
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
                            "status": "",
                            "annotations": []
                        }
                        saved_images.append(image_data)
                    else:
                        # Remove non-allowed files
                        os.remove(extracted_path)

        else:
            # Otherwise, treat as a normal image
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
                    "status": "",
                    "annotations": []
                }
                saved_images.append(image_data)
            else:
                return jsonify({"error": "Unsupported file extension"}), 400

    return jsonify({"message": "Upload successful", "data": saved_images})

# ------------- Single-mode route (/api/images/single) -------------
@app.route("/api/images/single", methods=["GET"])
def get_images_single():
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        search_text = request.args.get("search", "")

        skip = (page - 1) * per_page

        # Build SQL query
        sql_conditions = []
        params = []

        if search_text:
            # e.g. search both filename and original_name using LIKE
            sql_conditions.append("(filename LIKE ? OR original_name LIKE ?)")
            like_pattern = f"%{search_text}%"
            params.append(like_pattern)
            params.append(like_pattern)

        # If you want to exclude "discarded", do:  sql_conditions.append("status != 'discarded'")

        where_clause = ""
        if sql_conditions:
            where_clause = "WHERE " + " AND ".join(sql_conditions)

        query_sql = f"""
            SELECT *
            FROM images
            {where_clause}
            LIMIT ? OFFSET ?
        """

        # Add limit + offset to params
        params.append(per_page)
        params.append(skip)

        response = []
        with get_db_connection() as conn:
            rows = conn.execute(query_sql, tuple(params)).fetchall()
            for row in rows:
                image_id = row["id"]
                image_path = row["path"]

                # Base64 encode
                with open(image_path, "rb") as f:
                    image_b64_str = base64.b64encode(f.read()).decode("utf-8")

                response.append({
                    "id": image_id,
                    "filename": row["filename"],
                    "original_name": row["original_name"],
                    "base64": image_b64_str,
                    "status": row["status"] or "",
                })

        return jsonify(response)
    except Exception as e:
        print(f"Error fetching images (single): {e}")
        return jsonify({"error": "Error fetching images"}), 500

# ------------- Multiple-mode route (/api/images/multiple) -------------
@app.route("/api/images/multiple", methods=["GET"])
def get_images_multiple():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
        search_text = request.args.get("search", "")

        skip = (page - 1) * limit

        # Build SQL query
        sql_conditions = []
        params = []

        if search_text:
            sql_conditions.append("(filename LIKE ? OR original_name LIKE ?)")
            like_pattern = f"%{search_text}%"
            params.append(like_pattern)
            params.append(like_pattern)

        # If you want to exclude "discarded", do:  sql_conditions.append("status != 'discarded'")

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
                image_path = row["path"]

                with open(image_path, "rb") as f:
                    image_b64_str = base64.b64encode(f.read()).decode("utf-8")

                response.append({
                    "id": image_id,
                    "filename": row["filename"],
                    "original_name": row["original_name"],
                    "base64": image_b64_str,
                    "status": row["status"] or "",
                })
        print(query_sql, tuple(params))
        return jsonify(response)
    except Exception as e:
        print(f"Error fetching images (multiple): {e}")
        return jsonify({"error": "Error fetching images"}), 500

# ------------- Discard route (/api/images/<image_id>/discard) -------------
@app.route("/api/images/<int:image_id>/discard", methods=["PUT"])
def discard_image(image_id):
    """
    Mark an image's status as 'discarded' in SQLite.
    """
    try:
        with get_db_connection() as conn:
            # Update the row
            cur = conn.execute("""
                UPDATE images
                SET status = 'discarded'
                WHERE id = ?
            """, (image_id,))
            if cur.rowcount == 0:
                return jsonify({"error": "Image not found"}), 404

            # Return the updated row to confirm
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
        print(f"Error discarding image: {e}")
        return jsonify({"error": "Error discarding image"}), 500


@app.route("/api/reference_image", methods=["GET"])
def get_reference_image():
    """
    Returns the reference image (Base64) for a given search word, if it exists.
    E.g. /api/reference_image?search=cat looks for reference_images/cat.jpg
    """
    try:
        search_text = request.args.get("search", "").strip()
        if not search_text:
            return jsonify({"error": "No search word"}), 400

        # The reference images folder
        reference_folder = "reference_images"

        # Construct path like reference_images/<search_text>.jpg
        ref_filename = f"{search_text}.jpg"
        ref_path = os.path.join(reference_folder, ref_filename)

        if not os.path.exists(ref_path):
            # Reference image not found
            return jsonify({"error": "Reference not available"}), 404

        # Encode and return
        with open(ref_path, "rb") as f:
            image_b64_str = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"base64": image_b64_str})

    except Exception as e:
        print(f"Error fetching reference image: {e}")
        return jsonify({"error": "Error fetching reference image"}), 500


# ------------- Main -------------
if __name__ == "__main__":
    # Ensure db is set up
    if not os.path.exists(DB_PATH):
        open(DB_PATH, 'a').close()  # create empty file if not present
    initialize_db()

    app.run(debug=True)
