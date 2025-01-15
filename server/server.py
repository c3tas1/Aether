from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from PIL import Image
import os
import base64
import datetime
import zipfile

# Database connection
client = MongoClient("mongodb://localhost:27017/")
db = client["Images_DB"]
image_collection = db["images"]

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
allowed_extensions = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


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

                        # Insert into MongoDB
                        image_data = {
                            "filename": new_filename,
                            "original_name": extracted_filename,
                            "path": new_path,
                            "annotations": [],
                        }
                        insert_result = image_collection.insert_one(image_data)
                        # Convert ObjectId to string for JSON serialization
                        image_data["_id"] = str(insert_result.inserted_id)
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

                image_data = {
                    "filename": filename,
                    "original_name": file_obj.filename,
                    "path": image_path,
                    "annotations": [],
                }
                insert_result = image_collection.insert_one(image_data)
                # Convert ObjectId to string
                image_data["_id"] = str(insert_result.inserted_id)
                saved_images.append(image_data)
            else:
                return jsonify({"error": "Unsupported file extension"}), 400

    return jsonify({"message": "Upload successful", "data": saved_images})


# Example single-mode (search/pagination) route
@app.route("/api/images/single", methods=["GET"])
def get_images_single():
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        search_text = request.args.get("search", "")

        skip = (page - 1) * per_page
        query = {}
        if search_text:
            query = {
                "$or": [
                    {"filename": {"$regex": search_text, "$options": "i"}},
                    {"original_name": {"$regex": search_text, "$options": "i"}},
                ]
            }

        images_cursor = image_collection.find(query).skip(skip).limit(per_page)

        response = []
        for image in images_cursor:
            # Convert the Mongo _id to string
            image_id = str(image["_id"])
            image_path = image["path"]
            # Base64 encode
            with open(image_path, "rb") as f:
                image_b64_str = base64.b64encode(f.read()).decode("utf-8")

            response.append(
                {
                    "id": image_id,
                    "filename": image["filename"],
                    "original_name": image.get("original_name", ""),
                    "base64": image_b64_str,
                }
            )

        return jsonify(response)
    except Exception as e:
        print(f"Error fetching images: {e}")
        return jsonify({"error": "Error fetching images"}), 500


# Example multiple-mode route
@app.route("/api/images/multiple", methods=["GET"])
def get_images_multiple():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
        search_text = request.args.get("search", "")

        skip = (page - 1) * limit
        query = {}
        if search_text:
            query = {
                "$or": [
                    {"filename": {"$regex": search_text, "$options": "i"}},
                    {"original_name": {"$regex": search_text, "$options": "i"}},
                ]
            }

        images_cursor = image_collection.find(query).skip(skip).limit(limit)

        response = []
        for image in images_cursor:
            image_id = str(image["_id"])
            image_path = image["path"]
            with open(image_path, "rb") as f:
                image_b64_str = base64.b64encode(f.read()).decode("utf-8")

            response.append(
                {
                    "id": image_id,
                    "filename": image["filename"],
                    "original_name": image.get("original_name", ""),
                    "base64": image_b64_str,
                }
            )

        return jsonify(response)
    except Exception as e:
        print(f"Error fetching images: {e}")
        return jsonify({"error": "Error fetching images"}), 500


if __name__ == "__main__":
    app.run(debug=True)
