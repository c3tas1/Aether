import flask
from flask import Flask, render_template, request, send_file, jsonify, abort
import pickle
import os
import math
from PIL import Image # Pillow library for image processing
import io
import base64
import threading # For file locking

app = Flask(__name__)

# --- Configuration ---
PICKLE_FILE = 'image_paths.pkl'
BAD_FILE = 'bad.txt'
GOOD_FILE = 'good.txt'
IMAGES_PER_PAGE = 1000
THUMBNAIL_SIZE = (128, 128)
# IMPORTANT: Set this to the base directory where images are stored
# This adds a layer of security, preventing access outside this tree.
# Set to None or '/' if images are truly scattered, but be VERY careful.
ALLOWED_IMAGE_BASE_PATH = '/path/to/your/images/base/directory' # Or None

# --- Globals ---
# Load image paths once at startup
all_image_paths = []
try:
    with open(PICKLE_FILE, 'rb') as f:
        # Ensure paths are absolute and normalized for consistency
        all_image_paths = [os.path.abspath(p) for p in pickle.load(f)]
    print(f"Loaded {len(all_image_paths)} image paths from {PICKLE_FILE}")
except FileNotFoundError:
    print(f"Error: {PICKLE_FILE} not found. Please create it.")
    # Exit or provide default empty list? Depending on requirements.
    # For now, continue with empty list, but app might not work fully.
except Exception as e:
    print(f"Error loading pickle file {PICKLE_FILE}: {e}")

# File locks to prevent race conditions when writing
bad_file_lock = threading.Lock()
good_file_lock = threading.Lock()

# --- Helper Functions ---

def is_path_allowed(path_to_check):
    """Security check: Ensure the path is one we loaded or under the allowed base path."""
    abs_path_to_check = os.path.abspath(path_to_check)

    # 1. Check if it's exactly one of the paths loaded initially
    if abs_path_to_check in all_image_paths:
        return True

    # 2. (Optional but Recommended) Check if it's within the allowed base directory
    if ALLOWED_IMAGE_BASE_PATH:
        abs_allowed_base = os.path.abspath(ALLOWED_IMAGE_BASE_PATH)
        if os.path.commonpath([abs_path_to_check, abs_allowed_base]) == abs_allowed_base:
             # Additional check: Ensure it was in the original list to prevent arbitrary access
             # within the base path if the path wasn't originally intended.
             # This double-checks against path manipulation vulnerabilities.
             # Remove this line if you *trust* that any path under base is OK.
             return abs_path_to_check in all_image_paths
    
    # If neither check passes, deny access.
    print(f"Access denied for path: {path_to_check}") # Log denied attempts
    return False

def encode_path(path):
    """Encode path for safe use in URL."""
    return base64.urlsafe_b64encode(path.encode('utf-8')).decode('utf-8')

def decode_path(encoded_path):
    """Decode path from URL."""
    try:
        return base64.urlsafe_b64decode(encoded_path.encode('utf-8')).decode('utf-8')
    except Exception as e:
        print(f"Error decoding path '{encoded_path}': {e}")
        return None # Indicate decoding failure

# --- Routes ---

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    """Displays the paginated image grid."""
    if not all_image_paths:
        return "Error: No image paths loaded. Check pickle file.", 500

    total_images = len(all_image_paths)
    total_pages = math.ceil(total_images / IMAGES_PER_PAGE)
    page = max(1, min(page, total_pages)) # Ensure page is within valid range

    start_index = (page - 1) * IMAGES_PER_PAGE
    end_index = start_index + IMAGES_PER_PAGE
    current_page_paths = all_image_paths[start_index:end_index]

    # Encode paths for use in URLs
    encoded_page_paths = [(path, encode_path(path)) for path in current_page_paths]

    return render_template('index.html',
                           image_data=encoded_page_paths,
                           current_page=page,
                           total_pages=total_pages,
                           thumbnail_size=THUMBNAIL_SIZE)

@app.route('/image/<encoded_path>')
def serve_image(encoded_path):
    """Serves a resized thumbnail of the requested image."""
    image_path = decode_path(encoded_path)

    if not image_path:
        abort(400, "Invalid encoded path.") # Bad Request

    # SECURITY CHECK: Very important!
    if not is_path_allowed(image_path):
         abort(403, f"Access denied to image path: {image_path}") # Forbidden

    if not os.path.exists(image_path):
        abort(404, f"Image not found: {image_path}") # Not Found

    try:
        img = Image.open(image_path)
        img.thumbnail(THUMBNAIL_SIZE) # Resize while maintaining aspect ratio

        # Serve image from memory to avoid saving temporary files
        img_io = io.BytesIO()
        img_format = img.format if img.format else 'JPEG' # Use original format or default to JPEG
        # Handle potential issues with saving certain formats like WebP without specific libraries
        if img_format == 'MPO': img_format = 'JPEG' # Common issue format
        
        try:
            img.save(img_io, img_format, quality=85) # Adjust quality as needed
        except OSError as e:
             # If saving fails for the format, try JPEG as a fallback
             print(f"Warning: Could not save image in format {img_format} ({e}). Trying JPEG.")
             img_format = 'JPEG'
             # Convert to RGB if necessary for JPEG
             if img.mode in ("RGBA", "P"): 
                 img = img.convert("RGB")
             img.save(img_io, img_format, quality=85)

        img_io.seek(0)
        
        # Determine mimetype based on format
        mimetype = Image.MIME.get(img_format.upper(), 'image/jpeg')

        return send_file(img_io, mimetype=mimetype)

    except FileNotFoundError:
         abort(404, f"Image not found on disk (race condition?): {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        abort(500, "Error processing image.") # Internal Server Error

@app.route('/mark', methods=['POST'])
def mark_files():
    """Handles marking an image as bad and others as good."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid request format."}), 400

    bad_image_path_encoded = data.get('bad_image_path_encoded')
    displayed_paths_encoded = data.get('displayed_paths_encoded', [])

    if not bad_image_path_encoded:
        return jsonify({"status": "error", "message": "Missing 'bad_image_path_encoded'."}), 400

    bad_image_path = decode_path(bad_image_path_encoded)
    if not bad_image_path or not is_path_allowed(bad_image_path): # Security check
        return jsonify({"status": "error", "message": "Invalid or disallowed bad image path."}), 400

    displayed_paths = []
    for encoded in displayed_paths_encoded:
        decoded = decode_path(encoded)
        if decoded and is_path_allowed(decoded): # Security check
            displayed_paths.append(decoded)
        else:
            print(f"Warning: Skipping invalid/disallowed displayed path during mark: {encoded}")


    # --- Update bad.txt ---
    try:
        with bad_file_lock: # Acquire lock
            with open(BAD_FILE, 'a', encoding='utf-8') as f:
                f.write(bad_image_path + '\n')
    except Exception as e:
        print(f"Error writing to {BAD_FILE}: {e}")
        return jsonify({"status": "error", "message": f"Failed to write to {BAD_FILE}"}), 500
    finally:
        # Ensure lock is always released, though `with` handles this
        pass 

    # --- Update good.txt ---
    good_paths_to_write = []
    for p in displayed_paths:
        if p != bad_image_path:
            good_paths_to_write.append(p)

    if good_paths_to_write:
        try:
            with good_file_lock: # Acquire lock
                with open(GOOD_FILE, 'a', encoding='utf-8') as f:
                    for p in good_paths_to_write:
                        f.write(p + '\n')
        except Exception as e:
            print(f"Error writing to {GOOD_FILE}: {e}")
            # Continue even if good fails, but report it maybe?
            return jsonify({"status": "warning", "message": f"Marked bad, but failed to write to {GOOD_FILE}"}), 500
        finally:
            # Ensure lock is always released
             pass

    print(f"Marked bad: {bad_image_path}")
    print(f"Marked good ({len(good_paths_to_write)}): {good_paths_to_write[:5]}...") # Log first few

    return jsonify({"status": "ok", "message": "Files marked successfully."})

# --- Run Application ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    # Debug=True is helpful during development but SHOULD be False in production
    app.run(debug=True, host='0.0.0.0', port=5000)
