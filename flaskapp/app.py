import os, pickle
from flask import Flask, request, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Path to the directory where images are stored
IMAGE_DIRECTORY = '/path/to/your/images'  # Update this with your actual directory path

# Load the list of image paths from the pickled file
images_list = []
with open('images_list.pkl', 'rb') as f:
    images_list = pickle.load(f)

# Set a maximum number of images to show (1000 images in total)
MAX_IMAGES = 1000
images_list = images_list[:MAX_IMAGES]  # Only use the first 1000 images

# Pagination settings
IMAGES_PER_PAGE = 20

@app.route('/')
def index():
    """Gallery page: show paginated images."""
    page = request.args.get('page', 1, type=int)
    total_images = len(images_list)
    total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE  # Total pages (ceiling division)

    # Bound the page number within valid range
    page = max(1, min(page, total_pages))

    # Slice the list of images for the current page
    start = (page - 1) * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    page_images = images_list[start:end]

    return render_template('gallery.html', images=page_images, page=page, total_pages=total_pages)

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve an image from the specified directory."""
    try:
        return send_from_directory(IMAGE_DIRECTORY, filename)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404

@app.route('/mark_bad', methods=['POST'])
def mark_bad():
    """AJAX endpoint to mark an image as bad."""
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "No image path provided"}), 400

    img_path = data['path']
    # Move image from good list to bad list
    if img_path in images_list:
        images_list.remove(img_path)
        # Append to bad.txt
        with open('bad.txt', 'a') as bf:
            bf.write(img_path + '\n')
        # Rewrite good.txt with current remaining images
        with open('good.txt', 'w') as gf:
            for path in images_list:
                gf.write(path + '\n')
        return jsonify({"status": "ok"}), 200
    else:
        # Image not in remaining list (perhaps already marked)
        return jsonify({"status": "already_marked"}), 200

if __name__ == '__main__':
    app.run(debug=True)

