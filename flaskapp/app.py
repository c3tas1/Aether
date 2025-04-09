import os, random
from flask import Flask, render_template, request, abort
from PIL import Image

app = Flask(__name__)

# Globals to hold image paths and labeled sets
image_paths_by_class = {}   # { class_name: [list_of_image_paths] }
all_images_set = set()      # set of all displayed image paths (for good/bad tracking)
bad_set = set()             # set of marked-bad image paths

# Configuration
MAX_IMAGES_PER_CLASS = 1000        # maximum thumbnails to display per class
THUMB_SIZE = 150                   # max pixel size (width or height) for thumbnail

def reset_output_files():
    """Clear or create the output files at the start of a session."""
    open('bad.txt', 'w').close()
    open('good.txt', 'w').close()

def init_image_list_and_thumbs(base_dir):
    """
    Walk through base_dir to collect image paths per class folder.
    Generate thumbnails for each image for faster loading.
    """
    global image_paths_by_class, all_images_set
    image_paths_by_class = {}
    all_images_set.clear()
    # Allowed image file extensions
    allowed_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP'}
    # Ensure thumbnail output directory exists under static/
    thumb_root = os.path.join(os.getcwd(), 'static', 'thumbs')
    os.makedirs(thumb_root, exist_ok=True)
    # Iterate over each top-level class directory
    for class_name in sorted(os.listdir(base_dir)):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # skip files in base_dir, only look at directories
        # Recursively gather all image file paths in this class directory
        image_files = []
        for root, dirs, files in os.walk(class_dir):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in allowed_exts:
                    # Compute path relative to base_dir for storage
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    image_files.append(rel_path)
        # If more than MAX_IMAGES_PER_CLASS, pick a random subset
        if len(image_files) > MAX_IMAGES_PER_CLASS:
            image_files = random.sample(image_files, MAX_IMAGES_PER_CLASS)
        # Shuffle the order (so displayed thumbnails are in random order)
        random.shuffle(image_files)
        image_paths_by_class[class_name] = image_files
        # Add to global set and create a thumbnail for each image
        for rel_path in image_files:
            all_images_set.add(rel_path)
            full_path = os.path.join(base_dir, rel_path)
            # Generate thumbnail image
            try:
                img = Image.open(full_path)
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                # Save thumbnail to static/thumbs/<relative_path>
                thumb_path = os.path.join(thumb_root, rel_path)
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                img.save(thumb_path)
            except Exception as e:
                # If any issue in generating a thumbnail, skip it (and use full image if needed)
                print(f"Warning: could not create thumbnail for {rel_path} ({e})")
    return image_paths_by_class

@app.route('/')
def index():
    """
    Main page: displays the sampled thumbnails in a grid per class.
    Already-marked bad patches (if any) are highlighted.
    """
    # Load already marked bad paths (in case of page refresh or existing file)
    marked_bad = set()
    if os.path.exists('bad.txt'):
        with open('bad.txt', 'r') as f:
            marked_bad = {line.strip() for line in f if line.strip()}
    # Render the HTML template, passing the image list and bad mark set
    return render_template('index.html', image_paths_by_class=image_paths_by_class, bad_set=marked_bad)

@app.route('/mark_bad', methods=['POST'])
def mark_bad():
    """
    AJAX endpoint: receives a JSON `{ "path": "<rel_path>" }` when a thumbnail is clicked.
    Marks the patch as bad by adding to bad_set and writing to bad.txt.
    """
    data = request.get_json()
    if not data or 'path' not in data:
        abort(400, "No image path provided")
    img_path = data['path']
    # Validate the path is one of the known images (prevent any malicious input)
    if img_path not in all_images_set:
        abort(400, "Invalid image path")
    # If not already marked, record it as bad
    if img_path not in bad_set:
        bad_set.add(img_path)
        with open('bad.txt', 'a') as bad_file:
            bad_file.write(img_path + '\n')
    # Respond with no content (HTTP 204) to indicate success
    return ('', 204)

@app.route('/finish', methods=['POST'])
def finish():
    """
    Endpoint to finalize labeling. Writes all unmarked image paths to good.txt and returns a confirmation.
    """
    # Compute good patches as those displayed but not marked bad
    good_set = all_images_set - bad_set
    with open('good.txt', 'w') as good_file:
        for path in sorted(good_set):
            good_file.write(path + '\n')
    # Simple HTML response to confirm completion
    msg = f"Labeling complete. Marked {len(bad_set)} patches as bad and {len(good_set)} as good."
    return f"<h3>{msg} Results saved to <code>bad.txt</code> and <code>good.txt</code>.</h3><p><a href=\"/\">Back to start</a></p>"

def run_app(base_dir, host="0.0.0.0", port=5000):
    """Initialize data and run the Flask development server."""
    reset_output_files()
    init_image_list_and_thumbs(base_dir)
    app.run(host=host, port=port)

# If running this script directly, allow passing base directory and optional host/port
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the image patch labeling Flask app")
    parser.add_argument('base_dir', help="Path to the root directory containing class folders")
    parser.add_argument('--host', default='0.0.0.0', help="Host IP to bind (default 0.0.0.0 for external access)")
    parser.add_argument('--port', type=int, default=5000, help="Port number (default 5000)")
    args = parser.parse_args()
    run_app(args.base_dir, host=args.host, port=args.port)

