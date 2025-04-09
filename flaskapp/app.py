import os, pickle

# Attempt to import requests for remote file fetching
try:
    import requests
except ImportError:
    requests = None

from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the list of image paths from a pickled file (remote or local)
images_list = []
remote_url = 'http://example.com/path/to/images_list.pkl'  # TODO: replace with actual URL

if requests:
    try:
        resp = requests.get(remote_url, timeout=5)
        if resp.status_code == 200:
            # Unpickle the data from the remote content
            images_list = pickle.loads(resp.content)
    except Exception as e:
        print(f"Warning: Could not load remote pickle: {e}")

if not images_list:
    # Fallback to local pickle file if remote fetch failed or not used
    with open('images_list.pkl', 'rb') as f:
        images_list = pickle.load(f)

# Ensure we have a list of image path strings
if not isinstance(images_list, list):
    images_list = list(images_list)

# Initialize "good" and "bad" files
BAD_FILE = 'bad.txt'
GOOD_FILE = 'good.txt'
with open(BAD_FILE, 'w') as bf:
    pass  # start with empty bad.txt
with open(GOOD_FILE, 'w') as gf:
    for path in images_list:
        gf.write(path + '\n')

# Keep track of remaining (unclicked) images in memory
remaining_images = images_list[:]  # copy of the list

@app.route('/')
def index():
    """Gallery page: show paginated images."""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    total = len(remaining_images)
    total_pages = (total + per_page - 1) // per_page  # ceiling division for total pages

    # Bound the page number within valid range
    if page < 1:
        page = 1
    if total_pages > 0 and page > total_pages:
        page = total_pages

    # Slice the list of remaining images for the current page
    start = (page - 1) * per_page
    end = start + per_page
    page_images = remaining_images[start:end]

    return render_template('gallery.html', images=page_images, page=page, total_pages=total_pages)

@app.route('/mark_bad', methods=['POST'])
def mark_bad():
    """AJAX endpoint to mark an image as bad."""
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "No image path provided"}), 400

    img_path = data['path']
    if img_path in remaining_images:
        # Move image from good list to bad list
        remaining_images.remove(img_path)
        # Append to bad.txt
        with open(BAD_FILE, 'a') as bf:
            bf.write(img_path + '\n')
        # Rewrite good.txt with current remaining images
        with open(GOOD_FILE, 'w') as gf:
            for path in remaining_images:
                gf.write(path + '\n')
        return jsonify({"status": "ok"}), 200
    else:
        # Image not in remaining list (perhaps already marked)
        return jsonify({"status": "already_marked"}), 200

# (Optional) Run the Flask development server if this script is executed directly.
if __name__ == '__main__':
    app.run(debug=True)

