import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import random
from PIL import Image
import io
from pathlib import Path
import math # Needed for calculating grid size

# --- Configuration ---
NUM_SAMPLES = 100  # Number of random images to load per folder
IMAGE_WIDTH = 100  # Display width AND width of each cell in the output montage
IMAGE_HEIGHT = 100 # Display height AND height of each cell in the output montage
# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
# Directory to save labeled montage images (relative to notebook CWD)
OUTPUT_SAVE_DIR = Path("labeled_montage_images") # Changed name slightly for clarity
MONTAGE_BACKGROUND_COLOR = 'white' # Background color for the montage grid

# --- Data Storage ---
# This dictionary will store the labels persistently within the kernel session
folder_labels = {}
# Cache for loaded image samples AND their original paths
# Structure: {folder_path: ( [original_path_1, ...], [image_bytes_1, ...] ) }
# We primarily need image_bytes for the montage.
image_samples_cache = {}
# List of folder paths (replace with your actual list)
folder_list = [] # <<< --- !!! PUT YOUR LIST OF FOLDER PATHS HERE !!! --- >>>
# Example:
# folder_list = ['path/to/folder1', 'path/to/folder2', 'path/to/another/folder']
# Or use Path objects:
# folder_list = [Path('data/cats'), Path('data/dogs'), Path('data/birds')]

# Ensure folder_list contains absolute paths or paths relative to the notebook's CWD
# Resolve paths for consistency, especially for caching keys
if folder_list:
    folder_list = [Path(p).resolve() for p in folder_list]

# --- State ---
current_folder_index = -1 # Start at -1 so initial display call sets it to 0

# --- Helper Functions ---

def get_image_files(folder_path, num_samples):
    """Gets a list of random image file paths from a folder."""
    image_files = []
    try:
        p = Path(folder_path)
        if not p.is_dir():
            print(f"Error: Folder not found or is not a directory: {folder_path}")
            return []
        
        all_files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        
        if not all_files:
            print(f"Warning: No image files found in {folder_path}")
            return []
            
        if len(all_files) <= num_samples:
            image_files = all_files
            # Don't print warning here, let the caller handle sample size feedback
            # print(f"Warning: Found {len(all_files)} images in {folder_path}, requested {num_samples}. Using all found images.")
        else:
            image_files = random.sample(all_files, num_samples)
            
    except Exception as e:
        print(f"Error scanning folder {folder_path}: {e}")
        return []
        
    return image_files

def load_and_prepare_image(image_path, target_width, target_height):
    """Loads, resizes, and converts an image to bytes for display/montage."""
    try:
        img = Image.open(image_path)
        # Use thumbnail which resizes in place while maintaining aspect ratio,
        # fitting within the target box.
        img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Create a new background image with the exact target size
        # Paste the potentially smaller thumbnail onto the center of it
        # This ensures all montage cells are the same size, even if originals had different aspect ratios
        img_bg = Image.new('RGB', (target_width, target_height), MONTAGE_BACKGROUND_COLOR)
        paste_x = (target_width - img.width) // 2
        paste_y = (target_height - img.height) // 2
        img_bg.paste(img, (paste_x, paste_y))

        # Convert final image to RGB if necessary (though background is RGB already)
        if img_bg.mode != 'RGB':
             img_bg = img_bg.convert('RGB')
            
        # Save image to a byte buffer
        img_byte_arr = io.BytesIO()
        img_bg.save(img_byte_arr, format='JPEG') # Save display version as JPEG
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error loading/processing image {image_path}: {e}")
        return None

def create_and_save_montage(image_bytes_list, num_samples_actual, output_path, cell_width, cell_height):
    """Creates a grid montage from a list of image bytes and saves it."""
    if not image_bytes_list:
        print("Error: No image bytes provided for montage.")
        return False

    try:
        num_images = len(image_bytes_list)
        if num_images == 0:
             print("Error: Empty image list for montage.")
             return False
             
        # Calculate grid dimensions
        grid_cols = math.ceil(math.sqrt(num_images))
        # Calculate rows needed based on columns, ensuring all images fit
        grid_rows = math.ceil(num_images / grid_cols) 
        
        montage_width = grid_cols * cell_width
        montage_height = grid_rows * cell_height
        
        print(f"Creating a {grid_cols}x{grid_rows} montage ({montage_width}x{montage_height} pixels)...")
        
        # Create the blank montage canvas
        montage_image = Image.new('RGB', (montage_width, montage_height), MONTAGE_BACKGROUND_COLOR)
        
        # Paste each image onto the montage
        for i, img_bytes in enumerate(image_bytes_list):
            row = i // grid_cols
            col = i % grid_cols
            paste_x = col * cell_width
            paste_y = row * cell_height
            
            try:
                # Open the image from bytes
                img = Image.open(io.BytesIO(img_bytes))
                # Paste onto the montage
                montage_image.paste(img, (paste_x, paste_y))
            except Exception as paste_err:
                print(f"  Warning: Could not load/paste image #{i+1} into montage: {paste_err}")
                # Optionally draw a placeholder box or skip
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the final montage image
        montage_image.save(output_path, format='JPEG', quality=90) # Adjust quality as needed
        print(f"Successfully saved montage image to: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating or saving montage image to {output_path}: {e}")
        return False

# --- Widgets ---
output_area = widgets.Output() # Area to display images and messages
folder_label = widgets.Label(value="No folder loaded")
prev_button = widgets.Button(description="< Previous", icon='arrow-left', button_style='info')
next_button = widgets.Button(description="Next >", icon='arrow-right', button_style='info')
good_button = widgets.Button(description="Good 👍", button_style='success')
bad_button = widgets.Button(description="Bad 👎", button_style='danger')

# Layout for images (using HBox with wrap to create a grid-like feel)
images_layout = widgets.HBox(
    [], 
    layout=widgets.Layout(
        flex_flow='row wrap', # Allow items to wrap to the next line
        justify_content='flex-start', # Align items to the start
        align_items='center', # Center items vertically if rows have different heights
        width='100%' # Take full width
    )
)

# --- Display Logic ---
def display_folder(index):
    global current_folder_index, folder_list, image_samples_cache, folder_labels
    
    if not folder_list or not (0 <= index < len(folder_list)):
        with output_area:
            clear_output(wait=True)
            print("No folders specified or index out of bounds.")
        folder_label.value = "Invalid index or empty folder list"
        prev_button.disabled = True
        next_button.disabled = True
        good_button.disabled = True
        bad_button.disabled = True
        return

    current_folder_index = index
    current_folder_path = folder_list[current_folder_index]
    folder_name = current_folder_path.name # Display only the folder name
    folder_label.value = f"Folder ({current_folder_index + 1}/{len(folder_list)}): {folder_name}"

    # Enable/disable navigation buttons
    prev_button.disabled = (current_folder_index == 0)
    next_button.disabled = (current_folder_index == len(folder_list) - 1)
    good_button.disabled = False
    bad_button.disabled = False

    # Clear previous output and display loading message
    with output_area:
        clear_output(wait=True)
        print(f"Loading images from: {current_folder_path} ...")

        image_widgets = []
        original_image_paths = []
        image_byte_list = [] # Define here to ensure it's available later
        
        # Check cache first
        if current_folder_path in image_samples_cache:
            print("Loading images from cache...")
            original_image_paths, image_byte_list = image_samples_cache[current_folder_path]
        else:
            # Load, prepare and cache images
            print(f"Scanning for up to {NUM_SAMPLES} random images...")
            image_files = get_image_files(current_folder_path, NUM_SAMPLES)
            actual_num_loaded = len(image_files)
            print(f"Found {actual_num_loaded} images. Processing...")
            
            # Reset lists for this folder
            image_byte_list = [] 
            original_image_paths = [] 
            
            if image_files:
                for i, img_path in enumerate(image_files):
                    # Add progress feedback for large numbers of images
                    if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                         print(f"  Processed {i+1}/{len(image_files)}")
                    img_bytes = load_and_prepare_image(img_path, IMAGE_WIDTH, IMAGE_HEIGHT)
                    if img_bytes:
                        image_byte_list.append(img_bytes)
                        original_image_paths.append(img_path) # Store original path
                # Store tuple (original paths, byte data) in cache
                image_samples_cache[current_folder_path] = (original_image_paths, image_byte_list) 
            else:
                 print(f"No images loaded for {current_folder_path}")
                 image_samples_cache[current_folder_path] = ([], []) # Cache empty result

        # Create image widgets from bytes
        if image_byte_list:
             print(f"Creating {len(image_byte_list)} image widgets...")
             for img_bytes in image_byte_list:
                 img_widget = widgets.Image(
                     value=img_bytes,
                     format='jpeg', # Format we saved the bytes in
                     width=IMAGE_WIDTH,
                     height=IMAGE_HEIGHT,
                     layout=widgets.Layout(margin='5px') # Add some spacing
                 )
                 image_widgets.append(img_widget)
        else:
            print("No images to display.") # Message if no images were loaded/found

        # Update the HBox containing the images
        images_layout.children = image_widgets
        
        # Display the layout container
        clear_output(wait=True) # Clear the loading message
        if not image_byte_list:
             print("No images found or loaded for this folder.") # Show message within widget area too
        display(images_layout) # Display the box with images


# --- Event Handlers ---
def handle_labeling(label_value):
    """Handles common logic for good/bad button clicks including saving montage."""
    global folder_labels, image_samples_cache, OUTPUT_SAVE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT
    
    if not (0 <= current_folder_index < len(folder_list)):
        print("Error: No valid folder selected.")
        return
        
    current_folder_path = folder_list[current_folder_index]
    folder_key = current_folder_path.name # Use folder name as key
    
    # 1. Update the persistent dictionary
    folder_labels[folder_key] = label_value
    print(f"Labeled '{folder_key}' as {label_value.upper()}. Current labels: {folder_labels}")

    # 2. Save a representative montage image
    if current_folder_path in image_samples_cache:
        original_paths, image_byte_list = image_samples_cache[current_folder_path]
        
        if image_byte_list:
            # Construct the output filename
            output_filename = OUTPUT_SAVE_DIR / f"{folder_key}_{label_value}.jpg"
            
            # Create and save the montage
            create_and_save_montage(
                image_bytes_list=image_byte_list,
                num_samples_actual=len(image_byte_list), # Pass actual count
                output_path=output_filename,
                cell_width=IMAGE_WIDTH,
                cell_height=IMAGE_HEIGHT
            )
                 
        else:
            print(f"No sampled images available in cache for {folder_key} to create a montage.")
    else:
        print(f"Cannot save montage: Folder {folder_key} not found in image cache (this might indicate an issue).")

    # Optional: Automatically advance to the next folder after labeling
    # if current_folder_index < len(folder_list) - 1:
    #     on_next_button_clicked(None)

def on_prev_button_clicked(b):
    if current_folder_index > 0:
        display_folder(current_folder_index - 1)

def on_next_button_clicked(b):
    if current_folder_index < len(folder_list) - 1:
        display_folder(current_folder_index + 1)

def on_good_button_clicked(b):
    handle_labeling("good")

def on_bad_button_clicked(b):
    handle_labeling("bad")

# --- Connect Handlers ---
prev_button.on_click(on_prev_button_clicked)
next_button.on_click(on_next_button_clicked)
good_button.on_click(on_good_button_clicked)
bad_button.on_click(on_bad_button_clicked)

# --- Initial Display ---
# Check if folder_list is provided before trying to display
if folder_list:
    # Create the output directory proactively if it doesn't exist
    try:
        OUTPUT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_SAVE_DIR.resolve()}")
    except Exception as e:
        print(f"Warning: Could not create output directory '{OUTPUT_SAVE_DIR}': {e}")
        
    display_folder(0) # Display the first folder initially
else:
    folder_label.value = "Please provide a list of folders in 'folder_list'."
    prev_button.disabled = True
    next_button.disabled = True
    good_button.disabled = True
    bad_button.disabled = True


# --- Assemble the UI ---
controls_top = widgets.HBox([prev_button, next_button, folder_label], layout=widgets.Layout(justify_content='space-around'))
controls_bottom = widgets.HBox([good_button, bad_button], layout=widgets.Layout(justify_content='space-around', margin='10px 0 0 0'))

ui = widgets.VBox([controls_top, output_area, controls_bottom])

# --- Display the Widget ---
print("Widget Setup Complete. Make sure 'folder_list' is populated.")
print(f"Persistent labels will be stored in the 'folder_labels' dictionary.")
print(f"Labeled image montages will be saved to the '{OUTPUT_SAVE_DIR}' directory.")
display(ui)

# You can access the results later by checking the `folder_labels` dictionary
# print(folder_labels)

