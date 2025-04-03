import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import random
from PIL import Image
import io
from pathlib import Path
import math

# --- Configuration ---
NUM_SAMPLES = 100  # Number of random images to load per folder
IMAGE_WIDTH = 100  # Display width for each sample image
IMAGE_HEIGHT = 100 # Display height for each sample image
# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
# Directory to save labeled images (relative to notebook CWD)
OUTPUT_SAVE_DIR = Path("labeled_output_images") 

# --- Data Storage ---
# This dictionary will store the labels persistently within the kernel session
folder_labels = {}
# Cache for loaded image samples AND their original paths to speed up navigation & saving
# Structure: {folder_path: ( [original_path_1, ...], [image_bytes_1, ...] ) }
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
            print(f"Warning: Found {len(all_files)} images in {folder_path}, requested {num_samples}. Using all found images.")
        else:
            image_files = random.sample(all_files, num_samples)
            
    except Exception as e:
        print(f"Error scanning folder {folder_path}: {e}")
        return []
        
    return image_files

def load_and_prepare_image(image_path, target_width, target_height):
    """Loads, resizes, and converts an image to bytes for display."""
    try:
        img = Image.open(image_path)
        # Create a copy for thumbnail to avoid modifying the original object if needed elsewhere
        img_display = img.copy()
        img_display.thumbnail((target_width, target_height), Image.Resampling.LANCZOS) # Resize while maintaining aspect ratio
        
        # Convert to RGB if it's RGBA or P (Palette) for display bytes
        if img_display.mode in ('RGBA', 'P'):
            img_display = img_display.convert('RGB')
            
        # Save image to a byte buffer
        img_byte_arr = io.BytesIO()
        img_display.save(img_byte_arr, format='JPEG') # Save display version as JPEG
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error loading/processing image {image_path}: {e}")
        return None

def save_original_image_as_jpeg(original_path, output_path):
    """Loads the original image and saves it as JPEG to the output path."""
    try:
        print(f"Attempting to save {original_path} as {output_path}")
        img = Image.open(original_path)
        
        # Convert to RGB before saving as JPEG to handle transparency etc.
        if img.mode in ('RGBA', 'P', 'LA') or (img.mode == 'L' and 'transparency' in img.info):
             # If it has alpha or palette, convert to RGB
             print(f"  Converting image from mode {img.mode} to RGB for JPEG saving.")
             img = img.convert('RGB')
        elif img.mode != 'RGB':
             # Handle other modes like Grayscale ('L') if needed, or just try saving
             # For simplicity, we'll try converting other non-RGB modes too
             try:
                 print(f"  Attempting to convert image from mode {img.mode} to RGB.")
                 img = img.convert('RGB')
             except Exception as convert_err:
                 print(f"  Warning: Could not convert image mode {img.mode} to RGB: {convert_err}. Saving might fail or look incorrect.")


        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img.save(output_path, format='JPEG', quality=90) # Save as JPEG
        print(f"Successfully saved labeled image to: {output_path}")
        return True
    except FileNotFoundError:
         print(f"Error saving: Original file not found at {original_path}")
         return False
    except Exception as e:
        print(f"Error saving image {original_path} to {output_path}: {e}")
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
        
        # Check cache first
        if current_folder_path in image_samples_cache:
            print("Loading images from cache...")
            original_image_paths, image_byte_list = image_samples_cache[current_folder_path]
        else:
            # Load, prepare and cache images
            print(f"Scanning for {NUM_SAMPLES} random images...")
            image_files = get_image_files(current_folder_path, NUM_SAMPLES)
            image_byte_list = []
            original_image_paths = [] # Keep track of original paths corresponding to bytes
            
            if image_files:
                print(f"Loading {len(image_files)} images...")
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
             print("Creating image widgets...")
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
    """Handles common logic for good/bad button clicks."""
    global folder_labels, image_samples_cache, OUTPUT_SAVE_DIR
    
    if not (0 <= current_folder_index < len(folder_list)):
        print("Error: No valid folder selected.")
        return
        
    current_folder_path = folder_list[current_folder_index]
    folder_key = current_folder_path.name # Use folder name as key
    
    # 1. Update the persistent dictionary
    folder_labels[folder_key] = label_value
    print(f"Labeled '{folder_key}' as {label_value.upper()}. Current labels: {folder_labels}")

    # 2. Save a representative image
    if current_folder_path in image_samples_cache:
        original_paths, _ = image_samples_cache[current_folder_path]
        if original_paths:
            # Select the first sampled image's original path
            image_to_save_path = original_paths[0] 
            output_filename = OUTPUT_SAVE_DIR / f"{folder_key}_{label_value}.jpg"
            
            # Ensure output directory exists
            try:
                OUTPUT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
                save_original_image_as_jpeg(image_to_save_path, output_filename)
            except Exception as e:
                 print(f"Error during image save setup or call: {e}")
                 
        else:
            print(f"No sampled images available in cache for {folder_key} to save.")
    else:
        print(f"Cannot save image: Folder {folder_key} not found in image cache (should not happen).")

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
print(f"Labeled images will be saved to the '{OUTPUT_SAVE_DIR}' directory.")
display(ui)

# You can access the results later by checking the `folder_labels` dictionary
# print(folder_labels)
