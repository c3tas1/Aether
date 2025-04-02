import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import random
from collections import defaultdict
import io # For handling image bytes

# --- 1. Configuration & Data ---

# --- YOU NEED TO ADAPT THIS SECTION ---

# Placeholder: Your actual confusion map
confusion_map = {
    "cat": {
        "num of samples": 200,
        "num of miss classifications": 15,
        "classes confused on": {
            "dog": 8,
            "fox": 5,
            "lynx": 2,
        }
    },
    "dog": {
        "num of samples": 250,
        "num of miss classifications": 12,
        "classes confused on": {
            "cat": 6,
            "wolf": 4,
            "fox": 2,
        }
    },
    "fox": {
        "num of samples": 180,
        "num of miss classifications": 9,
        "classes confused on": {
            "dog": 4,
            "cat": 3,
            "coyote": 2,
        }
    },
    # Add other classes...
     "wolf": {"num of samples": 100, "num of miss classifications": 0, "classes confused on": {}},
     "lynx": {"num of samples": 80, "num of miss classifications": 0, "classes confused on": {}},
     "coyote": {"num of samples": 90, "num of miss classifications": 0, "classes confused on": {}},
}

# --- CRITICAL: Implement this function based on your dataset structure ---
def get_image_paths(class_name, num_samples=50):
    """
    Retrieves paths for a specified number of sample images for a given class.

    Args:
        class_name (str): The name of the class.
        num_samples (int): The maximum number of image paths to return.

    Returns:
        list: A list of file paths for the sample images.
              Returns an empty list if the directory doesn't exist or has no images.
    """
    # Example implementation: Assumes images are in folders named after classes
    # E.g., ./dataset/cat/image1.jpg, ./dataset/dog/img_abc.png etc.
    base_image_folder = "./dataset" # <-- ADAPT THIS PATH
    class_folder = os.path.join(base_image_folder, class_name)
    
    if not os.path.isdir(class_folder):
        print(f"Warning: Directory not found for class '{class_name}': {class_folder}")
        return []
        
    try:
        all_images = [
            os.path.join(class_folder, f) 
            for f in os.listdir(class_folder) 
            if os.path.isfile(os.path.join(class_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        # Ensure we don't request more samples than available
        num_samples = min(num_samples, len(all_images))
        return random.sample(all_images, num_samples)
    except Exception as e:
        print(f"Error reading images for class '{class_name}': {e}")
        return []
# --- End of Adaptation Section ---


# Dictionary to store the analysis reasons
# Using defaultdict for convenience
analysis_reasons = defaultdict(lambda: defaultdict(list))

# Predefined reasons for confusion
confusion_reasons = [
    "Looks Similar",
    "Dataset Not Consistent",
    "Low Sample Quality",
    "Background Confusion",
    "Same Shape",
    "Same Color",
    "Random Confusion",
    "Ambiguous Features",
    # Add more reasons as needed
]

# --- 2. UI Widgets ---

# Dropdown for selecting the primary class
primary_class_dropdown = widgets.Dropdown(
    options=list(confusion_map.keys()),
    description='Primary Class:',
    style={'description_width': 'initial'}
)

# Dropdown for selecting the confused class (options updated dynamically)
confused_class_dropdown = widgets.Dropdown(
    options=[], # Initially empty
    description='Confused With:',
    disabled=True, # Disabled until a primary class with confusions is selected
    style={'description_width': 'initial'}
)

# Output widgets to display images (using GridBox for better layout)
# We'll update the 'children' property of these VBoxes later
primary_images_display = widgets.VBox([widgets.Label("Primary class images will appear here.")])
confused_images_display = widgets.VBox([widgets.Label("Confused class images will appear here.")])

# Output widget to show feedback or the updated reasons dict
output_area = widgets.Output()
reasons_display_area = widgets.Output(layout={'border': '1px solid black', 'padding': '5px', 'margin_top': '10px'})


# Buttons for selecting confusion reasons
reason_buttons = [widgets.Button(description=reason, button_style='info', layout=widgets.Layout(margin='2px')) for reason in confusion_reasons]
buttons_box = widgets.HBox(reason_buttons, layout=widgets.Layout(flex_flow='row wrap', justify_content='center'))

# Function to create an image grid
def create_image_grid(image_paths, images_per_row=10):
    """Creates a GridBox widget containing images."""
    if not image_paths:
        return widgets.Label("No images found or path not configured correctly.")
        
    image_widgets = []
    for img_path in image_paths:
        try:
            with open(img_path, 'rb') as f:
                image_bytes = f.read()
            # Adjust format based on your image types if needed (e.g., 'png', 'jpeg')
            img_widget = widgets.Image(
                value=image_bytes, 
                # format='png', # Optional: specify format if needed
                width=80,      # Adjust size as needed
                height=80,     # Adjust size as needed
                layout=widgets.Layout(margin='2px')
            )
            image_widgets.append(img_widget)
        except FileNotFoundError:
            image_widgets.append(widgets.Label(f"Not Found: {os.path.basename(img_path)}", layout=widgets.Layout(width='80px', height='80px', border='1px dashed red')))
        except Exception as e:
            image_widgets.append(widgets.Label(f"Error: {os.path.basename(img_path)}", layout=widgets.Layout(width='80px', height='80px', border='1px dashed red')))
            print(f"Error loading image {img_path}: {e}")
            
    
    grid = widgets.GridBox(
        children=image_widgets,
        layout=widgets.Layout(
            grid_template_columns=f'repeat({images_per_row}, auto)', 
            grid_gap='5px',
            padding='5px',
            border='1px solid lightgrey' # Optional border
        )
    )
    return grid

# Function to display the current analysis reasons
def display_analysis_dict():
     with reasons_display_area:
         clear_output(wait=True)
         # Convert defaultdicts to regular dicts for cleaner printing
         print_dict = {k: dict(v) for k, v in analysis_reasons.items()}
         import json # Use JSON for pretty printing
         print("Current Analysis Reasons:")
         print(json.dumps(print_dict, indent=2))

# --- 3. Interaction Logic (Callback Functions) ---

def update_primary_images(change):
    """Callback when primary class dropdown changes."""
    primary_class = change['new']
    with output_area:
        clear_output()
        print(f"Loading images for primary class: {primary_class}...")

    # Update primary images display
    image_paths = get_image_paths(primary_class, num_samples=50)
    primary_images_display.children = [widgets.Label(f"Sample Images for: {primary_class} ({len(image_paths)} shown)"), create_image_grid(image_paths)]

    # Update confused class dropdown options
    confused_classes = list(confusion_map[primary_class]["classes confused on"].keys())
    if confused_classes:
        confused_class_dropdown.options = confused_classes
        confused_class_dropdown.disabled = False
        confused_class_dropdown.index = 0 # Select the first confused class by default
        # Manually trigger the update for the confused class display
        update_confused_images({'new': confused_class_dropdown.value, 'owner': confused_class_dropdown}) 
    else:
        confused_class_dropdown.options = []
        confused_class_dropdown.disabled = True
        confused_images_display.children = [widgets.Label("No misclassifications recorded for this class.")]
        # Disable reason buttons if no confusion
        for btn in reason_buttons:
            btn.disabled = True

    with output_area:
        clear_output()
        print(f"Displayed images for {primary_class}. Select a 'Confused With' class.")


def update_confused_images(change):
    """Callback when confused class dropdown changes."""
    confused_class = change['new']
    primary_class = primary_class_dropdown.value # Get current primary class
    
    if not confused_class or confused_class_dropdown.disabled:
         confused_images_display.children = [widgets.Label("Select a 'Confused With' class.")]
         # Disable reason buttons if no confusion class selected
         for btn in reason_buttons:
             btn.disabled = True
         return

    with output_area:
        clear_output()
        print(f"Loading images for confused class: {confused_class}...")

    # Update confused images display
    image_paths = get_image_paths(confused_class, num_samples=50)
    num_confusions = confusion_map[primary_class]["classes confused on"].get(confused_class, 0)
    confused_images_display.children = [widgets.Label(f"'{primary_class}' was confused with '{confused_class}' {num_confusions} times. Sample Images for: {confused_class} ({len(image_paths)} shown)"), create_image_grid(image_paths)]

    # Enable reason buttons
    for btn in reason_buttons:
        btn.disabled = False

    with output_area:
        clear_output()
        print(f"Displayed images for {confused_class}. Select a reason for confusion below.")
    
    # Refresh the display of reasons
    display_analysis_dict()


def on_reason_button_clicked(button):
    """Callback when a reason button is clicked."""
    primary_class = primary_class_dropdown.value
    confused_class = confused_class_dropdown.value
    reason = button.description

    if not primary_class or not confused_class or confused_class_dropdown.disabled:
        with output_area:
            clear_output()
            print("Error: Please ensure both primary and confused classes are selected.")
        return

    # Add the reason (avoid duplicates if desired, currently allows duplicates)
    if reason not in analysis_reasons[primary_class][confused_class]:
         analysis_reasons[primary_class][confused_class].append(reason)
    # Or simply append: analysis_reasons[primary_class][confused_class].append(reason)


    with output_area:
        clear_output()
        print(f"Reason '{reason}' added for {primary_class} -> {confused_class}.")

    # Update the display of the reasons dictionary
    display_analysis_dict()


# --- 4. Connect Widgets to Callbacks ---

primary_class_dropdown.observe(update_primary_images, names='value')
confused_class_dropdown.observe(update_confused_images, names='value')

for btn in reason_buttons:
    btn.on_click(on_reason_button_clicked)
    btn.disabled = True # Initially disable until confusion pair is selected

# --- 5. Layout and Display ---

# Combine UI elements vertically
ui_panel = widgets.VBox([
    widgets.HTML("<h2>Image Classification Confusion Analysis</h2>"),
    widgets.HBox([primary_class_dropdown, confused_class_dropdown]),
    widgets.HTML("<hr><h4>Primary Class Samples:</h4>"),
    primary_images_display,
    widgets.HTML("<hr><h4>Confused Class Samples:</h4>"),
    confused_images_display,
    widgets.HTML("<hr><h4>Select Reason(s) for Confusion:</h4>"),
    buttons_box,
    output_area, # For status messages
    reasons_display_area # To show the collected reasons dictionary
])

# Display the UI
display(ui_panel)

# --- 6. Initial Population ---
# Trigger the update for the initially selected primary class
update_primary_images({'new': primary_class_dropdown.value, 'owner': primary_class_dropdown})
display_analysis_dict() # Display the initial empty reasons dict
