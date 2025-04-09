import os
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QGridLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

class ClickableImageLabel(QLabel):
    """QLabel subclass that emits a signal with its file path when clicked."""
    clicked = pyqtSignal(str)
    def __init__(self, file_path, pixmap, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        # Set the label pixmap to the provided thumbnail image
        self.setPixmap(pixmap)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Emit the signal carrying this image's file path
            self.clicked.emit(self.file_path)
            # Disable the label to prevent toggling or multiple recordings
            self.setEnabled(False)
        # (No call to super().mousePressEvent, to avoid default behavior)

class ImageGridWindow(QMainWindow):
    """Main application window that displays images in a scrollable grid and handles user interaction."""
    def __init__(self, base_dir, thumb_size=128, max_per_class=1000):
        super().__init__()
        self.base_dir = base_dir
        self.thumb_size = thumb_size
        self.max_per_class = max_per_class

        # Sets to keep track of image paths for output classification
        self.clicked_paths = set()    # paths marked as bad (clicked)
        self.all_image_paths = []     # all paths displayed (for good vs bad determination)

        # Prepare the bad.txt file: truncate any existing content to start fresh
        open('bad.txt', 'w').close()

        # Set up the scrollable area and grid container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)   # allow resizing
        container = QWidget()                 
        grid = QGridLayout(container)          
        container.setLayout(grid)
        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

        # Optional window setup
        self.setWindowTitle("Image Patch Selector")
        self.resize(800, 600)  # initial window size

        # Populate the grid with image thumbnails
        cols = 8  # number of columns in the grid
        row = col = 0
        # Traverse each class directory and collect image patches
        if os.path.isdir(self.base_dir):
            class_dirs = [d for d in os.listdir(self.base_dir) 
                          if os.path.isdir(os.path.join(self.base_dir, d))]
            class_dirs.sort()
            for class_name in class_dirs:
                class_path = os.path.join(self.base_dir, class_name)
                # Find all image files under this class directory
                image_files = []
                for root, dirs, files in os.walk(class_path):
                    for fname in files:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            image_files.append(os.path.join(root, fname))
                # Randomly sample up to max_per_class images
                if len(image_files) > self.max_per_class:
                    image_files = random.sample(image_files, self.max_per_class)
                # Add selected files to the master list
                self.all_image_paths.extend(image_files)
                # Create a thumbnail label for each image file
                for file_path in image_files:
                    pixmap = QPixmap(file_path)
                    if pixmap.isNull():
                        # Skip files that cannot be opened as images
                        continue
                    # Scale the pixmap to a thumbnail (keeping aspect ratio)
                    thumbnail = pixmap.scaled(self.thumb_size, self.thumb_size, 
                                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    # Create a clickable label for the thumbnail
                    label = ClickableImageLabel(file_path, thumbnail, parent=container)
                    # Connect the label's clicked signal to the handler
                    label.clicked.connect(self.handle_image_click)
                    # Add the label widget to the grid layout
                    grid.addWidget(label, row, col)
                    col += 1
                    if col >= cols:
                        col = 0
                        row += 1
                # (Optional: one could insert a section header or spacing per class here if desired)
        else:
            print(f"Error: Base directory '{self.base_dir}' not found.")
        
    def handle_image_click(self, file_path):
        """Slot called when an image thumbnail is clicked. Marks the image as bad."""
        if file_path in self.clicked_paths:
            return  # already recorded this image
        self.clicked_paths.add(file_path)
        try:
            with open('bad.txt', 'a') as bad_file:
                bad_file.write(file_path + '\n')
        except Exception as e:
            print(f"Failed to write to bad.txt: {e}")
        # (The label is already disabled from further clicks in ClickableImageLabel)

    def closeEvent(self, event):
        """Override to write out 'good.txt' when the window is closed by the user."""
        try:
            with open('good.txt', 'w') as good_file:
                for path in self.all_image_paths:
                    if path not in self.clicked_paths:
                        good_file.write(path + '\n')
        except Exception as e:
            print(f"Failed to write to good.txt: {e}")
        event.accept()  # proceed with closing the window

# To run the application, create a QApplication and show the main window:
if __name__ == "__main__":
    import sys
    base_directory = "/path/to/root/directory"  # <<< Set this to your image base directory
    app = QApplication(sys.argv)
    window = ImageGridWindow(base_directory, thumb_size=128, max_per_class=1000)
    window.show()
    sys.exit(app.exec_())

