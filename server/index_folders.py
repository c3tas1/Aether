#!/usr/bin/env python3
import os
import sys
import sqlite3
import datetime

# ------------- CONFIG -------------
DB_PATH = "images.db"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ------------- UTILS -------------
def get_db_connection():
    """
    Returns a connection to the SQLite database (images.db).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """
    Creates the 'images' table if it doesn't exist.
    Adjust columns as needed for your app.
    """
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

def allowed_file(filename):
    """
    Returns True if 'filename' has an allowed extension.
    """
    return (
        "." in filename 
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def index_image_file(file_path, original_name=None):
    """
    Inserts the given file into the SQLite DB if it's not already indexed.
    - file_path: absolute path to the image
    - original_name: an optional "original name" if you want to store it
                     differently from the filename on disk.
    """
    # Derive 'filename' from file_path
    # e.g. "1674612341-cat.jpg"
    filename_on_disk = os.path.basename(file_path)
    # If not provided, we can treat original_name same as filename
    if not original_name:
        original_name = filename_on_disk

    # Insert into DB
    with get_db_connection() as conn:
        # Check if we already have this path in DB to avoid duplicates
        existing = conn.execute(
            "SELECT id FROM images WHERE path = ? LIMIT 1",
            (file_path,)
        ).fetchone()
        if existing:
            print(f"Already in DB: {file_path}")
            return

        conn.execute("""
            INSERT INTO images (filename, original_name, path, status, annotations)
            VALUES (?, ?, ?, ?, ?)
        """, (filename_on_disk, original_name, file_path, "", ""))
        conn.commit()
    print(f"Indexed: {file_path}")

# ------------- MAIN INDEX FUNCTION -------------
def index_folders(folder_list):
    """
    Recursively scans each folder in folder_list,
    and indexes all allowed image files into the DB.
    """
    for folder in folder_list:
        folder = os.path.abspath(folder)
        print(f"Scanning folder: {folder}")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(folder):
            for file in files:
                if allowed_file(file):
                    full_path = os.path.join(root, file)
                    index_image_file(full_path)
                else:
                    # Not an allowed image extension
                    pass

# ------------- ENTRY POINT -------------
if __name__ == "__main__":
    # 1) Initialize DB if needed
    if not os.path.exists(DB_PATH):
        open(DB_PATH, 'a').close()  # create empty file
    initialize_db()

    # 2) Parse command-line arguments for folders
    # Example usage: python index_folders.py reference_images uploads other_images
    if len(sys.argv) < 2:
        print("Usage: python index_folders.py <folder1> <folder2> ...")
        sys.exit(1)

    folders_to_scan = sys.argv[1:]
    # 3) Index them
    index_folders(folders_to_scan)

    print("Done indexing all folders!")

