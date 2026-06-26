 #!/usr/bin/env python3

import os
import hashlib
from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".webp", ".tiff", ".tif"
}


def file_hash(filepath, chunk_size=8192):
    """
    Compute SHA256 hash of a file.
    Exact duplicates will have identical hashes.
    """
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_and_delete_duplicates(folder_path):
    """
    Find exact duplicate image files and delete duplicates,
    keeping the first encountered original.
    """
    hashes = {}
    deleted = 0

    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder_path}")
        return

    for root, _, files in os.walk(folder):
        for filename in files:
            filepath = Path(root) / filename

            if filepath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                current_hash = file_hash(filepath)

                if current_hash in hashes:
                    original = hashes[current_hash]

                    print(f"Duplicate found:")
                    print(f"  Original : {original}")
                    print(f"  Deleting : {filepath}")

                    os.remove(filepath)
                    deleted += 1
                else:
                    hashes[current_hash] = filepath

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f"\nDone. Deleted {deleted} duplicate image(s).")


if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    find_and_delete_duplicates(folder)