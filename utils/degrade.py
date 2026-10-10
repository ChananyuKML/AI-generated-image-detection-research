import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
SOURCE_FOLDER = "dataset/genimage/sdv4"
OUTPUT_BASE = "dataset/genimage"  # Where to create the new folders (current dir)
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def apply_degradations(src_path, root_dir):
    """
    Reads an image and creates 6 degraded versions in corresponding folder structures.
    """
    # Read image
    img = cv2.imread(str(src_path))
    if img is None:
        return

    # Get relative path (e.g., "val/ai/image_01.png")
    rel_path = src_path.relative_to(root_dir)
    
    # --- Degradation Configurations ---
    degradations = [
        # Low Resolution
        ("LR_112", 112, 112, cv2.INTER_AREA, None),
        ("LR_64", 64, 64, cv2.INTER_AREA, None),
        # Gaussian Blur (0,0 ksize means compute ksize from sigma)
        ("Blur_3", 0, 0, None, 3), 
        ("Blur_5", 0, 0, None, 5),
        # JPEG Compression (Quality parameter)
        ("JPEG_65", None, None, None, 65),
        ("JPEG_30", None, None, None, 30),
    ]

    for name, size_w, size_h, interpolation, param in degradations:
        folder_name = f"{root_dir.name}_{name}"
        dest_path = Path(OUTPUT_BASE) / folder_name / rel_path
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # --- Apply Transformation ---
        
        if 'LR' in name:
            img_processed = cv2.resize(img, (size_w, size_h), interpolation=interpolation)
            cv2.imwrite(str(dest_path), img_processed)
        
        elif 'Blur' in name:
            sigma = param
            img_processed = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            cv2.imwrite(str(dest_path), img_processed)

        elif 'JPEG' in name:
            quality = param
            # Ensure file is saved as .jpg to apply JPEG compression
            save_path = dest_path.with_suffix('.jpg')
            cv2.imwrite(str(save_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def main():
    root_dir = Path(SOURCE_FOLDER)
    val_dir = root_dir / 'val' # <<< TARGETING THE 'val' SUBFOLDER ONLY
    
    if not val_dir.exists():
        print(f"Error: Validation folder '{val_dir}' not found. Please ensure your structure is correct.")
        return

    # Find all images recursively starting from the 'val' directory
    print(f"Scanning '{val_dir}' for images (TRAIN SET IGNORED)...")
    all_files = [
        p for p in val_dir.rglob("*") 
        if p.suffix.lower() in VALID_EXTENSIONS
    ]
    
    if not all_files:
        print("No images found in the 'val' directory.")
        return

    print(f"Found {len(all_files)} images in the validation set. Starting degradation...")

    # Process images with a progress bar
    for file_path in tqdm(all_files, desc="Processing Val Images"):
        # We pass the main root_dir here so relative path calculation is correct (e.g., starts with 'val/')
        apply_degradations(file_path, root_dir)

    print("\n? Done! Degraded validation datasets created successfully.")
    
if __name__ == "__main__":
    main()