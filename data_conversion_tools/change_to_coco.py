"""
YOLO to COCO Format Converter

This script converts YOLO annotations (txt format) to COCO format JSON files. It processes images and labels from specified directories
and generates a COCO-compliant JSON annotation file. The script does not support nested folders and assumes the presence of YOLO
annotation files for the corresponding images.

Usage:
    python script.py --images_dir <path_to_images> --labels_dir <path_to_labels> --output_dir <output_directory>

Arguments:
    images_dir (str): Directory containing the images.
    labels_dir (str): Directory containing the YOLO txt annotations (optional).
    output_dir (str): Directory where the COCO JSON file will be saved (default: current directory).

Output:
    The script outputs a COCO JSON file (train.json) in the specified output directory.

Dependencies:
    - os
    - json
    - cv2 (OpenCV)
    - shutil
    - argparse
    - tqdm
    - PIL (Python Imaging Library)
"""

import os
import json
import cv2
import shutil
import argparse
from tqdm import tqdm
from PIL import Image

def get_annotations(images_dir, labels_dir, output_dir, annotations_file_name, categories_file=None):
    """
    Convert YOLO annotations to COCO format.

    Args:
        images_dir (str): Directory containing the images.
        labels_dir (str): Directory containing the YOLO txt annotations.
        output_dir (str): Directory to save the COCO JSON annotations.
        annotations_file_name (str): Name of the output JSON file (without extension).
        categories_file (str): Optional file containing categories (one per line).

    Returns:
        dict: COCO annotations in JSON format.
    """
    # Load categories from the provided file or default to 'hand'
    if categories_file:
        with open(categories_file, 'r') as f1:
            lines1 = f1.readlines()
    else:
        lines1 = ['hand']  # Default category

    # Prepare the category structure for COCO
    categories = [{'id': j + 1, 'name': label.strip(), 'supercategory': label.strip()} for j, label in enumerate(lines1)]

    # Initialize COCO format structure
    coco_data = {
        'info': {'description': None, 'url': None, 'version': None, 'year': 2022, 'contributor': None, 'date_created': None},
        'licenses': [{'id': 1, 'name': None, 'url': None}],
        'categories': categories,
        'images': [],
        'annotations': [],
        'videos': []
    }

    directory_images = os.fsencode(images_dir)
    directory_labels = os.fsencode(labels_dir)
    
    # Add video information (COCO requires video ID)
    coco_data['videos'].append({'id': 1, 'file_name': os.fsdecode(directory_images)})

    # Sort the image files by name
    sorted_image_files = sorted(os.listdir(directory_images), key=lambda x: os.path.splitext(os.path.basename(x.decode("utf-8")))[0])

    image_id = 1
    bbox_id = 1

    # Process each image in the directory
    for frame_id, file in tqdm(enumerate(sorted_image_files), desc="Processing Images"):
        filename = os.fsdecode(file)
        if filename.endswith((".jpg", ".jpeg", ".png")):  # Filter image files
            img_path = os.path.join(images_dir, filename)
            img_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(img_name)[0]

            yolo_annotation_path = os.path.join(labels_dir, file_name_without_ext + ".txt")

            # Read image dimensions
            height, width = cv2.imread(img_path).shape[:2]

            # Add image data to COCO
            coco_data['images'].append({
                'file_name': img_name,
                'height': height,
                'width': width,
                'video_id': 1,
                'frame_id': frame_id,
                'date_captured': '',
                'id': image_id,
                'license': 1
            })

            try:
                # Read YOLO annotations for the image
                with open(yolo_annotation_path, 'r') as f2:
                    annotation_file_lines = f2.readlines()
            except FileNotFoundError:
                image_id += 1
                continue

            # Process each annotation
            for line in annotation_file_lines:
                line = line.split(' ')
                class_id, x_yolo, y_yolo, width_yolo, height_yolo = map(float, line[:5])
                class_id = int(class_id)

                # Convert YOLO format (relative) to COCO format (absolute)
                w = abs(width_yolo * width)
                h = abs(height_yolo * height)
                x_coco = round(x_yolo * width - w / 2)
                y_coco = round(y_yolo * height - h / 2)

                # Ensure bounding box does not exceed image boundaries
                x_coco = max(1, x_coco)
                y_coco = max(1, y_coco)

                # Add annotation data to COCO
                coco_data['annotations'].append({
                    'id': bbox_id,
                    'image_id': image_id,
                    'video_id': 1,
                    'category_id': class_id + 1,
                    'iscrowd': 0,
                    'occluded': 0,
                    'generated': 0,
                    'area': w * h,
                    'bbox': [x_coco, y_coco, w, h],
                    'segmentation': []
                })
                bbox_id += 1

            image_id += 1

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save COCO format JSON file
    coco_save_path = os.path.join(output_dir, annotations_file_name + '.json')
    with open(coco_save_path, 'w') as fw:
        json.dump(coco_data, fw)

    return coco_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format.")
    parser.add_argument('--images_dir', type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument('--labels_dir', type=str, required=True, help="Path to the directory containing YOLO annotations.")
    parser.add_argument('--output_dir', type=str, default="./", help="Path to the directory to save the COCO annotations.")
    args = parser.parse_args()

    # Call the annotation function
    get_annotations(args.images_dir, args.labels_dir, args.output_dir, annotations_file_name="train")

    print("Dataset successfully converted to COCO format.")