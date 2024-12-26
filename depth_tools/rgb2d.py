#!/usr/bin/env python
# Script to generate depth maps from images in a dataset using a pre-trained Huggingface model.
# The depth maps are saved in an output directory, which can be specified by the user.
# Example script
# python convert_depth_maps.py /path/to/images/ --output_dir /path/to/output/depth_pred --num_images 10

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline  # Huggingface's Transformers library for depth estimation
import argparse

def convert_images_to_depth(input_dir_path, output_dir_path, num_images=None):
    """
    Convert images to depth maps using a pre-trained depth estimation model.

    Args:
        input_dir_path (str): Path to the directory containing input images.
        output_dir_path (str): Path to the directory where depth maps will be saved.
        num_images (int, optional): Number of images to process. If None, all images are processed.
    """
    # Initialize the depth estimation pipeline from Huggingface's model hub
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)

    # Supported image file extensions
    extensions = ['*.png', '*.jpg', '*.jpeg']
    
    # Collect all image files from the input directory
    image_files = sorted([f for ext in extensions for f in input_dir.rglob(ext)])
    tqdm.write(f"Found {len(image_files)} images.")

    # If num_images is provided, limit the number of images to process
    if num_images is not None and num_images > 0:
        image_files = image_files[:num_images]

    # Iterate over all image files and process them
    for input_path in tqdm(image_files, desc="Processing images"):
        tqdm.write(f"Processing {input_path}")
        
        # Open the image file
        image = Image.open(input_path)
        
        # Estimate the depth using the pre-trained model
        depth = pipe(image)["depth"]
        depth = np.array(depth)

        # Normalize the depth values to an 8-bit range [0, 255]
        normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_rgba = cv2.merge([normalized_depth])

        # Define the output file path
        output_path = output_dir / input_path.relative_to(input_dir)
        
        # Create output directories if they do not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the depth image as a grayscale PNG
        cv2.imwrite(str(output_path), depth_rgba)

    print("Depth conversion completed.")

def main():
    """
    Main function to parse command-line arguments and call the image-to-depth conversion function.
    """
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Convert images to depth maps.')

    # Required argument: input directory
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing images.')
    
    # Optional argument: output directory. If not specified, a subdirectory "depth_pred" will be created.
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory. If not specified, a "depth_pred" subdirectory will be used in the input directory.')
    
    # Optional argument: number of images to process. If not specified, all images are processed.
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to process. If not specified or -1, all images will be processed.')

    # Parse the arguments
    args = parser.parse_args()

    # If output_dir is not provided, default to a subdirectory called "depth_pred" within the input directory
    if args.output_dir is None:
        args.output_dir = args.input_dir.replace("images", "depth_pred")

    # Call the function to convert images to depth maps
    convert_images_to_depth(args.input_dir, args.output_dir, args.num_images)

if __name__ == '__main__':
    main()