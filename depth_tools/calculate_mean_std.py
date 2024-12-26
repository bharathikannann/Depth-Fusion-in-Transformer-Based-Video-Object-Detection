#!/usr/bin/env python

# Script to calculate the mean and standard deviation of depth images in a dataset.
# Example usage:
# python calculate_depth_mean_std.py --data_dir /path/to/images/ --annotation_file /path/to/annotations.json --depth_dir depth_pred
# depth_dir: This name replaces 'images' in the --data_dir path.


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import cv2
import argparse

class CocoDepthDataset(Dataset):
    """
    Custom Dataset class for loading depth images using COCO-style annotations.

    Args:
        root_dir (str): Path to the root directory of the dataset containing images.
        annotation_file (str): Path to the COCO annotation file.
        depth_dir (str): Subdirectory where depth images are stored.
        transform (callable, optional): Optional transform to be applied on a depth image.
    """
    def __init__(self, root_dir, annotation_file, depth_dir='depth_pred', transform=None):
        self.root_dir = root_dir
        self.depth_dir = depth_dir
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Loads and returns the depth image corresponding to the given index.
        
        Args:
            index (int): Index of the image in the dataset.
        
        Returns:
            depth (PIL.Image): Normalized depth image as a grayscale PIL Image.
        """
        # Get the COCO image ID and its information
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Build the image and depth file paths
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        depth_path = img_path.replace('images', self.depth_dir)
        
        # Load the depth image
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Normalize the depth values to [0, 1]
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Convert the normalized depth array to a grayscale PIL image
        depth = Image.fromarray((depth_normalized * 255).astype(np.uint8), 'L')
        
        # Apply optional transformations (e.g., ToTensor)
        if self.transform:
            depth = self.transform(depth)
        
        return depth


def calculate_mean_std(dataset):
    """
    Calculate the mean and standard deviation of the depth images in the dataset.
    
    Args:
        dataset (Dataset): An instance of the dataset containing depth images.
    
    Returns:
        tuple: Mean and standard deviation of the depth values across the dataset.
    """
    # Convert the depth images to tensors for statistical calculations
    to_tensor = transforms.ToTensor()
    all_depth_values = []
    
    # Iterate over the dataset and accumulate all depth values
    for _, depth in tqdm(enumerate(dataset), total=len(dataset), desc="Processing images"):
        depth_tensor = to_tensor(depth)  # Convert PIL image to tensor
        all_depth_values.append(depth_tensor)
    
    # Stack all depth tensors into a single tensor for statistical calculation
    all_depth_values = torch.stack(all_depth_values)
    
    # Calculate the mean and standard deviation
    mean = torch.mean(all_depth_values)
    std = torch.std(all_depth_values)
    
    return mean.item(), std.item()


def main(data_dir, annotation_file, depth_dir):
    """
    Main function to set up dataset and calculate mean and standard deviation of the depth images.
    
    Args:
        data_dir (str): Path to the directory containing images.
        annotation_file (str): Path to the COCO annotations file.
        depth_dir (str): Subdirectory where depth images are stored.
    """
    # Create dataset instance
    dataset = CocoDepthDataset(root_dir=data_dir, annotation_file=annotation_file, depth_dir=depth_dir)
    
    # Calculate mean and standard deviation of depth images
    mean, std = calculate_mean_std(dataset)
    
    # Print the results
    print(f"Depth Mean: {mean}")
    print(f"Depth Standard Deviation: {std}")


if __name__ == "__main__":
    # Argument parser to allow command line input
    parser = argparse.ArgumentParser(description="Calculate the mean and standard deviation of depth images.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root directory of images.')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to the COCO annotation file.')
    parser.add_argument('--depth_dir', type=str, default='depth_pred', help='Subdirectory where depth images are stored (default: depth_pred).')

    args = parser.parse_args()

    # Call the main function with command line arguments
    main(data_dir=args.data_dir, annotation_file=args.annotation_file, depth_dir=args.depth_dir)