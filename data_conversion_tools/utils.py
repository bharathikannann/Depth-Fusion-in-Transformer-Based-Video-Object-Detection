"""
This script provides functions for loading, processing, and visualizing normal and depth images, 
as well as handling and visualizing COCO dataset annotations. The core functionalities include:
- Loading depth and normal images for visualization.
- Handling COCO dataset annotations to extract bounding boxes.
- Plotting the bounding boxes on the images.
- Optionally saving the visualizations as output images.

Functions:
1. load_and_process_depth_image: Loads and processes a depth image from the provided path.
2. load_and_process_normal_image: Loads and processes an RGB image from the provided path.
3. visualize_images: Visualizes side-by-side comparison of normal and depth images.
4. load_coco_clips: Loads images and annotations from a COCO dataset, filters, and visualizes.
5. plot_results: Plots bounding boxes on an image, optionally with probabilities and depth images.

Dependencies:
- numpy
- os
- matplotlib
- cv2 (OpenCV)
- PIL (Python Imaging Library)
- pycocotools (COCO API)
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pycocotools.coco import COCO

# Load and process a depth image from the specified path
def load_and_process_depth_image(path, visualize=False):
    """
    Loads a depth image using OpenCV and converts it to a grayscale PIL image. Optionally visualizes the image.
    
    Args:
        path (str): The path to the depth image.
        visualize (bool): If True, the image is visualized using matplotlib.
        
    Returns:
        depth_img (PIL.Image): The processed depth image in grayscale.
    """
    # Load the depth image (unchanged)
    depth_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Convert the depth image to a grayscale PIL image
    depth_img = Image.fromarray(depth_ori, 'L')
    
    if visualize:
        plt.title('Depth Image')
        plt.tight_layout()
        plt.imshow(depth_img, cmap='gray')
        plt.axis('off')
        plt.show()
    
    return depth_img

# Load and process an RGB image from the specified path
def load_and_process_normal_image(path, visualize=False):
    """
    Loads an RGB image using PIL and optionally visualizes the image.
    
    Args:
        path (str): The path to the RGB image.
        visualize (bool): If True, the image is visualized using matplotlib.
        
    Returns:
        normal_img (PIL.Image): The processed RGB image.
    """
    # Load the RGB image
    normal_img = Image.open(path).convert('RGB')
    
    if visualize:
        plt.title('Normal Image')
        plt.tight_layout()
        plt.imshow(normal_img)
        plt.axis('off')
        plt.show()
    
    return normal_img

# Visualize side-by-side comparison of normal and depth images
def visualize_images(normal_img_path, depth_img_path, savefig=False, output_dir='results', name='output'):
    """
    Visualizes a side-by-side comparison of normal and depth images. Optionally saves the output image.
    
    Args:
        normal_img_path (str): The path to the RGB image.
        depth_img_path (str): The path to the depth image.
        savefig (bool): If True, the result is saved as an image.
        output_dir (str): Directory to save the output image.
        name (str): Filename for the saved image.
    """
    normal_img = load_and_process_normal_image(normal_img_path)
    depth_img = load_and_process_depth_image(depth_img_path)

    # Create a figure with two subplots
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Display the normal and depth images
    ax1.imshow(normal_img)
    ax1.axis('off')
    ax2.imshow(depth_img, cmap='gray')
    ax2.axis('off')

    plt.tight_layout()
    
    # Save the figure if savefig is True
    if savefig:
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', pad_inches=0)

    plt.show()

# Load images and annotations from a COCO dataset, filter, and visualize
def load_coco_clips(coco_path, img_path, to_show=5, select_subset=None):
    """
    Loads images and annotations from a COCO dataset, filters by a subset, and visualizes a specified number of images.
    
    Args:
        coco_path (str): Path to the COCO annotations file.
        img_path (str): Directory containing the images.
        to_show (int): Number of images to visualize.
        select_subset (str): Substring to filter image files by name.
    """
    # Initialize COCO API
    coco = COCO(coco_path)
    img_ids = coco.getImgIds()
    img_files = []

    # Load image file paths
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_files.append(os.path.join(img_path, img_info['file_name']))

    img_files = np.array(img_files)
    img_ids = np.array(img_ids)

    # Filter image files by subset if provided
    if select_subset is not None:
        indices = np.char.find(img_files.astype(str), select_subset) != -1
        img_files = img_files[indices]
        img_ids = img_ids[indices]

    # Sort image files and limit the number of images to show
    sort_indices = np.argsort(img_files)
    img_files = img_files[sort_indices][:to_show]
    img_ids = img_ids[sort_indices][:to_show]

    # Load annotations for the selected images
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)

    # Group annotations by image ID
    anns_by_img_id = {img_id: [] for img_id in img_ids}
    for ann in anns:
        if ann['image_id'] in anns_by_img_id:
            anns_by_img_id[ann['image_id']].append(ann)

    # Visualize the images with bounding boxes
    for i, img_file in enumerate(img_files):
        img_id = img_ids[i]
        img = Image.open(img_file)
        annotations = [ann['bbox'] for ann in anns_by_img_id[img_id]]
        plot_results(img, None, annotations, savefig=False, output_file=f'output_{i}.png')

# Plot bounding boxes on an image, optionally with probabilities and depth images
def plot_results(pil_img, prob, boxes, color='r', linewidth=3, savefig=False, output_file='output.png', depth_img=None):
    """
    Plots bounding boxes and optional probabilities on the provided image. Can also overlay a depth image.
    
    Args:
        pil_img (PIL.Image): The image to display.
        prob (list): Optional list of probabilities for each bounding box.
        boxes (list): List of bounding boxes (xmin, ymin, width, height).
        color (str): Color for the bounding boxes.
        linewidth (int): Line width of the bounding boxes.
        savefig (bool): If True, saves the output image.
        output_file (str): Path to save the output image.
        depth_img (PIL.Image): Optional depth image to display alongside the main image.
    """
    fig = plt.figure(figsize=(16, 10))

    # Create subplot(s) based on whether a depth image is provided
    if depth_img is not None:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(depth_img)
        ax2.axis('off')
    else:
        ax1 = fig.add_subplot(1, 1, 1)

    ax1.imshow(pil_img)
    ax1.axis('off')

    # Plot bounding boxes with probabilities
    for p, (xmin, ymin, width, height) in zip(prob or [1.0] * len(boxes), boxes):
        ax1.add_patch(plt.Rectangle((xmin, ymin), width, height, fill=False, color=color, linewidth=linewidth))
        text = f'Hand: {p:.2f}'
        ax1.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    
    # Save figure if savefig is True
    if savefig:
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)