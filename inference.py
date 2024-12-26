# ------------------------------------------------------------------------
# Depth Fusion in Transformer-Based Video Object Detection
# Custom script developed for inference with RGB, RGB-D, RGB-V, RGB-D-V in a single script.
# Available through an api, and the example can be found in inference_playground
# ------------------------------------------------------------------------

import argparse
import os
import glob
import random
import sys
import time
from pathlib import Path
import gc

import torch
import torch.backends
import torchvision.transforms as T
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage, Resize, Compose
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from datasets.coco_video_parser import CocoVID
from models import build_model
import util.misc_multi as utils
# from captum.attr import IntegratedGradients

def get_args_parser():
    parser = argparse.ArgumentParser('TransVOD Detector', add_help=False)
    
    # Learning rate and optimizer parameters
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+',
                        help='Names of backbone layers to apply different learning rates.')
    parser.add_argument('--lr_backbone', default=2e-5, type=float, help='Learning rate for backbone.')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+',
                        help='Names of linear projection layers to apply different learning rates.')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float, 
                        help='Multiplier for learning rate of linear projection layers.')
    
    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training.')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer.')
    parser.add_argument('--epochs', default=15, type=int, help='Number of training epochs.')
    parser.add_argument('--lr_drop', default=5, type=int, help='Epoch at which to drop the learning rate.')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Maximum norm for gradient clipping.')

    # Model architecture parameters
    parser.add_argument('--num_ref_frames', default=4, type=int, help='Number of reference frames.')
    parser.add_argument('--sgd', action='store_true', help='Use SGD optimizer instead of Adam.')
    parser.add_argument('--with_box_refine', default=False, action='store_true',
                        help='Enable box refinement in Deformable DETR.')
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help='Use two-stage Deformable DETR.')

    # Backbone parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='Path to pretrained model. Only mask head will be trained if set.')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='Convolutional backbone to use.')
    parser.add_argument('--dilation', action='store_true',
                        help='Replace stride with dilation in the last convolutional block.')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help='Type of positional embedding.')
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help='Scale for positional embedding.')
    parser.add_argument('--num_feature_levels', default=4, type=int, help='Number of feature levels.')

    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int, help='Number of encoding layers.')
    parser.add_argument('--dec_layers', default=6, type=int, help='Number of decoding layers.')
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help='Feedforward network dimension.')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Transformer hidden dimension.')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate in transformer.')
    parser.add_argument('--nheads', default=8, type=int, help='Number of attention heads.')
    parser.add_argument('--num_queries', default=300, type=int, help='Number of query slots.')
    parser.add_argument('--dec_n_points', default=4, type=int, help='Number of points in decoder.')
    parser.add_argument('--enc_n_points', default=4, type=int, help='Number of points in encoder.')
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int, help='Number of temporal decoder layers.')
    parser.add_argument('--interval1', default=20, type=int, help='First interval parameter.')
    parser.add_argument('--interval2', default=60, type=int, help='Second interval parameter.')
    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true',
                        help='Use fixed pretrained model.')

    # Segmentation parameters
    parser.add_argument('--masks', action='store_true',
                        help='Enable segmentation head.')

    # Loss parameters
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help='Disable auxiliary decoding losses.')

    # Matcher parameters
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help='Class coefficient in matching cost.')
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help='L1 box coefficient in matching cost.')
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help='GIoU box coefficient in matching cost.')

    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float, help='Mask loss coefficient.')
    parser.add_argument('--dice_loss_coef', default=1, type=float, help='Dice loss coefficient.')
    parser.add_argument('--cls_loss_coef', default=2, type=float, help='Classification loss coefficient.')
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help='Bounding box loss coefficient.')
    parser.add_argument('--giou_loss_coef', default=2, type=float, help='GIoU loss coefficient.')
    parser.add_argument('--focal_alpha', default=0.25, type=float, help='Alpha parameter for focal loss.')

    # Dataset parameters
    parser.add_argument('--dataset_file', default='vid_multi', help='Dataset file identifier.')
    parser.add_argument('--coco_path', default='./data/coco', type=str, help='Path to COCO dataset.')
    parser.add_argument('--vid_path', default='./data/vid', type=str, help='Path to video dataset.')
    parser.add_argument('--coco_pretrain', default=False, action='store_true',
                        help='Use COCO pretraining.')
    parser.add_argument('--coco_panoptic_path', type=str, help='Path to COCO panoptic annotations.')
    parser.add_argument('--remove_difficult', action='store_true', help='Remove difficult samples.')

    # Output and device parameters
    parser.add_argument('--output_dir', default='', help='Directory to save outputs.')
    parser.add_argument('--device', default='cuda', help='Device to use for computation.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--resume', default='', help='Checkpoint to resume from.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch.')
    parser.add_argument('--eval', action='store_true', help='Evaluate model.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--cache_mode', default=False, action='store_true',
                        help='Cache images in memory.')

    # Number of classes
    parser.add_argument('--num_classes', default=31, type=int, help='Number of object classes.')

    # Inference parameters
    parser.add_argument('--inference_coco_path', default='', type=str,
                        help='Path to COCO annotation file for inference.')
    parser.add_argument('--img_path', default='', type=str,
                        help='Path to image or folder containing images.')
    parser.add_argument('--depth_path', default='', type=str,
                        help='Path to depth image or folder containing depth images.')
    parser.add_argument('--filter_key_img', default=True, action='store_true',
                        help='Filter key images during inference.')

    # Depth fusion parameters
    parser.add_argument('--use_depth', default=False, action='store_true',
                        help='Enable depth image usage.')
    parser.add_argument('--fusion_type', default="Baseline", type=str, help="Type of Fusion method used", 
                        choices=["Baseline", "LateFusion","Backbone_CrossFusion", "Encoder_CrossFusion"])
    
    # This is an argument internally used during research purposes #TODO: Remove this and only use --fusion-type
    # Internally, Baseline -> Baseline_rgb, LateFusion -> DepthDeform_latefusion_dformer
    #             Backbone_CrossFusion -> DepthDeform_dformer_crossfusion, Encoder_CrossFusion -> DepthDeform_encoder_cf_dformer
    parser.add_argument('--depth_type', default="Baseline_rgb", type=str, choices=[
        "Baseline_rgb","Baseline_rgbd", "Baseline_rgb_d_concat", 
        "DepthDeform_latefusion", "DepthDeform_crossfusion", 
        "DepthDeform_crossfusion_2way", "DepthDeform_crossfusion_2way_concat", 
        "DepthDeform_latefusion_2way", "DepthDeform_latefusion_2way_concat", 
        "DepthDeform_latefusion_noresidual", "DepthDeform_latefusion_dformer", 
        "DepthDeform_dformer_crossfusion", "DepthDeform_encoder_cf_dformer"],
        help='Type of depth fusion model.')
    parser.add_argument('--dformer_backbone', default=False, action='store_true',
                        help='Use DFormer backbone.')
    parser.add_argument('--keep_prob', default=0.5, type=float,
                        help='Probability threshold to keep detections.')
    parser.add_argument('--save_txt', default=False, action='store_true',
                        help='Save results in txt format.')
    parser.add_argument('--save_fig', default=False, action='store_true',
                        help='Visualize and save the results.')
    parser.add_argument('--save_depth_images', default=False, action='store_true',
                        help='Save depth images.')
    parser.add_argument('--num_images_to_show', default=1, type=int,
                        help='Number of images to display.')
    parser.add_argument('--img_no', default=0, type=int, help='Image number to start processing.')
    parser.add_argument('--close_fig', default=False, action='store_true',
                        help='Close figure after saving.')
    parser.add_argument('--dformer_weights', default=None, type=str,
                        help='Path to DFormer weights.')
    parser.add_argument('--dpth_n_points', default=4, type=int,
                        help='Number of points in depth transformer.')
    parser.add_argument('--img_max_size', default=1333, type=int,
                        help='Maximum size of the image.')
    parser.add_argument('--spatial_weights', default='', help='Path to spatial weights.')

    return parser

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, depth, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # Extract bounding boxes and convert to tensor
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # Convert to (x1, y1, x2, y2)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Extract class labels
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Extract masks if required
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # Extract keypoints if available
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # Keep only valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # Prepare target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # Compute area and crowd annotations
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # Add original image size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, depth, target

class ResizeWithMax(torch.nn.Module):
    """
    Resize image to a given size with an optional maximum size.

    Args:
        size (int or tuple): Desired output size.
        max_size (int, optional): Maximum size of the longer edge.
    """
    def __init__(self, size, max_size=None):
        super().__init__()
        self.size = size
        self.max_size = max_size

    def forward(self, img):
        """
        Apply resizing to the input image.

        Args:
            img (PIL.Image): Image to resize.

        Returns:
            PIL.Image: Resized image.
        """
        return resize(img, self.size, max_size=self.max_size)

def resize(image, size, max_size=None):
    """
    Resize image while maintaining aspect ratio.

    Args:
        image (PIL.Image): Image to resize.
        size (int or tuple): Desired output size.
        max_size (int, optional): Maximum size of the longer edge.

    Returns:
        PIL.Image: Resized image.
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    return rescaled_image

# -----------------------------------------------------------------------------
# Deformable DETR Inference Class
# -----------------------------------------------------------------------------
class DeformableDETR:
    """
    Deformable DETR class for inference.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    
    def __init__(self, args):
        # Set to evaluation mode
        args.eval = True   
        self.args = args
        
        # Initialize distributed mode if applicable
        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))
        
        # Initialize paths and directories
        self.inference_coco_path = args.inference_coco_path
        self.img_path = args.img_path
        self.depth_path = args.depth_path
        self.output_dir = args.output_dir
        self.dataset_file = args.dataset_file

        # Mode flags
        self.cache_mode = args.cache_mode
        self.is_train = False  # Inference mode
        self.depth_available = None
        self.use_depth = args.use_depth
        self.save_depth_images = args.save_depth_images
        self.save_txt = args.save_txt
        self.save_fig = args.save_fig
        self.filter_key_img = args.filter_key_img
        self.img_max_size = args.img_max_size

        # Data attributes
        self.coco = None
        self.img_files = np.array([])
        self.depth_files = np.array([])

        # Model parameters
        self.class_index = 1  # Class index for hand
        self.num_ref_frames = args.num_ref_frames
        self.num_images_to_show = args.num_images_to_show
        
        # Initialize dataset
        if args.dataset_file in ['vid_multi', 'vid_multi_plusplus']:
            self.cocovid = CocoVID(self.inference_coco_path)
        
        # Prepare target conversion
        self.prepare = ConvertCocoPolysToMask(args.masks)
        self.outputs = []
        self.output_imgs = np.array([])
        
        # Verify frozen weights
        if args.frozen_weights is not None:
            assert args.masks, "Frozen training is meant for segmentation only"
        print(args)
        self.device = torch.device(args.device)

        # Set random seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Define image transformations
        self.rgb_transform = T.Compose([
            ResizeWithMax(600, max_size=self.img_max_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = T.Compose([
            ResizeWithMax(600, max_size=self.img_max_size),
            T.ToTensor(),
            T.Normalize([0.48], [0.28])
        ]) if self.args.dformer_backbone else T.Compose([
            ResizeWithMax(600, max_size=self.img_max_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406, 0.48], [0.229, 0.224, 0.225, 0.28])
        ])
        
        # Combined transformation for RGB-D images
        if self.args.dformer_backbone:
            self.combined_transform = T.Compose([
                ResizeWithMax(600, max_size=self.img_max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406, 0.48], [0.229, 0.224, 0.225, 0.28])
            ])
        else:
            self.combined_transform = T.Compose([
                ResizeWithMax(600, max_size=self.img_max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406, 0.48], [0.229, 0.224, 0.225, 0.28])
            ])

    # -----------------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------------

    def box_cxcywh_to_xyxy(self, x):
        """
        Convert bounding boxes from center format to corner format.

        Args:
            x (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.

        Returns:
            torch.Tensor: Bounding boxes in (x1, y1, x2, y2) format.
        """
        x_c, y_c, w, h = x.unbind(1)
        boxes = [
            (x_c - 0.5 * w),
            (y_c - 0.5 * h),
            (x_c + 0.5 * w),
            (y_c + 0.5 * h)
        ]
        return torch.stack(boxes, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        """
        Rescale bounding boxes to the original image size.

        Args:
            out_bbox (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.
            size (tuple): Original image size (width, height).

        Returns:
            torch.Tensor: Rescaled bounding boxes in (x1, y1, x2, y2) format.
        """
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def plot_results(self, pil_img, prob, boxes, color='r', linewidth=3, savefig=False, 
                     output_file='output.png', depth_img=None):
        """
        Plot bounding boxes and probabilities on the image.

        Args:
            pil_img (PIL.Image): Original image.
            prob (list or torch.Tensor): List of probabilities for each box.
            boxes (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.
            color (str, optional): Color for bounding boxes. Defaults to 'r'.
            linewidth (int, optional): Line width for bounding boxes. Defaults to 3.
            savefig (bool, optional): Whether to save the figure. Defaults to False.
            output_file (str, optional): Path to save the figure. Defaults to 'output.png'.
            depth_img (PIL.Image, optional): Depth image to display alongside. Defaults to None.

        Returns:
            matplotlib.pyplot: The plotted figure.
        """
        fig = plt.figure(figsize=(16, 10))
        
        # If depth image is available, create subplots for both RGB and depth
        if self.depth_available and depth_img is not None:
            ax1 = fig.add_subplot(1, 2, 1)  # RGB image subplot
            ax2 = fig.add_subplot(1, 2, 2)  # Depth image subplot
            ax2.imshow(depth_img, cmap='gray')  # Display depth image in grayscale
            ax2.axis('off')
        else:
            ax1 = fig.add_subplot(1, 1, 1)  # Single subplot for RGB image

        ax1.imshow(pil_img)
        ax1.axis('off')
        
        # Plot each bounding box with its probability
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            ax1.add_patch(plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, color=color, linewidth=linewidth
            ))
            text = f'Hand: {p:0.2f}'
            ax1.text(xmin, ymin, text, fontsize=15, 
                     bbox=dict(facecolor='yellow', alpha=0.5))
            
        plt.tight_layout()
        
        # Save the figure if required
        if savefig:
            plt.axis('off')
            plt.subplots_adjust(wspace=0.02, hspace=0.02)  # Adjust space between subplots
            plt.savefig(f"{output_file}", bbox_inches='tight', pad_inches=0)
        
        # Close the figure to free memory if specified
        if self.args.close_fig:
            plt.close(fig)
        
        return plt

    def load_rgbd_image(self, img_path, depth_path=None):
        """
        Load RGB and Depth images from specified paths.

        Args:
            img_path (str): Path to the image or folder containing images.
            depth_path (str, optional): Path to the depth image or folder containing depth images.

        Raises:
            ValueError: If paths are invalid or counts mismatch.

        Sets:
            self.img_files (np.array): List of image file paths.
            self.depth_files (np.array): List of depth image file paths.
            self.depth_available (bool): Flag indicating availability of depth images.
        """
        if img_path and depth_path: 
            allowed_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            # Check if both paths are files
            if os.path.splitext(img_path)[1] in allowed_extensions and os.path.splitext(depth_path)[1] in allowed_extensions:
                self.input_format = "single_image_depth"
                self.img_files = np.array([img_path])
                self.depth_files = np.array([depth_path])
                self.depth_available = True
            # Check if both paths are directories
            elif os.path.splitext(img_path)[1] == '' and os.path.splitext(depth_path)[1] == '':
                self.input_format = "image_depth_folder"
                self.img_files = np.array(
                    glob.glob(os.path.join(img_path, '*.[jJ][pP][gG]')) +
                    glob.glob(os.path.join(img_path, '*.[pP][nN][gG]')) +
                    glob.glob(os.path.join(img_path, '*.[jJ][pP][eE][gG]'))
                )
                self.depth_files = np.array(
                    glob.glob(os.path.join(depth_path, '*.[jJ][pP][gG]')) +
                    glob.glob(os.path.join(depth_path, '*.[pP][nN][gG]')) +
                    glob.glob(os.path.join(depth_path, '*.[jJ][pP][eE][gG]'))
                )
                self.depth_available = True
            else:
                raise ValueError("Both img_path and depth_path should be either files or directories.")
        elif img_path:  # Only image path provided
            if os.path.isfile(img_path):
                self.input_format = "single_image"
                self.img_files = np.array([img_path])
            elif os.path.isdir(img_path):
                self.input_format = "image_folder"
                self.img_files = np.array(
                    glob.glob(os.path.join(img_path, '*.[jJ][pP][gG]')) +
                    glob.glob(os.path.join(img_path, '*.[pP][nN][gG]')) +
                    glob.glob(os.path.join(img_path, '*.[jJ][pP][eE][gG]'))
                )
            else:
                raise ValueError("img_path should be either a file or a directory.")
        
        # Verify depth image availability
        if self.depth_available:
            assert len(self.img_files) == len(self.depth_files), "Number of RGB and Depth images must be the same."
            assert self.depth_available == self.args.use_depth, (
                "Depth folder is given, but the model does not use depth images."
            )

        # Sort the image and depth file paths
        self.img_files = np.sort(self.img_files)
        if self.depth_available:
            self.depth_files = np.sort(self.depth_files)

    def load_coco_clips(self, coco_path):
        """
        Load image and depth file paths from COCO annotations.

        Args:
            coco_path (str): Path to the COCO annotation file.

        Raises:
            AssertionError: If the number of RGB and Depth images do not match.

        Sets:
            self.coco (COCO): COCO object.
            self.img_files (np.array): List of image file paths.
            self.depth_files (np.array): List of depth image file paths.
        """
        self.coco = COCO(coco_path)
        img_ids = self.coco.getImgIds()
        img_files = []
        depth_files = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_files.append(os.path.join(self.img_path, img_info['file_name']))
            if self.depth_available:
                # Replace 'images' with 'depth_pred' to get corresponding depth images
                depth_files.append(os.path.join(self.img_path, img_info['file_name'].replace('images', 'depth_pred')))
        self.img_files = np.array(img_files)
        self.depth_files = np.array(depth_files)

        if self.depth_available:
            assert len(self.img_files) == len(self.depth_files), "Number of RGB and Depth images must be the same."

        # Sort the image and depth file paths
        self.img_files = np.sort(self.img_files)
        if self.depth_available:
            self.depth_files = np.sort(self.depth_files)

    def correct_path(self, path):
        """
        Correct file path separators based on the operating system.

        Args:
            path (str): Original file path.

        Returns:
            str: Corrected file path.
        """
        corrected_path = path.replace('\\', os.sep)
        return corrected_path

    def get_image(self, path):
        """
        Load an RGB image from the specified path.

        Args:
            path (str): Path to the image file.

        Raises:
            NotImplementedError: If caching is enabled.

        Returns:
            PIL.Image: Loaded RGB image.
        """
        if self.cache_mode:
            raise NotImplementedError("Caching images is not supported.")
        img_path = os.path.join(self.img_path, path) if self.img_path != path else path
        img_path = self.correct_path(img_path)
        return Image.open(img_path).convert('RGB')

    def get_depth(self, path):
        """
        Load a depth image from the specified path.

        Args:
            path (str): Path to the depth image file.

        Raises:
            NotImplementedError: If caching is enabled.
            FileNotFoundError: If the depth file does not exist.
            ValueError: If the depth image has multiple channels.

        Returns:
            PIL.Image: Loaded depth image in grayscale.
        """
        if self.cache_mode:
            raise NotImplementedError("Caching depth images is not supported.")
        
        depth_path = os.path.join(self.img_path, path) if self.depth_path != path else path
        if 'images' in depth_path:
            depth_path = depth_path.replace('images', 'depth')
        depth_path = self.correct_path(depth_path)

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"No depth file found at {depth_path}")

        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Read depth data
        
        if depth_data is None:
            raise FileNotFoundError(f"Failed to read depth image at {depth_path}")
        
        if len(depth_data.shape) == 3 and depth_data.shape[-1] == 3:
            raise ValueError("Depth image has 3 channels. Consider only the first channel.")
        
        # Normalize depth data to [0, 255]
        depth_normalized = ((depth_data - depth_data.min()) * (255.0 / (depth_data.max() - depth_data.min()))).astype('uint8')
        depth_image = Image.fromarray(depth_normalized, 'L')  # Convert to PIL Image in grayscale

        return depth_image

    def get_image_and_reference_clips(self, idx):
        """
        Retrieve the main image and its reference frames for temporal context.

        Args:
            idx (int): Index of the current image in the dataset.

        Returns:
            tuple: Concatenated tensor of main and reference images, target annotations, and image path.
        """
        imgs = []
        depths = []
        ids = list(sorted(self.coco.imgs.keys()))
        img_id = ids[idx]
        depth_id = ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        depth_path = img_info['file_name']
        video_id = img_info['video_id']
        
        # Load main image and depth
        img = self.get_image(path)
        depth = self.get_depth(depth_path) if self.depth_available else None
        target = {'image_id': img_id, 'annotations': target}
        img, depth, target = self.prepare(img, depth, target)
        imgs.append(img)
        depths.append(depth)
        
        # Get all image IDs from the same video
        img_ids = self.cocovid.get_img_ids_from_vid(video_id)
        ref_img_ids = []
        interval = self.num_ref_frames  # Interval for sampling reference frames
        left = max(img_ids[0], img_id - interval)  # Left boundary
        right = min(img_ids[-1], img_id + interval)  # Right boundary
        sample_range = list(range(left, right + 1))  # Inclusive range

        # Optionally filter out the current image from reference frames
        if self.filter_key_img and img_id in sample_range:
            sample_range.remove(img_id)
        
        # Extend the sample range if necessary
        while len(sample_range) < self.num_ref_frames:
            sample_range.extend(sample_range)
        ref_img_ids = sample_range[:self.num_ref_frames]

        # Load reference frames
        for ref_img_id in ref_img_ids:
            ref_ann_ids = self.coco.getAnnIds(imgIds=ref_img_id)
            ref_img_info = self.coco.loadImgs(ref_img_id)[0]
            ref_img_path = ref_img_info['file_name']
            ref_depth_path = ref_img_info['file_name']
            ref_img = self.get_image(ref_img_path)
            ref_depth = self.get_depth(ref_depth_path) if self.depth_available else None
            imgs.append(ref_img)
            depths.append(ref_depth)
        
        # Apply transformations and concatenate images and depths
        final_imgs = [] 
        if self.depth_available:   
            for img, depth in zip(imgs, depths):
                img = self.rgb_transform(img)
                depth = self.depth_transform(depth)
                assert img.shape[0] == 3, "Image should have 3 RGB channels."
                assert depth.shape[0] == 1, "Depth should have 1 channel."
                combined_imgs = torch.cat([img, depth], dim=0)  # Concatenate along channel dimension
                final_imgs.append(combined_imgs) 
        else:
            for img in imgs:
                img_transformed = self.rgb_transform(img)
                final_imgs.append(img_transformed)
        
        return torch.cat(final_imgs, dim=0), target, os.path.join(self.img_path, path)

    def infer(self):
        """
        Perform inference on the loaded dataset/images.

        Returns:
            tuple: Output images and detections.
        """
        # Set the current device
        torch.cuda.set_device(int(self.args.device[-1]))  
        
        # Build the model and load checkpoint
        model, self.criterion, self.postprocessors = build_model(self.args)        
        checkpoint = torch.load(self.args.resume, map_location=self.args.device)
        state_dict = checkpoint['model']
        
        # Optionally update state dict with spatial weights, for loading spatial models separately
        if self.args.spatial_weights:
            spatial_checkpoint = torch.load(self.args.spatial_weights, map_location=self.args.device)
            spatial_state_dict = spatial_checkpoint['model']
            state_dict.update(spatial_state_dict)
        
        # Load state dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if missing_keys:
            print('Missing Keys: {}'.format(missing_keys))
        if unexpected_keys:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        
        # Determine if depth is available based on arguments
        self.depth_available = bool(self.depth_path or self.use_depth)
        if self.depth_path and not self.use_depth:
            print("Depth path is given, but the model does not use depth images (use_depth=False). Not using depth images.")
        
        # Load dataset based on provided paths
        if os.path.exists(self.inference_coco_path):
            print(f'Loading COCO clips from {self.inference_coco_path}')
            self.load_coco_clips(self.inference_coco_path)
        else:
            print(f'Loading RGB-D images from {self.img_path} and {self.depth_path}')
            self.load_rgbd_image(self.img_path, self.depth_path)
            
        print(f'Using {self.num_ref_frames} reference frames.')
        total = len(self.img_files) if self.num_images_to_show == -1 else self.num_images_to_show
        
        count = -1
        for i, img_file in enumerate(tqdm(self.img_files, total=total)):
            # Handle specific image number if provided, start with img_no and show num_images_to_show images
            if self.args.img_no != 0: 
                if i < self.args.img_no:
                    continue
                if i >= self.args.img_no:
                    count += 1
                    if count == self.num_images_to_show:
                        break
            elif i == self.num_images_to_show: 
                break

            img_file = self.correct_path(img_file)
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            
            if self.depth_available:
                depth_file = self.depth_files[i]
                depth_file = self.correct_path(depth_file)
                depth_name = os.path.splitext(os.path.basename(depth_file))[0]

            # Check if files exist
            for file_type, file in [('Image', img_file), ('Depth image', depth_file) if self.depth_available else (None, None)]:
                if file and not os.path.exists(file):
                    tqdm.write(f'{file_type} {file} does not exist')
                    continue

            # Load original images (RGB and Depth if provided)
            original_img = self.get_image(img_file)
            original_dpth = self.get_depth(depth_file) if self.depth_available else None

            # Get image with reference frames if using video dataset (RGB and Depth if provided)
            if self.dataset_file in ['vid_multi', 'vid_multi_plusplus']:
                img, _, img_file = self.get_image_and_reference_clips(i)
                original_img = self.get_image(img_file)
            else:
                img = original_img

            try:
                # Prepare image tensor for model
                if self.dataset_file in ['vid_multi', 'vid_multi_plusplus']:
                    img = img.to(self.device)
                    img = utils.nested_tensor_from_tensor_list([img], channel_size=4 if self.depth_available else 3)
                else:
                    img = self.rgb_transform(img).unsqueeze(0).to(self.device)
                    if self.depth_available:
                        depth = self.get_depth(depth_file)
                        depth = self.depth_transform(depth).unsqueeze(0).to(self.device)
                        img = torch.cat([img, depth], dim=1)

                # Move model to device
                model = model.to(self.args.device)
                t0 = time.time()
                model.eval()
                
                # ----------------- Integrated Gradients -----------------
                # Enable the gradients while calculating the attributions                
                # img2 = img.clone()
                # baseline = torch.zeros_like(img2).to(self.device)
                # ig = IntegratedGradients(model)
                # attribution, delta = ig.attribute(img2, baseline, n_steps=50, internal_batch_size=1, return_convergence_delta=True)
                # # print('IG Attributions:', attribution)
                # # print('Convergence Delta:', delta)
                # visualize_integrated_gradients(img2, attribution, self.args.device)
                # exit()

                # Perform model inference
                with torch.no_grad():
                    model_outputs = model(img)
                
                tqdm.write(f'Processing image {os.path.relpath(img_file, self.img_path)} took {time.time() - t0:.2f}s')
            except Exception as e:
                print("Error:", e)
                print(f'Error processing image through the model {img_file}')
                break

            # Extract probabilities and bounding boxes from model outputs
            probas = model_outputs['pred_logits'].softmax(-1)[0].detach()
            boxes = model_outputs['pred_boxes'][0].detach()

            # Filter detections based on probability threshold
            keep = probas[:,1] > self.args.keep_prob # [:1] corresponds to class index 1, which is the hand class
            keep_hand = keep  # Assuming class index 1 corresponds to 'hand'

            if sum(keep & keep_hand) == 0:
                tqdm.write(f'No hand detected in image {img_file}')
                continue

            probs_kept = probas[keep & keep_hand][:,1].to(torch.device('cpu'))
            normalized_boxes = boxes[keep & keep_hand].to(torch.device('cpu'))
            bboxes_scaled = self.rescale_bboxes(normalized_boxes, original_img.size)

            # Save or visualize results
            images_dir = os.path.join(self.output_dir, 'images')
            image_file_name = os.path.splitext(os.path.relpath(img_file, self.img_path))[0]
            full_images_dir = os.path.join(images_dir, os.path.dirname(image_file_name))
            Path(full_images_dir).mkdir(parents=True, exist_ok=True)
            output_img_name = os.path.join(images_dir, image_file_name + os.path.splitext(img_file)[1])
            
            # Plot results on the image
            figs = self.plot_results(
                original_img, probs_kept, bboxes_scaled, color='r', linewidth=3, 
                savefig=self.save_fig, output_file=output_img_name, 
                depth_img=original_dpth if self.depth_available else None
            )

            # Save detection results to text files if required
            if self.save_txt:
                labels_dir = os.path.join(self.output_dir, 'labels')
                label_file_name = os.path.splitext(os.path.relpath(img_file, self.img_path))[0]
                full_labels_dir = os.path.join(labels_dir, os.path.dirname(label_file_name))
                Path(full_labels_dir).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(labels_dir, label_file_name + '.txt'), 'w') as f:
                    for bbox, prob in zip(normalized_boxes.tolist(), probs_kept.tolist()):
                        cx, cy, width, height = bbox
                        f.write(f'Hand {cx:.8f} {cy:.8f} {width:.8f} {height:.8f} {prob:.8f}\n')

        if self.save_fig:
            print(f'Images saved in {self.output_dir}/images')
        if self.save_txt:
            print(f'Labels files saved in {self.output_dir}/labels')
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return self.output_imgs, self.outputs

# -----------------------------------------------------------------------------
# Visualization Function
# -----------------------------------------------------------------------------
def visualize_integrated_gradients(img, attribution, device):
    """
    Visualize the input image and Integrated Gradients attributions.

    Args:
        img (torch.Tensor): Input image tensor with 4 channels (e.g., RGB + Depth).
        attribution (torch.Tensor): Integrated Gradients attribution tensor with 4 channels.
        device (torch.device): Device for computations.

    Returns:
        matplotlib.pyplot.Figure: The visualization figure.
    """
    # Ensure input and attribution have the same shape
    assert img.shape == attribution.shape, "Input image and attribution must have the same shape."

    # Normalize attribution values to [0, 1]
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
    img = (img - img.min()) / (img.max() - img.min())

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # Top-left: RGB channels of the input image
    axs[0, 0].imshow(img[0, :3].permute(1, 2, 0).cpu())
    axs[0, 0].set_title('RGB Channels')
    axs[0, 0].axis('off')

    # Top-right: RGB Attributions
    rgb_attribution = attribution[0, :3].permute(1, 2, 0)
    axs[0, 1].imshow(rgb_attribution.cpu())
    axs[0, 1].set_title('RGB Attributions')
    axs[0, 1].axis('off')

    # Bottom-left: Depth channel of the input image
    axs[1, 0].imshow(img[0, 3:].permute(1,2,0).cpu())
    axs[1, 0].set_title('Depth Channel')
    axs[1, 0].axis('off')

    # Bottom-right: Depth Attribution
    depth_attribution = attribution[0, 3:].permute(1, 2, 0).cpu()
    depth_im = axs[1, 1].imshow(depth_attribution)
    axs[1, 1].set_title('Depth Attribution')
    axs[1, 1].axis('off')

    plt.tight_layout()
    
    # Add color bar for depth attribution
    cbar = fig.colorbar(depth_im, ax=axs, orientation='vertical', fraction=0.05, pad=0.01, shrink=0.5)
    cbar.set_label('Attribution Intensity')
    
    # Save and display the figure
    plt.savefig('integrated_gradients.png')
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# Batch Inference (Optional, not used in the main inference method)
# -----------------------------------------------------------------------------
def infer_batch(self, batch_size=1):
    """
    Perform inference on a batch of images.

    Args:
        batch_size (int, optional): Number of images per batch. Defaults to 1.

    Returns:
        tuple: Output images and detections.
    """
    self.depth_available = bool(self.depth_path and self.use_depth)
    if self.depth_path and not self.use_depth:
        print("Depth path is given, but the model does not use depth images (use_depth=False). Not using depth images.")
    
    # Load dataset based on provided paths
    if os.path.exists(self.inference_coco_path):
        print(f'Loading COCO clips from {self.inference_coco_path}')
        self.load_coco_clips(self.inference_coco_path)
    else:
        print(f'Loading RGB-D images from {self.img_path} and {self.depth_path}')
        self.load_rgbd_image(self.img_path, self.depth_path)
        
    print(f'Using {self.num_ref_frames} reference frames.')
    total = len(self.img_files) if self.num_images_to_show == -1 else self.num_images_to_show

    max_iterations = (self.num_images_to_show // batch_size) + 1
    for i in tqdm(range(0, min(total, max_iterations * batch_size), batch_size), 
                  desc='Processing image batches', total=max_iterations):
        batch_img_files = self.img_files[i:i+batch_size]
        batch_depth_files = self.depth_files[i:i+batch_size] if self.depth_available else None
        batch_images = []
        batch_depths = []
        model_outputs = []
        
        for j, img_file in enumerate(batch_img_files):
            img_file = self.correct_path(img_file)
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            if self.depth_available:
                depth_file = batch_depth_files[j]
                depth_file = self.correct_path(depth_file)
                depth_name = os.path.splitext(os.path.basename(depth_file))[0]
                assert img_name == depth_name, "RGB and Depth image names must be the same."
            original_img = self.get_image(img_file)
            original_dpth = self.get_depth(depth_file) if self.depth_available else None
            
            if self.dataset_file == 'vid_multi':
                img, _ = self.get_image_and_reference_clips(j)
            else:
                img = original_img
                batch_images.append(img)
            
            if self.depth_available:
                batch_depths.append(original_dpth)
        
        try:
            # Prepare batch tensor
            if self.dataset_file == 'vid_multi':
                batch_images = torch.stack(batch_images).to(self.device)
                batch_images = utils.nested_tensor_from_tensor_list(
                    [batch_images], channel_size=4 if self.depth_available else 3
                )
            else:
                batch_images = torch.stack([self.rgb_transform(img).unsqueeze(0).to(self.device) 
                                            for img in batch_images])
                if self.depth_available:
                    batch_depths = torch.stack([self.depth_transform(depth).unsqueeze(0).to(self.device) 
                                                for depth in batch_depths])
                    batch_images = torch.cat([batch_images, batch_depths], dim=1)
            
            # Move model to device
            model = model.to(self.device)
            t0 = time.time()
            
            # Perform model inference
            with torch.no_grad():
                model_outputs = model(batch_images)
            tqdm.write(f'Processing images {i}-{i+batch_size} took {time.time() - t0:.2f}s')
        except Exception as e:
            print(f'Error processing images through the model: {e}')
            break
        
        # Process each output in the batch
        for j, output in enumerate(model_outputs):
            img = batch_images[j]
            depth = batch_depths[j] if self.depth_available else None

            probas = output['pred_logits'].softmax(-1)[0].detach()
            boxes = output['pred_boxes'][0].detach()

            # Filter detections based on probability threshold
            keep = probas.max(-1)[0] > self.args.keep_prob
            keep_hand = probas.max(-1).indices == self.class_index

            if sum(keep & keep_hand) == 0:
                tqdm.write(f'No hand detected in image {batch_img_files[j]}')
                continue

            # Keep only high-probability detections
            probs_kept = probas[keep & keep_hand].max(-1)[0].to(torch.device('cpu'))
            normalized_boxes = boxes[keep & keep_hand].to(torch.device('cpu'))
            bboxes_scaled = self.rescale_bboxes(normalized_boxes, original_img.size)

            # Save or visualize results
            images_dir = os.path.join(self.output_dir, 'images')
            image_file_name = os.path.splitext(os.path.relpath(batch_img_files[j], self.img_path))[0]
            full_images_dir = os.path.join(images_dir, os.path.dirname(image_file_name))
            Path(full_images_dir).mkdir(parents=True, exist_ok=True)
            output_img_name = os.path.join(images_dir, image_file_name + os.path.splitext(batch_img_files[j])[1])
            
            # Plot results on the image
            figs = self.plot_results(
                original_img, probs_kept, bboxes_scaled, color='r', linewidth=3, 
                savefig=self.save_fig, output_file=output_img_name, 
                depth_img=original_dpth if self.depth_available else None
            )

            # Save detection results to text files if required
            if self.save_txt:
                labels_dir = os.path.join(self.output_dir, 'labels')
                label_file_name = os.path.splitext(os.path.relpath(batch_img_files[j], self.img_path))[0]
                full_labels_dir = os.path.join(labels_dir, os.path.dirname(label_file_name))
                Path(full_labels_dir).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(labels_dir, label_file_name + '.txt'), 'w') as f:
                    for bbox, prob in zip(normalized_boxes.tolist(), probs_kept.tolist()):
                        cx, cy, width, height = bbox
                        f.write(f'Hand {cx:.8f} {cy:.8f} {width:.8f} {height:.8f} {prob:.8f}\n')
                    
    # Notify user of saved outputs
    if self.save_fig:
        print(f'Images saved in {self.output_dir}/images')
    if self.save_txt:
        print(f'Labels files saved in {self.output_dir}/labels')
    
    return self.output_imgs, self.outputs

# -----------------------------------------------------------------------------
# Inference Function
# -----------------------------------------------------------------------------
def run_inference(model_path="checkpoint.pth", coco_path='', img_path=None, depth_path=None, output_dir='output', **kwargs):
    """
    Run inference using the Deformable DETR model.

    Args:
        model_path (str, optional): Path to the model checkpoint. Defaults to "checkpoint.pth".
        coco_path (str, optional): Path to the COCO annotation file. Defaults to ''.
        img_path (str, optional): Path to images or image folder. Defaults to None.
        depth_path (str, optional): Path to depth images or folder. Defaults to None.
        output_dir (str, optional): Directory to save outputs. Defaults to 'output'.
        **kwargs: Additional arguments to override default settings.

    Returns:
        tuple: Visualization figures and detection outputs.
    """
    # Initialize argument parser and parse arguments
    parser = get_args_parser()
    args = parser.parse_args([])  # Parse with empty list to allow setting manually

    # Override arguments with provided parameters
    args.inference_coco_path = coco_path
    args.img_path = img_path
    args.depth_path = depth_path
    args.output_dir = output_dir
    args.resume = model_path
    
    # Set additional arguments from kwargs
    for key, value in kwargs.items():
        setattr(args, key, value)

    # Create output directory if it doesn't exist
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Initialize Deformable DETR model
    detr_model = DeformableDETR(args)
    
    # # Perform batch inference if batch size > 1, else single inference
    # if args.batch_size > 1:
    #     figs, outputs = detr_model.infer_batch(args.batch_size)
    # else:
    if args.batch_size > 1:
        print("Batch size greater than 1 is not supported yet. Performing single inference.")
    figs, outputs = detr_model.infer()
    
    return figs, outputs

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'TransVOD inference script', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    # Create output directory
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set additional arguments for inference
    args.close_fig = True
    args.num_images_to_show = -1  # Show all images for command-line execution
    
    # Initialize and run inference
    detr_model = DeformableDETR(args)
    figs, outputs = detr_model.infer()