# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_single as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, depth_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, use_depth=False):
        super(CocoDetection, self).__init__(img_folder, depth_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size, use_depth=use_depth)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, depth, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, depth, target = self.prepare(img, depth, target)
        if self._transforms is not None:
            img, depth, target = self._transforms(img, depth, target)
        if self.use_depth: # if depth is used, concatenate the depth image with the RGB image
            assert img.shape[0] == 3, "Image should have 3 channels"
            assert depth.shape[0] == 1, "Depth should have 1 channel"
            img = torch.cat((img, depth), dim=0)
            return img, target
        return img, target


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

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, depth, target


def make_coco_transforms(image_set, args):
    
    # Normalize RGB and depth images, RGB based on ImageNet statistics and depth based on Dformer statistics
    if args.dformer_backbone:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406, 0.48], [0.229, 0.224, 0.225, 0.28])
        ])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406, 0.449], [0.229, 0.224, 0.225, 0.226])
        ])
    
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=1333),
            normalize,
            ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    # Use depth coco labels, else use RGB coco labels. Both can be the same. As the coc files will be read, and the image folder will be replaced by 'images' with 'depth'
    if args.use_depth:
        PATHS = {
        "train_joint": (root , root, root / 'coco' /  'annotations'  / 'train.json'),
        "val": (root , root , root / 'coco' / 'annotations' / 'train.json'),
    }
    else:
        PATHS = {
            "train_joint": (root , root, root / 'coco' /  'annotations_depth'  / 'train.json'),
            "val": (root , root , root / 'coco' / 'annotations_depth' / 'train.json'),
        }

    for key, value in PATHS.items():
        print(f"{key}:")
        print(f"Image Path: {value[0]}, \nDepth Path 2: {value[1]}, \nAnnotations Path: {value[2]}")
        print()

    img_folder, depth_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, depth_folder, ann_file, transforms=make_coco_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), use_depth=args.use_depth)
    return dataset