# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from torchvision.datasets.vision import VisionDataset
import numpy as np
import cv2
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from pathlib import Path


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, depth_folder, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1, use_depth=False):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.depth_folder = depth_folder
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        self.use_depth = use_depth
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()
                
    def correct_path(self, path):
        corrected_path = path.replace('\\', os.sep)
        return corrected_path

    def get_image(self, path):
        if self.cache_mode:
            path = self.correct_path(path)
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        img_path = os.path.join(self.root, path)
        img_path = self.correct_path(img_path)
        return Image.open(img_path).convert('RGB')

    def get_depth(self, path):
        """ Get the depth image from the depth folder
        Args:
            path (str): Path to the depth image
        Returns:
            PIL Image: The depth image in PIL Image format
        """
        
        if self.cache_mode: # caching depth images is not supported
            raise NotImplementedError("Caching depth images is not supported")
        
        depth_path = os.path.join(self.depth_folder, path)
        if 'images' in depth_path:
            depth_path = depth_path.replace('images', 'depth_pred')
        depth_path = self.correct_path(depth_path)

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"No depth file found at {depth_path}")

        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # Read the depth data
        
        if depth_data.shape[-1] == 3:
            # consider only the first channel if the depth data has 3 channels
            raise ValueError("Depth image has 3 channels. Consider only the first channel")
            # depth_data = depth_data[:, :, 0]
        
        depth_normalized = ((depth_data - depth_data.min()) * (1/(depth_data.max() - depth_data.min()) * 255)).astype('uint8')
        # depth_normalized = np.clip(depth_normalized, 80, 200)
        depth_image = Image.fromarray(depth_normalized, 'L') # Convert the depth data to a PIL Image
    
        return depth_image
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        depth_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        depth_path = coco.loadImgs(depth_id)[0]['file_name']

        img = self.get_image(path)
        depth = self.get_depth(depth_path) if self.use_depth else None
        
        if self.transforms is not None:
            if not self.use_depth:
                img, target = self.transforms(img, target)
            else:
                img, depth, target = self.transforms(img, depth, target)
                
        return img, depth, target

    def __len__(self):
        return len(self.ids)
