"""
Transforms and data augmentation for sequence level images, bboxes and masks.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh, box_iou
from util.misc import interpolate
import numpy as np
from numpy import random as rand
from PIL import Image
import cv2

def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    assert mode in ['iou', 'iof']
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def crop(clip, depth_clip, target, region):
    cropped_image = []
    cropped_depth = []
    for image in clip:
        cropped_image.append(F.crop(image, *region))
    if depth_clip is not None:
        for depth in depth_clip:
            if depth is not None:
                cropped_depth.append(F.crop(depth, *region))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    return cropped_image, cropped_depth, target


def hflip(clip, depth_clip, target):
    flipped_image = []
    flipped_depth = []
    for image in clip:
        flipped_image.append(F.hflip(image))
    if depth_clip is not None:
        assert len(clip) == len(depth_clip), "image and depth clip length not match in hflip"
        for depth in depth_clip:
            if depth is not None:
                flipped_depth.append(F.hflip(depth))

    w, h = clip[0].size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    
    return flipped_image, flipped_depth, target

def vflip(clip, depth, target):
    flipped_image = []
    flipped_depth = []
    for image in clip:
        flipped_image.append(F.vflip(image))
    if depth is not None:
        for depth in depth:
            flipped_depth.append(F.vflip(depth))
    
    w, h = clip[0].size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(1)

    return flipped_image, flipped_depth, target

def resize(clip, depth_clip, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

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

    size = get_size(clip[0].size, size, max_size)
    rescaled_image = []
    rescaled_depth = []
    for image in clip:
        rescaled_image.append(F.resize(image, size))
    if depth_clip is not None:
        for depth in depth_clip:
            if depth is not None:
                rescaled_depth.append(F.resize(depth, size))

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image[0].size, clip[0].size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        if target['masks'].shape[0]>0:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        else:
            target['masks'] = torch.zeros((target['masks'].shape[0],h,w))
    return rescaled_image, rescaled_depth, target


def pad(clip, depth_clip, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = []
    padded_depth = []
    for image in clip:
        padded_image.append(F.pad(image, (0, 0, padding[0], padding[1])))
    if depth_clip is not None:
        for depth in depth_clip:
            if depth is not None:
                padded_depth.append(F.pad(depth, (0, 0, padding[0], padding[1])))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[0].size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, padded_depth, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, depth, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, depth, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, depth: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img[0].width, self.max_size))
        h = random.randint(self.min_size, min(img[0].height, self.max_size))
        region = T.RandomCrop.get_params(img[0], [h, w])
        return crop(img, depth, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, depth, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, depth, target, (crop_top, crop_left, crop_height, crop_width))


class MinIoURandomCrop(object):
    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, depth, target):
        w,h = img.size
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return img, depth, target
            min_iou = mode
            boxes = target['boxes'].numpy()
            labels = target['labels']

            for i in range(50):
                new_w = rand.uniform(self.min_crop_size * w, w)
                new_h = rand.uniform(self.min_crop_size * h, h)
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue
                left = rand.uniform(w - new_w)
                top = rand.uniform(h - new_h)
                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue
                
                if len(overlaps) > 0:
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                        return mask
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if False in mask:
                        continue
                    #TODO: use no center boxes
                    #if not mask.any():
                    #    continue

                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    target['boxes'] = torch.tensor(boxes)
                
                img = np.asarray(img)[patch[1]:patch[3], patch[0]:patch[2]]
                img = Image.fromarray(img)
                depth = np.asarray(depth)[patch[1]:patch[3], patch[0]:patch[2]]
                depth = Image.fromarray(depth)
                width, height = img.size
                depth_width, depth_height = depth.size
                assert width == depth_width and height == depth_height, "image and depth size not match in min IOU random crop"
                target['orig_size'] = torch.tensor([height,width])
                target['size'] = torch.tensor([height,width])
                return img, depth, target 


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, depth, target):
        
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
            depth *= alpha
        return image, depth, target

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, image, depth, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
            depth += delta
        return image, depth, target

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth, target):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
            depth[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, depth, target

class RandomHue(object): #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, depth, target):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
            
            depth[:, :, 0] += rand.uniform(-self.delta, self.delta)
            depth[:, :, 0][depth[:, :, 0] > 360.0] -= 360.0
            depth[:, :, 0][depth[:, :, 0] < 0.0] += 360.0
        return image, depth, target

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, image, depth, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
            depth = shuffle(depth)
        return image, depth, target

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, depth, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            depth = cv2.cvtColor(depth, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, depth, target

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image, depth):
        image = image[:, :, self.swaps]
        depth = depth[:, :, self.swaps]
        return image, depth

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
    
    def __call__(self,clip, depth_clip, target):
        imgs = []
        depths = []
        for i, img in enumerate(clip):
            img = np.asarray(img).astype('float32')
            depth = np.asarray(depth[i]).astype('float32') if depth_clip is not None else None
            img, depth, target = self.rand_brightness(img, depth, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, depth, target = distort(img, depth, target)
            img, depth, target = self.rand_light_noise(img, depth, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
            depths.append(Image.fromarray(depth.astype('uint8')))

#NOTICE: if used for mask, need to change
class Expand(object):
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, clip, depth_clip, target):
        if rand.randint(2):
            return clip, depth_clip, target
        imgs = []
        dpths = []
        masks = []
        image = np.asarray(clip[0]).astype('float32')
        dpth = np.asarray(depth_clip[0]).astype('float32')
        height, width, depth = image.shape
        depth_height, depth_width, depth_depth = dpth.shape
        assert height == depth_height and width == depth_width, "image and depth size not match in expand"
        ratio = rand.uniform(1, 4)
        left = rand.uniform(0, width*ratio - width)
        top = rand.uniform(0, height*ratio - height)
        for i in range(len(clip)):
            image = np.asarray(clip[i]).astype('float32')
            dpth = np.asarray(depth_clip[i]).astype('float32')
            expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=image.dtype)
            expand_dpth = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=dpth.dtype)
            expand_image[:, :, :] = self.mean
            expand_dpth[:, :, :] = self.mean
            expand_image[int(top):int(top + height),int(left):int(left + width)] = image
            expand_dpth[int(top):int(top + height),int(left):int(left + width)] = dpth
            imgs.append(Image.fromarray(expand_image.astype('uint8')))
            dpths.append(Image.fromarray(expand_dpth.astype('uint8')))
            expand_mask = torch.zeros((int(height*ratio), int(width*ratio)),dtype=torch.uint8)
            expand_mask[int(top):int(top + height),int(left):int(left + width)] = target['masks'][i]
            masks.append(expand_mask)
        boxes = target['boxes'].numpy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = torch.tensor(boxes)
        target['masks']=torch.stack(masks)
        return imgs, dpths, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return hflip(img, depth, target)
        return img, depth, target

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return vflip(img, depth, target)
        return img, depth, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, depth, target=None):
        size = random.choice(self.sizes)
        return resize(img, depth, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, depth, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, depth, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return self.transforms1(img, depth, target)
        return self.transforms2(img, depth, target)


class ToTensor(object):
    def __call__(self, clip, depth_clip, target):
        img = []
        depth = []
        for im in clip:
            img.append(F.to_tensor(im))
        if depth_clip is not None:
            for d in depth_clip:
                depth.append(F.to_tensor(d))
        return img, depth, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, depth, target):
        return self.eraser(img), self.eraser(depth), target


class Normalize(object):
    def __init__(self, mean, std):
        self.has_depth_val = len(mean) == 4 and len(std) == 4
        self.mean = mean
        self.std = std

    def __call__(self, clip, depth_clip, target=None):
        image = []
        depth = []
        for im in clip:
            image.append(F.normalize(im, mean=self.mean[:3], std=self.std[:3]))
            if self.has_depth_val == False and depth_clip is not None:
                raise ValueError("Depth mean and std values are not provided")
            if depth_clip is not None:
                if self.has_depth_val == False:
                    raise ValueError("Depth mean and std values are not provided")
                for d in depth_clip:
                    depth.append(F.normalize(d, mean=self.mean[-1], std=self.std[-1]))
        if target is None:
            return image, None
        target = target.copy()
        h, w = image[0].shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, depth, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, target):
        for t in self.transforms:
            image, depth, target = t(image, depth, target)
        return image, depth, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string