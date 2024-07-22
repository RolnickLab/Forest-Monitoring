import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

"""
Modified from: https://github.com/jfzhang95/pytorch-deeplab-xception
"""


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(
        self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        # Image norm
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img, "label": mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        mask = sample["label"]

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.uint8).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {"image": img, "label": mask}


class RandomHorizontalFlip(object):
    def __init__(self, random_val, random_int):
        self.random_val = random_val
        self.random_int = random_int

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if self.random_val < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "label": mask}


class RandomVerticalFlip(object):
    def __init__(self, random_val, random_int):
        self.random_val = random_val
        self.random_int = random_int

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if self.random_val < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {"image": img, "label": mask}


class RandomRotate90(object):
    """Rotate by 90 degrees to prevent blackspace on the edges."""

    def __init__(self, random_val, random_int):
        self.random_val = random_val
        self.random_int = random_int

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        #        rotate_degree = random.choice(self.degrees)

        img = img.rotate(self.random_int, Image.BILINEAR)
        mask = mask.rotate(self.random_int, Image.NEAREST)

        return {"image": img, "label": mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {"image": img, "label": mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, "label": mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.0))
        y1 = int(round((h - self.crop_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, "label": mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {"image": img, "label": mask}


# Non-geometric augmentations


class RandomGaussianBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < self.prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {"image": img, "label": mask}


class RandomBrightnessContrast(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        # Random Brightness
        if random.random() < self.prob:
            brightness_factor = random.uniform(-0.7, 1.3)  # Might need to fix these values later

            filter = ImageEnhance.Brightness(img)
            img = filter.enhance(brightness_factor)

        # Random Contrast
        if random.random() < self.prob:
            contrast_factor = random.uniform(-0.7, 1.3)  # Might need to fix these values later

            filter = ImageEnhance.Contrast(img)
            img = filter.enhance(contrast_factor)

        return {"image": img, "label": mask}


class RandomSaturation(object):
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        # Random Brightness
        if random.random() < self.prob:
            saturation_factor = random.uniform(-0.7, 1.5)  # Might need to fix these values later

            filter = ImageEnhance.Color(img)
            img = filter.enhance(saturation_factor)

        return {"image": img, "label": mask}
