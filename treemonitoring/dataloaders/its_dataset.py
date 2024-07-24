import json
import os
import random
from typing import Dict

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import treemonitoring.dataloaders.custom_transforms as tr
from treemonitoring.utils.paths import Paths
from treemonitoring.utils.utils import glob_recursive, stich_tile


REPO_ROOT = Path(__file__).parents[2]  # Go up two levels to reach the repo root
TREE_DATA_DIR = REPO_ROOT / "data" / "tree_data"

class ImageTimeSeriesDataset(Dataset):
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)

    def __init__(self, split: str, mode: str, size: int, cv: str):
        self.split = split
        self.transformations = None
        self.dataset_path = TREE_DATA_DIR
        self.base_size = 768

        if mode == "train":
            self.file_name = 'train_768.csv'
        elif mode == "val":
            self.file_name = 'val_768.csv'
        else:
            self.file_name = 'test_768.csv'

        self.dataset = pd.read_csv(TREE_DATA_DIR / self.file_name)
        print(f"Loading dataset from: {TREE_DATA_DIR / self.file_name}")

        self.class_names = self._load_class_names()
        self.mode = mode
        self.final_size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_paths = self._get_image_paths(idx)
        mask_path = TREE_DATA_DIR / self.dataset.iloc[idx]["labels"]
        
        images = self._load_and_transform_images(image_paths, mask_path)
        
        tensor_concat = self._concatenate_images(images)
        dates = torch.tensor([0, 77, 103, 112])  # Calculated manually from the start date.
        
        return {
            "image": tensor_concat,
            "label": images["sept2"]["label"],
            "dates": dates
        }

    def _get_image_paths(self, idx: int) -> Dict[str, Path]:
        return {
            "june": TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_june"],
            "sept2": TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_main"],
            "sept28": TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_september"],
            "oct": TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_october"],
        }

    def _load_and_transform_images(self, image_paths: Dict[str, Path], mask_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
        mask = Image.open(mask_path)
        images = {}
        
        if self.mode == "train":
            random_val = random.random()
            random_int = random.choice([0, 90, 180, 270])
            transform = self._get_train_transforms(random_val, random_int)
        else:
            transform = self._get_val_transforms()

        for name, path in image_paths.items():
            img = Image.open(path)
            sample = {"image": img, "label": mask}
            images[name] = transform(sample)
        
        return images

    def _get_train_transforms(self, random_val: float, random_int: int) -> transforms.Compose:
        return transforms.Compose([
            tr.RandomHorizontalFlip(random_val, random_int),
            tr.RandomRotate90(random_val, random_int),
            tr.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD),
            tr.ToTensor(),
        ])

    def _get_val_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            tr.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD),
            tr.ToTensor(),
        ])

    def _concatenate_images(self, images: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.cat([
            images[name]["image"].unsqueeze(0)
            for name in ["june", "sept2", "sept28", "oct"]
        ], dim=0)

    def _load_class_names(self) -> Dict[int, str]:
        classes_path = TREE_DATA_DIR / "classes_onlymyriam_augmented.json"
        with open(classes_path, "r") as fp:
            names_class = json.load(fp)
        class_names = {v: k for k, v in names_class.items()}
        class_names[0] = "Background"
        return class_names

    @property
    def get_names(self) -> Dict[int, str]:
        return self.class_names
