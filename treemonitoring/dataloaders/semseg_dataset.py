import json
import os
from typing import Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import treemonitoring.dataloaders.custom_transforms as tr
from treemonitoring.utils.paths import Paths
from treemonitoring.utils.utils import glob_recursive, stich_tile


class SemsegDataset(Dataset):
    def __init__(self, split: str, mode: str, size: int, cv: str):
        self.split = split
        self.mode = mode
        self.transformations = None
        self.dataset_path = Paths().get()["quebectrees"]
        self.base_size = 768
        
        self.cv_version = cv
        if mode == "train":
            self.file_name = os.path.join(self.dataset_path, 'train_768.csv')        
        elif mode == "val":
            self.file_name = self.file_name = os.path.join(self.dataset_path, 'val_768.csv') 
        else:
            self.file_name = os.path.join(self.dataset_path, 'test_768.csv') 

        self.dataset = pd.read_csv(os.path.join(self.dataset_path, "splits", self.file_name))
        print(os.path.join(self.dataset_path, "splits", self.file_name))
        self.class_names = self._load_class_names()
        self.final_size = size

    def __len__(self):
        return len(self.dataset)

    def transform_tr(self, sample):
        composed_transforms = self.get_train_transforms("geometric")
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def get_train_transforms(self, group="geometric"):
        if group == "geometric":
            return transforms.Compose(
                [
                    tr.RandomHorizontalFlip(),
                    tr.RandomRotate90(),
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor(),
                ]
            )
        else:
            return transforms.Compose(
                [
                    tr.RandomHorizontalFlip(),
                    tr.RandomRotate90(),
                    tr.RandomGaussianBlur(),
                    tr.RandomBrightnessContrast(),
                    tr.RandomSaturation(),
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor(),
                ]
            )

    def __getitem__(self, idx: int) -> (Dict[str, torch.Tensor]):

        img_path = self.dataset.iloc[idx]["tiles"]
        mask_path = self.dataset.iloc[idx]["labels"]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        sample = {"image": img, "label": mask}

        if self.mode == "train":
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)

        return sample

    def _load_class_names(self):
        classes_path = self.dataset_path / "classes_onlymyriam_augmented.json"
        with open(classes_path, "r") as fp:
            names_class = json.load(fp)
        class_names = {v: k for k, v in names_class.items()}
        class_names[0] = "Background"
        return class_names

    @property
    def get_names(self):
        return self.class_names
