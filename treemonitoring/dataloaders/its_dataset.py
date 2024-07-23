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
    def __init__(self, split: str, mode: str, size: int, cv: str):
        self.split = split
        self.mode = mode
        self.transformations = None
        #self.dataset_path = Paths().get()["quebectrees"]
        self.dataset_path = TREE_DATA_DIR
        self.base_size = 768
        
#        self.cv_version = cv
#        if mode == "train":
#            self.file_name = os.path.join(self.dataset_path, 'train_768.csv')        
#        elif mode == "val":
#            self.file_name = self.file_name = os.path.join(self.dataset_path, 'val_768.csv') 
#        else:
#            self.file_name = os.path.join(self.dataset_path, 'test_768.csv') 
#
#        self.dataset = pd.read_csv(os.path.join(self.dataset_path, "splits", self.file_name))
#        print(os.path.join(self.dataset_path, "splits", self.file_name))

        if mode == "train":
            self.file_name = 'train_768.csv'
        elif mode == "val":
            self.file_name = 'val_768.csv'
        else:
            self.file_name = 'test_768.csv'

        self.dataset = pd.read_csv(TREE_DATA_DIR / self.file_name)
        print(TREE_DATA_DIR / self.file_name)

        self.class_names = self._load_class_names()
        self.final_size = size

    def __len__(self):
        return len(self.dataset)

    def transform_tr(self, sample, random_val, random_int):
        composed_transforms = self.get_train_transforms(random_val, random_int, "geometric")
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def get_train_transforms(self, random_val, random_int, group="geometric"):
        if group == "geometric":
            return transforms.Compose(
                [
                    tr.RandomHorizontalFlip(random_val, random_int),
                    tr.RandomRotate90(random_val, random_int),
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor(),
                ]
            )
        else:
            return transforms.Compose(
                [
                    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    tr.ToTensor(),
                ]
            )

#    def __getitem__(self, idx: int) -> (Dict[str, torch.Tensor]):
#        img_path_may = self.dataset.iloc[idx]["tiles_may"]
#        img_path_june = self.dataset.iloc[idx]["tiles_june"]
#        img_path_july = self.dataset.iloc[idx]["tiles_july"]
#        img_path_aug = self.dataset.iloc[idx]["tiles_august"]
#        img_path_sept2 = self.dataset.iloc[idx]["tiles_main"]
#        img_path_sept28 = self.dataset.iloc[idx]["tiles_september"]
#        img_path_oct = self.dataset.iloc[idx]["tiles_october"]
#        mask_path = self.dataset.iloc[idx]["labels"]

        # img = stich_tile(img_path, self.final_size, False)
        # mask = stich_tile(mask_path, self.final_size, True)

#        img_may = Image.open(img_path_may)
#        img_june = Image.open(img_path_june)
#        img_july = Image.open(img_path_july)
#        img_aug = Image.open(img_path_aug)
#        img_sept2 = Image.open(img_path_sept2)
#        img_sept28 = Image.open(img_path_sept28)
#        img_oct = Image.open(img_path_oct)
#        mask = Image.open(mask_path)

    def __getitem__(self, idx: int) -> (Dict[str, torch.Tensor]):
        img_path_may = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_may"]
        img_path_june = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_june"]
        img_path_july = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_july"]
        img_path_aug = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_august"]
        img_path_sept2 = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_main"]
        img_path_sept28 = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_september"]
        img_path_oct = TREE_DATA_DIR / self.dataset.iloc[idx]["tiles_october"]
        mask_path = TREE_DATA_DIR / self.dataset.iloc[idx]["labels"]

        img_may = Image.open(img_path_may)
        img_june = Image.open(img_path_june)
        img_july = Image.open(img_path_july)
        img_aug = Image.open(img_path_aug)
        img_sept2 = Image.open(img_path_sept2)
        img_sept28 = Image.open(img_path_sept28)
        img_oct = Image.open(img_path_oct)
        mask = Image.open(mask_path)

        mask_array = np.array(mask)

        # Hacky fix for the LALA higher-level taxon.
        # Need to re-create the dataset channel.
        # Get the 3rd channel of the mask array
        channel_3 = mask_array[:, :, 2]

        # Change the values in the 3rd channel
        channel_3[channel_3 == 3] = 1
        channel_3[channel_3 == 4] = 3

        mask_array[:, :, 2] = channel_3

        # Convert the modified NumPy array back to a PIL image
        mask = Image.fromarray(mask_array)

        # Adding masks in all dicts for compatibility with transforms
        sample_may = {"image": img_may, "label": mask}
        sample_june = {"image": img_june, "label": mask}
        sample_july = {"image": img_july, "label": mask}
        sample_aug = {"image": img_aug, "label": mask}
        sample_sept2 = {"image": img_sept2, "label": mask}
        sample_sept28 = {"image": img_sept28, "label": mask}
        sample_oct = {"image": img_oct, "label": mask}

        if self.mode == "train":
            random_val = random.random()
            random_int = random.choice([0, 90, 180, 270])

            sample_may = self.transform_tr(sample_may, random_val, random_int)
            sample_june = self.transform_tr(sample_june, random_val, random_int)
            sample_july = self.transform_tr(sample_july, random_val, random_int)
            sample_aug = self.transform_tr(sample_aug, random_val, random_int)
            sample_sept2 = self.transform_tr(sample_sept2, random_val, random_int)
            sample_sept28 = self.transform_tr(sample_sept28, random_val, random_int)
            sample_oct = self.transform_tr(sample_oct, random_val, random_int)
        else:
            sample_may = self.transform_val(sample_may)
            sample_june = self.transform_val(sample_june)
            sample_july = self.transform_val(sample_july)
            sample_aug = self.transform_val(sample_aug)
            sample_sept2 = self.transform_val(sample_sept2)
            sample_sept28 = self.transform_val(sample_sept28)
            sample_oct = self.transform_val(sample_oct)

        #        print('Tensor', torch.unique(sample["label"]))
        tensor_may = sample_may["image"]
        tensor_may = tensor_may.unsqueeze(0)

        tensor_june = sample_june["image"]
        tensor_june = tensor_june.unsqueeze(0)

        tensor_july = sample_july["image"]
        tensor_july = tensor_july.unsqueeze(0)

        tensor_aug = sample_aug["image"]
        tensor_aug = tensor_aug.unsqueeze(0)

        tensor_sept2 = sample_sept2["image"]
        tensor_sept2 = tensor_sept2.unsqueeze(0)

        tensor_sept28 = sample_sept28["image"]
        tensor_sept28 = tensor_sept28.unsqueeze(0)

        tensor_oct = sample_oct["image"]
        tensor_oct = tensor_oct.unsqueeze(0)

        tensor_concat = torch.cat((tensor_june, tensor_sept2, tensor_sept28, tensor_oct,), dim=0)

        # Adding this line for compatibility with processor
        # TO-DO: Comment this line for UTAE & Unet3d
        #        tensor_concat = tensor_concat.permute(1, 0, 2, 3)

        #        print(tensor_concat.shape)
        # Changing dates to 4 dates
        #        dates = torch.tensor([0, 21, 55, 83, 98, 124, 133])
        dates = torch.tensor([0, 77, 103, 112]) # Calculated manually from the start date.

        label_species = sample_sept2["label"] 

        sample = {"image": tensor_concat, "label": label_species, "dates": dates}

        return sample

    def _load_class_names(self):
        classes_path = TREE_DATA_DIR / "classes_onlymyriam_augmented.json"
        with open(classes_path, "r") as fp:
            names_class = json.load(fp)
        class_names = {v: k for k, v in names_class.items()}
        class_names[0] = "Background"
        return class_names

#    def _load_class_names(self):
#        classes_path = self.dataset_path / "classes_onlymyriam_augmented.json"
#        with open(classes_path, "r") as fp:
#            names_class = json.load(fp)
#        class_names = {v: k for k, v in names_class.items()}
#        class_names[0] = "Background"
#        return class_names

    @property
    def get_names(self):
        return self.class_names
