import json

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils.paths import Paths


class InstsegDataset(Dataset):
    """
    #TODO: Create an instance segmentation dataloader after creating the data.
    Adding a template semseg dataloader for now.

    """

    def __init__(self, split, size, transformations):
        self.split = split
        self.transformations = transformations
        self.dataset_path = Paths().get()["quebectrees"]
        # Change this to load from folders and using glob instead of csv.
        self.file_name = split + "_" + str(size) + ".csv"
        self.dataset = pd.read_csv(self.dataset_path / "splits" / self.file_name)
        self.class_names = self._load_class_names()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset.iloc[idx]["tiles"]
        mask_path = self.dataset.iloc[idx]["labels"]
        img = np.array(Image.open(img_path))

        # TODO: Write custom augmentation lists for segmentation
        img = self.transformations(img)
        mask = np.array(Image.open(mask_path))
        mask = (mask / 10).astype(np.int8)
        frame = {"img": img, "mask": mask}
        return frame

    @property()
    def _load_class_names(self):
        classes_path = self.dataset_path / "classes.json"
        with open(classes_path, "r") as fp:
            names_class = json.load(fp)
        class_names = {v: k for k, v in names_class.items()}
        class_names[0] = "Background"
        return class_names

    @property
    def get_names(self):
        return self.class_names
