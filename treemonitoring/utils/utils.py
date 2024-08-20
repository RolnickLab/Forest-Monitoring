import glob
import json
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from tqdm import tqdm

from treemonitoring.utils.paths import Paths

paths = Paths().get()

classes_path = paths["quebectrees"] / "classes_onlymyriam_augmented.json"
with open(classes_path, "r") as fp:
    names_class = json.load(fp)
class_names = {v: k for k, v in names_class.items()}
class_names[0] = "Background"


def glob_recursive(path: str, ext: str) -> List[str]:
    """
    Inputs:

    path: The rooot file path.
    ext: Extension of the files.

    Returns:

    files: A list with all the file names.
    """

    ext = ext.replace(".", " ")
    files = glob.glob(path + "/**/*." + ext, recursive=True)
    return files


def calculate_label_weights(
    dataloader: torch.utils.data.DataLoader, num_classes: int
) -> np.ndarray:
    """From here: https://github.com/jfzhang95/pytorch-deeplab-
    xception/blob/master/utils/calculate_weights.py."""
    # Create an instance from the data loader
    if os.path.isfile(os.path.join(paths["class_weights"], "treemonitoring_classes_weights.npy")):
        print(
            "Loading pre-existing weight file! Please manually delete this file if there is a change in training dataset!"
        )
        return
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print("Calculating classes weights")

    for sample in tqdm_batch:
        y = sample["label"]
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # TO-DO: Change this later.
    classes_weights_path = os.path.join(
        paths["class_weights"], "treemonitoring_classes_weights.npy"
    )
    np.save(classes_weights_path, ret)


def stich_tile(img: str, size: int, isMask: bool) -> Image:
    """
    Input:
    img: Image file path
    size: Size of the resulting image. An odd multiple of of 256 so
            that (0, 0) tile is the centre of the stiched image.

    Returns:
    result: Stiched Image of desired resolution.
    """
    base_size = 256

    splits = img.split("/")
    z, x, y = splits[-3], splits[-2], splits[-1].split(".")[0]

    result = Image.new("RGB", (size, size), color=0)

    i_count = 0
    j_count = 0

    folders = img.split("/")[:-2]
    image_path = "/" + os.path.join(*folders)

    assert (
        size // base_size
    ) % 2 != 0, "Size should be an odd multiple of of 256 so that (0, 0) tile is the centre of the stiched image."
    # To loop from -i to i centered around 0. Might be a more elegant way to do this?
    loop_range = int((size // base_size) / 2)

    for i in range(-loop_range, loop_range + 1, 1):

        j_count = 0
        for j in range(-loop_range, loop_range + 1, 1):
            try:
                temp_img = Image.open(
                    os.path.join(image_path, str(int(x) + i), str(int(y) + j) + ".png")
                )
            except:
                temp_img = Image.new("RGB", (base_size, base_size), color=0)

            result.paste(im=temp_img, box=(i_count * base_size, j_count * base_size))

            j_count += 1
        i_count += 1

    if isMask:
        result = Image.fromarray(np.array(result)[:, :, 0])

    return result


def _make_confusion_matrix(cm_array):
    cm_df = pd.DataFrame(cm_array.astype("uint8"), class_names.values(), class_names.values())
    cm_fig = sn.heatmap(cm_df, annot=False, cbar=False)
    return cm_fig


def map_to_colors(arr):
    # Define a color map with 16 distinct colors
    color_map = np.array(
        [
            [0, 0, 0],
            [230, 126, 34],
            [41, 128, 185],
            [142, 68, 173],
            [39, 174, 96],
            [241, 196, 15],
            [231, 76, 60],
            [26, 188, 156],
            [243, 156, 18],
            [155, 89, 182],
            [22, 160, 133],
            [247, 220, 111],
            [192, 57, 43],
            [52, 152, 219],
            [211, 84, 0],
            [44, 62, 80],
        ]
    )

    #    color_map = np.array(
    #        [
    #            [0, 0, 0],  # Black
    #            [230, 25, 75],  # Red
    #            [60, 180, 75],  # Green
    #            [255, 225, 25],  # Yellow
    #            [0, 130, 200],  # Blue
    #            [245, 130, 48],  # Orange
    #            [145, 30, 180],  # Purple
    #            [70, 240, 240],  # Cyan
    #            [240, 50, 230],  # Magenta
    #            [210, 245, 60],  # Lime
    #            [250, 190, 212],  # Pink
    #            [0, 128, 128],  # Teal
    #            [220, 190, 255],  # Lavender
    #            [170, 110, 40],  # Brown
    #            [255, 250, 200],  # Beige
    #            [128, 0, 0],  # Maroon
    #        ]
    #    )
    # Create a 3D array of the same shape as the input array
    color_array = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    # Map each integer to its corresponding color
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color_array[i, j] = color_map[arr[i, j]]

    return color_array
