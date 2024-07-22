import cv2
import matplotlib.image as pltimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class TreeSegmentationVisualizer:
    """
    PARAMETERS
    ----------
    tile: np.array
    mask: np.array
    classes: dict
    """

    def __init__(self, tile, mask, classes=None):
        self.tile = tile
        self.mask = mask
        self.classes = classes
        self.color_code = self._build_color_code()
        self.colored_mask = self._create_colored_mask()
        self.masked_tile = self._create_masked_tile()

    def _build_color_code(self):
        if self.classes:
            n_colors = len(list(self.classes.keys()))
        else:
            n_colors = len(np.unique(self.mask))
        # colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
        colors = plt.cm.gist_ncar(np.linspace(0, 1, n_colors))
        color_code = {value: colors[value] for value in range(n_colors)}
        color_code[0] = [1.0, 1.0, 1.0, 1.0]  # Special background
        return color_code

    def _create_colored_mask(self):
        colored_mask = np.zeros((self.mask.shape[0], self.mask.shape[1], 4))
        for class_val in self.color_code.keys():
            colored_mask[self.mask == class_val] = self.color_code[class_val]
        colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
        return colored_mask

    def _create_masked_tile(self):
        masked_tile = cv2.addWeighted(self.tile, 0.8, self.colored_mask, 0.4, 0.0)
        return masked_tile

    def get_color_code(self):
        return self.color_code

    def get_colored_mask(self):
        return self.colored_mask

    def save_colored_mask(self, path):
        pltimg.imsave(path, self.colored_mask)

    def save_masked_tile(self, path, put_legend=False):
        if put_legend:
            plt.figure(figsize=(60, 60))
            plt.imshow(self.masked_tile)
            patches = []
            for class_name in self.classes.keys():
                class_val = self.classes[class_name]
                color = self.color_code[class_val]
                patch = mpatches.Patch(color=color, label=class_name)
                patches.append(patch)
            plt.legend(
                handles=patches,
                prop={"size": 45},
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            plt.savefig(path)
        else:
            pltimg.imsave(path, self.masked_tile)
