"""Class to create Wandb Visualization during training."""
import json

import numpy as np
import torch

import wandb
from treemonitoring.utils.paths import Paths


class WandbVisualizer:
    """Class to generate WandB visualisation."""

    def __init__(self, project_name, entity, cfg, n_classes, class_names, accelerate):
        self.project_name = "quebec_forest"  # project_name
        self.cfg = cfg
        self.n_classes = n_classes
        self.class_names = class_names
        self.accelerator = accelerate
        self.wb = self._wandb_init()
        run = self.accelerator.get_tracker("wandb", unwrap=True)
        run.tags = list(self.cfg["wandb"]["tags"])

        paths = Paths().get()
        classes_path = paths["quebectrees"] / "classes_onlymyriam_augmented.json"
        with open(classes_path, "r") as fp:
            names_class = json.load(fp)
        self.class_list = {v: k for k, v in names_class.items()}
        self.class_list[0] = "Background"

    def _wandb_init(self):
        #        init_kwargs = {'entity':'venka97'}
        return self.accelerator.init_trackers(
            project_name=self.project_name, config=self.cfg
        )  # , tags=self.cfg["wandb"]["tags"])

    def update_losses(self, split, global_loss, losses, step):
        self.accelerator.log({"{}_losses/global".format(split): global_loss})
        for loss_id, loss_value in losses.items():
            self.accelerator.log({"{}_losses/{}".format(split, loss_id): loss_value})

    def update_lr(self, lr, step):
        self.accelerator.log({"parameters/learning_rate": lr})

    def update_metrics(self, split, metrics, iteration):
        for metric_name in metrics.keys():
            self.accelerator.log(
                {
                    "{}/{}/mean".format(split, metric_name): metrics[metric_name][0],
                    "step": iteration,
                }
            )
            for i_class in range(len(metrics[metric_name][1])):
                if self.class_names:
                    self.accelerator.log(
                        {
                            "{}/{}/class_{}".format(
                                split, metric_name, self.class_names[i_class]
                            ): metrics[metric_name][1][i_class],
                            "step": iteration,
                        }
                    )
                else:
                    self.accelerator.log(
                        {
                            "{}/{}/class_{}".format(split, metric_name, str(i_class)): metrics[
                                metric_name
                            ][1][i_class],
                            "step": iteration,
                        }
                    )

    def wand_mask_gt_composite(self, bg_img, pred_mask, true_mask):
        return wandb.Image(
            bg_img,
            masks={
                "prediction": {"mask_data": pred_mask, "class_labels": self.class_list},
                "ground truth": {"mask_data": true_mask, "class_labels": self.class_list},
            },
        )

    def update_input_output_labels(self, inputs, outputs, labels, step):
        # Get the first sample from the batch
        mask_composite = []

        if isinstance(inputs, tuple):
            inputs = inputs[0]

        # print(inputs.shape, outputs.shape, labels.shape)c
        inputs = self.tensor2img(inputs[0, :, :, :])
        outputs = torch.argmax(outputs, dim=1)
        outputs = self.tensor2img(outputs[0, :, :])
        labels = self.tensor2img(labels[0, :, :])

        mask_composite.append(self.wand_mask_gt_composite(inputs, outputs, labels))

        self.log_list_to_wandb(mask_composite)

    def log_list_to_wandb(self, pred_list):
        self.accelerator.log({"Image/Preds/Labels": pred_list})

    def update_confusion_matrix(self, confusion_matrix, step):
        self.log_images_to_wandb("Confusion Matrix", confusion_matrix)

    def log_images_to_wandb(self, key_name, image_array):
        self.accelerator.log({key_name: wandb.Image(image_array)})

    def tensor2img(self, tensor, out_type=np.uint8, min_max=(-1, 1)):
        """
        Converts a torch Tensor into an image Numpy array
        Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
        """

        tensor = tensor.squeeze().cpu()
        n_dim = tensor.dim()

        if n_dim == 4:
            tensor = tensor[0, :, :, :]
            tensor = tensor.clamp_(*min_max)
            tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            img_np = tensor.permute(1, 2, 0).numpy()  # HWC, RGB
            img_np = (img_np * 255.0).round()
        if n_dim == 3:
            tensor = tensor.clamp_(*min_max)
            tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            img_np = tensor.permute(1, 2, 0).numpy()  # HWC, RGB
            img_np = (img_np * 255.0).round()
        elif n_dim == 2:
            img_np = tensor.numpy()
        else:
            raise TypeError(
                "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(n_dim)
            )
        return img_np.astype(out_type)
