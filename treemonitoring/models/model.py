"""
Model class for training and evaluating PyTorch models for tree monitoring.

This module provides a flexible framework for training various PyTorch models
on tree monitoring datasets, with support for different loss functions,
optimizers, and evaluation metrics.
"""

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from PIL import Image
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from treemonitoring.models.base import BaseExperiment
from treemonitoring.utils.loss import Loss
from treemonitoring.utils.metrics import Evaluator
from treemonitoring.utils.utils import map_to_colors
from treemonitoring.utils.visualizer import WandbVisualizer


class Model(BaseExperiment):
    """
    A class for training and evaluating PyTorch models.

    This class extends BaseExperiment and provides methods for training,
    evaluation, and prediction using various PyTorch models.
    """

    def __init__(
        self,
        net: nn.Module,
        pretrained: bool = True,
        feature_extractor: Optional[Any] = None
    ):
        """
        Initialize the Model instance.

        Args:
            net (nn.Module): The PyTorch model to be trained/evaluated.
            pretrained (bool): Whether to use pretrained weights.
            feature_extractor (Optional[Any]): Feature extractor for the model.
        """
        self.state: Dict[str, Any] = {}
        self.net = net
        self.feature_extractor = feature_extractor
        super().__init__(pretrained)

        self._initialize_accelerator()
        self._setup_model()
        self._setup_loss_and_metrics()
        self._setup_visualizer()

    def _initialize_accelerator(self) -> None:
        """Initialize the Accelerator for distributed training."""
        self.accelerator = Accelerator(log_with="wandb")
        self.device = self.accelerator.device

    def _setup_model(self) -> None:
        """Set up the model, optimizer, and scheduler."""
        self.net.to(self.device)
        self.name = self.cfg["model"]["name"]
        self._build_optim()
        self.lr_step = self.cfg["model"]["lr_step"]

        total_params = sum(p.numel() for p in self.net.parameters())
        param_count_millions = total_params / 1e6
        print(f"The model has {param_count_millions:.2f} million parameters.")

        self._prepare_distributed_components()

    def _setup_loss_and_metrics(self) -> None:
        """Set up loss function and evaluation metrics."""
        self.loss = Loss(loss_name=self.cfg["model"]["loss"])
        self.loss_name = self.cfg["model"]["loss"]

        self.evaluator = Evaluator(
            self.cfg["experiment"]["eval_metrics"], self.task, self.n_classes
        )
        self.evaluator_genus = Evaluator(
            self.cfg["experiment"]["eval_metrics_genus"], self.task, self.n_classes_genus
        )
        self.evaluator_family = Evaluator(
            self.cfg["experiment"]["eval_metrics_family"], self.task, self.n_classes_family
        )

        self._check_eval_metric()

    def _setup_visualizer(self) -> None:
        """Set up the visualizer for logging and visualization."""
        self.project_name = (
            f"{self.cfg['dataset']['name']}_{self.cfg['model']['name']}_"
            f"{self.cfg['model']['loss']}{self.cfg['base_exp']}"
        )

        self.visualizer = WandbVisualizer(
            self.project_name,
            self.cfg["experiment"]["name"],
            cfg=self.cfg,
            n_classes=self.n_classes,
            class_names=self.class_names,
            accelerate=self.accelerator,
        )

    def _prepare_distributed_components(self) -> None:
        """Prepare components for distributed training."""
        (
            self.net,
            self.optimizer,
            self.loaders["train"],
            self.loaders["val"],
            self.loaders["test"],
            self.scheduler,
        ) = self.accelerator.prepare(
            self.net,
            self.optimizer,
            self.loaders["train"],
            self.loaders["val"],
            self.loaders["test"],
            self.scheduler,
        )

        self.accelerator.register_for_checkpointing(self.optimizer)
        self.accelerator.register_for_checkpointing(self.scheduler)
        self.accelerator.register_for_checkpointing(self.net)

    def train(self) -> None:
        """
        Train the model for the specified number of epochs.
        """
        epoch_iterator = range(self.epoch, self.n_epochs)
        for epoch in epoch_iterator:
            self.next_epoch()

            for i, data in enumerate(tqdm(self.loaders["train"]), 0):
                self.next_step()

                self.optimizer.zero_grad()

                inputs, labels = self._format_inputs(data)

                outputs = self._format_outputs(self.net(inputs))

                if self.loss_name == "HLoss":
                    _, _ = self.loss.compute(outputs, labels)
                else:
                    _, _ = self.loss.compute(outputs, labels[:, 0, :, :])

                if self.name == "mask2former":
                    for l in self.loss._values:
                        agg_val = sum(list(l.values()))
                    self.accelerator.backward(agg_val)
                else:
                    self.accelerator.backward(sum(self.loss._values))
                self.optimizer.step()
                self._train_stepper

            self.visualizer.update_lr(self.scheduler.get_last_lr()[0], self.step)


    def evaluate(self, split: str, record_pred: bool = False) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        self.net.eval()
        eval_loss = Loss(self.cfg["model"]["loss"])
        rand_frame = np.random.randint(len(self.loaders[split]))

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.loaders[split])):
                inputs, labels = self._format_inputs(data)
                
                outputs = self._format_outputs(
                    self.net(inputs[0], batch_positions=inputs[1])
                    if self.data_mode == "SITS"
                    else self.net(inputs)
                )

                if self.loss_name == "HLoss":
                    genus_tensor, family_tensor = eval_loss.compute(outputs, labels)
                    genus_tensor, family_tensor = self.accelerator.gather_for_metrics(
                        (torch.argmax(genus_tensor, axis=1), torch.argmax(family_tensor, axis=1))
                    )
                else:
                    eval_loss.compute(outputs, labels)

                labels, preds = self.accelerator.gather_for_metrics(
                    (labels, torch.argmax(outputs, axis=1))
                )

                if self.loss_name == "HLoss":
                    self.evaluator.add_batch(labels[:, 0, :, :].cpu(), preds.cpu())
                    self.evaluator_genus.add_batch(labels[:, 1, :, :].cpu(), genus_tensor.cpu())
                    self.evaluator_family.add_batch(labels[:, 2, :, :].cpu(), family_tensor.cpu())
                    logging_image = labels[:, 0, :, :]
                else:
                    logging_image = labels[:, 0, :, :].cpu()
                    self.evaluator.add_batch(logging_image, preds.cpu())

                if self.vis_step % self.step == 0 and i == rand_frame and self.task == "Segmentation":
                    inputs = inputs[0]
                    self.visualizer.update_input_output_labels(
                        inputs[:, 2, :, :, :], outputs, logging_image, self.step
                    )

        metrics_species = self.evaluator.get_metrics()
        metrics_genus = self.evaluator_genus.get_metrics() if self.loss_name == "HLoss" else None
        metrics_family = self.evaluator_family.get_metrics() if self.loss_name == "HLoss" else None

        self._save(metrics_species, split)

        self.accelerator.print(
            f"Performances on the {split} set: "
            f"loss={eval_loss.running_value:.4f}, "
            f"accuracy={metrics_species['Accuracy'][0]:.4f}, "
            f"recall={metrics_species['Recall'][0]:.4f}, "
            f"dice={metrics_species['Dice'][0]:.4f}"
        )

        self.visualizer.update_losses(
            split,
            eval_loss.running_value,
            eval_loss.running_hloss_values if self.loss_name == "HLoss" else eval_loss.running_values,
            self.step
        )

        eval_loss.reset()
        self.evaluator.reset()
        self.net.train()

        return metrics_species, metrics_genus, metrics_family if self.loss_name == "HLoss" else (metrics_species, None, None)


    def get_predictions(self, ckpt_path: str, save_path: str) -> None:
        """
        Generate and save predictions using a checkpoint.

        Args:
            ckpt_path (str): Path to the checkpoint file.
            save_path (str): Directory to save the predictions.
        """
        # Load checkpoint
        if os.path.isfile(ckpt_path):
            self.net.load_state_dict(torch.load(ckpt_path))
        else:
            self.accelerator.load_state(ckpt_path)
        self.net.eval()

        save_index = 0

        # Create output directories
        for subdir in ["inputs", "labels", "outputs"]:
            os.makedirs(os.path.join(save_path, subdir), exist_ok=True)

        with torch.no_grad():
            for data in tqdm(self.loaders["test"], desc="Generating predictions"):
                inputs, labels = self._format_inputs(data)
                
                # Forward pass
                if self.data_mode == "SITS":
                    outputs = self._format_outputs(self.net(inputs[0], batch_positions=inputs[1]))
                else:
                    outputs = self._format_outputs(self.net(inputs))

                # Taking the second index for September 2nd
                if self.name in ["processor_unet", "processor_deeplab"]:
                    inputs = inputs[0][:, :, 2, :, :]
                else:
                    inputs = inputs[0][:, 2, :, :, :]
                
                labels = labels[:, 0, :, :]  # Taking the species class labels
                outputs = torch.argmax(outputs, dim=1)

                self.save_predictions(inputs, labels, outputs, save_path, save_index)
                save_index += self.batch_size

    def save_predictions(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor, save_path: str, save_index: int) -> None:
        """
        Save input images, ground truth labels, and prediction outputs as PNG files.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            labels (torch.Tensor): Ground truth labels tensor
            outputs (torch.Tensor): Prediction outputs tensor
            save_path (str): Base path to save the images
            save_index (int): Starting index for saved images
        """
        inputs_path = os.path.join(save_path, "inputs")
        labels_path = os.path.join(save_path, "labels")
        outputs_path = os.path.join(save_path, "outputs")

        means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).numpy()
        stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).numpy()

        for i in range(inputs.shape[0]):
            input_img = inputs[i].cpu().detach().numpy()
            label_img = labels[i].cpu().detach().numpy()
            output_img = outputs[i].cpu().detach().numpy()

            # Convert input tensor back to image
            inv_array = input_img * stds + means
            inv_array = np.transpose(inv_array, (1, 2, 0))
            inv_array = np.clip(inv_array * 255, 0, 255).astype(np.uint8)

            image_out = Image.fromarray(inv_array)
            gt_out = Image.fromarray(map_to_colors(label_img.astype(np.uint8)))
            preds_out = Image.fromarray(map_to_colors(output_img.astype(np.uint8)))

            image_out.save(os.path.join(inputs_path, f"{save_index + i}.png"))
            gt_out.save(os.path.join(labels_path, f"{save_index + i}.png"))
            preds_out.save(os.path.join(outputs_path, f"{save_index + i}.png"))

    @property
    def _train_stepper(self):
        if self.step % self.loss_step == 0:
            self.accelerator.print(
                "epoch [{}, {}] loss: {}".format(self.epoch, self.n_epochs, self.loss.running_value)
            )

            self.visualizer.update_losses(
                "train", self.loss.running_value, self.loss.running_hloss_values, self.step
            )
            self.loss.reset
        if self.step % self.lr_step == 0:
            self.scheduler.step()

        # Evaluate model
        # Hypothesis: self.val_step % self.ckp_step == 0 !
        if self.step % self.val_step == 0:
            val_metrics_species, val_metrics_genus, val_metrics_family = self.evaluate("val")
            self.visualizer.update_metrics("val", val_metrics_species, self.step)

            if self.loss_name == "HLoss":
                self.visualizer.update_metrics("val_genus", val_metrics_genus, self.step)
                self.visualizer.update_metrics("val_family", val_metrics_family, self.step)

            if self.step % self.ckp_step == 0:
                if self.selection_metric:
                    # Hypothesis: metric['xxx'][0] is the average over the classes
                    if self.task == "Segmentation":
                        eval_criteria = (
                            val_metrics_species[self.selection_metric][0] > self._best_eval_metric
                        )
                    else:
                        raise Exception("Task {} is not supported.".format(self.task))
    
                    torch.save(
                        self.net.state_dict(),
                        Path(self.cfg["paths"]["checkpoint_dir"])
                        / "checkpoint_iter_{}.pt".format(str(self.step).zfill(8)),
                    )

                    self._best_eval_metric = val_metrics_species[self.selection_metric][0]

                    if self.loss_name == "HLoss":
                        (
                            test_metrics_species,
                            test_metrics_genus,
                            test_metrics_family,
                        ) = self.evaluate("test")
                    else:
                        test_metrics_species, _, _ = self.evaluate("test")

                    self.visualizer.update_metrics("test", test_metrics_species, self.step)

                    if self.loss_name == "HLoss":
                        self.visualizer.update_metrics("test_genus", test_metrics_genus, self.step)
                        self.visualizer.update_metrics(
                            "test_family", test_metrics_family, self.step
                        )
         
    def _build_optim(self):
        optim_name = self.cfg["model"]["optim"]
        if optim_name == "SGD":
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.cfg["model"]["lr"], momentum=0.9
            )
        elif optim_name == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg["model"]["lr"])
        else:
            raise Exception("Optimization method {} is not supported yet.".format(optim_name))
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0.0, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def recursive_todevice(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: self.recursive_todevice(v, device) for k, v in x.items()}
        else:
            return [self.recursive_todevice(c, device) for c in x]

    def _format_inputs(self, data):
        if self.task == "Segmentation":
            if self.data_mode == "SITS":
                image, dates, labels = data["image"], data["dates"], data["label"]

                image = image.to(self.device).float()

                if self.name == "processor_unet" or self.name == "processor_deeplab":
                    image = image.permute(0, 2, 1, 3, 4)
                dates = dates.to(self.device).float()
                inputs = (image, dates)

                labels = labels.to(self.device).long()

            else:
                inputs, labels = data["image"], data["label"]
                labels = labels.to(self.device).long()
                inputs = inputs.to(self.device).float()
        else:
            inputs, labels = data
            labels = labels.to(self.device)
            inputs = inputs.to(self.device).float()

        return inputs, labels

    def _format_outputs(self, outputs):
        if self.name in ["deeplabv3", "deeplabv3resnet101", "deeplabv3resnet50"]:
            outputs = outputs["out"]
        elif self.name in [
            "unetresnet50",
            "unet",
            "dualgcnresnet50",
            "unet3d",
            "processor_unet",
            "processor_deeplab",
            "utae",
            "fpn",
            "convlstm",
        ]:
            outputs = outputs
        else:
            raise Exception(
                "Output formatting is not supported yet for model {}.".format(self.name)
            )
        return outputs

    def _save(self, metrics, split):
        """save metrics."""
        path = Path(self.cfg["paths"]["results_dir"])
        file_name = "metrics_{}_{}.json".format(split, str(self.step).zfill(8))
        for metric_name in metrics.keys():
            if self.task == "Segmentation":
                metrics[metric_name] = [
                    metrics[metric_name][0],
                    metrics[metric_name][1].tolist(),
                ]
            elif self.task in ("Regression", "MultiRegression"):
                pass
            else:
                raise Exception("Task {} is not supported.".format(self.task))
        with open(path / file_name, "w") as stream:
            json.dump(metrics, stream)

    def _check_eval_metric(self):
        if self.selection_metric:
            metric_names = self.evaluator.get_metric_names()
            assert self.selection_metric in metric_names
