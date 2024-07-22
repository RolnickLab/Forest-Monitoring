"""Class to train a PyTorch model."""
import json
import os
import traceback
from pathlib import Path

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
    def __init__(self, net, pretrained=True, feature_extractor=None):
        self.state = {}
        self.net = net
        self.feature_extractor = feature_extractor
        super().__init__(pretrained)

        # Init accelerator and device
        self.accelerator = Accelerator(log_with="wandb")
        self.device = self.accelerator.device

        self.net.to(self.device)
        self.loss = Loss(loss_name=self.cfg["model"]["loss"])
        self.loss_name = self.cfg["model"]["loss"]

        self.d_sizes = (self.cfg["dataset"]["w_size"], self.cfg["dataset"]["h_size"])
        self.name = self.cfg["model"]["name"]
        self._build_optim()
        self.lr_step = self.cfg["model"]["lr_step"]

        # Get the total number of parameters in the model
        total_params = sum(p.numel() for p in self.net.parameters())
        param_count_millions = total_params / 1e6
        print(f"The model has {param_count_millions:.2f} million parameters.")

        self.project_name = (
            self.cfg["dataset"]["name"]
            + "_"
            + self.cfg["model"]["name"]
            + "_"
            + self.cfg["model"]["loss"]
            + self.cfg["base_exp"]
        )

        self.visualizer = WandbVisualizer(
            self.project_name,
            self.cfg["experiment"]["name"],
            cfg=self.cfg,
            n_classes=self.n_classes,
            class_names=self.class_names,
            accelerate=self.accelerator,
        )

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

    def train(self) -> None:
        epoch_iterator = range(self.epoch, self.n_epochs)
        for epoch in epoch_iterator:
            self.next_epoch()
            for i, data in enumerate(tqdm(self.loaders["train"]), 0):
                self.next_step()
                inputs, labels = self._format_inputs(data)
                self.optimizer.zero_grad()
                if self.data_mode == "SITS":
                    outputs = self._format_outputs(
                        self.net(inputs[0], batch_positions=inputs[1])
                    )
                else:
                    outputs = self._format_outputs(self.net(inputs))
                _, _ = self.loss.compute(outputs, labels)
                self.accelerator.backward(sum(self.loss._values))
                self.optimizer.step()
                self._train_stepper

            self.visualizer.update_lr(self.scheduler.get_last_lr()[0], self.step)


    def evaluate(self, split):
        self.net.eval()
        eval_loss = Loss(self.cfg["model"]["loss"])
        rand_frame = np.random.randint(len(self.loaders[split]))
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.loaders[split]), 0):
                inputs, labels = self._format_inputs(data)
                self.optimizer.zero_grad()
                
                if self.data_mode == "SITS":
                    outputs = self._format_outputs(
                        self.net(inputs[0], batch_positions=inputs[1])
                    )
                else:
                    outputs = self._format_outputs(self.net(inputs))

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

                    if (
                        self.vis_step % self.step
                        and i == rand_frame
                        and self.task == "Segmentation"
                    ):

                        inputs = inputs[0]

                        self.visualizer.update_input_output_labels(
                            inputs[:, 2, :, :, :],
                            outputs,
                            labels[:, 0, :, :],
                            self.step,  # 2nd index is the image from September
                        )
                else:
                    #                    if self.data_mode != "SITS":
                    logging_image = labels[:, 0, :, :].cpu()
                    #                    else:
                    # logging_image = labels.cpu()

                    self.evaluator.add_batch(logging_image, preds.cpu())

                    if (
                        self.vis_step % self.step
                        and i == rand_frame
                        and self.task == "Segmentation"
                    ):

                        inputs = inputs[0]  # To get the first element of the tuple

                        self.visualizer.update_input_output_labels(
                            inputs[:, 2, :, :, :], outputs, logging_image, self.step
                        )

        metrics_species = self.evaluator.get_metrics()
        if self.loss_name == "HLoss":
            metrics_genus = self.evaluator_genus.get_metrics()
            metrics_family = self.evaluator_family.get_metrics()


        self._save(metrics_species, split)

        self.accelerator.print(
            "Performances on the {} set: loss={}, accuracy={}, "
            "recall={}, dice={}".format(
                split,
                round(eval_loss.running_value, 4),
                round(metrics_species["Accuracy"][0], 4),
                round(metrics_species["Recall"][0], 4),
                round(metrics_species["Dice"][0], 4),
            )
        )

        if self.loss_name == "HLoss":
            self.visualizer.update_losses(
                split, eval_loss.running_value, eval_loss.running_hloss_values, self.step
            )
        else:
            self.visualizer.update_losses(
                split, eval_loss.running_value, eval_loss.running_values, self.step
            )

        eval_loss.reset
        self.evaluator.reset
        self.net.train()
        if self.loss_name == "HLoss":
            return metrics_species, metrics_genus, metrics_family
        else:
            return metrics_species, None, None

    def get_predictions(self, ckpt_path, save_path):

        if os.path.isfile(ckpt_path):
            self.net.load_state_dict(torch.load(ckpt_path))
        else:
            self.accelerator.load_state(ckpt_path)
        self.net.eval()

        save_index = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.loaders["test"]), 0):
                inputs, labels = self._format_inputs(data)
                self.optimizer.zero_grad()
                if self.data_mode == "SITS":
                    outputs = self._format_outputs(self.net(inputs[0], batch_positions=inputs[1]))
                else:
                    outputs = self._format_outputs(self.net(inputs))

                if self.name == "processor_unet" or self.name == "processor_deeplab":
                    inputs = inputs[0][:, :, 2, :, :]
                # Taking the second index for September 2nd
                else:
                    inputs = inputs[0][:, 2, :, :, :]
                labels = labels[:, 0, :, :]  # Taking the species class labels
                outputs = torch.argmax(outputs, dim=1)

                # Create dirs
                inputs_path = os.path.join(save_path, "inputs")
                labels_path = os.path.join(save_path, "labels")
                outputs_path = os.path.join(save_path, "outputs")

                # Make directories
                # Add check for path if it doesn't exist
                if not os.path.exists(inputs_path):
                    os.mkdir(inputs_path)
                if not os.path.exists(labels_path):
                    os.mkdir(labels_path)
                if not os.path.exists(outputs_path):
                    os.mkdir(outputs_path)

                self.save_predictions(inputs, labels, outputs, save_path, save_index)
                save_index = save_index + self.batch_size

    def save_predictions(self, inputs, labels, outputs, save_path, save_index):

        inputs_path = os.path.join(save_path, "inputs")
        labels_path = os.path.join(save_path, "labels")
        outputs_path = os.path.join(save_path, "outputs")

        # Create means
        means = torch.tensor([0.485, 0.456, 0.406])
        stds = torch.tensor([0.229, 0.224, 0.225])

        # Reshape to image size
        means = means.view(3, 1, 1)
        stds = stds.view(3, 1, 1)

        means = means.numpy()
        stds = stds.numpy()

        for i in range(inputs.shape[0]):
            inputs_idx = inputs[i]
            labels_idx = labels[i]
            outputs_idx = outputs[i]

            labels_idx = labels_idx.cpu().detach().numpy()
            outputs_idx = outputs_idx.cpu().detach().numpy()

            # Convert tensor back to image
            inputs_idx = inputs_idx.cpu().detach().numpy()
            inv_array = inputs_idx * stds + means
            # C, H, W -> H, W, C
            inv_array = np.transpose(inv_array, (1, 2, 0))

            inv_array = np.clip(inv_array, 0, 1)

            inv_array = (inv_array * 255).astype(np.uint8)

            image_out = Image.fromarray(inv_array)
            gt_out = Image.fromarray(map_to_colors(labels_idx.astype("uint8")))
            preds_out = Image.fromarray(map_to_colors(outputs_idx.astype("uint8")))

            image_out.save(os.path.join(inputs_path, str(save_index + i) + ".png"))
            gt_out.save(os.path.join(labels_path, str(save_index + i) + ".png"))
            preds_out.save(os.path.join(outputs_path, str(save_index + i) + ".png"))

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
            else:
                raise Exception("Task {} is not supported.".format(self.task))
        with open(path / file_name, "w") as stream:
            json.dump(metrics, stream)

    def _check_eval_metric(self):
        if self.selection_metric:
            metric_names = self.evaluator.get_metric_names()
            assert self.selection_metric in metric_names