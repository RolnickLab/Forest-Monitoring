"""Build and track a loss."""
import os

import numpy as np
import torch
import torch.nn as nn

from treemonitoring.utils.diceloss import VanillaDiceCE, WeightedDiceCE
from treemonitoring.utils.hierarchical_loss import Hierarchical_Loss
from treemonitoring.utils.paths import Paths


class Loss:
    def __init__(self, loss_name, aggregation="average", **kwargs):
        self.paths = Paths().get()
        self.aggregation = aggregation
        self.loss_names = self._split_names(loss_name)
        self.n_losses = len(self.loss_names)
        # self.data_mode = data_mode
        self.loss_ref = {}
        self._criterions = {}
        self._value = None
        self._values = None
        self._running_value = None
        self._running_values = {}
        self._running_hloss_values = {}
        self.w1 = 0.7
        self.w2 = 0.3
        self.weight_file = os.path.join(
            self.paths["class_weights"], "treemonitoring_classes_weights.npy"
        )
        self.weights = self._load_weights()
        self._build_references()
        self._build_losses()
        self._test_builders()

    def __len__(self):
        return self.n_losses

    def _split_names(self, loss_name):
        return loss_name.split("/")

    def _build_references(self):
        self.loss_ref["CrossEntropy"] = nn.CrossEntropyLoss()
        self.loss_ref["CrossEntropy1"] = nn.CrossEntropyLoss()
        self.loss_ref["BinaryCrossEntropy"] = nn.BCELoss()
        self.loss_ref["MSE"] = nn.MSELoss()
        self.loss_ref["DiceCE"] = WeightedDiceCE(  
            self.w1, self.w2, weight=self.weights
        )  # VanillaDiceCE(self.w1, self.w2, weight=self.weights)
        self.loss_ref["HLoss"] = Hierarchical_Loss(w1=1.0, w2=1.0, w3=1.0, weight=self.weights)

    def _build_losses(self):
        for loss_name in self.loss_names:
            try:
                if loss_name in self._criterions.keys():
                    raise KeyError("Cannot add the same loss several times.")
                self._criterions[loss_name] = self.loss_ref[loss_name]
                self._running_values[loss_name] = []
            except KeyError:
                print("Loss {} is not supported yet".format(loss_name))

    def _test_builders(self):
        assert len(self._criterions) == self.n_losses, "Losses have been wrongly built"
    
    def _load_weights(self):
        weights = None
        if os.path.isfile(self.weight_file):
            weights = torch.from_numpy(np.load(self.weight_file))
        return weights

    def compute(self, outputs, labels):
        losses = []
        for loss_name in self.loss_names:

            if "HLoss" in self.loss_names:
                loss, sp_loss, ge_loss, fa_loss, genus_tensor, family_tensor = self.criterions[
                    loss_name
                ](outputs, labels)
                losses.append(loss)
                self._running_values[loss_name].append(loss.item())

                self._running_hloss_values["species_loss"] = sp_loss
                self._running_hloss_values["genus_loss"] = ge_loss
                self._running_hloss_values["family_loss"] = fa_loss
            else:
                #             if self.data_mode == 'SITS':
                #                    loss = self.criterions[loss_name](outputs, labels)
                #                else:
                loss = self.criterions[loss_name](outputs, labels[:, 0, :, :])
                losses.append(loss)
                self._running_values[loss_name].append(loss.item())

        self._values = losses

        if "HLoss" in self.loss_names:
            return genus_tensor, family_tensor
        else:
            return None, None

    def set_values(self, values):
        losses = []
        for i in range(len(self.loss_names)):
            losses.append(values[i])
            self._running_values[self.loss_names[i]].append(values[i].item())
        self._values = losses

    @property
    def backward(self):
        if self._values is None:
            raise Exception("Cannot backprogate the loss before computing it.")
        if self.aggregation == "average":
            self._value = sum(self._values) / self.n_losses
            self._value.backward()
        elif self.aggregation == "sum":
            self._value = sum(self._values)
            self._value.backward()
        else:
            raise Exception("The loss aggregation method {} is not supported yet.")

    def _agg_running_values(self):
        # Default: mean of all the loss of a given number of steps

        # Aggregate hierarchical loss items
        if "HLoss" in self.loss_names:
            for loss_id, loss_value in self._running_hloss_values.items():
                self._running_hloss_values[loss_id] = np.mean(self._running_hloss_values[loss_id])

        for loss_name in self.loss_names:
            self._running_values[loss_name] = np.mean(self._running_values[loss_name])
        if not self._running_values:
            raise Exception("Losses have not been computed yet.")
        if self.aggregation == "average":
            self._running_value = np.mean(list(self._running_values.values()))
        elif self.aggregation == "sum":
            self._running_value = np.sum(list(self._running_values.values()))
        else:
            raise Exception("The loss aggregation method {} is not supported yet.")

    @property
    def reset(self):
        self._running_value = None
        for loss_name in self.loss_names:
            self._running_values[loss_name] = []
        if "HLoss" in self.loss_names:
            for loss_id, loss_value in self._running_hloss_values.items():
                self._running_hloss_values[loss_id] = []

    @property
    def criterions(self):
        return self._criterions

    @property
    def running_values(self):
        if self._running_value is None:
            self._agg_running_values()
        return self._running_values

    @property
    def running_hloss_values(self):
        if self._running_hloss_values is None:
            self._agg_running_values()
        return self._running_hloss_values

    @property
    def running_value(self):
        if self._running_value is None:
            self._agg_running_values()
        return self._running_value  # Case where the value is asked several times

    @property
    def values(self):
        return self._values

    @property
    def value(self):
        return self._value
