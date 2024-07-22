"""Class to computes metrics for TreeMonitoring."""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix
from torchmetrics import R2Score

from treemonitoring.utils.utils import _make_confusion_matrix


class Evaluator:
    """Class to evaluate a model with quantitative metrics using a ground truth mask and a predicted
    mask.

    PARAMETERS
    ----------
    num_class: int
    """

    def __init__(self, metric_names, task, num_class, device="cuda"):
        self.metric_names = metric_names.split("/")
        self.task = task
        self.device = device
        if self.task == "Segmentation":
            self.num_class = num_class
            self.confusion_matrix = np.zeros((self.num_class,) * 2)
        elif self.task in ("Regression", "MultiRegression"):
            self.buff = [[], []]
        else:
            raise Exception("Task {} is not supported.".format(self.task))

    def get_pixel_prec_class(self, harmonic_mean=False):
        """Pixel Precision."""
        prec_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=0)
        prec_by_class = np.nan_to_num(prec_by_class)
        if harmonic_mean:
            prec = hmean(prec_by_class)
        else:
            prec = np.mean(prec_by_class)
        return prec, prec_by_class

    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall."""
        recall_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=1)
        recall_by_class = np.nan_to_num(recall_by_class)
        if harmonic_mean:
            recall = hmean(recall_by_class)
        else:
            recall = np.mean(recall_by_class)
        return recall, recall_by_class

    def get_pixel_acc_class(self, harmonic_mean=False):
        """Pixel Accuracy."""
        acc_by_class = np.diag(self.confusion_matrix).sum() / (
            np.nansum(self.confusion_matrix, axis=1)
            + np.nansum(self.confusion_matrix, axis=0)
            + np.diag(self.confusion_matrix).sum()
            - 2 * np.diag(self.confusion_matrix)
        )
        acc_by_class = np.nan_to_num(acc_by_class)
        if harmonic_mean:
            acc = hmean(acc_by_class)
        else:
            acc = np.mean(acc_by_class)
        return acc, acc_by_class

    def get_miou_class(self, harmonic_mean=False):
        """Mean Intersection over Union."""
        miou_by_class = np.diag(self.confusion_matrix) / (
            np.nansum(self.confusion_matrix, axis=1)
            + np.nansum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        miou_by_class = np.nan_to_num(miou_by_class)
        if harmonic_mean:
            miou = hmean(miou_by_class)
        else:
            miou = np.mean(miou_by_class)
        return miou, miou_by_class

    def get_dice_class(self, harmonic_mean=False):
        """Dice."""
        _, prec_by_class = self.get_pixel_prec_class()
        _, recall_by_class = self.get_pixel_recall_class()
        # Add epsilon term to avoid /0
        dice_by_class = (
            2 * prec_by_class * recall_by_class / (prec_by_class + recall_by_class + 1e-8)
        )
        if harmonic_mean:
            dice = hmean(dice_by_class)
        else:
            dice = np.mean(dice_by_class)
        return dice, dice_by_class

    def get_mse(self):
        """MSE."""
        labels = torch.stack(self.buff[0], dim=1)
        predictions = torch.stack(self.buff[1], dim=1)
        mse_loss = nn.MSELoss(reduction="none")(predictions, labels)
        mse_by_cat = torch.mean(mse_loss, axis=1)
        mse_mean = torch.mean(mse_by_cat)
        return mse_mean.item(), mse_by_cat.tolist()

    def get_rmse(self):
        """RMSE."""
        labels = torch.stack(self.buff[0], dim=1)
        predictions = torch.stack(self.buff[1], dim=1)
        mse_loss = nn.MSELoss(reduction="none")(predictions, labels)
        rmse_by_cat = torch.sqrt(torch.mean(mse_loss, axis=1))
        rmse_mean = torch.sqrt(nn.MSELoss()(predictions, labels))
        return rmse_mean.item(), rmse_by_cat.tolist()

    def get_r2(self):
        """R2"""
        labels = torch.stack(self.buff[0], dim=0)
        predictions = torch.stack(self.buff[1], dim=0)
        n_outputs = labels.shape[1]
        r2_score = R2Score(num_outputs=n_outputs, multioutput="raw_values").to(self.device)
        r2_by_cat = r2_score(predictions, labels)
        r2_mean = torch.mean(r2_by_cat)
        return r2_mean.item(), r2_by_cat.tolist()

    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(
            labels.flatten(), predictions.flatten(), labels=list(range(self.num_class))
        )
        return matrix

    def plot_confusion_matrix(self):
        cm = _make_confusion_matrix(self.confusion_matrix)
        return cm

    def _update_buff(self, labels, predictions):
        self.buff[0] += labels
        self.buff[1] += predictions

    def add_batch(self, labels, predictions):
        """Method to add ground truth and predicted masks by batch and update the global confusion
        matrix (entire dataset)

        PARAMETERS
        ----------
        labels: torch tensor or numpy array
            Ground truth masks
        predictions: torch tensor or numpy array
            Predicted masks
        """
        assert labels.shape == predictions.shape
        if self.task == "Segmentation":
            self.confusion_matrix += self._generate_matrix(labels, predictions)
        elif self.task in ("Regression", "MultiRegression"):
            self._update_buff(labels, predictions)
        else:
            raise Exception("Task {} is not supported.".format(self.task))

    @property
    def reset(self):
        """Method to reset the confusion matrix."""
        if self.task == "Segmentation":
            self.confusion_matrix = np.zeros((self.num_class,) * 2)
        elif self.task in ("Regression", "MultiRegression"):
            self.buff = [[], []]
        else:
            raise Exception("Task {} is not supported.".format(self.task))

    def get_metrics(self, harmonic_mean=False):
        metrics = {}
        for metric_name in self.metric_names:
            if metric_name == "Accuracy":
                metrics[metric_name] = self.get_pixel_acc_class(harmonic_mean)
            elif metric_name == "Recall":
                metrics[metric_name] = self.get_pixel_recall_class(harmonic_mean)
            elif metric_name == "Precision":
                metrics[metric_name] = self.get_pixel_prec_class(harmonic_mean)
            elif metric_name == "mIoU":
                metrics[metric_name] = self.get_miou_class(harmonic_mean)
            elif metric_name == "Dice":
                metrics[metric_name] = self.get_dice_class(harmonic_mean)
            elif metric_name == "MSE":
                metrics[metric_name] = self.get_mse()
            elif metric_name == "RMSE":
                metrics[metric_name] = self.get_rmse()
            elif metric_name == "R2":
                metrics[metric_name] = self.get_r2()
            else:
                raise Exception("Metric {} is not supported yet.".format(metric_name))
        return metrics

    def get_metric_names(self):
        return self.metric_names
