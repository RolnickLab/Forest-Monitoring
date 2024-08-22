import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from treemonitoring.dataloaders.custom_dataloaders import CustomDataloaders
from treemonitoring.utils.paths import Paths


class BaseExperiment:
    def __init__(self, pretrained=False):
        self.paths = Paths().get()
        self.args = self._parse_args()
        self.cfg = self._load_cfg(self.args.cfg)
        self.ckp = None
        # if not pretrained:
        self._build_expe_dir()
        self._step = None
        self._epoch = None
        self._set_seeds
        self.device = self.cfg["experiment"]["device"]
        self.n_epochs = self.cfg["experiment"]["n_epochs"]
        self.selection_metric = self.cfg["experiment"]["selection_metric"]
        self.loss_step = self.cfg["log_steps"]["loss"]
        self.val_step = self.cfg["log_steps"]["val"]
        self.vis_step = self.cfg["log_steps"]["visualization"]
        self.ckp_step = self.cfg["log_steps"]["checkpoint"]
        self.n_classes = self.cfg["dataset"]["n_classes"]
        self.n_classes_genus = self.cfg["dataset"]["n_classes_genus"]
        self.n_classes_family = self.cfg["dataset"]["n_classes_family"]
        self.task = self.cfg["dataset"]["task"]
        self.data_mode = self.cfg["dataset"]["data_mode"]
        self.data = CustomDataloaders(self.cfg)
        self.loaders = self.data.get_loaders()
        self.batch_size = self.cfg["experiment"]["batch_size"]
        self.class_names = self._get_class_names()
        self._best_eval_metric = self._build_best_eval_metric()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg", help="Path to config file.", default="tester.yaml")
        parser.add_argument("--ckp", help="Path to checkpoint to load.", default=None)
        parser.add_argument(
            "--debug", help="Debug mode to switch wand mode offline", action="store_true"
        )
        parser.add_argument("--savedir", help="Path to save output images.", default=None
        )
        args = parser.parse_args()
        return args

    def _load_cfg(self, cfg_path):
        with open(cfg_path, "r") as stream:
            cfg = yaml.safe_load(stream)
        return cfg

    def _load_ckp(self, ckp_path):
        if ckp_path is None:
            return ckp_path
        return torch.load(ckp_path)

    def _build_expe_dir(self):
        self.cfg["paths"] = {}
        logs_path = self.paths["logs"]
        folder_name = self.cfg["dataset"]["name"] + "_" + self.cfg["model"]["name"]
        exp_name = (
            self.cfg["experiment"]["name"] + "_" + str(self.cfg["experiment"]["version"]).zfill(3)
        )
        dir_path = logs_path / folder_name / exp_name
        # Update the version to avoid raise existing expe
        while dir_path.exists():
            self.cfg["experiment"]["version"] += 1
            exp_name = (
                self.cfg["experiment"]["name"]
                + "_"
                + str(self.cfg["experiment"]["version"]).zfill(3)
            )
            dir_path = logs_path / folder_name / exp_name
        dir_path.mkdir(parents=True, exist_ok=True)
        self.cfg["paths"]["dir"] = str(dir_path)
        results_dir_path = dir_path / "results"
        results_dir_path.mkdir(parents=True, exist_ok=True)
        self.cfg["paths"]["results_dir"] = str(results_dir_path)
        logs_dir_path = dir_path / "logs"
        logs_dir_path.mkdir(parents=True, exist_ok=True)
        self.cfg["paths"]["logs_dir"] = str(logs_dir_path)
        checkpoints_dir_path = dir_path / "checkpoints"
        checkpoints_dir_path.mkdir(parents=True, exist_ok=True)
        self.cfg["paths"]["checkpoint_dir"] = str(checkpoints_dir_path)
        # Save the config file with the paths
        with open(dir_path / "config.yaml", "w") as stream:
            yaml.dump(self.cfg, stream)

    @property
    def step(self):
        """The current (global) step."""
        if self._step is None:
            if self.ckp is None:
                self._step = 0
            else:
                self._step = self.ckp["step"]
        return self._step

    def next_step(self):
        """Increments the global step counter."""
        if self._step is None:
            if self.ckp is None:
                self._step = 0
            else:
                self._step = self.ckp["step"]
        else:
            self._step
        self._step += 1
        return self

    @property
    def epoch(self):
        """The current epoch."""
        if self._epoch is None:
            if self.ckp is None:
                self._epoch = 0
            else:
                self._epoch = self.ckp["epoch"]
        return self._epoch

    def next_epoch(self):
        """Increments the epoch counter."""
        if self._epoch is None:
            if self.ckp is None:
                self._epoch = 0
            else:
                self._epoch = self.ckp["epoch"]
        else:
            self._epoch
        self._epoch += 1
        return self

    def get_cfg(self):
        """Get the config file."""
        return self.cfg

    @property
    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.cfg["experiment"]["torch_seed"])
        torch.manual_seed(self.cfg["experiment"]["torch_seed"])
        np.random.seed(self.cfg["experiment"]["numpy_seed"])

    def checkpoint(self, global_state, force=True):
        ckp_step = self.cfg["log_steps"]["checkpoint"]
        if force:
            do_checkpoint = True
        elif isinstance(ckp_step, int):
            do_checkpoint = (self.step % ckp_step) == 0
        else:
            do_checkpoint = False
        if do_checkpoint:
            torch.save(
                global_state,
                Path(self.cfg["paths"]["checkpoint_dir"])
                / "checkpoint_iter_{}.pt".format(str(self.step).zfill(8)),
            )

    def _build_best_eval_metric(self):
        if self.task == "Segmentation" or self.task == "Segmentation":
            return 0.0
        elif self.task in ("Regression", "MultiRegression"):
            return np.inf
        else:
            raise Exception("Task {} is not supported.".format(self.task))

    def _get_class_names(self):
        sets = self.data.get_sets()
        try:
            class_names = sets["train"].get_names()
        except:
            print("Method get_names() is not implemented in dataloader")
            class_names = None
        return class_names


def test_yaml_cfg():
    # paths = Paths().get()
    # cfg_path = paths['logs'] / 'tests' / 'config_files' / 'tester.yaml'
    exp = BaseExperiment()
    cfg = exp.get_cfg()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    test_yaml_cfg()
