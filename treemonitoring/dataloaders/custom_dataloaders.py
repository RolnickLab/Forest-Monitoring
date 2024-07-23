from torch.utils.data import DataLoader

from treemonitoring.dataloaders.its_dataset import ImageTimeSeriesDataset

# from treemonitoring.dataloaders.semseg_dataset import SemsegDataset
from treemonitoring.utils.utils import calculate_label_weights


class CustomDataloaders:
    """Utility function to create custom dataloaders for multiple datasets."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg["dataset"]["name"]
        self.shuffle = self.cfg["dataset"]["shuffle"]
        self.batch_size = self.cfg["experiment"]["batch_size"]
        self.n_workers = self.cfg["experiment"]["n_workers"]
        self.final_size = self.cfg["dataset"]["final_size"]
        self.cv = self.cfg["dataset"]["cv"]
        self.num_classes = self.cfg["dataset"]["n_classes"]

        self.dataset_folder = self.cfg["dataset"]["dataset_path"]

        # self.mono_date = self.cfg["dataset"]["mono_date"]
        self.pad_value = 0

        # Build loaders and sets
        self.sets = self._build_sets(self.dataset_name)
        self.loaders = self._build_loaders(self.dataset_name)
        self.loss_weights = self._loss_weights()

    def _build_sets(self, dataset_name):
        if dataset_name == "quebectrees":
            sets = {
                "train": ImageTimeSeriesDataset("pretiled", "train", self.final_size, self.cv),
                "val": ImageTimeSeriesDataset("pretiled", "val", self.final_size, self.cv),
                "test": ImageTimeSeriesDataset("pretiled", "test", self.final_size, self.cv),
            }
        else:
            raise ValueError("Dataset not supported")

        return sets

    def _build_loaders(self, dataset_name):
        splits = list(self.sets.keys())
        loaders = {}

        if dataset_name == "quebectrees":
            loaders["train"] = DataLoader(
                self.sets["train"],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.n_workers,
            )
            if "val" in splits:
                loaders["val"] = DataLoader(
                    self.sets["val"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.n_workers,
                )
            else:
                loaders["val"] = None

            if "test" in splits:
                loaders["test"] = DataLoader(
                    self.sets["test"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.n_workers,
                )
            else:
                loaders["test"] = None
        else:
            raise ValueError("Dataset not supported")

        return loaders

    def _loss_weights(self):
        calculate_label_weights(self.loaders["train"], self.num_classes)

    def get_sets(self):
        return self.sets

    def get_loaders(self):
        return self.loaders

    def get_class_names(self):
        class_names = self.sets["train"].get_names()
        return class_names
