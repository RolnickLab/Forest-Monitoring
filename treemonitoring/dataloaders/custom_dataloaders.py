from torch.utils.data import DataLoader

from treemonitoring.dataloaders.its_dataset import ImageTimeSeriesDataset
from treemonitoring.dataloaders.pastis_dataset import PASTIS_Dataset

# from treemonitoring.dataloaders.semseg_dataset import SemsegDataset
from treemonitoring.utils.pastis_dataset_utils import pad_collate
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

        # Pastis dataset configs (Can be added to model config file)
        self.dataset_folder = self.cfg["dataset"]["dataset_path"]

        if self.dataset_name == "pastis":
            self.ref_date = self.cfg["dataset"]["ref_date"]
        # self.mono_date = self.cfg["dataset"]["mono_date"]
        self.fold_sequence = fold_sequence = [[1, 2, 3], [4], [5]]
        self.pad_value = 0

        # Build loaders and sets
        self.sets = self._build_sets(self.dataset_name)
        self.loaders = self._build_loaders(self.dataset_name)
        self.loss_weights = self._loss_weights()

    def _build_sets(self, dataset_name):
        if dataset_name == "pastis":

            dt_args = dict(
                folder=self.dataset_folder,
                norm=True,
                reference_date=self.ref_date,
                # mono_date=self.mono_date,
                target="semantic",
                sats=["S2"],
            )

            sets = {
                "train": PASTIS_Dataset(**dt_args, folds=self.fold_sequence[0]),
                "val": PASTIS_Dataset(**dt_args, folds=self.fold_sequence[1]),
                "test": PASTIS_Dataset(**dt_args, folds=self.fold_sequence[2]),
            }

        elif dataset_name == "quebectrees":
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

        elif dataset_name == "pastis":

            def collate(x):
                return pad_collate(x, pad_value=self.pad_value)

            loaders["train"] = DataLoader(
                self.sets["train"],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.n_workers,
                drop_last=True,
                collate_fn=collate,
            )
            if "val" in splits:
                loaders["val"] = DataLoader(
                    self.sets["val"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.n_workers,
                    drop_last=True,
                    collate_fn=collate,
                )
            else:
                loaders["val"] = None

            if "test" in splits:
                loaders["test"] = DataLoader(
                    self.sets["test"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.n_workers,
                    drop_last=True,
                    collate_fn=collate,
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
