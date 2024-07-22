import os
import pathlib
from pathlib import Path

from treemonitoring.utils.configurable import Configurable

MILA_HOME = Path(__file__).parent.parent.parent
# MILA_HOME = "/home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/"  # TO-DO: Find a better solution to find the config file. Path.resolve is not generalizable


class Paths(Configurable):
    def __init__(self):
        # self.config_path = MILA_HOME / "config.ini"
        self.config_path = os.path.join(MILA_HOME, "config.ini")
        super().__init__(self.config_path)
        print(self.config_path)
        self.paths = dict()
        self.scratch = self.config["scratch"]
        self.storage = self.config["storage"]
        self._build()

    def _build(self):
        # location
        self.paths["application"] = os.path.join(MILA_HOME, Path(self.scratch["application"]))
        self.paths["temp"] = os.path.join(MILA_HOME, Path(self.scratch["temp"]))
        self.paths["logs"] = os.path.join(MILA_HOME, Path(self.scratch["logs"]))

        # storage
        self.paths["datasets"] = os.path.join(MILA_HOME, Path(self.storage["datasets"]))
        self.paths["archive"] = os.path.join(MILA_HOME, Path(self.storage["archive"]))

        # datasets
        self.paths["quebectrees"] = os.path.join(MILA_HOME, Path(self.storage["quebectrees"]))
        self.paths["images_dir"] = os.path.join(MILA_HOME, Path(self.storage["images_dir"]))
        self.paths["class_weights"] = os.path.join(MILA_HOME, Path(self.storage["class_weights"]))

    def get(self):
        return self.paths


if __name__ == "__main__":
    paths = Paths().get()
    print(paths)
