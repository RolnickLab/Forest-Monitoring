import os
import pathlib
from pathlib import Path

from treemonitoring.utils.configurable import Configurable

# MILA_HOME = pathlib.Path(__name__).resolve().parents[0]
MILA_HOME = "/home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/"  # TO-DO: Find a better solution to find the config file. Path.resolve is not generalizable


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
        self.paths["application"] = Path(self.scratch["application"])
        self.paths["temp"] = Path(self.scratch["temp"])
        self.paths["logs"] = Path(self.scratch["logs"])

        # storage
        self.paths["datasets"] = Path(self.storage["datasets"])
        self.paths["archive"] = Path(self.storage["archive"])

        # datasets
        self.paths["quebectrees"] = Path(self.storage["quebectrees"])
        self.paths["images_dir"] = Path(self.storage["images_dir"])
        self.paths["class_weights"] = Path(self.storage["class_weights"])

    def get(self):
        return self.paths


if __name__ == "__main__":
    paths = Paths().get()
    import ipdb

    ipdb.set_trace()
