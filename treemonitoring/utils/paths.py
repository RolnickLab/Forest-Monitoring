import os
import pathlib
from pathlib import Path
from treemonitoring.utils.configurable import Configurable

MILA_HOME = Path(__file__).parent.parent.parent

class Paths(Configurable):
    def __init__(self):
        self.config_path = MILA_HOME / "config.ini"
        super().__init__(self.config_path)
        print(self.config_path)
        self.paths = dict()
        self.scratch = self.config["scratch"]
        self.storage = self.config["storage"]
        self._build()

    def _build(self):
        # location
        self.paths["application"] = MILA_HOME / Path(self.scratch["application"])
        self.paths["temp"] = MILA_HOME / Path(self.scratch["temp"])
        self.paths["logs"] = MILA_HOME / Path(self.scratch["logs"])
        # storage
        self.paths["datasets"] = MILA_HOME / Path(self.storage["datasets"])
        self.paths["archive"] = MILA_HOME / Path(self.storage["archive"])
        # datasets
        self.paths["quebectrees"] = MILA_HOME / Path(self.storage["quebectrees"])
        self.paths["images_dir"] = MILA_HOME / Path(self.storage["images_dir"])
        self.paths["class_weights"] = MILA_HOME / Path(self.storage["class_weights"])

        # Convert all paths to Path objects
        self.paths = {key: Path(value) for key, value in self.paths.items()}

    def get(self):
        return self.paths

if __name__ == "__main__":
    paths = Paths().get()
    print(paths)
