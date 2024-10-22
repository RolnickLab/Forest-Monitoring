import configparser


class Configurable:
    def __init__(self, path_to_config):
        self.path_to_config = path_to_config
        self.config = self._get()

    def _get(self):
        config = configparser.ConfigParser()
        config.read(self.path_to_config)
        return config
