import os
from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass


class ImageLabelDatasetLoader(DatasetLoader):
    def __init__(self, **kwargs):
        self.images_file_path = kwargs.get("images_file_path", None)
        self.labels_file_path = kwargs.get("labels_file_path", None)

    def load(self):
        if not self.images_file_path or not os.path.exists(self.images_file_path):
            raise FileNotFoundError(f"Images file path '{self.images_file_path}' does not exist.")
        if not self.labels_file_path or not os.path.exists(self.labels_file_path):
            raise FileNotFoundError(f"Labels file path '{self.labels_file_path}' does not exist.")