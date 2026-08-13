import struct
import numpy as np
from array import array

from .base import ImageLabelDatasetLoader


class MnistDatasetLoader(ImageLabelDatasetLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "MNIST"

    def load(self):
        super().load()

        with open(self.labels_file_path, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

        with open(self.images_file_path, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            images = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(size, rows, cols) / 255.

        return images, labels