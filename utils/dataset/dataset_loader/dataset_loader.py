from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass