import torch
import torch.utils.data as data

from utils.dataset.dataset_loader import DatasetLoader

class Dataset(data.Dataset):
    def __init__(self, dataset_loader: DatasetLoader):
        self.dataset_loader = dataset_loader
        
        self.x, self.y = self.dataset_loader.load()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]