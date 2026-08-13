import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from utils.dataset import MnistDatasetLoader
from utils.dataset import Dataset

dataset_loader = MnistDatasetLoader(images_file_path="assets/dataset/mnist-dataset/train-images.idx3-ubyte",
                                   labels_file_path="assets/dataset/mnist-dataset/train-labels.idx1-ubyte")

dataset = Dataset(dataset_loader)

data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

image, label = next(iter(data_loader))
print(f"Image shape: ({image.shape}), Label shape: {label.shape}")

image = image[0].squeeze()
label = label[0]

plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.show()