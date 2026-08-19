import os
import datetime
import numpy as np
from typing import Any

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import get_device, get_activation_fn
from utils.logging import save_images, logging_progress
from utils.models import VariationalAutoEncoder
from utils.dataset import MnistDatasetLoader, Dataset


class VaeExperiment:
    def __init__(
        self,
        e_kwargs: dict[str, Any],
        d_kwargs: dict[str, Any],
        device: torch.device | str = "auto",
        optimizer_kwargs: dict[str, Any] | None = None,
        batch_size: int = 16,
    ) -> None:
        self.device = get_device(device)
        self.batch_size = batch_size
        self.latent_dim = e_kwargs['latent_dim']

        e_kwargs['activation_fn'] = get_activation_fn(e_kwargs['activation_fn'])
        d_kwargs['activation_fn'] = get_activation_fn(d_kwargs['activation_fn'])

        self.vae = VariationalAutoEncoder(e_kwargs, d_kwargs, self.device).to(self.device)

        self.criterion = self.loss_function

        self.optimizer = optim.Adam(self.vae.parameters(), **optimizer_kwargs)

        dataset_loader = MnistDatasetLoader(images_file_path="assets/dataset/mnist-dataset/train-images.idx3-ubyte",
                                            labels_file_path="assets/dataset/mnist-dataset/train-labels.idx1-ubyte")
        dataset = Dataset(dataset_loader)
        self.data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def loss_function(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = 0.5 * torch.sum(mu.pow(2) + torch.exp(2 * log_std) - 1 - 2 * log_std)
        return reproduction_loss + KLD

    def train(self, epochs: int):
        if not os.path.exists("logs/vae/"):
            os.makedirs("logs/vae/")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"logs/vae/{timestamp}/samples")
        os.makedirs(f"logs/vae/{timestamp}/models")
        os.makedirs(f"logs/vae/{timestamp}/tensorboard")

        writer = SummaryWriter(f"logs/vae/{timestamp}/tensorboard")

        total_batches = len(self.data_loader)

        for epoch in range(epochs):
            steps = 0
            self.vae.train()

            for i, data in enumerate(self.data_loader):
                images = data[0].to(self.device, dtype=torch.float32).view(self.batch_size, -1)

                self.optimizer.zero_grad()
                x_hat, mu, log_std = self.vae(images)
                loss = self.criterion(images, x_hat, mu, log_std)

                loss.backward()
                self.optimizer.step()

                logging_progress(epoch, steps, total_batches=total_batches, print_str=f"Loss: {loss.item():.4f}")
                steps += 1

                writer.add_scalar("Loss/Total", loss.item(), epoch*total_batches + steps)

            self.vae.eval()

            with torch.no_grad():
                test_noise = torch.randn(self.batch_size, self.latent_dim, device=self.device, dtype=torch.float32)
                test_images = self.vae.decoder(test_noise)
                test_images = test_images.view(self.batch_size, 1, 28, 28)
                test_images = test_images.clamp(0, 1).cpu()

                test_img_grids = torchvision.utils.make_grid(test_images, nrow=16)
                writer.add_image("Generated Images", test_img_grids, epoch*total_batches + steps)

                test_images_numpy = (test_images.squeeze(1).mul(255).byte().numpy().reshape(self.batch_size, 28, 28))
                save_images(test_images_numpy, 4, 4, f"logs/vae/{timestamp}/samples/epoch_{epoch+1}.png", epoch+1)

            torch.save(self.vae.state_dict(), f"logs/vae/{timestamp}/models/vae_epoch_{epoch+1}.pth")
            writer.close()

    def test(self, epochs: int, length: int, path: str):
        assert os.path.exists(path)

        if not os.path.exists(f"{path}/test"):
            os.makedirs(f"{path}/test")

        model_path = os.path.join(path, f"models/vae_epoch_{epochs}.pth")
        assert os.path.exists(model_path)
        self.vae.load_state_dict(torch.load(model_path, map_location=self.device))

        u = torch.linspace(0, 1, length+2)
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u[1:-1])
        z2 = normal.icdf(u[1:-1])

        images_array = np.zeros((length, length, 28, 28))
        for i in range(length):
            for j in range(length):
                noise = torch.stack([z1[i], z2[j]], dim=0)
                noise = noise.repeat(self.batch_size, 1).to(self.device, dtype=torch.float32)
                images = self.vae.decoder(noise)
                images = images.view(self.batch_size, 1, 28, 28).clamp(0, 1).cpu()

                images_numpy = (images.squeeze(1).mul(255).byte().numpy().reshape(self.batch_size, 28, 28))
                images_array[i, j] = images_numpy[0]

        save_images(images_array.reshape(length*length, 28, 28), length, length, f"{path}/test/result_{epochs}.png", epochs)