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
from utils import logging_progress, save_progress_image, save_gan_test_image
from utils.models import Generator, Discriminator
from utils.dataset import MnistDatasetLoader, Dataset


class GanExperiment:
    def __init__(
        self,
        g_kwargs: dict[str, Any],
        d_kwargs: dict[str, Any],
        device: torch.device | str = "auto",
        optimizer_kwargs: dict[str, Any] | None = None,
        batch_size: int = 16,
    ) -> None:
        self.device = get_device(device)
        self.batch_size = batch_size
        self.latent_dim = g_kwargs['latent_dim']

        g_kwargs['activation_fn'] = get_activation_fn(g_kwargs['activation_fn'])
        d_kwargs['activation_fn'] = get_activation_fn(d_kwargs['activation_fn'])

        self.generator = Generator(**g_kwargs).to(self.device)
        self.discriminator = Discriminator(**d_kwargs).to(self.device)

        self.criterion = nn.BCELoss()

        self.g_optimizer = optim.Adam(self.generator.parameters(), **optimizer_kwargs)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), **optimizer_kwargs)

        dataset_loader = MnistDatasetLoader(images_file_path="assets/dataset/mnist-dataset/train-images.idx3-ubyte",
                                            labels_file_path="assets/dataset/mnist-dataset/train-labels.idx1-ubyte")
        dataset = Dataset(dataset_loader)
        self.data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, epochs: int, k: int):
        if not os.path.exists("logs/gan/"):
            os.makedirs("logs/gan/")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"logs/gan/{timestamp}/samples")
        os.makedirs(f"logs/gan/{timestamp}/models")
        os.makedirs(f"logs/gan/{timestamp}/tensorboard")

        writer = SummaryWriter(f"logs/gan/{timestamp}/tensorboard")

        real_labels = torch.ones(self.batch_size, device=self.device, dtype=torch.float32)
        fake_labels = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)

        total_batches = len(self.data_loader)
        generator_update_steps = k

        for epoch in range(epochs):
            steps = 0
            self.generator.train()
            self.discriminator.train()

            for i, data in enumerate(self.data_loader):
                real_images = data[0].to(self.device, dtype=torch.float32).view(self.batch_size, -1)

                noise = torch.randn(self.batch_size, self.latent_dim, device=self.device, dtype=torch.float32)
                fake_images = self.generator(noise)

                self.discriminator.zero_grad()
                real_output = self.discriminator(real_images).view(-1)
                fake_output = self.discriminator(fake_images.detach()).view(-1)

                d_loss = (
                    self.criterion(real_output, real_labels) +
                    self.criterion(fake_output, fake_labels)
                )
                d_loss.backward()
                self.d_optimizer.step()

                if generator_update_steps == k:
                    self.generator.zero_grad()
                    output = self.discriminator(fake_images).view(-1)
                    g_loss = self.criterion(output, real_labels)
                    g_loss.backward()
                    self.g_optimizer.step()
                    generator_update_steps = 0
                
                logging_progress(epoch, steps, total_batches=total_batches, print_str=f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
                generator_update_steps += 1
                steps += 1

                writer.add_scalar("Loss/D", d_loss.item(), epoch*total_batches + steps)
                writer.add_scalar("Loss/G", g_loss.item(), epoch*total_batches + steps)

            self.generator.eval()

            with torch.no_grad():
                test_noise = torch.randn(self.batch_size, self.latent_dim, device=self.device, dtype=torch.float32)
                test_images = self.generator(test_noise)
                test_images = test_images.view(self.batch_size, 1, 28, 28)
                test_images = test_images.clamp(0, 1).cpu()

                test_img_grids = torchvision.utils.make_grid(test_images, nrow=16)
                writer.add_image("Generated Images", test_img_grids, epoch*total_batches + steps)

                test_images_numpy = (test_images.squeeze(1).mul(255).byte().numpy().reshape(self.batch_size, 28, 28))
                save_progress_image(test_images_numpy, 4, 4, f"logs/gan/{timestamp}/samples/epoch_{epoch+1}.png", epoch+1)

            torch.save(self.generator.state_dict(), f"logs/gan/{timestamp}/models/generator_epoch_{epoch+1}.pth")
            torch.save(self.discriminator.state_dict(), f"logs/gan/{timestamp}/models/discriminator_epoch_{epoch+1}.pth")

    def test(self, epochs: int, stride: int, path: str):
        assert os.path.exists(path)

        if not os.path.exists(f"{path}/test"):
            os.makedirs(f"{path}/test")

        n_col = int(epochs / stride)

        images_array = np.zeros((4, n_col, 28, 28))
        for i in range(n_col):
            epoch = (i + 1) * stride
            model_path = os.path.join(path, f"models/generator_epoch_{epoch}.pth")
            assert os.path.exists(model_path)
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device))

            noise = torch.randn(4, self.latent_dim, device=self.device, dtype=torch.float32)
            images = self.generator(noise)
            images = images.view(4, 1, 28, 28).clamp(0, 1).cpu()

            images_numpy = (images.squeeze(1).mul(255).byte().numpy().reshape(4, 28, 28))
            images_array[:, i] = images_numpy

        save_gan_test_image(images_array.reshape(4 * n_col, 28, 28), 4, n_col, f"{path}/test/result.png", epochs)

        