import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_fn: type[nn.Module],
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_layer_dim = latent_dim

        for curr_layer_dim in hidden_dims:
            layers.append(nn.Linear(last_layer_dim, curr_layer_dim))
            layers.append(activation_fn())
            last_layer_dim = curr_layer_dim

        layers.append(nn.Linear(last_layer_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        features_dim: int,
        hidden_dims: list[int],
        activation_fn: type[nn.Module],
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_layer_dim = features_dim

        for curr_layer_dim in hidden_dims:
            layers.append(nn.Linear(last_layer_dim, curr_layer_dim))
            layers.append(activation_fn())
            last_layer_dim = curr_layer_dim

        layers.append(nn.Linear(last_layer_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)