from typing import Any

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        features_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation_fn: type[nn.Module],
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_layer_dim = features_dim

        for curr_layer_dim in hidden_dims:
            layers.append(nn.Linear(last_layer_dim, curr_layer_dim))
            layers.append(activation_fn())
            last_layer_dim = curr_layer_dim

        self.network = nn.Sequential(*layers)
        self.mean = nn.Linear(last_layer_dim, latent_dim)
        self.log_std = nn.Linear(last_layer_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.network(x)
        mu = self.mean(x)
        log_std = self.log_std(x)

        return mu, log_std


class Decoder(nn.Module):
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


class VariationalAutoEncoder(nn.Module):
    def __init__(self, e_kwargs: dict[str, Any], d_kwargs: dict[str, Any], device: torch.device):
        super().__init__()
        self.encoder = Encoder(**e_kwargs)
        self.decoder = Decoder(**d_kwargs)

        self.device = device

    def reparameterize_latent(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        e = torch.randn_like(mu).to(self.device)
        return mu + e * torch.exp(log_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_std = self.encoder(x)
        z = self.reparameterize_latent(mu, log_std)
        x_hat = self.decoder(z)

        return x_hat, mu, log_std