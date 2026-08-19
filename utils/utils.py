import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def get_device(device: torch.device | str = "auto") -> torch.device:
    # Cuda by default
    if device == "auto":
        device = "cuda"
    device = torch.device(device)

    # If cuda in not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_activation_fn(name: str) -> type[nn.Module]:
    activation_functions = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
    }
    return activation_functions[name]