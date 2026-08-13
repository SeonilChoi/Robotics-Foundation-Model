import matplotlib.pyplot as plt

import torch


def get_device(device: torch.device | str = "auto") -> torch.device:
    # Cuda by default
    if device == "auto":
        device = "cuda"
    device = torch.device(device)

    # If cuda in not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device