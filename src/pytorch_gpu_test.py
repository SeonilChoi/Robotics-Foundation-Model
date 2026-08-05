import torch

print(torch.cuda.get_device_name(device=0) + " is " + "available." if torch.cuda.is_available() else "not available.")