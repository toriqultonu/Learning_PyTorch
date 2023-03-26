import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Getting a gpu

# print(torch.cuda.is_available())

## setup device agonostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# ## count number of devices
# print(torch.cuda.device_count())

## puting tensor models on gpu

tensor = torch.tensor([1, 2, 3], device="cpu")

print(tensor, tensor.device)

tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu, tensor_on_gpu.device)

## moving tensor back to cpu

tensor_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_on_cpu)
