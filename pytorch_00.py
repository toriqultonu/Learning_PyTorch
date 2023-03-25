import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(torch.__version__)

# creating random tensor
# random_tensor = torch.rand(3, 4)

# print(random_tensor)

# float_32_tensor = torch.tensor(
#     [3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False
# )

# int_32_tensor = torch.tensor([3, 5, 7], dtype=torch.long)

# print(int_32_tensor * float_32_tensor)

## Getting info from tensors..

some_tensor = torch.rand(3, 4)

print(f"Datatype of the tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Size of tensor: {some_tensor.size()}")
print(f"Device of tensor: {some_tensor.device}")
