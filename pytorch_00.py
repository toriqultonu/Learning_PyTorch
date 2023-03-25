import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(torch.__version__)

# creating random tensor
# random_tensor = torch.rand(3, 4)

# print(random_tensor)

float_32_tensor = torch.tensor(
    [3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False
)

int_32_tensor = torch.tensor([3, 5, 7], dtype=torch.int32)

print(int_32_tensor * float_32_tensor)
