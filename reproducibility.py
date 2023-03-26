import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## reproducibility -> trying to take random out of random

# create two random tensor
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A == random_tensor_B)

# creating random but reproducible tensors

Random_Seed = 42
torch.manual_seed(Random_Seed)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(Random_Seed)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C == random_tensor_D)
