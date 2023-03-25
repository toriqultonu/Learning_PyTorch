import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# tensor1 = torch.tensor([1, 2, 3])

# tensor2 = torch.tensor([4, 5, 6])

# ###  element wise multiplication
# print(tensor1 * tensor2)

# ## matrix multiplication
# ## @ stands for matrix multiplications
# print(tensor2 @ tensor1)
# print(torch.matmul(tensor1, tensor2))

## shapes for matrix multiplication

tensorA = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensorB = torch.tensor([[6, 7], [4, 8], [8, 9]])
## torch.mm == torch.matmul

print(tensorA.T)
print(tensorA.shape)
print(tensorA.T.shape)
print(torch.matmul(tensorA.T, tensorB)) 
