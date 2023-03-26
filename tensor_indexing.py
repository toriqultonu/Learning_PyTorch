import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = torch.arange(1, 10).reshape(1, 3, 3)

# print(x)
# print(x.shape)

## indexing our tensor

print(x[0])

## indexing to middle bracket or dim = 1
print(x[0][1])

## indexing to last bracket or dim = 2
print(x[0][1][1])
print(x[0][2][2])
print(x[:, :, 1])
