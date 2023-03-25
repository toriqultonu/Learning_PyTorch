import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = torch.arange(1.0, 10.0)
# print(x)
# print(x.shape)

# ## Reshaping

# x_reshaped = x.reshape(3, 3)
# x_reshaped2 = x.reshape(9, 1)

# print(x_reshaped)
# print(x_reshaped2)

## Change view
## changing z, changes x cz view of a tensor share the same memory as the input tensor.

z = x.view(1, 9)
print(z)
print(z.shape)

z[0] = 9    
print(z)
print(x)

## stacking tensors on top each other

x_stacked = torch.stack([x,x,x,x,x])