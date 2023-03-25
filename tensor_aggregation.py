import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create a tensor

x = torch.arange(0, 100, 10)

# print(x)
# print(x.min())
# print(x.dtype)

# ## torch.mean() functions requires a tensor of dtype float32 to work properly
# print(torch.mean(x.type(torch.float32)))
# print(x.type(torch.float32).mean())

# print(x.sum())

## Finding positional aggregation
print(x.argmax())
print(x[x.argmax()])
