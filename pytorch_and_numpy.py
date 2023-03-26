import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## numpy array to tensor

# array = np.arange(1.0, 8.0)
# # converting from numpy to torch the datatype will be default which is float64 so we need to convert it ..
# tensor = torch.from_numpy(array).type(torch.float32)

# print(array)
# print(tensor)
# print(tensor.dtype)


## tensor to numpy array

tensor1 = torch.ones(7)
numpy_tensor = tensor1.numpy()

print(tensor1)
print(numpy_tensor)
print(numpy_tensor.dtype)
