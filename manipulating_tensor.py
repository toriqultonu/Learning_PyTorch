import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tensor = torch.tensor([1, 2, 3])

print(tensor + 10)

print(tensor * 10)

# pytorch in-build functions:
tensor = torch.add(tensor, 10)
tensor = torch.mul(tensor, 30)
print(tensor)
