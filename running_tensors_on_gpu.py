import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Getting a gpu

print(torch.cuda.is_available())

## setup device agonostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

## count number of devices
print(torch.cuda.device_count())
