import torch
from torch import nn
import matplotlib.pyplot as plt

## create known parameters
weight = 0.7
bias = 0.3

## create
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print(X[:10], y[:10], len(X), len(y))

## createing a training and test split

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))
