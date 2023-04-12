import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

## Make classification data and get it ready

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

# print(len(X), len(y))
# print(f"First 5 sample of X: {X[:5]}")
# print(f"First 5 sample of y: {y[:5]}")

# Make dataframe of circle data using pandas
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

# print(circles.head(100))

# Visualizing data
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

## Turn data into tensor and create train and test splits

# Turn data in tensor
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# print(X[:5], y[:5])

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(len(X_train), len(X_test), len(y_train), len(y_test))

## Building model
# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Constructing  a model that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


# Instantiating and device agnostic
model_0 = CircleModelV0().to(device=device)

# print(model_0)
# print(next(model_0.parameters()).device)

# replicating model
model_1 = nn.Sequential(
    nn.Linear(in_features=2, out_features=4), nn.Linear(in_features=5, out_features=1)
).to(device=device)

# Make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions: {untrained_preds[:10]}")
print(f"\nFirst 10 labels: {y_test[:10]}")

# setup loss function and optimizer

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params= model_0.parameters(), lr=0.1)

