import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path

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

# Impoved version of the model
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))

# Instantiating and device agnostic
# model_0 = CircleModelV0().to(device=device)
model_0 = CircleModelV1().to(device=device)

# print(model_0)
# print(next(model_0.parameters()).device)

# replicating model
model_1 = nn.Sequential(
    nn.Linear(in_features=2, out_features=4), nn.Linear(in_features=5, out_features=1)
).to(device=device)

# Make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
# print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
# print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
# print(f"\nFirst 10 predictions: {untrained_preds[:10]}")
# print(f"\nFirst 10 labels: {y_test[:10]}")

# setup loss function and optimizer

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


## Train
# view the first 5 outputs of the forward pass on the test data
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)  # the output of y_logits are the called the logits..

# Target : going from raw logits -> prediction probabilities -> prediction labels

# using the sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
y_preds = torch.round(y_pred_probs)
y_preds.squeeze()
print(y_preds)

# Training Loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set number of epoch
epochs = 1000

# put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# buiding training and evaluation loop
for epoch in range(epochs):
    ## Training
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_preds = torch.round(
        torch.sigmoid(y_logits)
    )  # turns logits -> pred probs -> pred labels

    # Calculate loss/ accuracy
    loss = loss_fn(
        y_logits, y_train
    )  # here y_logits is used rather than y_pred because BCEwithLogitLoss is use as a loss function
    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss Backward / Backpropagation
    loss.backward()

    # Optimizer step
    optimizer.step()

    ## Testing
    model_0.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model_0(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        # Calculate Test Loss/Accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

    # printing result
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}"
        )

## Download helper function from Learn Pytorch repo if it's not already downloaded...
# if Path("helper_functions.py").is_file():
#     print("helper_functions.py already exists, skipping download")
# else:
#     print("Download helper functions.py")
#     request = requests.get(
#         "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
#     )

#     with open("helper_functions.py", "wb") as f:
#         f.write(request.content)
