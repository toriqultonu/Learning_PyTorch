import torch
from torch import nn
import matplotlib.pyplot as plt
import build_model
import regression_formula

## The steps:
# Loop through the data
# Forward pass
# Calculate the loss
# Optimize zero grad
# Loss backward
# Optimizer step


## an epoch is one loop through the data (it's a hyper parameter)
epochs = 100

## Training
# Loop through the data
for epoch in range(epochs):
    ## set the model to training mode
    build_model.model_0.train()

    # Forward pass
    y_pred = build_model.model_0(regression_formula.X_train)

    # Calculate the loss
    loss = build_model.loss_fn(y_pred, regression_formula.y_train)

    # Optimized zero grad
    build_model.optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Step the optimizer
    build_model.optimizer.step()

print(f"Loss: {loss}")
print(build_model.model_0.state_dict())
