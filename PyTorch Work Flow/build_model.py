import torch
from torch import nn
import matplotlib.pyplot as plt
import regression_formula

## creating a linear regression model class


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # forward method to define the computation in the model
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias


model_0 = LinearRegressionModel()

print(model_0.state_dict())

print(regression_formula.X_test)
