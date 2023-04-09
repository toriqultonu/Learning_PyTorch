import torch
from torch import nn
import matplotlib.pyplot as plt
import build_model
import regression_formula
import building_training_loop

## Testing
build_model.model_0.eval()

with torch.inference_mode():
    test_pred = build_model.model_0(regression_formula.X_test)

    test_loss = build_model.loss_fn(test_pred, regression_formula.y_test)

print(
    f"Epoch: {building_training_loop.epoch} Loss: {building_training_loop.loss} Test Loss: {test_loss}"
)

print(build_model.model_0.state_dict())
