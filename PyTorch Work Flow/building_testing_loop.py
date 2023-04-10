import torch
from torch import nn
import matplotlib.pyplot as plt
import build_model
import regression_formula
import building_training_loop

## Testing
build_model.model_0.eval()

for building_training_loop.epoch in range(building_training_loop.epochs):
    with torch.inference_mode():
        test_pred = build_model.model_0(regression_formula.X_test)

        test_loss = build_model.loss_fn(test_pred, regression_formula.y_test)

        building_training_loop.epoch_count.append(building_training_loop.epoch)
        building_training_loop.loss_values.append(building_training_loop.loss)
        building_training_loop.test_loss_values.append(test_loss)

    print(
        f"Epoch: {building_training_loop.epoch} Loss: {building_training_loop.loss} Test Loss: {test_loss}"
    )

print(build_model.model_0.state_dict())
print(building_training_loop.epoch_count)
print(building_training_loop.loss_values)
print(building_training_loop.test_loss_values)
