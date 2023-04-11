from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import first_model
import save_model

## Loading model

# Create a new instance of the model
loaded_model_1 = first_model.LinearRegressionModelV2()

# Load the saved model_1 sate dict
loaded_model_1.load_state_dict(torch.load(save_model.MODEL_SAVE_PATH))

# Put the loaded model to device
loaded_model_1.to(device=first_model.device)

print(loaded_model_1)
print(loaded_model_1.state_dict())

# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_pred = loaded_model_1(first_model.X_train)

print(first_model.y_pred == loaded_model_1_pred)
