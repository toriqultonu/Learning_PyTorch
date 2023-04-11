from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import first_model

## Saving and loading trained model

# Create a model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=first_model.model_1.state_dict(), f=MODEL_SAVE_PATH)
