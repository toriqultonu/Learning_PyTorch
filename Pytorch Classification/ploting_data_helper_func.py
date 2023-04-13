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
from helper_functions import plot_predictions, plot_decision_boundary
import neural_network_classification as nnc

## Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(nnc.model_0, nnc.X_train, nnc.y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(nnc.model_0, nnc.X_test, nnc.y_test)
