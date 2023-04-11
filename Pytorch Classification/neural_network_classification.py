import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_circles

## Make classification data and get it ready

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

print(len(X), len(y))
print(f"First 5 sample of X: {X[:5]}")
print(f"First 5 sample of y: {y[:5]}")

# Make dataframe of circle data using pandas
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

print(circles.head(100))


