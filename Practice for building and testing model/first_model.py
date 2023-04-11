import torch
from torch import nn
import matplotlib.pyplot as plt


## check pytorch version
# print(torch.__version__)

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

## Create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and label)
X = torch.arange(start, end, step).unsqueeze(
    dim=1
)  # without unsqueeze error will pop up
y = weight * X + bias

# print(X[:10], y[:10])

# Split the data
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))


# Create plot prediction function
def plot_predictions(
    train_data,
    train_label,
    test_data,
    test_label,
    predictions=None,
):
    plt.figure(figsize=(10, 7))

    # plot training data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")

    # plot testing data in green
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # showing the plt
    plt.legend(prop={"size": 14})
    plt.show()


## Building a pytorch linear model


# Create a linear model by subclassing nn.Module
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for createing the model parameters.
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
# print(model_1, model_1.state_dict())

# Using plot prediction
# plot_predictions(X_train, y_train, X_test, y_test)

# Check the model current device
# print(next(model_1.parameters()).device)

# Set the model to use the target device
model_1.to(device=device)
# print(next(model_1.parameters()).device)

## Training

# Set up the loss function
loss_fn = nn.L1Loss()  # same as MAE

# Set up the optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# Writing a training loop
torch.manual_seed(42)

epochs = 200

# put data on the target device (device agnostic code for data)
X_train = X_train.to(device=device)
y_train = y_train.to(device=device)
X_test = X_test.to(device=device)
y_test = y_test.to(device=device)

for epoch in range(epochs):
    model_1.train()

    # Forward pass
    y_pred = model_1(X_train)

    # Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Perform backpropagation
    loss.backward()

    # Optimizer step
    optimizer.step()

    ## Testing

    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    # Printing everything
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss} ")

print(model_1.state_dict())
print(weight, bias)


