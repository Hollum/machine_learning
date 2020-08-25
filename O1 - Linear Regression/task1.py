import torch
import matplotlib.pyplot as plt
import pandas as pd

# Visualize result
data = pd.read_csv("./data/length_weight.csv")

weight = data["weight"].tolist()
length = data["length"].tolist()

# Numpy: length = data["length"].to_numpy()

# Observed/training input and output
x_train = torch.tensor(length).reshape(-1, 1)  # x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
y_train = torch.tensor(weight).reshape(-1, 1)  # y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.5]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)
for epoch in range(150000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
plt.legend()

plt.savefig("task1")
plt.show()
