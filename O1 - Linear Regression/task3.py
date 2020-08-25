import torch
import matplotlib.pyplot as plt
import pandas as pd

# Visualize result
data = pd.read_csv("./data/day_head_circumference.csv")

day = data["day"].tolist()
head_circumference = data["head_circumference"].tolist()

# Numpy: length = data["length"].to_numpy()

# Observed/training input and output
x_train = torch.tensor(day).reshape(-1, 1)  # x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
y_train = torch.tensor(head_circumference).reshape(-1, 1)  # y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        m = x @ self.W + self.b
        s = torch.sigmoid(m)
        y = 20.0 * s + 31.0
        return y

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.000001)
for epoch in range(75000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result

plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.mean(x_train)], [torch.max(x_train)]])
plt.scatter(x_train, model.f(x_train).detach(), label='$y = f(x) = 20*sigmoid(xW+b)+31$', color="red")


plt.legend()
plt.savefig("task3")
plt.show()
