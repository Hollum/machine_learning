import torch
import matplotlib.pyplot as plt
import numpy as np

class SigmoidModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.]], requires_grad=True)
        self.b = torch.tensor([[0.]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

model = SigmoidModel()

x_train = torch.tensor([[0.0], [1.0]])
y_train = torch.tensor([[1.0], [0.0]])

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

fig = plt.figure("Logistic regression: the logical NOT operator")

plot1 = fig.add_subplot()

plot1.plot(x_train.detach(),
           y_train.detach(),
           'o',
           label="$(\\hat x_1^{(i)},\\hat y^{(i)})$",
           color="blue")

out = torch.reshape(torch.tensor(np.linspace(0, 1, 100).tolist()), (-1, 1))

plt.plot(out, model.f(out).detach())

plt.savefig("a")
plt.show()