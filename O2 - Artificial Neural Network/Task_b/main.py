import torch
import matplotlib.pyplot as plt
import numpy as np


class SigmoidModel:
    def __init__(self):
        # Model variables
        self.W = torch.rand((2, 1), dtype=torch.float, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float, requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)



model = SigmoidModel()

x_train = torch.tensor([[0., 0.], [1., 1], [1., 0.], [0., 1.]])
y_train = torch.tensor([[1.], [0.], [1.], [1.]])

learning_rate = 0.1
optimizer = torch.optim.SGD([model.W, model.b], learning_rate)

for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # perform optimization by adjusting W and b.
    optimizer.zero_grad()  # Clear gradients for the next step.

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

fig = plt.figure("NAND operator")
plot1 = fig.add_subplot(projection='3d') # Type 3d

print(x_train[:, 0])
# Plot data points
plot1.plot(x_train[:, 0].squeeze().detach(),
           x_train[:, 1].squeeze().detach(),
           y_train[:, 0].squeeze().detach(),
           'o',
           label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
           color="blue")

x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10),
                                            np.linspace(-0.25, 1.25, 10)), dtype=torch.float)
y_grid = torch.tensor(np.empty([10, 10]), dtype=torch.float)
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()
plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

plt.savefig("b")
plt.show()