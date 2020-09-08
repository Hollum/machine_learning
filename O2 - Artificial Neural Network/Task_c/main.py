import torch
import matplotlib.pyplot as plt
import numpy as np


class SigmoidModel:
    def __init__(self):
        '''
         Example of XOR that does not converge
         self.W1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float, requires_grad=True)
         self.b1 = torch.tensor([[0.0, 0.0]], dtype=torch.float, requires_grad=True)
         self.W2 = torch.tensor([[0.0], [0.0]], dtype=torch.float, requires_grad=True)
         self.b2 = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)
         '''

        # Converging modell
        self.W1 = torch.tensor([[1.0, 0.5], [0.5, 0.6]], dtype=torch.float, requires_grad=True)
        self.b1 = torch.tensor([[1.0, 1.0]], dtype=torch.float, requires_grad=True)

        self.W2 = torch.tensor([[1.0], [-1.0]], dtype=torch.float, requires_grad=True)
        self.b2 = torch.tensor([[1.0]], dtype=torch.float, requires_grad=True)

    def logits2(self, h):
        return h @ self.W2 + self.b2

    # First layer function
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits2(self.f1(x)), y)



model = SigmoidModel()

x_train = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
y_train = torch.tensor([[1.], [1.], [0.], [0.]])

learning_rate = 5
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], learning_rate)

for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # perform optimization by adjusting W and b.
    optimizer.zero_grad()  # Clear gradients for the next step.

print("W = %s, b = %s, loss = %s" % (model.W1, model.b1, model.loss(x_train, y_train)))
print("W = %s, b = %s, loss = %s" % (model.W2, model.b2, model.loss(x_train, y_train)))
print("[0,1] is 1 = ", round(model.f(torch.tensor([0., 1.])).item()))

fig = plt.figure("Logistic regression: the logical XOR operator")
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

plt.savefig("c")
plt.show()