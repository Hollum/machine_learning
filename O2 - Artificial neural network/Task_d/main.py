import torch
import torchvision
import matplotlib.pyplot as plt

class SoftmaxModel:
    def __init__(self):
        # Model variables
        self.W = torch.rand((784, 10), requires_grad=True)
        self.b = torch.rand((1, 10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = SoftmaxModel()

print("Loading data..")
# Load training set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

# Load and test set
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

print("Training model")
learning_rate = 0.2
optimizer = torch.optim.SGD([model.W, model.b], learning_rate, momentum=0.5)

epoch = 0
while model.accuracy(x_test, y_test).item() < 0.9:
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch: %s, Loss: %s, Accuracy: %s" % (
        epoch +1, model.loss(x_test, y_test).item(), model.accuracy(x_test, y_test).item()))
    epoch += 1

print("Finished training")

print("After training = Loss: %s, Accuracy: %s" % (
    model.loss(x_test, y_test).item(), model.accuracy(x_test, y_test).item()))

# ****VISUALS****
# Show the input of the first observation in the training set
plt.imshow(x_train[0, :].reshape(28, 28))

# Print the classification of the first observation in the training set
print(y_train[0, :])

# Save the input of the first observation in the training set
plt.imsave('x_train_%i.png', x_train[0, :].reshape(28, 28))

for i in range(10):
    plt.imsave("%i.png" % i, model.W[:, i].reshape(28, 28).detach())

plt.show()