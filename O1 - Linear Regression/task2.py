import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d

matplotlib.rcParams.update({'font.size': 11})

# regarding the notations, see http://stats.stackexchange.com/questions/193908/in-machine-learning-why-are-superscripts-used-instead-of-subscripts

W_init = np.array([[-0.2], [0.53]])
b_init = np.array([[3.1]])


# Visualize result
data = pd.read_csv("./data/day_length_weight.csv")

day = data["day"].tolist()
weight = data["weight"].tolist()
length = data["length"].tolist()

train_x = []
train_y = []
for i in range(weight.__len__()):
    temp_x = [weight[i], length[i]]
    temp_y = [day[i]]

    train_x.append(temp_x)
    train_y.append(temp_y)

"""
Should be????
    temp_x = [weight[i], length[i]]
    temp_y = [day[i]]
    
    train_x.append(temp_x)
    train_y.append(temp_y)
"""




class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0],[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# observed/training input and output


"""
x_train = np.array(train_x)
y_train = np.array(train_y)
"""
x_train = torch.tensor(train_x)  # x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
y_train = torch.tensor(train_y)  # y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.0000001)
for epoch in range(75000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step


print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

fig = plt.figure().gca(projection='3d')

fig.scatter(day, length, weight, c='red')
fig.scatter(day, length, model.f(x_train).detach(), label='$y = f(x) = xW+b$')

fig.set_xlabel('Day')
fig.set_ylabel('Length')
fig.set_zlabel('Weight')


plt.show()
