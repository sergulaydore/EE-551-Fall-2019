import torch
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, training):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x, training):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


class Solver(object):
    def __init__(self, net, learning_rate, n_epochs,
                 batch_size, seed, lambda_l2=0, lambda_l1=0,
                 ):

        self.net = net
        self.params = list(net.parameters())
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = check_random_state(seed)
        random.seed(seed)
        self.loss = []
        self.test_loss = []
        self.lambda_l2 = lambda_l2 # L2 regularizer
        self.lambda_l1 = lambda_l1  # L2 regularizer


    def batch_update(self, X_batch, y_batch):

        input = torch.from_numpy(X_batch).float() if type(X_batch) == np.ndarray else X_batch
        target = torch.from_numpy(y_batch) if type(y_batch) == np.ndarray else y_batch

        out = self.net.forward(input, training=True)

        # add l2
        l2_reg = autograd.Variable(torch.FloatTensor(1), requires_grad=True)
        l1_reg = autograd.Variable(torch.FloatTensor(1), requires_grad=True)
        for idx, p in enumerate(self.params):
            l2_reg = l2_reg + p.norm(2)
            l1_reg = l1_reg + p.norm(1)
        loss = self.criterion(out, target.long()) + self.lambda_l2 * l2_reg + self.lambda_l1 * l1_reg
        self.net.zero_grad()
        loss.backward()

        # manual sgd
        for idx, p in enumerate(self.params):
            p.data.sub_(p.grad.data * self.learning_rate)
            p.grad.zero_()

        batch_loss = self.criterion(out, target.long())

        return float(batch_loss.data)

    def train_tensor(self, train_loader, test_loader):
        n_samples = train_loader.dataset.train_data.shape[0]
        for idx in range(self.n_epochs):
            # if idx %2 == 0:
            print("Epoch", idx)
            # fig, axes = plt.subplots(4, 4)
            # weights = self.params[0].data.numpy()
            # # use global min / max to ensure all weights are shown on the same scale
            # # vmin, vmax = weights.min(), weights.max()
            # for coef, ax in zip(weights, axes.ravel()):
            #     ax.imshow(coef.reshape(28, 28), cmap=plt.cm.gray)
            #     ax.set_xticks(())
            #     ax.set_yticks(())
            # plt.show()

            accumulated_loss = 0.0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = autograd.Variable(X_batch).float(), autograd.Variable(y_batch).float()
                batch_loss = self.batch_update(X_batch, y_batch)
                accumulated_loss += batch_loss

            iteration_loss = accumulated_loss / n_samples
            self.loss.append(iteration_loss)
            pred, accuracy, iteration_test_loss = self.predict_tensor(test_loader)
            self.test_loss.append(iteration_test_loss)

            print("Accuracy", accuracy)

    def predict_tensor(self, test_loader):
        iteration_test_loss = 0
        correct = 0
        with torch.no_grad():
            for input, target in test_loader:
                out = self.net.forward(input.float(), training=False)
                iteration_test_loss += self.criterion(out, target)
                pred = out.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred).long()).sum().item()

        accuracy = correct / len(test_loader.dataset)
        iteration_test_loss = iteration_test_loss / len(test_loader.dataset)
        return pred, accuracy, iteration_test_loss
