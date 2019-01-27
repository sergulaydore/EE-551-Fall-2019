from __future__ import print_function
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils import LeNet, Solver


torch.set_default_tensor_type('torch.FloatTensor')
batch_size = 16


transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                                 ])

train_loader = torch.utils.data.DataLoader(
                                            datasets.MNIST('../data', train=True, download=True, transform=transforms),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory = True
                                           )

test_loader = torch.utils.data.DataLoader(
                                            datasets.MNIST('../data', train=False, transform=transforms),
                                            batch_size=batch_size, shuffle=True,
                                            pin_memory=True
                                         )




## SOLVER
learning_rate = 0.01; n_epochs = 11; n_phi = 1000; n_sample_rena = 20; seed = 10003

net = LeNet()
solver = Solver(net, learning_rate=learning_rate, n_epochs=n_epochs,
                batch_size=batch_size, seed=seed, lambda_l2=0, lambda_l1=0)
solver.train_tensor(train_loader, test_loader)
print(solver.loss[-1])
print(solver.test_loss[-1])

fig, ax = plt.subplots(1, 1)
plt.plot(solver.loss, label="Training loss")
plt.plot(solver.test_loss, label="Test Loss")
plt.legend()
plt.show()

fig, axes = plt.subplots(2, 5)
weights = solver.params[0].data.numpy()
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = weights.min(), weights.max()
for coef, ax in zip(weights, axes.ravel()):
    ax.imshow(coef.reshape(5, 5), cmap=plt.cm.gray, vmin=0.5*vmin, vmax=0.5*vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()