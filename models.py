from collections import OrderedDict

import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# build network
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(216, 256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(128, 128, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding='same'),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


net1 = Net1()

# hyper params
learning_rate = 1e-5
batch_size = 16

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net1.parameters(), lr=learning_rate)

print(net1)
