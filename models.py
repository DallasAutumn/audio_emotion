from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import DataLoader


class Flatten(nn.Module):
    """Note that batch_size is the first dimension"""

    def forward(self, x):
        return x.view(x.size(0), -1)  # [batch, seq_len*input_size]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=13, stride=13),
            #             nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            #             nn.Dropout(0.1),
            Flatten(),
            nn.Linear(136, 6),

            nn.Softmax()
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        return x


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    # data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data.unsqueeze(-1)


cnn = CNN()
print(cnn)

if __name__ == "__main__":
    train_set = AudioDataset(train=True, transform=ToTensor())
    test_set = AudioDataset(train=False, transform=ToTensor())

    # hyper params
    learning_rate = 1e-5
    batch_size = 16
    epochs = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader = DataLoader(dataset=train_set, num_workers=2,
                              collate_fn=collate_fn, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_set, num_workers=2,
                             collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    cnn.to(device)
    # training
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            T = 5
            if i % T == 0:    # print every T mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / T))
                running_loss = 0.0

        print('Finished Training')
