import torch.nn as nn


class MnistCnn(nn.Module):
    def __init__(self):
        super(MnistCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # n=> 1, 28, 28
        x = self.relu(self.conv1(x))  # n=> 6, 24, 24
        x = self.pool(x)  # n=> 6, 12, 12

        x = self.relu(self.conv2(x))  # n=> 16, 8, 8
        x = self.pool(x)  # n=> 16, 4, 4

        x = x.view(-1, 16 * 4 * 4)  # n=> 256
        x = self.relu(self.fc1(x))  # n=> 128
        x = self.relu(self.fc2(x))  # n=> 128

        x = self.output_layer(x)  # n=> 10
        x = self.softmax(x)

        return x
