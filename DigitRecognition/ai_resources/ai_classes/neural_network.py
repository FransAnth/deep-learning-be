import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(784, 128)
        self.hidden_layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # initial hidden layer
        x = self.hidden_layer(x)
        x = self.relu(x)

        x = self.hidden_layer2(x)
        x = self.relu(x)

        x = self.hidden_layer2(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x
