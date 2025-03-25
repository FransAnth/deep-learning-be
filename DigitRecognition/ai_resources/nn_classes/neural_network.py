import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_count):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.hidden_layer_count = hidden_layer_count

    def forward(self, x):
        # initial hidden layer
        x = self.hidden_layer(x)
        x = self.relu(x)

        for _ in range(self.hidden_layer_count - 1):
            x = self.hidden_layer2(x)
            x = self.relu(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x
