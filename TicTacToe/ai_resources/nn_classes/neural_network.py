import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_func="tanh"):
        super().__init__()

        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        if activation_func == "tanh":
            self.act_func = nn.Tanh()
        else:
            self.act_func = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.act_func(x)

        x = self.hidden_layer2(x)
        x = self.act_func(x)

        x = self.hidden_layer2(x)
        x = self.act_func(x)

        x = self.hidden_layer2(x)
        x = self.act_func(x)

        x = self.hidden_layer2(x)
        x = self.act_func(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x
