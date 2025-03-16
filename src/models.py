import torch
import torch.nn as nn


class MLPClassifier(nn.Module):

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(MLPClassifier, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        self.layer2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer4 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = torch.relu(x)

        x = self.layer4(x)

        return x
