import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device State:', device)
    return device


class Hidden(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(Hidden, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        return self.hidden(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, max_hidden_dim=128):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, max_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(max_hidden_dim),
            *[Hidden(max_hidden_dim, max_hidden_dim) for i in range(hidden_layers)],
        )

        self.final = nn.Linear(max_hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return self.final(x)


class ClassifierInverseTriangle(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, max_hidden_dim=128):
        super(ClassifierInverseTriangle, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, max_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(max_hidden_dim),
            *[Hidden(max_hidden_dim // (2**i), max_hidden_dim // (2 ** (i + 1))) for i in range(hidden_layers)],
        )

        self.final = nn.Sequential(
            nn.Linear(max_hidden_dim // 2**hidden_layers, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return self.final(x)


class Down(nn.Module):
    def __init__(self, hidden_layers=1, max_hidden_dim=128) -> None:
        super(Down, self).__init__()

        self.hidden_layers = hidden_layers

        self.nets = []
        for i in range(self.hidden_layers):
            self.nets.append(
                nn.Sequential(Hidden(max_hidden_dim // (2**i), max_hidden_dim // (2 ** (i + 1)))).to(get_device()),
            )

    def forward(self, x):
        features = []
        for i in range(self.hidden_layers):
            x = self.nets[i](x)
            features.append(x)
        return features


class Up(nn.Module):
    def __init__(self, hidden_layers=1, max_hidden_dim=128) -> None:
        super(Up, self).__init__()

        self.hidden_layers = hidden_layers

        self.nets = []
        for i in range(self.hidden_layers, 0, -1):
            self.nets.append(
                nn.Sequential(Hidden(max_hidden_dim // (2**i), max_hidden_dim // (2 ** (i - 1)))).to(get_device()),
            )

    def forward(self, features):
        for i in range(self.hidden_layers):
            if i == 0:
                x = self.nets[i](features[self.hidden_layers - 1])
                continue
            x = self.nets[i](x + features[self.hidden_layers - i - 1])
        return x


class ClassResidual(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, max_hidden_dim=128) -> None:
        super(ClassResidual, self).__init__()

        self.hidden_layers = hidden_layers
        self.max_hidden_dim = max_hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.max_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.max_hidden_dim),
        )
        self.down = Down(self.hidden_layers, self.max_hidden_dim)
        self.up = Up(self.hidden_layers, self.max_hidden_dim)
        self.final = nn.Linear(max_hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.down(x)
        x = self.up(x)
        return self.final(x)
