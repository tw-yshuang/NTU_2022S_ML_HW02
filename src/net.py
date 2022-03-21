import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, max_hidden_dim=128):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, max_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(max_hidden_dim),
            *[self.hidden(max_hidden_dim, max_hidden_dim) for i in range(hidden_layers)],
        )

        self.final = nn.Linear(max_hidden_dim, output_dim)

    def hidden(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

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
            *[self.hidden(max_hidden_dim // 2**i, max_hidden_dim // 2 ** (i + 1)) for i in range(hidden_layers)],
        )

        self.final = nn.Linear(max_hidden_dim // 2**hidden_layers, output_dim)

    def hidden(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        return self.final(x)
