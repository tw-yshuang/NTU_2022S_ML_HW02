import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HiddenBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=2, max_hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, max_hidden_dim),
            nn.ReLU(),
            *[HiddenBlock(max_hidden_dim // 2**i, max_hidden_dim // 2 ** (i + 1)) for i in range(hidden_layers)],
            # *[HiddenBlock(max_hidden_dim, max_hidden_dim) for i in range(hidden_layers)],
            # *[self.hidden(max_hidden_dim // 2**i, max_hidden_dim // 2 ** (i + 1)) for i in range(hidden_layers)],
        )
        self.final = nn.Linear(max_hidden_dim // 2**hidden_layers, output_dim)
        # self.final = nn.Linear(max_hidden_dim, output_dim)

    def hidden(self, input_dim, output_dim):
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        return self.final(x)

    # def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
    #     super(Classifier, self).__init__()

    #     self.fc = nn.Sequential(
    #         BasicBlock(input_dim, hidden_dim),
    #         *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
    #         nn.Linear(hidden_dim, output_dim)
    #     )

    # def forward(self, x):
    #     x = self.fc(x)
    #     return x
