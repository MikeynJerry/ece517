"""
DQN Model for training
"""

import torch
from torch import nn


model_registry = {}


def register(key):
    def decorator(cls):
        model_registry[key] = cls
        return cls

    return decorator


@register("stanford")
class StanfordDQNModel(nn.Module):
    def __init__(self, width, height, in_channels=3, nb_actions=4):
        super().__init__()

        # Network described by Stanford project
        # Conv2d = (in channels, out channels, kernel width)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
        )

        flat_features = (
            self.conv(torch.zeros(1, in_channels, height, width)).view(1, -1).size(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 256),
            nn.ReLU(),
            nn.Linear(256, nb_actions),
        )

    def forward(self, inp):
        out = self.conv(inp)
        out = self.head(out)
        return out


@register("small")
class SmallGridDQN(nn.Module):
    def __init__(self, width, height, in_channels=6, nb_actions=4):
        super().__init__()

        # Conv2d = (in channels, out channels, kernel width)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 2), nn.ReLU()
        )

        flat_features = (
            self.conv(torch.zeros(1, in_channels, height, width)).view(1, -1).size(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 512),
            nn.ReLU(),
            nn.Linear(512, nb_actions),
        )

    def forward(self, inp):
        out = self.conv(inp)
        out = self.head(out)
        return out
