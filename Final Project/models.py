"""
Models for training (DQN, Double DQN)
"""


from torch import nn


class DQNModel(nn.Module):
    def __init__(self, width, height, nb_actions=5):
        super().__init__()

        # Network described by Stanford project
        # Conv2d = (in channels, out channels, kernel width)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
        )

        # The initial image loses 2 pixels per Convolutional layer pass in each dim
        # We're using 3 Convolutional layers, so (dim - # pixels lost * # layers)
        # Multiply that by the out-channels of the Convolutional sequence (32)
        flat_features = (width - 2 * 3) * (height - 2 * 3) * 32

        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_features, 256), nn.ReLU(), nn.Linear(256, 4)
        )

    def forward(self, inp):
        out = self.conv(inp)
        out = self.head(out)
        return out
