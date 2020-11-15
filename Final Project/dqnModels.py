"""
Models for training (DQN, Double DQN)
"""


from torch import nn


class DQNModel(nn.Module):
  def __init__(self):
    super().__init__()

    # Network described by Stanford project
    self.conv = nn.Sequential(
      nn.Conv2d(3, 8, 3),
      nn.ReLU(),
      nn.Conv2d(8, 16, 3),
      nn.ReLU(),
      nn.Conv2d(16, 32, 3),
      nn.ReLU(),
    )

    # Missing Q(s, a) at the end
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2240, 256),
      nn.ReLU(),
    )

  def forward(self, inp):
    out = self.conv(inp)
    print('after conv', out.size())
    out = self.head(out)
    print('after head', out.size())
    pass