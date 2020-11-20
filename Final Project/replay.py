"""
Classes meant to handle replaying experienced games
"""

import random
from collections import deque, namedtuple

import torch


TimeStep = namedtuple(
    "TimeStep", ["state", "action", "reward", "next_state", "terminal"]
)


class Replay:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.steps = deque(maxlen=50_000)

    def push(self, state, action, reward, next_state, terminal):
        step = TimeStep(state, action, reward, next_state, terminal)
        self.steps.append(step)


class BasicReplay(Replay):
    def __init__(self):
        super().__init__()

    def sample(self, batch_size):
        steps = random.sample(self.steps, batch_size)
        # States: (1, Channels, Height, Width) -> (Batch Size, Channels, Height, Width)
        states = torch.cat([step.state for step in steps])
        # Actions: () -> (Batch Size)
        actions = torch.tensor([step.action for step in steps], device=self.device)
        # Rewards: () -> (Batch Size)
        rewards = torch.tensor([step.reward for step in steps], device=self.device)
        # Next States: (1, Channels, Height, Width) -> (Batch Size, Channels, Height, Width)
        next_states = torch.cat([step.next_state for step in steps])
        # Terminal: () -> (Batch Size)
        terminal = torch.tensor([step.terminal for step in steps], device=self.device)

        return states, actions, rewards, next_states, terminal

