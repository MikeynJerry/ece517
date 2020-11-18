"""
Classes meant to handle replaying experienced games
"""

import random
from collections import deque, namedtuple

import torch


TimeStep = namedtuple(
    "TimeStep", ["state", "action", "reward", "next_state", "next_actions"]
)


class Replay:
    def __init__(self):
        self.device = "cuda"
        self.steps = deque(maxlen=100_000)

    def push(self, state, action, reward, next_state, next_actions):
        step = TimeStep(state, action, reward, next_state, next_actions)
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
        # Next Actions is a (Batch Size, # Actions) boolean mask that is True for legal actions at T + 1
        # Example: next_actions[i][j] is True if action j in sample i is legal
        next_actions = torch.full(
            (batch_size, 4), False, dtype=torch.bool, device=self.device
        )
        for i, step in enumerate(steps):
            if step.next_actions is not None:
                next_actions[i][step.next_actions] = True

        return states, actions, rewards, next_states, next_actions

