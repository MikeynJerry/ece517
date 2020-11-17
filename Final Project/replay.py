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
        states = torch.cat([step.state for step in steps])
        actions = torch.tensor([step.action for step in steps], device=self.device)
        rewards = torch.tensor([step.reward for step in steps], device=self.device)
        next_states = torch.cat([step.next_state for step in steps])
        next_actions = [step.next_actions for step in steps]
        return states, actions, rewards, next_states, next_actions

