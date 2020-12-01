"""
Classes meant to handle replaying experienced games
"""

import random
from collections import deque, namedtuple
from scipy.stats import rankdata
import numpy as np
import torch

replay_registry = {}


def register(key):
    def decorator(cls):
        replay_registry[key] = cls
        return cls

    return decorator


TimeStep = namedtuple(
    "TimeStep", ["state", "action", "reward", "next_state", "terminal"]
)


class Replay:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def sample(self, batch_size):
        raise NotImplementedError()

    def push(self, *args):
        raise NotImplementedError()

    def weight_losses(self, losses, *args):
        return losses


@register("basic")
class BasicReplay(Replay):
    def __init__(self, capacity=50_000, **kwargs):
        super().__init__()
        self.steps = deque(maxlen=capacity)

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

        return states, actions, rewards, next_states, terminal, None, None

    def push(self, state, action, reward, next_state, terminal, *extra):
        step = TimeStep(state, action, reward, next_state, terminal)
        self.steps.append(step)


class PrioritizedReplay(Replay):
    def __init__(self, *, alpha, beta, capacity=50_000, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.steps = deque(maxlen=capacity)
        self.priorities = torch.zeros(
            (capacity,), dtype=torch.float32, device=self.device
        )

    def push(self, state, action, reward, next_state, terminal):
        max_priority = self.priorities.max() if self.steps else 1.0

        step = TimeStep(state, action, reward, next_state, terminal)
        self.steps.append(step)

        if len(self.steps) == self.steps.maxlen:
            self.priorities = self.priorities.roll(-1)

        self.priorities[len(self.steps) - 1] = max_priority

    def sample(self, batch_size):
        # Get non-empty priorities
        priorities = self.priorities[: len(self.steps)]

        # Paper formula: https://arxiv.org/pdf/1511.05952.pdf
        # P(i) = (p_i^alpha) / (sum[p_i^alpha])
        probabilities = priorities ** self.alpha
        probabilities /= torch.sum(probabilities)

        # Choose mini-batch using calculated probabilities
        indices = np.random.choice(
            len(self.steps), batch_size, p=probabilities.cpu().numpy()
        )
        steps = [self.steps[index] for index in indices]

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

        # Importance sampling parameter
        weights = (len(self.steps) * probabilities[indices]) ** -self.beta
        weights = weights / torch.max(weights)

        return states, actions, rewards, next_states, terminal, indices, weights

@register("priority-proportional")
class PrioritizedProportionalReplay(PrioritizedReplay):
  def weight_losses(self, losses, indices, weights):
      losses = losses * weights
      priorities = losses + 1e-6
      self.priorities[indices] = priorities.detach()
      return losses

@register("priority-rank")
class PrioritizedRankReplay(PrioritizedReplay):
    def weight_losses(self, losses, indices, weights):
        losses = losses * weights
        numpy_losses = losses.detach().numpy()
        priorities = 1 / rankdata(numpy_losses)
        self.priorities[indices] = torch.from_numpy(priorities).to(self.device)
        return losses
