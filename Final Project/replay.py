"""
Classes meant to handle replaying experienced games
"""

import random
from collections import deque, namedtuple
import numpy as np


import torch


TimeStep = namedtuple(
    "TimeStep", ["state", "action", "reward", "next_state", "terminal"]
)


class Replay:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

    

class BasicReplay(Replay):
    def __init__(self):
        super().__init__()
        self.steps = deque(maxlen=50_000)

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
    
    def push(self, state, action, reward, next_state, terminal):
        step = TimeStep(state, action, reward, next_state, terminal)
        self.steps.append(step)



class PrioritizedExperienceReplay(Replay):
    def __init__(self, alpha=0.6, beta=0.4):
        super().__init__()
        self.capacity = 50_000
        self.prob_alpha = alpha
        self.beta = beta
        self.steps = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        
    
    def push(self, state, action, reward, next_state, terminal):
        #temporarly assing the max priority to this sample
        max_priority = self.priorities.max() if self.steps else 1.0
        
        #circular buffer
        if len(self.steps) < self.capacity:
            step = TimeStep(state, action, reward, next_state, terminal)
            self.steps.append(step)
        else:
            self.steps[self.pos] = TimeStep(state, action, reward, next_state, terminal)

        #update priorities and buffer pointer
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    
    def sample(self, batch_size):
        #get non-empty priorities
        if len(self.steps) == self.capacity: 
            prios = self.priorities
        else: 
            prios = self.priorities[:self.pos]
        
        #formula
        #P(i) = (p_i^alpha) / (sum[p_i^alpha])
        probabilities  = prios ** self.prob_alpha
        probabilities /= probabilities.sum()

        #choose the minibatch using probabilities
        indices = np.random.choice(len(self.steps), batch_size, p=probabilities)
        steps = [self.steps[idx] for idx in indices]
        
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
        
        #Importance sampling parameter
        N = len(self.steps)
        weights  = (N * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        weights  = np.array(weights, dtype=np.float32)
        return states, actions, rewards, next_states, terminal, indices, weights


    def update_priorities_replay(self, batch_indices, batch_priorities):
        
        for index, priority in zip(batch_indices, batch_priorities):
            self.priorities[index] = priority

    def __len__(self):
        return len(self.steps)