"""
Classes for playing the game
"""

import math
import random
from enum import Enum

import torch
import matplotlib.pyplot as plt

from pacman.game import Directions
from pacman import game

from models import DQNModel
from replay import BasicReplay

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


directions_to_values = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
}

values_to_directions = dict([(val, key) for key, val in directions_to_values.items()])


class Agent(game.Agent):
    def __init__(self):
        super().__init__()
        self.device = "cuda"

        # White
        self.capsule_color = torch.tensor(
            [255, 255, 255], dtype=torch.uint8, device=self.device
        )
        # Green
        self.food_color = torch.tensor(
            [0, 128, 0], dtype=torch.uint8, device=self.device
        )
        # Blue
        self.wall_color = torch.tensor(
            [0, 0, 255], dtype=torch.uint8, device=self.device
        )
        # Yellow
        self.pacman_color = torch.tensor(
            [255, 255, 0], dtype=torch.uint8, device=self.device
        )
        # Red
        self.ghost_color = torch.tensor(
            [255, 0, 0], dtype=torch.uint8, device=self.device
        )
        # Gray
        self.ghost_scared_color = torch.tensor(
            [128, 128, 128], dtype=torch.uint8, device=self.device
        )

    def state_to_image(self, state):
        """
          IMPORTANT: State is indexed based on the bottom left, i.e. bottom left = (0, 0)
        """

        # List of (x, y) tuples and represent the 'large' dots
        capsules = state.getCapsules()

        # (width, height) array where True values represent locations of 'small' dots
        food = torch.tensor(state.getFood().data, device=self.device)

        # (width, height) array where True values represent locations of 'walls'
        walls = torch.tensor(state.getWalls().data, device=self.device)

        # (x, y) tuple of Pacman's positon
        pacman = state.getPacmanPosition()

        # List of (x, y) tuples representing each ghost's position
        ghosts = state.getGhostPositions()
        # List of True/False values to determine if the ghosts are scared (Pacman ate a capsule)
        ghost_states = [ghost.scaredTimer > 0 for ghost in state.getGhostStates()]

        # Represent the board as a (width, height, 3) image
        image = torch.zeros(
            (state.data.layout.width, state.data.layout.height, 3),
            dtype=torch.uint8,
            device=self.device,
        )

        # Set the capsule locations
        for capsule in capsules:
            image[capsule] = self.capsule_color

        # Food locations
        image[food] = self.food_color

        # Wall locations
        image[walls] = self.wall_color

        # Pacman's location
        image[pacman] = self.pacman_color

        # Ghost positions
        for ghost, scared in zip(ghosts, ghost_states):
            """
              IMPORTANT: Ghost locations aren't always, nice, round numbers. Often they end up as floats
              because we get state updates when Pacman has reached another position and not when the ghosts have.
              Currently we're truncating them by converting them to ints but in the future it might work better if
              we round the number to the closest side or take into account the direction the ghosts are taking when rounding.
            """
            if type(ghost[0]) != int:
                ghost = int(ghost[0]), int(ghost[1])

            # Set ghost color based on if the ghost hasn't been eaten since Pacman ate a 'capsule'
            image[ghost] = self.ghost_scared_color if scared else self.ghost_color

        # Models expect (Batch Size, Channels, Height, Width)
        return image.permute(2, 1, 0).unsqueeze(0)


class DQNAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        print("agent kwargs", kwargs)
        self.replay = BasicReplay()
        self.target_update_period = 50
        self.train_start = 50
        self.iteration = 0
        self.built = False

    def build(self, state):
        self.policy = DQNModel(
            width=state.data.layout.width, height=state.data.layout.height
        ).to(self.device)
        self.target = DQNModel(
            width=state.data.layout.width, height=state.data.layout.height
        ).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.RMSprop(self.policy.parameters())

        self.built = True

    def registerInitialState(self, state):
        if not self.built:
            self.build(state)

    def train_step(self):
        states, actions, rewards, next_states, next_actions = self.replay.sample(32)
        state_action_values = self.policy(states.type(torch.float32) / 255).gather(
            1, actions.unsqueeze(0)
        )
        next_state_action_values = self.target(states.type(torch.float32) / 255).max(1)[
            0
        ]
        expected_state_action_values = next_state_action_values * GAMMA + rewards

        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(0)
        )

        # Backprop step
        self.policy.zero_grad()
        loss.backward()
        # Clamp gradients between -1 and 1 to prevent explosion
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def getAction(self, state):

        self.iteration += 1

        # Shape = (1, Width, Height, Channels)
        state_as_image = self.state_to_image(state)

        # 0 is Pacman's agent id
        legal_actions = [
            action for action in state.getLegalActions(0) if action != Directions.STOP
        ]

        # Exponentially decaying epsilon threshold
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.iteration / EPS_DECAY
        )

        # Either pick a random action or ask our model for one
        if random.random() < eps_threshold:
            (action,) = random.sample(
                [action for action in legal_actions if action != Directions.STOP], 1
            )
        else:
            state_action_values = self.policy(state_as_image.type(torch.float32) / 255)
            action = values_to_directions[
                torch.argmax(state_action_values, dim=1).item()
            ]

        # Start saving data for replaying later
        if self.iteration > 1:
            self.replay.push(
                state=self.last_state,
                action=self.last_action,
                reward=state.getScore() - self.last_score,
                next_state=state_as_image,
                next_actions=legal_actions,
            )

        self.last_action = directions_to_values[action]
        self.last_state = state_as_image
        self.last_score = state.getScore()

        # Delay training by some number of steps so our replay cache has data
        if self.iteration >= self.train_start:
            self.train_step()

        # Update target network parameters every `target_update_period` iterations
        if self.iteration % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # If we chose an illegal action, just stop instead
        # This is a bit iffy as to why, but other people do this
        if action not in legal_actions:
            action = Directions.STOP

        return action
