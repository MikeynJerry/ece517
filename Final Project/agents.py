"""
Classes for playing the game
"""

import json
import math
import random

import torch

from pacman.game import Directions
from pacman import game

from models import DQNModel
from replay import BasicReplay

GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 100


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

    def state_to_tensor(self, state):
        """
          IMPORTANT: State is indexed based on the bottom left, i.e. bottom left = (0, 0)
        """

        # List of (x, y) tuples and represent the 'large' dots
        capsules = state.getCapsules()

        # (width, height) array where True values represent locations of 'small' dots
        food = torch.tensor(state.getFood().data, dtype=torch.bool, device=self.device)

        # (width, height) array where True values represent locations of 'walls'
        walls = torch.tensor(
            state.getWalls().data, dtype=torch.bool, device=self.device
        )

        # (x, y) tuple of Pacman's positon
        pacman = state.getPacmanPosition()

        # List of (x, y) tuples representing each ghost's position
        ghosts = state.getGhostPositions()
        # List of True/False values to determine if the ghosts are scared (Pacman ate a capsule)
        ghost_states = [ghost.scaredTimer > 0 for ghost in state.getGhostStates()]

        # Represent the board as a (6, width, height) image
        image = torch.zeros(
            (6, state.data.layout.width, state.data.layout.height),
            dtype=torch.bool,
            device=self.device,
        )

        # Set the capsule locations
        for capsule in capsules:
            image[0, capsule] = True

        # Food locations
        image[1] = food

        # Wall locations
        image[2] = walls

        # Pacman's location
        image[3][pacman] = True

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
            if not scared:
                image[4][ghost] = True
            else:
                image[5][ghost] = True

        # Models expect (Batch Size, Channels, Height, Width)
        return image.permute(0, 2, 1).unsqueeze(0)


class DQNAgent(Agent):
    def __init__(self, numTraining=0, **kwargs):
        super().__init__()
        # Passed on the command line using -x
        self.training_epsiodes = numTraining

        self.replay = BasicReplay()
        # Target network update period
        self.target_update_period = 100
        # Iteration to start training from
        self.train_start = 1000
        self.built = False

        # Episode = the number of the pacman game we're on
        self.episode = 0
        # Iteration = the iteration number of the current episode
        self.iteration = 0
        # Total Iterations = the number of iterations we've went through over all episodes
        self.total_iterations = 0

        # Used for stats
        self.eps_threshold = EPS_START
        self.eps = []
        self.losses = []

    def build(self, state):
        # Create duplicate policy and target models
        self.policy = DQNModel(
            width=state.data.layout.width, height=state.data.layout.height
        ).to(self.device)
        self.target = DQNModel(
            width=state.data.layout.width, height=state.data.layout.height
        ).to(self.device)
        # Clone their weights
        self.target.load_state_dict(self.policy.state_dict())
        # Freeze the target model
        self.target.eval()

        # Suggested optimization method
        # TODO: Maybe use a scheduled learning rate?
        self.optimizer = torch.optim.RMSprop(self.policy.parameters())

        self.built = True

    def registerInitialState(self, state):
        if not self.built:
            self.build(state)

        self.episode += 1
        self.iteration = 0

        self.last_state = self.state_to_tensor(state)
        self.last_score = 0

    def train_step(self):
        batch_size = 32
        states, actions, rewards, next_states, legal_next_actions = self.replay.sample(
            batch_size
        )

        # Q(S_t, A_t), we request Q(S_t) from our policy model and then
        #  index those values using the actions we actually took during the episode
        state_action_values = self.policy(states.type(torch.float32))
        state_action_values = state_action_values.gather(1, actions.unsqueeze(1))
        # Old code relating to preventing predictions on next states that were terminal
        #
        # non_terminal_mask = torch.tensor(
        #     tuple(legal_actions is not None for legal_actions in legal_next_actions),
        #     dtype=torch.bool,
        #     device=self.device,
        # )

        # # These are the corresponding next states (S_t+1) that aren't terminal
        # non_terminal_next_states = torch.stack(
        #     [
        #         state
        #         for state, legal_actions in zip(next_states, legal_next_actions)
        #         if legal_actions is not None
        #     ]
        # )

        # We predict on all next states (S_t+1) for convenience
        target_state_action_values = self.target(next_states.type(torch.float32))
        # But if the next state is terminal, we don't want to use predicted Q values for it --- they should default to 0
        # Non-terminal: (Batch Size, 4) where non_terminal[i] is all True if the next state isn't terminal and all False if it is
        non_terminal = (
            (torch.sum(legal_next_actions, axis=1) != 0)
            .unsqueeze(1)
            .expand(batch_size, 4)
        )
        # We want to set all action values that have non-terminal next states and are illegal moves to -infinity
        target_state_action_values[non_terminal & ~legal_next_actions] = -float("inf")
        # We want to set all action values that have terminal next states to 0
        target_state_action_values[~non_terminal] = 0
        # We now take the max of each sample
        target_state_action_values = target_state_action_values.max(1)[0]

        # Old code relating to preventing predictions on next states that were terminal
        #
        # We only want the max over A_t+1 so we have one value per sample
        # Initialized to 0 so that terminal values are 0
        # next_state_action_values = torch.zeros(batch_size, device=self.device)
        # next_state_action_values[non_final_mask] = (
        #     target_state_action_values[non_final_mask].max(1)[0].detach()
        # )

        # r + gamma * max_a Q(S_{t+1}, a)
        expected_state_action_values = rewards + GAMMA * target_state_action_values

        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Saved for stats
        self.losses.append(loss.item())

        # Backprop step
        self.policy.zero_grad()
        loss.backward()
        # Clamp gradients between -1 and 1 to prevent explosion
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def getAction(self, state):

        self.iteration += 1
        self.total_iterations += 1

        # Shape = (1, Channels, Height, Width)
        state_as_tensor = self.state_to_tensor(state)

        legal_actions = [
            action for action in state.getLegalActions() if action != Directions.STOP
        ]

        # Exponentially decaying epsilon threshold
        # TODO: Replace this with linear decline?
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.episode / EPS_DECAY
        )

        # Saved for stats
        self.eps_threshold = eps_threshold

        # Either pick a random action (as long as we're still training)
        if random.random() < eps_threshold and self.episode <= self.training_epsiodes:
            (action,) = random.sample(
                [action for action in legal_actions if action != Directions.STOP], 1
            )
        # Or ask our model for one
        else:
            state_action_values = self.policy(state_as_tensor.type(torch.float32) / 255)
            action = values_to_directions[
                torch.argmax(state_action_values, dim=1).item()
            ]

        # Start saving data for replaying later
        self.replay.push(
            state=self.last_state,
            action=self.last_action
            if self.iteration > 1
            else directions_to_values[action],
            reward=state.getScore() - self.last_score,
            next_state=state_as_tensor,
            next_actions=[directions_to_values[a] for a in legal_actions],
        )

        self.last_action = directions_to_values[action]
        self.last_state = state_as_tensor
        self.last_score = state.getScore()

        # Delay training by some number of steps so our replay cache has data
        if self.total_iterations >= self.train_start:
            self.train_step()

        # Update target network parameters periodically
        if self.total_iterations % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # If we chose an illegal action, it's because those actions have a higher reward
        #  than moving legally, thus we'd rather just stop
        # Example: Pacman is in a corner and not moving is more advantageous than moving
        if action not in legal_actions:
            action = Directions.STOP

        return action

    def save_stats(self):
        stats = {}
        stats["losses"] = self.losses
        stats["eps"] = self.eps
        with open("stats.json", "w+") as f:
            json.dump(stats, f)

    # Called when the game is over with the final state
    def final(self, state):
        # This is solely done for convenience to create tensors in the replay
        state_as_tensor = self.state_to_tensor(state)

        # Save final state transitions
        self.replay.push(
            state=self.last_state,
            action=self.last_action,
            reward=100 if state.isWin() else state.getScore() - self.last_score,
            next_state=state_as_tensor,
            next_actions=None,
        )

        # Kept for stats
        self.eps.append(
            EPS_END + (EPS_START - EPS_END) * math.exp(-self.episode / EPS_DECAY)
        )

        # Update the stats file every episode
        self.save_stats()
