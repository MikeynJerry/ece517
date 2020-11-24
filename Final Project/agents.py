"""
Classes for playing the game
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd

from pacman.game import Directions
from pacman import game

from models import model_registry
from replay import replay_registry

from default_config import *

directions_to_values = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
}

values_to_directions = dict([(val, key) for key, val in directions_to_values.items()])

loss_functions = {
    "mse": torch.nn.functional.mse_loss,
    "huber": torch.nn.functional.smooth_l1_loss,
}


class DQNAgent(game.Agent):
    def __init__(
        self,
        *,
        # Eps Args
        gamma,
        eps_start,
        eps_end,
        eps_decay,
        eps_schedule,
        # Train Args
        train_start,
        target_update,
        numTraining,
        # Stat Args
        log_dir,
        logging_period,
        # Model Args
        model_type,
        model_dir,
        model_period,
        batch_size,
        # Replay Args
        replay_type,
        replay_alpha,
        replay_beta,
        # Loss / Optimizer Args,
        loss_type,
        # Testing Args
        is_training=True,
        model_paths=[],
        nb_testing_episodes=100,
        settings={},
        **kwargs,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Used for training stats
        self.eps = []
        self.all_losses = []
        self.avg_losses = []
        self.episode_losses = []

        # Used for testing stats
        self.training_stats = {
            episode_number: {"wins": [], "scores": []}
            for episode_number in [
                model_period * model_number
                for model_number in range(1, len(model_paths) + 1)
            ]
        }

        # Epsilon Arguments
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_schedule = eps_schedule

        # Training Arguments
        self.train_start = train_start
        self.target_update_period = target_update
        self.training_epsiodes = numTraining

        # Stats and Logging Arguments
        self.logging_period = logging_period
        if log_dir is not None:
            self.log_path = Path(DEFAULT_LOG_ROOT) / log_dir
            self.log_path.mkdir(parents=True, exist_ok=True)
            with open(self.log_path / "config.json", "w+") as f:
                json.dump(settings, f)
            self.log_file = open(self.log_path / "log.txt", "w+", buffering=1)
        else:
            self.log_path = None

        # Model Arguments
        self.model_type = model_type
        if model_dir is not None:
            self.model_dir = Path(DEFAULT_MODEL_ROOT) / model_dir
            self.model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.model_dir = None
        self.model_period = model_period
        self.batch_size = batch_size

        # Replay Arguments
        self.replay_type = replay_type
        self.replay = replay_registry[replay_type](alpha=replay_alpha, beta=replay_beta)

        # Loss and Optimizer Arguments
        self.loss_fn = loss_functions[loss_type]

        # Testing Arguments
        self.is_training = is_training
        self.model_paths = model_paths
        self.nb_testing_episodes = nb_testing_episodes
        self.model_number = 0

        # Episode = the number of the pacman game we're on
        self.episode = 0
        # Iteration = the iteration number of the current episode
        self.iteration = 0
        # Total Iterations = the number of iterations we've went through over all episodes
        self.total_iterations = 0
        # Helps us build our models on first time receiving state
        self.built = False

    @staticmethod
    def add_args(parser):

        # Epsilon Arguments
        parser.add_argument("--gamma", default=DEFAULT_GAMMA, type=float)
        parser.add_argument("--eps-start", default=DEFAULT_EPS_START, type=float)
        parser.add_argument("--eps-end", default=DEFAULT_EPS_END, type=float)
        parser.add_argument("--eps-decay", default=DEFAULT_EPS_DECAY, type=float)
        parser.add_argument(
            "--eps-schedule",
            default=DEFAULT_EPS_SCHEDULE,
            choices=["linear", "exponential"],
        )

        # Training Arguments
        parser.add_argument("--train-start", default=DEFAULT_TRAIN_START, type=int)
        parser.add_argument("--target-update", default=DEFAULT_TARGET_UPDATE, type=int)

        # Stats and Logging Arguments
        parser.add_argument("--log-dir")
        parser.add_argument(
            "--log-frequency",
            default=DEFAULT_LOGGING_PERIOD,
            dest="logging_period",
            type=int,
        )

        # Model Arguments
        parser.add_argument(
            "--model-type", default=DEFAULT_MODEL_TYPE, choices=model_registry.keys()
        )
        parser.add_argument("--model-dir")
        parser.add_argument(
            "--model-save-rate",
            default=DEFAULT_MODEL_PERIOD,
            dest="model_period",
            type=int,
        )
        parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)

        # Replay Arguments
        parser.add_argument(
            "--replay-type", default=DEFAULT_REPLAY_TYPE, choices=replay_registry.keys()
        )
        parser.add_argument("--replay-alpha", default=DEFAULT_REPLAY_ALPHA, type=float)
        parser.add_argument("--replay-beta", default=DEFAULT_REPLAY_BETA, type=float)

        # Loss and Optimizer Arguments
        parser.add_argument(
            "--loss-type", default=DEFAULT_LOSS_TYPE, choices=loss_functions.keys()
        )

    def build(self, state):
        raise NotImplementedError()

    def getAction(self, state):
        self.iteration += 1
        self.total_iterations += 1

        # Shape = (1, Channels, Height, Width)
        state_as_tensor = self.get_state_tensor(state)

        # Either pick a random action (as long as we're still training)
        if (
            random.random() < self.epsilon
            and self.episode <= self.training_epsiodes
            and self.is_training
        ):
            (action,) = random.sample(
                [action for value, action in values_to_directions.items()], 1
            )
        # Or ask our model for one
        else:
            state_action_values = self.policy(state_as_tensor)
            action = values_to_directions[
                torch.argmax(state_action_values, dim=1).item()
            ]

        # Start saving data for replaying later
        if self.iteration > 1 and self.is_training:
            self.replay.push(
                state=self.last_state,
                action=self.last_action,
                reward=state.getScore() - self.last_score,
                next_state=state_as_tensor,
                terminal=False,
            )

        self.last_action = directions_to_values[action]
        self.last_state = state_as_tensor
        self.last_score = state.getScore()

        # Delay training by some number of steps so our replay cache has data
        self.train_step()

        # If we chose an illegal action, it's because those actions have a higher reward
        #  than moving legally, thus we'd rather just stop
        # Example: Pacman is in a corner and not moving is more advantageous than moving
        if action not in state.getLegalActions():
            action = Directions.STOP

        return action

    def get_state_tensor(self):
        raise NotImplementedError()

    @property
    def epsilon(self):
        if self.eps_schedule == "linear":
            return max(self.eps_end, self.eps_start - self.episode / self.eps_decay)

        if self.eps_schedule == "exponential":
            return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -self.episode / self.eps_decay
            )

    # Called when the game is over with the final state
    def final(self, state):

        if not self.is_training:
            self.training_stats[self.model_number * self.model_period]["wins"].append(
                state.isWin()
            )
            self.training_stats[self.model_number * self.model_period]["scores"].append(
                state.getScore()
            )
            self.save_testing_stats()
            return

        # This is solely done for convenience to create tensors in the replay
        state_as_tensor = self.get_state_tensor(state)

        # Save final state transitions
        self.replay.push(
            state=self.last_state,
            action=self.last_action,
            reward=100 if state.isWin() else state.getScore() - self.last_score,
            next_state=state_as_tensor,
            terminal=True,
        )

        self.train_step()

        # Kept for stats
        self.eps.append(self.epsilon)
        self.avg_losses.append(
            sum(self.episode_losses) / (len(self.episode_losses) + 1e-10)
        )

        if (
            self.episode % self.logging_period == 0
            and self.log_path is not None
            and self.episode < self.training_epsiodes
        ):
            # Update the stats file every episode
            self.save_training_stats()

            self.log_file.write(
                " | ".join(
                    [
                        f"Episode {self.episode:>5}",
                        f"Won: {str(state.isWin()):>5}",
                        f"Reward: {state.getScore():>6}",
                        f"Epsilon: {self.eps[-1]:.4f}",
                    ]
                )
                + "\n"
            )

        # Update target network parameters periodically
        if self.episode % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())

        if self.episode % self.model_period == 0 and self.model_dir is not None:
            torch.save(
                self.policy.state_dict(),
                self.model_dir / DEFAULT_POLICY_FILE_NAME.format(self.episode),
            )
            torch.save(
                self.target.state_dict(),
                self.model_dir / DEFAULT_TARGET_FILE_NAME.format(self.episode),
            )

    def registerInitialState(self, state):
        if not self.built:
            self.build(state)

        if not self.is_training and self.episode % self.nb_testing_episodes == 0:
            random.seed("cs188")
            np.random.seed(seed=0)
            torch.manual_seed(0)
            self.policy.load_state_dict(torch.load(self.model_paths[self.model_number]))
            self.model_number += 1

        self.episode += 1
        self.iteration = 0

        self.last_state = self.get_state_tensor(state)
        self.last_score = 0

        self.episode_losses = []

    def save_training_stats(self):
        if self.log_path is None:
            return
        stats = {}
        stats["loss_vs_iteration"] = self.all_losses
        stats["avg_loss"] = self.avg_losses
        stats["eps"] = self.eps
        with open(self.log_path / "training_stats.json", "w+") as f:
            json.dump(stats, f)

    def save_testing_stats(self):
        if self.log_path is None:
            return
        with open(self.log_path / "testing_stats.json", "w+") as f:
            json.dump(self.training_stats, f)

    def train_step(self):
        if not self.is_training:
            return

        if self.episode < self.train_start:
            self.all_losses.append(0)
            return

        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, terminal, indices, weights = batch

        # Q(S_t, A_t), we request Q(S_t) from our policy model and then
        #  index those values using the actions we actually took during the episode
        state_action_values = self.policy(states)
        state_action_values = state_action_values.gather(1, actions.unsqueeze(1))

        # We predict on all next states (S_t+1) for convenience
        target_state_action_values = self.target(next_states)
        # But if the next state is terminal, we don't want to use predicted Q values for it --- they should default to 0
        # We want to set all action values that have terminal next states to 0
        target_state_action_values[terminal] = 0
        # We now take the max of each sample
        target_state_action_values = target_state_action_values.detach().max(1)[0]

        # r + gamma * max_a Q(S_{t+1}, a)
        expected_state_action_values = rewards + self.gamma * target_state_action_values

        losses = self.loss_fn(
            state_action_values.squeeze(),
            expected_state_action_values,
            reduction="none",
        )
        weighted_losses = self.replay.weight_losses(losses, indices, weights)
        loss = torch.mean(weighted_losses)

        # Saved for stats
        self.all_losses.append(loss.item())
        self.episode_losses.append(loss.item())

        # Backprop step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ChannelDQNAgent(DQNAgent):
    """
      DQN that uses the converted TF Model and converts the state to a 6-channel tensor
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, state):
        # Create duplicate policy and target models
        self.policy = model_registry[self.model_type](
            width=state.data.layout.width,
            height=state.data.layout.height,
            in_channels=6,
        ).to(self.device)
        self.target = model_registry[self.model_type](
            width=state.data.layout.width,
            height=state.data.layout.height,
            in_channels=6,
        ).to(self.device)

        # Suggested optimization method
        # TODO: Maybe use a scheduled learning rate?
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(), lr=0.00025, alpha=0.95, eps=0.01
        )

        self.built = True

    def get_state_tensor(self, state):
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
        return image.permute(0, 2, 1).unsqueeze(0).type(torch.float32)


class ImageDQNAgent(DQNAgent):
    """
      Stanford DQN that uses the Stanford Model and converts the state to an image
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def build(self, state):
        # Create duplicate policy and target models
        self.policy = model_registry[self.model_type](
            width=state.data.layout.width,
            height=state.data.layout.height,
            in_channels=3,
        ).to(self.device)
        self.target = model_registry[self.model_type](
            width=state.data.layout.width,
            height=state.data.layout.height,
            in_channels=3,
        ).to(self.device)

        # Suggested optimization method
        # TODO: Maybe use a scheduled learning rate?
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(), lr=0.00025, alpha=0.95, eps=0.01
        )

        self.built = True

    def get_state_tensor(self, state):
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
        return image.permute(2, 1, 0).unsqueeze(0).type(torch.float32) / 255.0
