"""
Classes for playing the game
"""


import torch
import matplotlib.pyplot as plt

from pacman import Directions
import game

from dqnModels import DQNModel

class Agent(game.Agent):
  def __init__(self):
    super().__init__()
    self.device = "cuda"

    # White
    self.capsule_color = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
    # Green
    self.food_color = torch.tensor([0, 128, 0], dtype=torch.uint8, device=self.device)
    # Blue
    self.wall_color = torch.tensor([0, 0, 255], dtype=torch.uint8, device=self.device)
    # Yellow
    self.pacman_color = torch.tensor([255, 255, 0], dtype=torch.uint8, device=self.device)
    # Red
    self.ghost_color = torch.tensor([255, 0, 0], dtype=torch.uint8, device=self.device)
    # Gray
    self.ghost_scared_color = torch.tensor([128, 128, 128], dtype=torch.uint8, device=self.device)

  def state_to_image(self, state):
    """
      IMPORTANT: State is indexed based on the bottom left, i.e. bottom left = (0, 0)
    """

    # List of (x, y) tuples and represent the 'large' dots
    capsules = state.getCapsules()

    # (width, height) array where True values represent locations of 'small' dots
    food = torch.tensor(state.getFood().data, device=self.device)

    # (width, height) array where True values represent locations of 'walls'
    wall_grid = state.getWalls()
    walls = torch.tensor(wall_grid.data, device=self.device)

    # (x, y) tuple of Pacman's positon
    pacman = state.getPacmanPosition()

    # List of (x, y) tuples representing each ghost's position
    ghosts = state.getGhostPositions()
    # List of True/False values to determine if the ghosts are scared (Pacman ate a capsule)
    ghost_states = [ghost.scaredTimer > 0 for ghost in state.getGhostStates()]

    # Represent the board as a (width, height, 3) image
    image = torch.zeros((wall_grid.width, wall_grid.height, 3), dtype=torch.uint8, device=self.device)

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

    return image


class DQNAgent(Agent):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.model = DQNModel().to(self.device)

    def getAction(self, state):
        # Shape = (Width, Height, Channels)
        state_as_image = self.state_to_image(state)

        # Model expects (Batch Size, Channels, Height, Width)
        self.model(
          state_as_image.permute(2, 1, 0).unsqueeze(0).type(torch.float32)
        )

        # Basic set of actions to cause Pacman to eat a capsule
        action = [
          Directions.EAST,
          Directions.EAST,
          Directions.EAST,
          Directions.EAST,
          Directions.NORTH,
          Directions.NORTH,
          Directions.EAST,
          Directions.EAST,
          Directions.SOUTH,
          Directions.SOUTH,
          Directions.EAST,
          Directions.EAST,
          Directions.EAST,
          Directions.NORTH
        ][self.i]
        self.i += 1
        if self.i > 13:
            plt.imshow(state_as_image.permute(1, 0, 2).cpu(), origin='lower')
            plt.show()
        return action