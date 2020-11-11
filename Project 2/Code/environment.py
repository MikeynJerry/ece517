import numpy as np

from mdp import Action, State


class Environment:
    def __init__(self, grid_size, reward_structure):
        self.grid_size = grid_size

        if reward_structure not in ['constant', 'dynamic']:
            raise ValueError("`reward_structure` must be one of ['constant', 'dynamic']")

        self.reward_structure = reward_structure

    def start(self):
        indices = np.random.choice(self.grid_size ** 2, size=2, replace=False)
        robot_pos, bomb_pos = map(tuple, np.unravel_index(indices, (self.grid_size, self.grid_size)))
        self.state = robot_pos, bomb_pos
        return self.state

    def move(self, action):
        reward = self.reward(action)
        robot_pos, bomb_pos = self.state
        next_robot_pos = tuple(np.clip(self.next_position(robot_pos, action), 0, self.grid_size - 1, dtype=int))

        # If the next robot position is the current bomb position, we've moved the bomb
        # If the bomb is outside the grid after moving, then the simulation is solved
        next_bomb_pos = bomb_pos
        if next_robot_pos == bomb_pos:
            next_bomb_pos = self.next_position(bomb_pos, action)
            x, y = next_bomb_pos
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                return (State.Terminal, State.Terminal), reward

        self.state = next_robot_pos, next_bomb_pos
        return self.state, reward

    def reward(self, action):
        # If we're using a constant reward structure
        if self.reward_structure == 'constant':
            return -1

        # If the next robot position isn't where the bomb is, we didn't move the bomb
        robot_pos, bomb_pos = self.state
        next_robot_pos = tuple(np.clip(self.next_position(robot_pos, action), 0, self.grid_size - 1, dtype=int))
        if next_robot_pos != bomb_pos:
            return -1

        # If the bomb was moved and the next bomb position is outside the grid, we won
        next_bomb_pos = self.next_position(bomb_pos, action)
        x, y = next_bomb_pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return 10

        # If we moved the bomb further away from the center
        if self.dist(next_bomb_pos) > self.dist(bomb_pos):
            return 1

        # We moved the bomb closer to the center
        return -1

    def dist(self, pos):
        center = (self.grid_size / 2 - 0.5, self.grid_size / 2 - 0.5)
        return np.linalg.norm(np.subtract(center, pos))

    def next_position(self, pos, action):
        x, y = pos

        if action == Action.E:
            x += 1

        if action == Action.W:
            x -= 1

        if action == Action.S:
            y += 1

        if action == Action.N:
            y -= 1

        return x, y
