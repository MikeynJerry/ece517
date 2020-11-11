import numpy as np
from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt
from time import time

from mdp import Action, State


class Learner:
    def __init__(self, alpha, eps, env, grid_size, max_iters):
        self.alpha = alpha
        self.eps = eps
        self.env = env
        self.grid_size = grid_size
        self.max_iters = max_iters

        # Create Q and policy for each state, action pair
        state_action_space = (grid_size, grid_size, grid_size, grid_size, len(Action))
        self.Q = np.zeros(state_action_space)
        self.policy = np.full(state_action_space, 1 / len(Action))

        # This is to speed up making the same probability arrays each iteration
        self.policy_lookup = np.full((len(Action), len(Action)), eps / len(Action))
        np.fill_diagonal(self.policy_lookup, 1 - eps + eps / len(Action))

    def episode(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

    def run(self, nb_episodes):
        rewards, times = [], []
        for i in range(nb_episodes):
            start = time()
            rewards.append(self.learn())
            end = time()
            times.append(end - start)
        return rewards, times

    def final_policy(self):
        raise NotImplementedError()

    def plot(self, starting_state, title):
        self.env.state = starting_state
        robot_pos, bomb_pos = starting_state
        board = np.zeros((self.env.grid_size, self.env.grid_size))

        policy = self.final_policy()

        positions = []
        for i in range(20):
            probabilities = policy[tuple([*robot_pos, *bomb_pos])]
            action = Action(np.random.choice(np.arange(len(probabilities)), p=probabilities))

            (next_robot_pos, next_bomb_pos), reward = self.env.move(action)
            positions.append((robot_pos, bomb_pos, action))
            if next_robot_pos == State.Terminal:
                positions.append((next_robot_pos, next_bomb_pos, action))
                break
            robot_pos, bomb_pos = next_robot_pos, next_bomb_pos

        fig, axes = plt.subplots((len(positions) - 1) // 3 + 1, 3, figsize=(8, 3 * len(positions) // 3))
        for i in range(len(axes.ravel())):
            ax = axes[i // 3, i % 3]
            if i >= len(positions):
                ax.set_visible(False)
                continue
            robot_pos, bomb_pos, action = positions[i]
            current_board = np.copy(board)
            if robot_pos == State.Terminal:
                last_robot_pos, last_bomb_pos, action = positions[i - 1]
                robot_pos = self.env.next_position(last_robot_pos, action)
            else:
                current_board[bomb_pos] = 20

            current_board[robot_pos] = 10
            ax.imshow(current_board.T, vmin=0, vmax=20)
            ax.set_title(f"Time step {i+1}")
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        plt.suptitle(title)
        plt.savefig(title.lower().replace(' ', '-'), dpi=300)
        plt.show()

    def q_values(self, starting_state):
        self.env.state = starting_state
        robot_pos, bomb_pos = starting_state
        policy = self.final_policy()
        for i in range(20):
            q = self.Q[tuple([*robot_pos, *bomb_pos])]
            print(f"Q((robotX: {robot_pos[0]}, robotY: {robot_pos[1]}), (bombX: {bomb_pos[0]}, bombY: {bomb_pos[1]})) = [N: {q[Action.N]:.2f}, E: {q[Action.E]:.2f}, S: {q[Action.S]:.2f}, W: {q[Action.W]:.2f}]")
            probabilities = policy[tuple([*robot_pos, *bomb_pos])]
            action = Action(np.random.choice(np.arange(len(probabilities)), p=probabilities))
            (next_robot_pos, next_bomb_pos), reward = self.env.move(action)
            if next_robot_pos == State.Terminal:
                break
            robot_pos, bomb_pos = next_robot_pos, next_bomb_pos


TimeStep = namedtuple('TimeStep', ['state', 'action', 'reward'])
class MCLearner(Learner):
    def __init__(self, env, grid_size, max_iters, eps, alpha):
        super().__init__(
            alpha=alpha,
            eps=eps,
            env=env,
            grid_size=grid_size,
            max_iters=max_iters
        )

    def learn(self):
        timesteps = []
        robot_pos, bomb_pos = self.env.start()

        # Generate Episode
        for i in range(self.max_iters):
            # Select which action to take based on pi
            probabilities = self.policy[tuple([*robot_pos, *bomb_pos])]
            action = Action(np.random.choice(np.arange(len(probabilities)), p=probabilities))

            # Move the robot
            (next_robot_pos, next_bomb_pos), reward = self.env.move(action)

            # Save the timestep for updating Q
            timesteps.append(
                TimeStep(state=(robot_pos, bomb_pos), action=action, reward=reward)
            )

            # If the next robot position is the terminal state, then we're done with the episode
            if next_robot_pos == State.Terminal:
                break

            # Update the robot and bomb position for the next step
            robot_pos, bomb_pos = next_robot_pos, next_bomb_pos

        # Keep a list of all state, action pairs we've seen
        seen = [(timestep.state, timestep.action) for timestep in timesteps]
        G = 0

        # Iterate over the timesteps backwards
        for t, timestep in zip(reversed(range(len(timesteps))), reversed(timesteps)):

            # Keep a sum of the total return from T-1 to t
            G = G + timestep.reward

            # If we've seen this state, action pair before then this isn't the first visit
            if (timestep.state, timestep.action) in seen[:t]:
                continue

            robot_pos, bomb_pos = timestep.state
            current_ind = tuple([*robot_pos, *bomb_pos, timestep.action])

            # Update Q
            self.Q[current_ind] = self.Q[current_ind] + self.alpha * (G - self.Q[current_ind])

            # Update the policy
            best_action = Action(np.argmax(self.Q[tuple([*robot_pos, *bomb_pos])]))
            self.policy[tuple([*robot_pos, *bomb_pos])] = self.policy_lookup[best_action]

        return G

    def final_policy(self):
        return self.policy


class QLearner(Learner):
    def __init__(self, env, grid_size, policy_type, alpha, max_iters, eps):
        super().__init__(
            alpha=alpha,
            eps=eps,
            env=env,
            grid_size=grid_size,
            max_iters=max_iters
        )

        if policy_type not in ['greedy', 'uniform']:
            raise ValueError("`policy_type` must be one of ['greedy', 'uniform']")

        self.policy_type = policy_type

    def learn(self):
        reward_sum = 0
        robot_pos, bomb_pos = self.env.start()
        for i in range(self.max_iters):
            # Select which action to take based on pi
            probabilities = self.policy[tuple([*robot_pos, *bomb_pos])]
            action = Action(
                np.random.choice(np.arange(len(probabilities)), p=probabilities)
            )

            # Move the robot
            (next_robot_pos, next_bomb_pos), reward = self.env.move(action)

            # Add the reward for that step to our total reward
            reward_sum += reward

            # Update Q based on the current state, action and the next state, action
            current_ind = tuple([*robot_pos, *bomb_pos, action])

            if next_robot_pos == State.Terminal:
                next_Q = 0
            else:
                next_ind = tuple([*next_robot_pos, *next_bomb_pos])
                next_Q = np.max(self.Q[next_ind])
            self.Q[current_ind] = self.Q[current_ind] + self.alpha * (reward + next_Q - self.Q[current_ind])

            # If we're using a greedy policy (over uniform), update our policy
            if self.policy_type == 'greedy':
                best_action = Action(
                    np.argmax(self.Q[tuple([*robot_pos, *bomb_pos])])
                )
                self.policy[tuple([*robot_pos, *bomb_pos])] = self.policy_lookup[best_action]

            # If the next robot position is the terminal state, then we're done with the episode
            if next_robot_pos == State.Terminal:
                break

            # Update the robot and bomb position for the next step
            robot_pos, bomb_pos = next_robot_pos, next_bomb_pos

        return reward_sum

    def final_policy(self):
        policy = np.zeros_like(self.policy)
        for a, b, c, d in product(range(self.env.grid_size), repeat=4):
            policy[(a, b, c, d, np.argmax(self.Q[(a, b, c, d)]))] = 1

        return policy