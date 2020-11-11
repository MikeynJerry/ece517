import argparse
import matplotlib.pyplot as plt
import numpy as np

from environment import Environment
from learners import MCLearner, QLearner


parser = argparse.ArgumentParser()
parser.add_argument(
    "grid_size",
    metavar="d",
    type=int
)
parser.add_argument(
    "alpha",
    metavar="a",
    type=float
)
parser.add_argument(
    "epsilon",
    metavar="e",
    type=float
)
parser.add_argument(
    "nb_episodes",
    metavar="n",
    type=int
)
parser.add_argument(
    "learner",
    choices=[1, 2],
    metavar="m",
    type=int
)
args = parser.parse_args()

environment = Environment(
    grid_size=args.grid_size,
    reward_structure='dynamic'
)
if args.learner == 1:
    learner = MCLearner(
        env=environment,
        grid_size=args.grid_size,
        max_iters=1000,
        eps=args.eps,
        alpha=args.alpha
    )
elif args.learner == 2:
    learner = QLearner(
        env=environment,
        grid_size=args.grid_size,
        policy_type='greedy',
        alpha=args.alpha,
        max_iters=1000,
        eps=args.epsilon
    )

rewards, times = learner.run(args.nb_episodes)

plt.figure(figsize=(10, 7.5))
plt.plot(range(len(rewards)), rewards)
plt.title('Reward vs Episode')
plt.show()

avg_rewards = [r / (i + 1) for i, r in enumerate(np.cumsum(rewards))]
plt.figure(figsize=(10, 7.5))
plt.plot(range(len(avg_rewards)), avg_rewards)
plt.title('Average Reward vs Episode')
plt.show()

plt.figure(figsize=(10, 7.5))
plt.plot(range(len(times)), np.cumsum(times))
plt.title('Time vs Episode')
plt.show()

starting_state = (0, 5), (1, 4)
learner.plot(starting_state, 'Robot Path - Robot = Blue-Green, Bomb = Yellow')
learner.q_values(starting_state)