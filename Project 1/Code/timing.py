import argparse
from enum import IntEnum
from itertools import product
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('p_min', type=float)
parser.add_argument('p_max', type=float)
parser.add_argument('p_same', type=float)
parser.add_argument('t_drive', type=int)
parser.add_argument('t_walk', type=int)
parser.add_argument('t_wait', type=int)
parser.add_argument('r_no', type=int)
args = parser.parse_args()


class Action(IntEnum):
    WAIT = 0
    PARK = 1
    NEXT = 2


class Status(IntEnum):
    EMPTY = 0
    FULL = 1


def reward_decorator(num_spaces):
    def reward_with_spaces(state, action):
        space, status = state
        if action == Action.WAIT:
            return -args.t_wait
        elif action == Action.PARK:
            if status == Status.EMPTY:
                return -args.t_walk * (num_spaces - space - 1)
            elif status == Status.FULL:
                return -args.r_no
        elif action == Action.NEXT:
            if space == num_spaces - 1:
                return -args.r_no
            else:
                return -args.t_drive
    return reward_with_spaces


def value_iteration(v, q, p, num_spaces):
    reward = reward_decorator(num_spaces)
    delta = 1
    threshold = 1e-6
    max_iters = 100
    iters = 0
    while delta > threshold and iters < max_iters:
        last_v = np.copy(v)
        for space, status, action in product(range(num_spaces), Status, Action):
            def same_status(status):
                return args.p_same * (reward((space, status), action) + v[space, status])

            def diff_status(old, new):
                return (1 - args.p_same) * (reward((space, old), action) + v[space, new])

            if status == Status.EMPTY and action == Action.WAIT:
                q[space, status, action] = same_status(status) + diff_status(status, Status.FULL)

            if status == Status.FULL and action == Action.WAIT:
                q[space, status, action] = same_status(status) + diff_status(status, Status.EMPTY)

            def terminal(status, action):
                return reward((space, status), action)

            if status == Status.EMPTY and action == Action.PARK:
                q[space, status, action] = terminal(status, action)

            if status == Status.FULL and action == Action.PARK:
                q[space, status, action] = terminal(status, action)

            def same_next_status(status):
                return reward((space, status), action) + v[space + 1, status]

            def diff_next_status(old, new):
                return reward((space, old), action) + v[space + 1, new]

            if status == Status.EMPTY and action == Action.NEXT:
                if space != num_spaces - 1:
                    q[space, status, action] = (1 - p[space + 1]) * same_next_status(status) + p[space + 1] * diff_next_status(status, Status.FULL)
                else:
                    q[space, status, action] = terminal(status, action)

            if status == Status.FULL and action == Action.NEXT:
                if space != num_spaces - 1:
                    q[space, status, action] = p[space + 1] * same_next_status(status) + (1 - p[space + 1]) * diff_next_status(status, Status.EMPTY)
                else:
                    q[space, status, action] = terminal(status, action)

        v = np.max(q, axis=-1)

        delta = np.max(np.abs(v - last_v))
        iters += 1

    return v, q


def policy(q):
    return np.argmax(q, axis=-1)


steps = range(10, 1000, 5)
times = []
for spaces in tqdm(steps):
    start = time.time()
    q = np.zeros((spaces, 2, 3))
    v = np.zeros((spaces, 2))
    p = np.linspace(args.p_min, args.p_max, spaces)
    value_iteration(v, q, p, spaces)
    end = time.time()
    times.append(end-start)


plt.figure(figsize=(16, 12))
plt.title('Time vs Number of Parking Spaces')
plt.xlabel('Number of Parking Spaces')
plt.ylabel('Time (s)')
plt.plot(steps, times, label='Time')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('time_vs_spaces.png', dpi=300)

plt.figure(figsize=(16, 12))
plt.title('Average Time per Space')
plt.xlabel('Number of Parking Spaces')
plt.ylabel('Time (s) per Space')
plt.plot(steps, [t / s for t, s in zip(times, steps)], label='Time per Space')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('time_per_space.png', dpi=300)

plt.figure(figsize=(16, 12))
plt.title('Time vs Number of Parking Spaces')
plt.xlabel('Number of Parking Spaces')
plt.ylabel('Time (s)')
plt.plot(steps, times, label='Time')
plt.plot(steps, [(x / 100) for x in steps], label='$0.01 \cdot x$')
plt.plot(steps, [(x / (1 / 0.0055)) for x in steps], label='$0.0055 \cdot x$')
plt.plot(steps, [(x / 1000) for x in steps], label='$0.001 \cdot x$')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('time_vs_spaces_with_estimates.png', dpi=300)
