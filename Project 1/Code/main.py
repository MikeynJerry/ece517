import argparse
from enum import IntEnum
from itertools import product

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('num_spaces', metavar='m', type=int)
parser.add_argument('p_min', type=float)
parser.add_argument('p_max', type=float)
parser.add_argument('p_same', type=float)
parser.add_argument('t_drive', type=int)
parser.add_argument('t_walk', type=int)
parser.add_argument('t_wait', type=int)
parser.add_argument('r_no', type=int)
args = parser.parse_args()


Q = np.zeros((args.num_spaces, 2, 3))
V = np.zeros((args.num_spaces, 2))

p = np.linspace(args.p_min, args.p_max, args.num_spaces)


class Action(IntEnum):
    WAIT = 0
    PARK = 1
    NEXT = 2


class Status(IntEnum):
    EMPTY = 0
    FULL = 1


def reward(state, action):
    space, status = state
    if action == Action.WAIT:
        return -args.t_wait
    elif action == Action.PARK:
        if status == Status.EMPTY:
            return -args.t_walk * (args.num_spaces - space - 1)
        elif status == Status.FULL:
            return -args.r_no
    elif action == Action.NEXT:
        if space == args.num_spaces - 1:
            return -args.r_no
        else:
            return -args.t_drive


def value_iteration(v, q):
    delta = 1
    threshold = 1e-6
    max_iters = 100
    iters = 0
    while delta > threshold and iters < max_iters:
        last_v = np.copy(v)
        for space, status, action in product(range(args.num_spaces), Status, Action):
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
                if space != args.num_spaces - 1:
                    q[space, status, action] = (1 - p[space + 1]) * same_next_status(status) + p[space + 1] * diff_next_status(status, Status.FULL)
                else:
                    q[space, status, action] = terminal(status, action)

            if status == Status.FULL and action == Action.NEXT:
                if space != args.num_spaces - 1:
                    q[space, status, action] = p[space + 1] * same_next_status(status) + (1 - p[space + 1]) * diff_next_status(status, Status.EMPTY)
                else:
                    q[space, status, action] = terminal(status, action)

        v = np.max(q, axis=-1)

        delta = np.max(np.abs(v - last_v))
        iters += 1

    return v, q


def policy(q):
    return np.argmax(q, axis=-1)


new_v, new_q = value_iteration(V, Q)

print('Learned Value Function')
for space in range(1, args.num_spaces+1):
    print(f'At space {space}')
    print(f'    If space is full: {new_v[space-1, Status.FULL]:.2f}')
    print(f'    If space is empty: {new_v[space-1, Status.EMPTY]:.2f}')

pi = policy(new_q)
print('Learned Policy Pi')
for space in range(1, args.num_spaces+1):
    print(f'At space {space}')
    print(f'    If space is full:', Action(pi[space-1, Status.FULL]))
    print(f'    If space is empty:', Action(pi[space-1, Status.EMPTY]))