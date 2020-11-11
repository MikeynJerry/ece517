from enum import IntEnum


class Action(IntEnum):
    N = 0
    S = 1
    E = 2
    W = 3


class State(IntEnum):
    Terminal = -1