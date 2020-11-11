from environment import Environment
from mdp import Action, State

grid_size = 8

envs = [
    Environment(grid_size, 'constant'),
    Environment(grid_size, 'dynamic')
]

move_sets = [
    [
        Action.E,
        Action.E,
        Action.N,
        Action.W,
        Action.W
    ],
    [
        Action.E,
        Action.N,
        Action.N,
        Action.N,
        Action.N,
        Action.N
    ]
]

starting_state = (0, 5), (1, 4)
for env in envs:
    for move_set in move_sets:
        env.state = starting_state
        robot_pos, bomb_pos = starting_state
        total_reward = 0
        for move in move_set:
            (robot_pos, bomb_pos), reward = env.move(move)
            total_reward += reward
            if robot_pos == State.Terminal:
                break
        print(total_reward, env.reward_structure, move_set)