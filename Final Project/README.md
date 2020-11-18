# Pacman DQN

## Code Locations

- `agents.py` contains our DQN agent
- `models.py` contains our DQN model
- `replay.py` contains our replay selectors

## Running Code

- Run `python play.py` to start a game

- Run `python play.py -p DQNAgent` to run a game using our agent

- Run `python play.py -p DQNAgent -l smallGrid -n <# episodes> -x <# training episodes> -f -q` to run a lot of games
  - `-l` chooses the layout and should be one of `{smallGrid, mediumGrid, mediumClassic}`
  - `-n` is the number of total episodes to run
  - `-x` is the number of episodes to run that are solely for training (no output is provided for these)
  - `-f` uses a fixed seed and should be set so we can recreate results
  - `-q` causes output of the `n - x` non-training episodes to be text only (`e.g. Pacman died! Score: ___`)
    - Remove this option if you want to see our model's actions played out on the actual board
