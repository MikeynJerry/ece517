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

## Agent Command Line Arguments

All of the defaults can be changed in `default_config.py`

### Epsilon Arguments

- `--gamma <gamma>`
  - Discount factor when calculating the return of the next state
  - Default: `DEFAULT_GAMMA = 0.95`
- `--eps-start <eps_start>`
  - The `eps` value to start training at
  - Default: `DEFAULT_EPS_START = 1`
- `--eps-end <eps_end>`
  - The `eps` value to end training at
  - Default: `DEFAULT_EPS_END = 0.05`
- `--eps-decay <num_episodes>`
  - The rate at which `eps` goes down
  - Default: `DEFAULT_EPS_DECAY = 7500`
- `--eps-schedule <schedule>`
  - Whether to use a linear or exponential `eps` decay
  - Default: `DEFAULT_EPS_SCHEDULE = linear`

### Training Arguments

- `--train-start <episode_num>`
  - The episode to start training on
  - Default: `DEFAULT_TRAIN_START = 300`
- `--target-update <num_episodes>`
  - Copy the policy weights to the target network every `<num_episodes>`
  - Default: `DEFAULT_TARGET_UPDATE = 100`

### Stats and Logging Arguments

- `--log-dir <log_dir>`
  - The directory to save stats files to
  - If you omit this, then no stats are saved
  - The directory root is `DEFAULT_LOG_ROOT = experiments`
  - The place stats will be saved is then `<DEFAULT_LOG_ROOT>/<log_dir>`
- `--log-frequency <num_episodes>`
  - Save stats every `<num_episodes>`
  - Default: `DEFAULT_LOGGING_PERIOD = 10`

### Model Arguments

- `--model-type <model_type>`
  - Pick which type of model to use
  - New models can be added using the `@register(model_type)` decorator in `models.py`
  - Default: `DEFAULT_MODEL_TYPE = small`
- `--model-dir <model_dir>`
  - The directory to save model files to
  - If you omit this, then no model files are saved
  - The directory root is `DEFAULT_MODEL_ROOT = saved_models`
  - The place models will be saved is then `<DEFAULT_MODEL_ROOT>/<model_dir>`
  - The filenames for models are `DEFAULT_POLICY_FILE_NAME = policy_episode_{:05d}.th` and `DEFAULT_TARGET_FILE_NAME = target_episode_{:05d}.th`
- `--model-save-rate <num_episodes>`
  - Save policy and target models every `<num_episodes>`
  - Default: `DEFAULT_MODEL_PERIOD = 500`
- `--batch-size <batch_size>`
  - The size of learning batches to sample from the replay
  - Default: `DEFAULT_BATCH_SIZE = 32`

### Replay Arguments

- `--replay-type <replay_type>`
  - The type of replay memory to use
  - New replays can be added using the `@register(replay_type)` decorator in `replays.py`
  - Default: `DEFAULT_REPLAY = "basic"`
- `--replay-alpha <replay_alpha>`
  - The value of alpha in the priority replay formula.
  - Default: `DEFAULT_ALPHA_EXPERIENCE_REPLAY = 0.9`
- `--replay-beta <replay_beta>`
  - The value of beta in the priority replay formula.
  - Default: `DEFAULT_BETA_EXPERIENCE_REPLAY = 0.4`

### Loss and Optimizer Arguments

- `--loss-type <loss_type>`
  - The type of loss to use
  - Default: `DEFAULT_LOSS_TYPE = "huber"`

## Testing Arguments

- `--from-experiment <log_dir>`
  - The directory containing a `config.json` file for a ran experiment
- `--nb-testing-episodes <nb_testing_episodes>`
  - The number of episodes to run on each model saved from the experiment defined by `<log_dir>/config.json`
