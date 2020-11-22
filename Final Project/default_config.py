DEFAULT_GAMMA = 0.95
DEFAULT_EPS_START = 1
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY = 7500 
DEFAULT_EPS_SCHEDULE = 'linear'

DEFAULT_TRAIN_START = 300
DEFAULT_TARGET_UPDATE = 100

DEFAULT_LOGGING_PERIOD = 10

DEFAULT_POLICY_FILE_NAME = "policy_episode_{:05d}.th"
DEFAULT_TARGET_FILE_NAME = "target_episode_{:05d}.th"

DEFAULT_MODEL_TYPE = 'small'
DEFAULT_MODEL_PERIOD = 500
DEFAULT_BATCH_SIZE = 32

DEFAULT_LOG_ROOT = "experiments"
DEFAULT_MODEL_ROOT = "saved_models"

DEFAULT_ALPHA_EXPERIENCE_REPLAY = 0.6
DEFAULT_BETA_EXPERIENCE_REPLAY = 0.4
DEFAULT_REPLAY = "basic"
