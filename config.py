# RL Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99

# Environment Settings
MAX_ATTEMPTS = 10
REWARD_SCALING = "exponential"

# Reward Settings
SUCCESS_REWARD = 100
INTERMEDIATE_SCALING = 10
FAILURE_PENALTY = -100

# Data Paths
WORD_FREQS_FILE = "data/4letter_word_freqs.json"
ALPHA_BETA_MAPPING_FILE = "rl_stats/alpha_beta_mapping.json"
EPISODE_REWARDS_FILE = "rl_stats/episode_rewards.json"
EPISODE_GUESSES_FILE = "rl_stats/episode_guesses.json"
MODEL_PATH = "rl_model/trained_rl_agent.pth"