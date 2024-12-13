import os
import gym
from stable_baselines3 import PPO
from rl_env import AlphaLockEnv

# Load data
with open("4letter_word_freqs.json", "r") as f:
    word_frequencies = json.load(f)

allowed_words = list(word_frequencies.keys())
solution = "sync"  # Example solution

# Initialize environment
env = AlphaLockEnv(allowed_words, solution)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/alphalock_rl")