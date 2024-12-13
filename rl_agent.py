import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from alphalock_env import AlphaLockEnv
from custom_ppo_policy import CustomPPOPolicy

# Load word lists and frequencies
with open("4letter_word_freqs.json") as f:
    word_frequencies = json.load(f)
allowed_words = list(word_frequencies.keys())

# Define the secret code
secret_code = "sync"

# Create the AlphaLock environment
env = AlphaLockEnv(allowed_words, word_frequencies, secret_code)
env = make_vec_env(lambda: env, n_envs=1)

# Train PPO agent with custom policy
model = PPO(CustomPPOPolicy, env, verbose=1)
model.learn(total_timesteps=10)

# Save the trained model
model.save("ppo_alphalock_with_weights")

# Test the agent
obs = env.reset()
done = False
while not done:
    action, alpha_beta = model.predict(obs)
    obs, reward, done, _ = env.step(action, alpha_beta[0], alpha_beta[1])
    env.render()
