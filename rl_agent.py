from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from alphalock_env import AlphaLockEnv
from custom_ppo_policy import CustomPPOPolicy
import json

# Load word lists and frequencies
with open("4letter_word_freqs.json") as f:
    word_frequencies = json.load(f)
allowed_words = list(word_frequencies.keys())

# Define the secret code
secret_code = "rose"

# Create the AlphaLock environment
env = AlphaLockEnv(allowed_words, word_frequencies, secret_code)
env = make_vec_env(lambda: env, n_envs=1)

# Train PPO agent with custom policy
model = PPO(CustomPPOPolicy, env, verbose=1)
model.learn(total_timesteps=2)

# Save the trained model
model.save("ppo_alphalock_with_weights")

# Test the agent
obs = env.reset()
done = False
print("Testing the trained model...")
while not done:
    action, _ = model.predict(obs)
    alpha_beta = model.policy.get_alpha_beta()  # Get alpha and beta from the policy
    env.envs[0].set_alpha_beta(alpha_beta[0].item(), alpha_beta[1].item())  # Set alpha and beta in the environment
    obs, reward, done, _ = env.step(action)
    env.render()
