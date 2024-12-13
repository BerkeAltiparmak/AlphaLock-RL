from stable_baselines3 import PPO
from alphalock_env import AlphaLockEnv
from custom_ppo_policy import CustomPPOPolicy
import json

# Load word lists and frequencies
with open("4letter_word_freqs.json") as f:
    word_frequencies = json.load(f)
allowed_words = list(word_frequencies.keys())

# Define the secret code
secret_code = "rose"

# Create the environment
env = AlphaLockEnv(allowed_words, word_frequencies, secret_code)

# Train PPO agent with the Custom PPO Policy
model = PPO(CustomPPOPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_alphalock_with_weights")

# Test the trained model
obs = env.reset()
done = False
print("Testing the trained model...")
while not done:
    action, _ = model.predict(obs)
    alpha_beta = model.policy.get_alpha_beta().detach().numpy()
    alpha, beta = alpha_beta[0, 0], alpha_beta[0, 1]

    print(f"Alpha: {alpha}, Beta: {beta}")

    obs, reward, done, _ = env.step(action)
    env.render()
