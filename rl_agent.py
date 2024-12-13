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
cumulative_reward = 0
print("Testing the trained model...")
while not done:
    action, _ = model.predict(obs)
    if action[0] < len(env.envs[0].unwrapped.current_pool):  # Ensure valid action
        word_guess = env.envs[0].unwrapped.current_pool[action[0]]
    else:
        print("Invalid action index:", action[0])
        word_guess = "Invalid"
    print(f"Action (word index): {action}, Predicted Word: {word_guess}")

    alpha_beta = model.policy.get_alpha_beta().detach().numpy()
    alpha, beta = alpha_beta[0, 0], alpha_beta[0, 1]
    print(f"Alpha: {alpha}, Beta: {beta}")

    env.envs[0].unwrapped.set_alpha_beta(alpha, beta)
    obs, reward, done, _ = env.step(action)
    cumulative_reward += reward
    env.render()

    print(f"Remaining Pool Size: {len(env.envs[0].unwrapped.current_pool)}")

print(f"Cumulative Reward: {cumulative_reward}")




