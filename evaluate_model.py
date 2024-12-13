from stable_baselines3 import PPO
from rl_env import AlphaLockEnv

# Load model
model = PPO.load("models/alphalock_rl")

# Initialize environment
env = AlphaLockEnv(allowed_words, solution)

# Evaluate performance
for _ in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
