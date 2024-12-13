import os
import time
import numpy as np
import json
from rl_environment import AlphalockEnvironment
from rl_agent import RLAgent
from reward_calculator import select_best_word
from utils import flatten_state, load_json, save_json, load_alpha_beta_mapping, save_alpha_beta_mapping

def train_agent(episodes=1000, batch_size=4, state_dim=3, hidden_dim=128, lr=0.001, gamma=0.99, model_path="trained_rl_agent.pth"):
    """
    Train the RL agent on the Alphalock game.

    Parameters:
    - episodes (int): Number of training episodes.
    - batch_size (int): Number of episodes per policy update.
    - state_dim (int): Dimension of the state space.
    - hidden_dim (int): Number of hidden units in the policy network.
    - lr (float): Learning rate for the RL agent.
    - gamma (float): Discount factor.
    - model_path (str): Path to the saved model file.

    Returns:
    - RLAgent: Trained RL agent.
    """
    # Initialize the environment
    env = AlphalockEnvironment()
    
    # Initialize the agent
    agent = RLAgent(state_dim, hidden_dim, lr, gamma)

    # Load the model if it exists
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading...")
        agent.load_model(model_path)
        print("Pretrained model loaded successfully.")
    else:
        print(f"No existing model found at {model_path}. Starting fresh training.")

    alpha, beta = 1.0, 0  # Initial alpha and beta values (defined to explore early)

    # Load existing JSON files if they exist
    alpha_beta_mapping = load_alpha_beta_mapping("alpha_beta_mapping.json")
    episode_rewards = load_json("episode_rewards.json")
    episode_guesses = load_json("episode_guesses.json")

    # Adjust episode numbers based on existing data
    final_episode_in_file = max(map(int, episode_rewards.keys()), default=0)

    batch_rewards = []
    batch_guesses = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_guesses = 0
        episode_total_rewards = []

        current_episode = final_episode_in_file + episode + 1

        print(f"Episode {current_episode}/{final_episode_in_file + episodes}")

        while not done:
            start_time = time.time()
            
            if len(env.feedback_history) == 0:  # First guess
                guess = "sera"  # Precomputed optimal first guess through Information Theory
            else:
                # Flatten state and select action
                flat_state = flatten_state(state)
                action = agent.select_action(flat_state)
                alpha, beta = action

                # Store alpha, beta values mapped to state keys
                # Convert tuple keys to unique strings
                alpha_beta_mapping[f"{current_episode}:{10 - state['attempts_remaining']},{state['pool_entropy']}"] = {
                    "alpha": alpha,
                    "beta": beta
                }

                # Pick the best word using the score calculator
                guess = select_best_word(
                    env.allowed_words,
                    env.possible_words,
                    env.word_frequencies,
                    alpha,
                    beta
                )

            # Step through the environment
            next_state, reward, done = env.step(guess, alpha, beta)
            agent.store_reward(reward)
            episode_total_rewards.append(reward)
            total_guesses += 1
            end_time = time.time()

            # Log guess details
            print(f"Episode {current_episode}, Guess #{len(env.feedback_history)}:")
            print(f"  Word: {guess}")
            print(f"  Alpha: {alpha}, Beta: {beta}")
            print(f"  Pool Size: {len(env.possible_words)}")
            print(f"  Word to Solve: {env.solution}")
            print(f"  Time Taken: {end_time - start_time:.4f} seconds")

            state = next_state

        # Log the results of the episode
        total_reward = sum(episode_total_rewards)
        print(f"Episode {current_episode} finished with total reward: {total_reward}")
        print(f"Total guesses needed: {total_guesses}")

        # Store episode data
        episode_rewards[current_episode] = total_reward
        episode_guesses[current_episode] = total_guesses
        batch_rewards.append(total_reward)
        batch_guesses.append(total_guesses)

        # Perform policy update after a batch of episodes
        if (episode + 1) % batch_size == 0:
            agent.update_policy()
            avg_batch_reward = np.mean(batch_rewards)
            avg_batch_guesses = np.mean(batch_guesses)
            print(f"Policy updated after batch of {batch_size} episodes.")
            print(f"Average reward in last batch: {avg_batch_reward}")
            print(f"Average guess in last batch: {avg_batch_guesses}")
            batch_rewards = []  # Reset batch rewards
            batch_guesses = []  # Reset batch guesses

    # Save updated JSON files
    save_alpha_beta_mapping(alpha_beta_mapping, "alpha_beta_mapping.json")
    save_json(episode_rewards, "episode_rewards.json")
    save_json(episode_guesses, "episode_guesses.json")

    return agent

if __name__ == "__main__":
    # Train the agent
    trained_agent = train_agent(episodes=1, batch_size=4)
    trained_agent.save_model("trained_rl_agent.pth")
