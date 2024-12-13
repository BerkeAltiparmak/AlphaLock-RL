import time
import numpy as np
import json
from rl_environment import AlphalockEnvironment
from rl_agent import RLAgent
from reward_calculator import select_best_word

def flatten_state(state):
    """
    Flatten the state dictionary into a single-dimensional list.
    
    Parameters:
    - state (dict): State dictionary from the environment.
    
    Returns:
    - list: Flattened state representation.
    """
    return [
        state["pool_entropy"],  # Numeric value
        state["attempts_remaining"],  # Numeric value
        len(state["feedback_history"]),  # Derived numeric value
    ]

def train_agent(episodes=1000, batch_size=4, state_dim=3, hidden_dim=128, lr=0.001, gamma=0.99):
    """
    Train the RL agent on the Alphalock game.

    Parameters:
    - episodes (int): Number of training episodes.
    - batch_size (int): Number of episodes per policy update.
    - state_dim (int): Dimension of the state space.
    - hidden_dim (int): Number of hidden units in the policy network.
    - lr (float): Learning rate for the RL agent.
    - gamma (float): Discount factor.

    Returns:
    - RLAgent: Trained RL agent.
    """
    # Initialize the environment and the agent
    env = AlphalockEnvironment()
    agent = RLAgent(state_dim, hidden_dim, lr, gamma)
    alpha, beta = 0.9, 0.1  # Initial alpha and beta values (defined to explore early)

    # Data tracking
    alpha_beta_mapping = {}
    episode_rewards = {}
    episode_guesses = {}
    batch_rewards = []
    batch_guesses = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_guesses = 0
        episode_total_rewards = []

        print(f"Episode {episode + 1}/{episodes}")

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
                alpha_beta_mapping[(state["pool_entropy"], state["attempts_remaining"])] = {
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
            print(f"Episode {episode + 1}, Guess #{len(env.feedback_history)}:")
            print(f"  Word: {guess}")
            print(f"  Alpha: {alpha}, Beta: {beta}")
            print(f"  Pool Size: {len(env.possible_words)}")
            print(f"  Word to Solve: {env.solution}")
            print(f"  Time Taken: {end_time - start_time:.4f} seconds")

            state = next_state

        # Log the results of the episode
        total_reward = sum(episode_total_rewards)
        print(f"Episode {episode} finished with total reward: {total_reward}")
        print(f"Total guesses needed: {total_guesses}")

        # Store episode data
        episode_rewards[episode] = total_reward
        episode_guesses[episode] = total_guesses
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

    # Save data to JSON files
    with open("alpha_beta_mapping.json", "w") as f:
        json.dump(alpha_beta_mapping, f, indent=4)

    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f, indent=4)

    with open("episode_guesses.json", "w") as f:
        json.dump(episode_guesses, f, indent=4)

    return agent

def evaluate_agent(agent, episodes=100):
    """
    Evaluate the performance of the trained RL agent.

    Parameters:
    - agent (RLAgent): Trained RL agent.
    - episodes (int): Number of evaluation episodes.

    Returns:
    - float: Average reward over evaluation episodes.
    - float: Success rate (percentage of games won).
    """
    env = AlphalockEnvironment()
    total_rewards = []
    successes = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = []

        while not done:
            # Agent selects action (alpha, beta)
            action = agent.select_action(list(state.values()))
            alpha, beta = action

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
            episode_rewards.append(reward)

            state = next_state

        # Track success and total rewards
        total_rewards.append(sum(episode_rewards))
        if reward > 0:  # Positive reward indicates success
            successes += 1

    avg_reward = np.mean(total_rewards)
    success_rate = (successes / episodes) * 100
    print(f"Evaluation results: Avg Reward = {avg_reward}, Success Rate = {success_rate}%")

    return avg_reward, success_rate

if __name__ == "__main__":
    # Train the agent
    trained_agent = train_agent(episodes=1000, batch_size=4)
    trained_agent.save_model("trained_rl_agent.pth")

    # Evaluate the agent
    #evaluate_agent(trained_agent, episodes=100)
