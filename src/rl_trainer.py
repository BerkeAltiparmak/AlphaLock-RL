import numpy as np
from rl_environment import AlphalockEnvironment
from rl_agent import RLAgent
from score_calculator import select_best_word
from config import MAX_ATTEMPTS, SUCCESS_REWARD, FAILURE_PENALTY

def train_agent(episodes=1000, state_dim=3, hidden_dim=128, lr=0.001, gamma=0.99):
    """
    Train the RL agent on the Alphalock game.

    Parameters:
    - episodes (int): Number of training episodes.
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

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = []

        print(f"Episode {episode + 1}/{episodes}")

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
            agent.store_reward(reward)
            episode_rewards.append(reward)

            state = next_state

        # Update the agent's policy
        agent.update_policy()

        # Log the results of the episode
        total_reward = sum(episode_rewards)
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

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
    trained_agent = train_agent(episodes=1000)

    # Evaluate the agent
    #evaluate_agent(trained_agent, episodes=100)
