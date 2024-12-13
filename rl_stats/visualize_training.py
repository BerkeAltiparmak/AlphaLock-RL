import json
import matplotlib.pyplot as plt

def load_json(file_path):
    """
    Load a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Loaded JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def visualize_training(guesses_file, rewards_file):
    """
    Visualize episode guesses and rewards in two subplots.

    Parameters:
    - guesses_file (str): Path to the episode guesses JSON file.
    - rewards_file (str): Path to the episode rewards JSON file.
    """
    # Load data
    episode_guesses = load_json(guesses_file)
    episode_rewards = load_json(rewards_file)

    # Sort data by episode number
    episodes = sorted(map(int, episode_guesses.keys()))
    guesses = [episode_guesses[str(ep)] for ep in episodes]
    rewards = [episode_rewards[str(ep)] for ep in episodes]

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot guesses
    axs[0].plot(episodes, guesses, marker="o", label="Guesses per Episode", color="blue")
    axs[0].set_title("Guesses per Episode")
    axs[0].set_ylabel("Number of Guesses")
    axs[0].grid(True)
    axs[0].legend()

    # Plot rewards
    axs[1].plot(episodes, rewards, marker="o", label="Rewards per Episode", color="green")
    axs[1].set_title("Rewards per Episode")
    axs[1].set_xlabel("Episode Number")
    axs[1].set_ylabel("Reward")
    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    visualize_training("episode_guesses.json", "episode_rewards.json")