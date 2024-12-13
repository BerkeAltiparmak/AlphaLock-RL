import json
import seaborn as sns
import matplotlib.pyplot as plt
import re

def load_alpha_beta_mapping(file_path):
    """
    Load the alpha_beta_mapping JSON file.

    Parameters:
    - file_path (str): Path to the alpha_beta_mapping.json file.

    Returns:
    - dict: Parsed JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def parse_alpha_beta_mapping(data):
    """
    Parse the alpha_beta_mapping data to extract guess numbers, pool sizes, and alpha values.

    Parameters:
    - data (dict): Alpha-beta mapping data from JSON.

    Returns:
    - list: Guess numbers.
    - list: Pool sizes.
    - list: Alpha values.
    """
    guess_numbers = []
    pool_sizes = []
    alphas = []

    for key, values in data.items():
        match = re.match(r"(\d+):(\d+),(\d+)", key)
        if match:
            episode, guess_number, pool_size = map(int, match.groups())
            if guess_number != 1:  # Ignore data points where guess_number is 1 because we always guessed sare
                guess_numbers.append(guess_number)
                pool_sizes.append(pool_size)
                alphas.append(values["alpha"])

    return guess_numbers, pool_sizes, alphas

def visualize_alpha_beta_mapping(file_path):
    """
    Visualize alpha values against guess number and pool size.

    Parameters:
    - file_path (str): Path to the alpha_beta_mapping.json file.
    """
    # Load and parse the data
    data = load_alpha_beta_mapping(file_path)
    guess_numbers, pool_sizes, alphas = parse_alpha_beta_mapping(data)

    # Create the first plot: Guess number vs Alpha
    plt.figure(figsize=(10, 5))
    plt.plot(guess_numbers, alphas, marker="o", linestyle="--", color="b")
    plt.title("Guess Number vs Alpha")
    plt.xlabel("Guess Number")
    plt.ylabel("Alpha")
    plt.grid(True)
    plt.show()

    # Create the second plot: Pool size vs Alpha
    plt.figure(figsize=(10, 5))
    plt.plot(pool_sizes, alphas, marker="o", linestyle="--", color="r")
    plt.title("Pool Size vs Alpha")
    plt.xlabel("Pool Size")
    plt.ylabel("Alpha")
    plt.grid(True)
    plt.show()

def visualize_alpha_beta_mapping_confidence_interval(file_path):
    """
    Visualize alpha values against guess number and pool size with the confidence interval of 95% (default).

    Parameters:
    - file_path (str): Path to the alpha_beta_mapping.json file.
    """
    # Load and parse the data
    data = load_alpha_beta_mapping(file_path)
    guess_numbers, pool_sizes, alphas = parse_alpha_beta_mapping(data)

    # Create the first plot: Guess number vs Alpha
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=guess_numbers, y=alphas, marker="o", linestyle="--", color="b")
    plt.title("Guess Number vs Alpha", fontsize=16)
    plt.xlabel("Guess Number", fontsize=14)
    plt.ylabel("Alpha", fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

    # Create the second plot: Pool size vs Alpha
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=pool_sizes, y=alphas, marker="o", linestyle="--", color="r")
    plt.title("Pool Size vs Alpha", fontsize=16)
    plt.xlabel("Pool Size", fontsize=14)
    plt.ylabel("Alpha", fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def visualize_alpha_vs_pool_and_guess(file_path):
    """
    Visualize alpha values against guess number and pool size on a 2D scatter plot using seaborn.

    Parameters:
    - file_path (str): Path to the alpha_beta_mapping.json file.
    """
    # Load and parse the data
    data = load_alpha_beta_mapping(file_path)
    guess_numbers, pool_sizes, alphas = parse_alpha_beta_mapping(data)

    # Create a 2D scatter plot using seaborn
    plt.figure(figsize=(12, 8))
    scatter_plot = sns.scatterplot(
        x=guess_numbers,
        y=pool_sizes,
        hue=alphas,
        palette="coolwarm",
        size=alphas,
        sizes=(20, 200),
        edgecolor="k",
        alpha=0.5
    )
    scatter_plot.set_title("Guess Number vs Pool Size with Alpha as Color", fontsize=16)
    scatter_plot.set_xlabel("Guess Number", fontsize=14)
    scatter_plot.set_ylabel("Pool Size", fontsize=14)
    scatter_plot.legend(title="Alpha Value", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "alpha_beta_mapping.json"
    visualize_alpha_beta_mapping(file_path)
    visualize_alpha_beta_mapping_confidence_interval(file_path)
    visualize_alpha_vs_pool_and_guess(file_path)
