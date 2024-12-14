import json
import matplotlib.pyplot as plt
import numpy as np

def plot_compare_it_rl_results(file_path, bin_size=50, output_image="compare_it_rl_plot.png"):
    """
    Generate aggregated plots of words vs guess count for both IT and RL solvers with confidence intervals, and save the plot.

    Parameters:
    - file_path (str): Path to the compare_it_rl_results.json file.
    - bin_size (int): Number of words per bin for aggregation.
    - output_image (str): Filename to save the generated plot.
    """
    # Load the JSON data
    with open(file_path, "r") as f:
        results = json.load(f)

    # Prepare data for aggregation
    words = list(results.keys())
    it_guesses = [results[word]["IT"][0] for word in words]
    rl_guesses = [results[word]["RL"][0] for word in words]

    # Aggregate data into bins
    bins = list(range(0, len(words), bin_size))
    bin_labels = [f"{i + 1}-{i + bin_size} ({words[i]})" for i in bins[:-1]]  # Bin ranges with example word
    it_aggregates = [np.mean(it_guesses[i:i + bin_size]) for i in bins[:-1]]
    rl_aggregates = [np.mean(rl_guesses[i:i + bin_size]) for i in bins[:-1]]
    it_std_devs = [np.std(it_guesses[i:i + bin_size]) for i in bins[:-1]]
    rl_std_devs = [np.std(rl_guesses[i:i + bin_size]) for i in bins[:-1]]

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Plot IT results with error bars
    plt.errorbar(
        bin_labels, 
        it_aggregates, 
        yerr=it_std_devs, 
        fmt="o--", 
        color="blue", 
        ecolor="blue", 
        capsize=7, 
        elinewidth=2, 
        alpha=0.5,
        label="Information Theory (IT)"
    )

    # Plot RL results with error bars
    plt.errorbar(
        bin_labels, 
        rl_aggregates, 
        yerr=rl_std_devs, 
        fmt="o--", 
        color="orange", 
        ecolor="orange", 
        capsize=7, 
        elinewidth=2,
        alpha=0.5,
        label="Reinforcement Learning (RL)"
    )

    # Customize the plot
    plt.title("Aggregated Guess Counts per Word Frequency Bin of the 1000 Most Common Words: IT vs RL", fontsize=16)
    plt.xlabel("Word Frequency Bins (with Example Words)", fontsize=14)
    plt.ylabel("Average Guess Count", fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Add mean and standard deviation statistics to the plot
    it_mean = np.mean(it_guesses)
    it_std = np.std(it_guesses)
    rl_mean = np.mean(rl_guesses)
    rl_std = np.std(rl_guesses)
    stats_text = (
        f"IT: Mean = {it_mean:.2f}, Std Dev = {it_std:.2f}\n"
        f"RL: Mean = {rl_mean:.2f}, Std Dev = {rl_std:.2f}"
    )
    plt.gcf().text(0.80, 0.20, stats_text, fontsize=12, bbox=dict(facecolor="white", alpha=0.8))

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved as {output_image}")

    # Provide insights
    print("Insights:")
    print(f"IT - Mean Guess Count: {it_mean:.2f}, Standard Deviation: {it_std:.2f}")
    print(f"RL - Mean Guess Count: {rl_mean:.2f}, Standard Deviation: {rl_std:.2f}")

# Example usage
if __name__ == "__main__":
    file_path = "compare_it_rl_results.json"
    plot_compare_it_rl_results(file_path, bin_size=50)
