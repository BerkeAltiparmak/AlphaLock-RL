import json
import time
import random
import numpy as np
from src.infotheory_solver import simulate_game
from src.rl_agent import RLAgent
from src.rl_environment import AlphalockEnvironment
from src.config import WORD_FREQS_FILE, MODEL_PATH

def generate_random_solution(word_frequencies):
    """
    Generate a realistic random solution based on frequency with smoothing.
    """
    words = list(word_frequencies.keys())
    frequencies = list(word_frequencies.values())

    # Apply log smoothing
    smoothed_weights = [np.log(freq + 0.0001) for freq in frequencies]

    # Normalize the smoothed weights
    total_weight = sum(smoothed_weights)
    normalized_weights = [w / total_weight for w in smoothed_weights]

    # Select a random word based on smoothed weights
    solution = random.choices(words, weights=normalized_weights, k=1)[0]
    return solution

def compare_models(allowed_words, word_frequencies, num_trials=10):
    """
    Compare the performance of the Information Theory-based solver and the RL model.

    Parameters:
    - allowed_words (list): List of allowed words for guessing.
    - word_frequencies (dict): Word frequency data.
    - num_trials (int): Number of trials for comparison.
    """
    # Load RL agent
    rl_agent = RLAgent(state_dim=3, hidden_dim=128)
    rl_agent.load_model(MODEL_PATH)

    it_results = []
    rl_results = []

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        # Generate a random solution
        solution = generate_random_solution(word_frequencies)
        print(f"Randomly chosen solution: {solution}")

        # Information Theory-based solver
        print("Running Information Theory-based solver...")
        it_start = time.time()
        it_guesses = simulate_game(allowed_words, allowed_words.copy(), solution, first_guess="sera")
        it_time = time.time() - it_start
        it_results.append((len(it_guesses), it_time))

        print(f"IT-based solver finished in {len(it_guesses)} guesses and {it_time:.2f} seconds.")

        # RL-based solver
        print("Running RL-based solver...")
        env = AlphalockEnvironment()
        env.solution = solution  # Set the solution in the environment

        rl_start = time.time()
        state = env.reset()
        done = False
        rl_guesses = []

        while not done:
            if len(env.feedback_history) == 0:
                guess = "sera"  # First guess
            else:
                flat_state = [
                    state["pool_entropy"],
                    state["attempts_remaining"],
                    len(state["feedback_history"]),
                ]
                action = rl_agent.select_action(flat_state)
                alpha, beta = action

                # Pick the best word based on the RL model's alpha and beta values
                guess = select_best_word(
                    env.allowed_words,
                    env.possible_words,
                    env.word_frequencies,
                    alpha,
                    beta
                )

            rl_guesses.append(guess)
            state, _, done = env.step(guess, alpha, beta)

        rl_time = time.time() - rl_start
        rl_results.append((len(rl_guesses), rl_time))

        print(f"RL-based solver finished in {len(rl_guesses)} guesses and {rl_time:.2f} seconds.")

    # Aggregate and print results
    avg_it_guesses = np.mean([r[0] for r in it_results])
    avg_it_time = np.mean([r[1] for r in it_results])
    avg_rl_guesses = np.mean([r[0] for r in rl_results])
    avg_rl_time = np.mean([r[1] for r in rl_results])

    print("\nComparison Results:")
    print(f"Information Theory-based solver: Avg Guesses = {avg_it_guesses:.2f}, Avg Time = {avg_it_time:.2f} seconds.")
    print(f"RL-based solver: Avg Guesses = {avg_rl_guesses:.2f}, Avg Time = {avg_rl_time:.2f} seconds.")

if __name__ == "__main__":
    # Load data
    with open(WORD_FREQS_FILE) as f:
        word_frequencies = json.load(f)

    allowed_words = list(word_frequencies.keys())

    # Compare the models
    compare_models(allowed_words, word_frequencies, num_trials=10)
