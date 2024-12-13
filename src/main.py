import json
import time
import random
import numpy as np
from infotheory_solver import simulate_game
from rl_agent import RLAgent
from rl_environment import AlphalockEnvironment
from reward_calculator import select_best_word
from config import WORD_FREQS_FILE, MODEL_PATH

def generate_random_solution(word_frequencies):
    """
    Generate a realistic random solution based on frequency with smoothing.
    """
    words = list(word_frequencies.keys())
    frequencies = list(word_frequencies.values())
    weights = [freq for freq in frequencies]

    # Normalize the weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Select a random word based on weights. This mimics the choice of the AlphaLock's creator
    # as humans tend to choose common words for the solutions of word-based games.
    solution = random.choices(words, weights=normalized_weights, k=1)[0]
    return solution

def log_guess_details(guess_num, guess, pool_size, alpha=None, model_type="IT"):
    """
    Log details for each guess in a uniform format.

    Parameters:
    - guess_num (int): Guess number.
    - guess (str): The guessed word.
    - pool_size (int): The remaining pool size.
    - alpha (float, optional): Information theory coefficient (for RL-based solver).
    - model_type (str): Model type ("IT" or "RL").
    """
    print(f"[{model_type}] Guess #{guess_num}: Word = {guess}, Pool Size = {pool_size}")
    if alpha is not None:
        print(f"  Information Theory Coefficient: {alpha}")

def compare_models(allowed_words, word_frequencies, num_trials=10):
    """
    Compare the performance of the Information Theory-based solver and the RL model.

    Parameters:
    - allowed_words (list): List of allowed words for guessing.
    - word_frequencies (dict): Word frequency data.
    - num_trials (int): Number of trials for comparison.
    """
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

        for i, guess in enumerate(it_guesses, start=1):
            log_guess_details(i, guess, len(allowed_words), model_type="IT")

        print(f"IT-based solver's final prediction: {it_guesses[-1]}")
        print(f"IT-based solver finished in {len(it_guesses)} guesses and {it_time:.2f} seconds.")

        # RL-based solver
        print("Running RL-based solver...")

        # Load RL agent
        rl_agent = RLAgent(state_dim=3, hidden_dim=128)
        rl_agent.load_model(MODEL_PATH)

        env = AlphalockEnvironment()
        alpha, beta = 1.0, 0  # Initial alpha and beta values (defined to explore early)

        rl_start = time.time()
        state = env.reset()
        env.solution = solution  # Set the solution in the environment
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

            # Log guess details
            log_guess_details(len(rl_guesses), guess, len(env.possible_words), alpha=alpha, model_type="RL")

        rl_time = time.time() - rl_start
        rl_results.append((len(rl_guesses), rl_time))

        print(f"RL-based solver finished in {len(rl_guesses)} guesses and {rl_time:.2f} seconds.")
        print(f"RL-based solver's final prediction: {rl_guesses[-1]}")

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
