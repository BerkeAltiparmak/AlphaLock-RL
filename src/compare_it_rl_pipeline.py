import json
import time
import numpy as np
from infotheory_solver import simulate_game
from rl_agent import RLAgent
from rl_environment import AlphalockEnvironment
from reward_calculator import select_best_word
from config import WORD_FREQS_FILE, MODEL_PATH
from pattern_utils import generate_pattern

def log_guess_details(guess_num, guess, pool_size, feedback=None, alpha=None, model_type="IT"):
    """
    Log details for each guess in a uniform format.

    Parameters:
    - guess_num (int): Guess number.
    - guess (str): The guessed word.
    - pool_size (int): The remaining pool size.
    - feedback (tuple, optional): Feedback for the guess.
    - alpha (float, optional): Information theory coefficient (for RL-based solver).
    - model_type (str): Model type ("IT" or "RL").
    """
    if feedback == (2, 2, 2, 2):
        print(f"[{model_type}] Guess #{guess_num}: Word = {guess}, Pool Size = SOLVED")
    else:
        print(f"[{model_type}] Guess #{guess_num}: Word = {guess}, Pool Size = {pool_size}")
    print(f"  Feedback: {feedback}")

def append_results_to_file(file_path, word, it_score, it_time, rl_score, rl_time):
    """
    Append results to a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.
    - word (str): The solution word.
    - it_score (int): Number of guesses for IT-based solver.
    - it_time (float): Time taken by IT-based solver.
    - rl_score (int): Number of guesses for RL-based solver.
    - rl_time (float): Time taken by RL-based solver.
    """
    try:
        with open(file_path, "r") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    results[word] = {
        "IT": [it_score, it_time],
        "RL": [rl_score, rl_time]
    }

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

def compare_models_top_1000(allowed_words, word_frequencies, output_file="compare_it_rl_results.json"):
    """
    Compare the performance of the Information Theory-based solver and the RL model
    for the top 1000 most frequent words in the dataset.

    Parameters:
    - allowed_words (list): List of allowed words for guessing.
    - word_frequencies (dict): Word frequency data.
    - output_file (str): Path to the file where results will be saved.
    """
    # Load RL agent
    rl_agent = RLAgent(state_dim=3, hidden_dim=128)
    rl_agent.load_model(MODEL_PATH)

    # Get the top 1000 most frequent words
    top_1000_words = list(word_frequencies.keys())[1:1000]

    for solution in top_1000_words:
        print(f"Testing solution: {solution}")

        # Information Theory-based solver
        print("Running Information Theory-based solver...")
        it_start = time.time()
        it_guesses, _, _ = simulate_game(allowed_words, allowed_words.copy(), solution, first_guess="sera")
        it_time = time.time() - it_start
        it_score = len(it_guesses)

        print(f"IT-based solver finished in {it_score} guesses and {it_time:.2f} seconds.")

        # RL-based solver
        print("Running RL-based solver...")
        env = AlphalockEnvironment()

        rl_start = time.time()
        state = env.reset()
        env.solution = solution  # Set the solution in the environment
        alpha, beta = 1.0, 0  # Initial alpha and beta values (defined to explore early)

        done = False
        rl_guesses = []

        while not done:
            if len(env.feedback_history) == 0:
                guess = "sera"  # First guess
                feedback = generate_pattern(guess, solution)
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
                feedback = generate_pattern(guess, solution)

            rl_guesses.append(guess)
            state, _, done = env.step(guess, alpha, beta)

        rl_time = time.time() - rl_start
        rl_score = len(rl_guesses)

        print(f"RL-based solver finished in {rl_score} guesses and {rl_time:.2f} seconds.")

        # Append results to file
        append_results_to_file(output_file, solution, it_score, it_time, rl_score, rl_time)

if __name__ == "__main__":
    # Load data
    with open(WORD_FREQS_FILE) as f:
        word_frequencies = json.load(f)

    allowed_words = list(word_frequencies.keys())

    # Compare the models on the top 1000 words
    compare_models_top_1000(allowed_words, word_frequencies)
