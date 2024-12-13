import json
from entropy_calculation import calculate_entropies
from pattern_utils import generate_pattern

def update_possible_words(guess, feedback, possible_words):
    """Update the pool of possible answers based on feedback."""
    return [word for word in possible_words if generate_pattern(guess, word) == feedback]

def select_best_guess(allowed_words, possible_words):
    """Select the best guess based on entropy."""
    if not possible_words:
        raise ValueError("No possible words remain. Check feedback logic.")
    entropies = calculate_entropies(allowed_words, possible_words)
    return max(entropies, key=entropies.get)

import time

def simulate_game(allowed_words, possible_words, solution, first_guess=None, max_attempts=10):
    """Simulate the game using the IT-based approach, with an optional first guess."""
    guesses = []
    possible_words_list = []
    feedback_list = []
    attempts = 0

    # Use the provided first guess if available
    if first_guess:
        start_time = time.time()
        guess = first_guess
        feedback = generate_pattern(guess, solution)
        possible_words = update_possible_words(guess, feedback, possible_words)
        guesses.append(guess)
        attempts += 1
        end_time = time.time()
        print(f"Attempt {attempts}: Guess = {guess}, Feedback = {feedback}, Remaining Words = {len(possible_words)}")
        print(f"Time for Guess {attempts}: {end_time - start_time:.2f} seconds")

    # Continue with entropy-based guesses for subsequent rounds
    while len(possible_words) > 1 and attempts < max_attempts:
        start_time = time.time()
        attempts += 1
        guess = select_best_guess(allowed_words, possible_words)
        feedback = generate_pattern(guess, solution)
        possible_words = update_possible_words(guess, feedback, possible_words)
        guesses.append(guess)
        end_time = time.time()
        print(f"Attempt {attempts}: Guess = {guess}, Feedback = {feedback}, Remaining Words = {len(possible_words)}")
        print(f"Time for Guess {attempts}: {end_time - start_time:.2f} seconds")

    # Handle the final guess if only one word remains
    if len(possible_words) == 1:
        guesses.append(possible_words[0])

    if attempts >= max_attempts:
        print("Warning: Maximum attempts reached. The game might not have converged.")

    return guesses, 

