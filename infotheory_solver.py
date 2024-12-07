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

def simulate_game(allowed_words, possible_words, solution, max_attempts=10):
    """Simulate the game using the IT-based approach."""
    guesses = []
    attempts = 0

    while len(possible_words) > 1 and attempts < max_attempts:
        attempts += 1
        guess = select_best_guess(allowed_words, possible_words)
        feedback = generate_pattern(guess, solution)
        possible_words = update_possible_words(guess, feedback, possible_words)
        guesses.append(guess)
        print(f"Attempt {attempts}: Guess = {guess}, Feedback = {feedback}, Remaining Words = {len(possible_words)}")

    if len(possible_words) == 1:
        guesses.append(possible_words[0])  # Add the last word to the guesses

    if len(guesses) > max_attempts:
        print("Warning: Maximum attempts reached. The game might not have converged.")

    return guesses
