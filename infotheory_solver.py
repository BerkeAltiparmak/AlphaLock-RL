import json
from entropy_calculation import calculate_entropies
from pattern_utils import generate_pattern

def update_possible_words(guess, feedback, possible_words):
    """Update the pool of possible answers based on feedback."""
    return [word for word in possible_words if generate_pattern(guess, word) == feedback]

def select_best_guess(allowed_words, possible_words):
    """Select the best guess based on entropy."""
    entropies = calculate_entropies(allowed_words, possible_words)
    return max(entropies, key=entropies.get)

def simulate_game(allowed_words, possible_words, solution):
    """Simulate the game using the IT-based approach."""
    guesses = []
    while len(possible_words) > 1:
        guess = select_best_guess(allowed_words, possible_words)
        feedback = generate_pattern(guess, solution)
        possible_words = update_possible_words(guess, feedback, possible_words)
        guesses.append(guess)
    return guesses
