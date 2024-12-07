import json
from infotheory_solver import simulate_game

# Load data
with open("4letter_word_freqs.json") as f:
    word_frequencies = json.load(f)

allowed_words = list(word_frequencies.keys())
possible_words = allowed_words.copy()

# Simulate a game
solution = "rose"  # Example solution
guesses = simulate_game(allowed_words, possible_words, solution)

print("Guesses:", guesses)
