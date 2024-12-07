import json
from infotheory_solver import simulate_game
from pattern_utils import precompute_patterns

# Load data
with open("4letter_word_freqs.json") as f:
    word_frequencies = json.load(f)

allowed_words = list(word_frequencies.keys())
possible_words = allowed_words.copy()

# Precompute feedback patterns
pattern_dict = precompute_patterns(allowed_words)

# Simulate a game
solution = "rose"  # Example solution
guesses = simulate_game(allowed_words, possible_words, solution, pattern_dict=pattern_dict)

print("Guesses:", guesses)
