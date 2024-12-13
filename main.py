import json
import time
from infotheory_solver import simulate_game

# Start timing
total_start = time.time()

# Load data
data_start = time.time()
with open("4letter_word_freqs.json") as f:
    word_frequencies = json.load(f)
data_end = time.time()
print(f"Time to load data: {data_end - data_start:.2f} seconds")

allowed_words = list(word_frequencies.keys())
possible_words = allowed_words.copy()

# Set the solution and first guess
solution = "that"  # Example solution
first_guess = "sare"  # Experimentally found optimal first word with Information Theory

# Simulate a game
simulate_start = time.time()
guesses = simulate_game(allowed_words, possible_words, solution, first_guess=first_guess)
simulate_end = time.time()
print(f"Time to simulate game: {simulate_end - simulate_start:.2f} seconds")

# End timing
total_end = time.time()
print(f"Total execution time: {total_end - total_start:.2f} seconds")

# Output results
print("Guesses:", guesses)
