import csv
import json

def create_word_frequency_dict(words_file, ngram_file, output_file, word_length=4):
    """
    Create dictionaries for absolute and relative word frequencies,
    filtered by word length, and save them as JSON files.

    Args:
    - words_file (str): Path to the words_alpha.txt file.
    - ngram_file (str): Path to the ngram_freq.csv file.
    - output_file (str): Path to the output JSON file for absolute frequencies.
    - word_length (int): Filter words by this length (default is 4).
    """
    # Load words from words_alpha.txt filtered by length
    with open(words_file, 'r') as f:
        valid_words = set(
            word.strip().lower() for word in f if word.strip() and len(word.strip()) == word_length
        )

    # Create a dictionary for word frequencies from ngram_freq.csv
    word_frequencies = {}
    total_frequency = 0
    with open(ngram_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row if present
        for row in reader:
            word = row[0].strip().lower()
            freq = int(row[1]) if row[1].isdigit() else 0
            if word in valid_words:
                word_frequencies[word] = freq
                total_frequency += freq

    # Calculate relative frequencies
    relative_frequencies = {
        word: freq / total_frequency for word, freq in word_frequencies.items()
    }

    # Save absolute frequencies as JSON
    with open(output_file, 'w') as f:
        json.dump(word_frequencies, f, indent=4)

    # Save relative frequencies as JSON
    relative_output_file = output_file.replace(".json", "_relative.json")
    with open(relative_output_file, 'w') as f:
        json.dump(relative_frequencies, f, indent=4)

    print(f"Word frequency dictionary saved to {output_file}")
    print(f"Relative frequency dictionary saved to {relative_output_file}")

# Paths to the input files and output file
words_alpha_path = "words_alpha.txt"
ngram_freq_path = "ngram_freq.csv"
output_json_path = "4letter_word_freqs.json"

# Create the word frequency dictionaries
create_word_frequency_dict(words_alpha_path, ngram_freq_path, output_json_path, word_length=4)