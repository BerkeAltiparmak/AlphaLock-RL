import csv
import json

def create_word_frequency_dict(words_file, ngram_file, output_file):
    """
    Create a dictionary mapping words to their usage frequencies
    and save it as a JSON file.
    
    Args:
    - words_file (str): Path to the words_alpha.txt file.
    - ngram_file (str): Path to the ngram_freq.csv file.
    - output_file (str): Path to the output JSON file.
    """
    # Load words from words_alpha.txt
    with open(words_file, 'r') as f:
        valid_words = set(word.strip().lower() for word in f if word.strip())

    # Create a dictionary for word frequencies from ngram_freq.csv
    word_frequencies = {}
    with open(ngram_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row if present
        for row in reader:
            word = row[0].strip().lower()
            freq = int(row[1]) if row[1].isdigit() else 0
            if word in valid_words:
                word_frequencies[word] = freq

    # Save the dictionary as a JSON file
    with open(output_file, 'w') as f:
        json.dump(word_frequencies, f, indent=4)

    print(f"Word frequency dictionary saved to {output_file}")

# Paths to the input files and output file
words_alpha_path = "words_alpha.txt"
ngram_freq_path = "ngram_freq.csv"
output_json_path = "word_freqs.json"

# Create the word frequency dictionary
create_word_frequency_dict(words_alpha_path, ngram_freq_path, output_json_path)
