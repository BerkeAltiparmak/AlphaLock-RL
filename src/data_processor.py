import json

def load_dictionaries(file_path):
    """
    Load and validate the word frequency dictionary.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Loaded word frequency data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid data format. Expected a dictionary.")
    return data

def normalize_frequencies(word_freqs):
    """
    Normalize word frequencies to [0, 1].

    Parameters:
    - word_freqs (dict): Word frequency dictionary.

    Returns:
    - dict: Normalized word frequencies.
    """
    max_freq = max(word_freqs.values())
    return {word: freq / max_freq for word, freq in word_freqs.items()}
