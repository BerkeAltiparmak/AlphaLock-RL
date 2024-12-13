def calculate_relative_word_frequencies(possible_words, word_frequencies):
    """
    Calculate the Relative Word Frequencies (RWF) for words in the current pool of possible words.

    Parameters:
    - possible_words (list): Current pool of possible words.
    - word_frequencies (dict): Absolute frequencies of words.

    Returns:
    - dict: A dictionary mapping words in the pool to their normalized relative frequencies.
    """
    # Extract frequencies for possible words
    pool_frequencies = {word: word_frequencies.get(word, 0) for word in possible_words}
    
    # Get minimum and maximum frequencies
    min_freq = min(pool_frequencies.values())
    max_freq = max(pool_frequencies.values())

    # Normalize frequencies to [0, 1]
    if max_freq > min_freq:
        return {
            word: (freq - min_freq) / (max_freq - min_freq)
            for word, freq in pool_frequencies.items()
        }
    else:
        # If all frequencies are the same, return 1 for all words
        return {word: 1 for word in possible_words}