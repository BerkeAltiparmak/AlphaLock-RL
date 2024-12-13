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
    total_frequency = sum(pool_frequencies.values())

    # Normalize frequencies
    if total_frequency > 0:
        return {word: freq / total_frequency for word, freq in pool_frequencies.items()}
    else:
        return {word: 0 for word in possible_words}