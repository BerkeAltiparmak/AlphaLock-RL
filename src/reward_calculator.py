from entropy_calculation import calculate_entropies
from rwf_calculation import calculate_relative_word_frequencies

def compute_reward(success, moves_remaining, max_attempts, success_reward=100, failure_penalty=-50, scaling="linear"):
    """
    Compute the reward for a game outcome.

    Parameters:
    - success (bool): Whether the game was successfully completed.
    - moves_remaining (int): Number of moves left (if successful).
    - max_attempts (int): Total allowed attempts.
    - success_reward (int): Base reward for success.
    - failure_penalty (int): Penalty for failure.
    - scaling (str): "linear" or "exponential" reward scaling.

    Returns:
    - float: Calculated reward.
    """
    if success:
        if scaling == "linear":
            return success_reward * (moves_remaining / max_attempts)
        elif scaling == "exponential":
            return success_reward * (2 ** (moves_remaining / max_attempts))
        else:
            raise ValueError("Invalid scaling method. Choose 'linear' or 'exponential'.")
    else:
        return failure_penalty

def calculate_scores(allowed_words, possible_words, word_frequencies, alpha, beta):
    """
    Calculate scores for each word based on Information Theory (IT) and Relative Word Frequency (RWF).

    Parameters:
    - allowed_words (list): List of all allowed words for guessing.
    - possible_words (list): Current pool of possible words.
    - word_frequencies (dict): Absolute frequencies of words.
    - alpha (float): Weight for Information Theory component.
    - beta (float): Weight for Relative Word Frequency component.

    Returns:
    - dict: A dictionary mapping each allowed word to its calculated score.
    """
    # Calculate entropies for all allowed words
    entropies = calculate_entropies(allowed_words, possible_words)

    # Calculate normalized relative word frequencies
    rwf = calculate_relative_word_frequencies(possible_words, word_frequencies)

    # Combine IT and RWF into a single score
    scores = {}
    for word in allowed_words:
        it_score = entropies.get(word, 0)
        rwf_score = rwf.get(word, 0)
        scores[word] = alpha * it_score + beta * rwf_score

    return scores

def select_best_word(allowed_words, possible_words, word_frequencies, alpha, beta):
    """
    Select the best word to guess based on the calculated scores.

    Parameters:
    - allowed_words (list): List of all allowed words for guessing.
    - possible_words (list): Current pool of possible words.
    - word_frequencies (dict): Absolute frequencies of words.
    - alpha (float): Weight for Information Theory component.
    - beta (float): Weight for Relative Word Frequency component.

    Returns:
    - str: The word with the highest score.
    """
    scores = calculate_scores(allowed_words, possible_words, word_frequencies, alpha, beta)
    return max(scores, key=scores.get)

