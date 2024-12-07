from scipy.stats import entropy
import numpy as np
from pattern_utils import generate_pattern
from multiprocessing import Pool

def get_pattern_distributions(allowed_words, possible_words):
    """Compute the distribution of patterns for each allowed guess."""
    pattern_counts = {}

    for guess in allowed_words:
        counts = {}
        for answer in possible_words:
            pattern = generate_pattern(guess, answer)
            counts[pattern] = counts.get(pattern, 0) + 1
        pattern_counts[guess] = counts

    return pattern_counts

def calculate_entropies(allowed_words, possible_words):
    """Calculate entropy for all allowed guesses."""
    pattern_distributions = get_pattern_distributions(allowed_words, possible_words)
    entropies = {}

    for word, distribution in pattern_distributions.items():
        total = sum(distribution.values())
        probabilities = np.array(list(distribution.values())) / total
        entropies[word] = entropy(probabilities, base=2)

    return entropies


def get_pattern_distributions_vectorized(allowed_words, possible_words, pattern_dict):
    """Compute the distribution of patterns for each allowed guess using precomputed patterns."""
    pattern_counts = {word: {} for word in allowed_words}
    
    for guess in allowed_words:
        for answer in possible_words:
            pattern = pattern_dict[(guess, answer)]
            pattern_counts[guess][pattern] = pattern_counts[guess].get(pattern, 0) + 1
    
    return pattern_counts

def calculate_entropies_vectorized(allowed_words, possible_words, pattern_dict):
    """Calculate entropy for all allowed guesses using precomputed patterns."""
    pattern_distributions = get_pattern_distributions_vectorized(allowed_words, possible_words, pattern_dict)
    entropies = {}

    for word, distribution in pattern_distributions.items():
        total = sum(distribution.values())
        probabilities = np.array(list(distribution.values())) / total
        entropies[word] = entropy(probabilities, base=2)

    return entropies


def calculate_entropy_for_word(args):
    """Helper function to calculate entropy for a single word."""
    word, possible_words, pattern_dict = args
    counts = {}
    for answer in possible_words:
        pattern = pattern_dict[(word, answer)]
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    probabilities = np.array(list(counts.values())) / total
    return word, entropy(probabilities, base=2)

def calculate_entropies_parallel(allowed_words, possible_words, pattern_dict):
    """Calculate entropy for all allowed guesses in parallel."""
    with Pool() as pool:
        results = pool.map(calculate_entropy_for_word, 
                           [(word, possible_words, pattern_dict) for word in allowed_words])
    return dict(results)