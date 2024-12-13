from scipy.stats import entropy
import numpy as np
from pattern_utils import generate_pattern

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

    # Normalize the entropies to the range [0, 1]
    entropy_values = np.array(list(entropies.values()))
    min_entropy = np.min(entropy_values)
    max_entropy = np.max(entropy_values)

    normalized_entropies = {
        word: (ent - min_entropy) / (max_entropy - min_entropy) if max_entropy > min_entropy else 1
        for word, ent in entropies.items()
    }

    return normalized_entropies