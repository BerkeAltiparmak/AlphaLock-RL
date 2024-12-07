import numpy as np

MISS = 0
MISPLACED = 1
EXACT = 2

def generate_pattern(guess, answer):
    """Generate the feedback pattern for a given guess and answer."""
    pattern = [MISS] * len(guess)
    used = [False] * len(answer)

    # Green (EXACT) pass
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            pattern[i] = EXACT
            used[i] = True

    # Yellow (MISPLACED) pass
    for i, g in enumerate(guess):
        if pattern[i] == MISS:
            for j, a in enumerate(answer):
                if g == a and not used[j]:
                    pattern[i] = MISPLACED
                    used[j] = True
                    break

    return tuple(pattern)


def precompute_patterns(allowed_words):
    """Precompute feedback patterns for all pairs of allowed words."""
    pattern_dict = {}
    for guess in allowed_words:
        for answer in allowed_words:
            pattern = generate_pattern(guess, answer)
            pattern_dict[(guess, answer)] = pattern
    return pattern_dict


def pattern_to_int(pattern):
    """Convert a feedback pattern to an integer."""
    return sum(p * (3 ** i) for i, p in enumerate(pattern))

def generate_pattern_as_int(guess, answer):
    """Generate the feedback pattern as an integer."""
    return pattern_to_int(generate_pattern(guess, answer))

