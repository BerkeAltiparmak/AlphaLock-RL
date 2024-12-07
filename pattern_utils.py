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
