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
