import random
from pattern_utils import generate_pattern
from reward_calculator import compute_reward
from data_preprocessor import load_dictionaries, normalize_frequencies
from config import MAX_ATTEMPTS, SUCCESS_REWARD, FAILURE_PENALTY, REWARD_SCALING, WORD_FREQS_FILE

class AlphalockEnvironment:
    def __init__(self):
        """
        Initialize the Alphalock environment.
        """
        # Load and preprocess the word frequencies
        word_freqs = load_dictionaries(WORD_FREQS_FILE)
        self.allowed_words = list(word_freqs.keys())
        self.word_frequencies = normalize_frequencies(word_freqs)

        # Initialize environment state
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - dict: Initial state.
        """
        self.solution = random.choice(self.allowed_words)  # Randomly select a solution
        self.possible_words = self.allowed_words.copy()  # All words are initially possible
        self.attempts_remaining = MAX_ATTEMPTS
        self.feedback_history = []

        # Return the initial state
        return self.get_state()

    def get_state(self):
        """
        Retrieve the current environment state.

        Returns:
        - dict: Current state.
        """
        state = {
            "pool_entropy": len(self.possible_words),
            "attempts_remaining": self.attempts_remaining,
            "feedback_history": self.feedback_history,
        }
        return state

    def step(self, guess, alpha, beta):
        """
        Take a step in the environment based on the agent's guess and weights.

        Parameters:
        - guess (str): The guessed word.
        - alpha (float): Weight for information theory (IT).
        - beta (float): Weight for relative word frequency (RWF).

        Returns:
        - dict: New state.
        - float: Reward for the step.
        - bool: Whether the episode is done.
        """
        if guess not in self.allowed_words:
            raise ValueError(f"Invalid guess: {guess}. Must be one of the allowed words.")

        # Generate feedback for the guess
        feedback = generate_pattern(guess, self.solution)
        self.feedback_history.append((guess, feedback))

        # Check if the game is over
        if feedback == (2, 2, 2, 2):  # Correct solution
            reward = compute_reward(True, self.attempts_remaining, MAX_ATTEMPTS, SUCCESS_REWARD, FAILURE_PENALTY, REWARD_SCALING)
            done = True
        else:
            # Update the possible word pool based on feedback
            self.possible_words = [
                word for word in self.possible_words if generate_pattern(guess, word) == feedback
            ]

            # Decrement remaining attempts
            self.attempts_remaining -= 1

            # Check if the game is over due to attempts running out
            if self.attempts_remaining == 0:
                reward = compute_reward(False, self.attempts_remaining, MAX_ATTEMPTS, SUCCESS_REWARD, FAILURE_PENALTY, REWARD_SCALING)
                done = True
            else:
                reward = 0  # No reward for intermediate steps
                done = False

        # Return the new state, reward, and done flag
        return self.get_state(), reward, done

    def render(self):
        """
        Print the current state for debugging purposes.
        """
        print("Solution:", self.solution)
        print("Attempts Remaining:", self.attempts_remaining)
        print("Feedback History:", self.feedback_history)
        print("Remaining Words in Pool:", len(self.possible_words))
