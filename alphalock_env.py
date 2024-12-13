import gym
from gym import spaces
import numpy as np
import random
from entropy_calculation import calculate_entropies
from pattern_utils import generate_pattern

class AlphaLockEnv(gym.Env):
    def __init__(self, allowed_words, word_frequencies, secret_code):
        super(AlphaLockEnv, self).__init__()
        self.allowed_words = allowed_words
        self.word_frequencies = word_frequencies
        self.secret_code = secret_code
        self.state = None
        self.current_pool = allowed_words.copy()
        self.turn = 0
        self.max_turns = 10
        self.seed_value = None

        # Action and observation space
        self.action_space = spaces.Discrete(len(allowed_words))
        self.observation_space = spaces.Dict({
            "pool_size": spaces.Discrete(len(allowed_words) + 1),  # Include 0 as a valid value
            "turn": spaces.Discrete(self.max_turns + 1),  # Turn can range from 0 to max_turns
            "alpha": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "beta": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def seed(self, seed=None):
        """Set the seed for reproducibility."""
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """Reset the environment for a new game."""
        if self.seed_value is not None:
            self.seed(self.seed_value)
        self.current_pool = self.allowed_words.copy()
        self.turn = 0
        self.state = {
            "pool_size": len(self.current_pool),
            "turn": self.turn,
            "alpha": np.array([0.5]),  # Start with equal weights
            "beta": np.array([0.5])
        }
        return self.state

    def step(self, action, alpha, beta):
        """Execute one step in the environment."""
        guess = self.allowed_words[action]
        feedback = generate_pattern(guess, self.secret_code)
        self.current_pool = [word for word in self.current_pool if generate_pattern(guess, word) == feedback]
        self.turn += 1
        correct_guess = (guess == self.secret_code)

        # Compute rewards
        pool_size_penalty = -len(self.current_pool)
        it_score = calculate_entropies([guess], self.current_pool).get(guess, 0)
        rwf_score = self.word_frequencies.get(guess, 0) / sum(self.word_frequencies.values())
        reward = (100 / self.turn if correct_guess else 0) + alpha * it_score + beta * rwf_score + pool_size_penalty

        done = correct_guess or self.turn >= self.max_turns
        self.state = {
            "pool_size": len(self.current_pool),
            "turn": self.turn,
            "alpha": np.array([alpha]),
            "beta": np.array([beta])
        }
        return self.state, reward, done, {}

    def render(self):
        print(f"Turn: {self.turn}, Pool Size: {len(self.current_pool)}")
