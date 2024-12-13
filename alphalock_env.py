import gym
from gym import spaces
import numpy as np
import random
from entropy_calculation import calculate_entropies
from pattern_utils import generate_pattern

class AlphaLockEnv(gym.Env):
    def __init__(self, allowed_words, word_frequencies, secret_code, alpha=0.5, beta=0.5, render_mode=None):
        super(AlphaLockEnv, self).__init__()
        self.allowed_words = allowed_words
        self.word_frequencies = word_frequencies
        self.secret_code = secret_code
        self.state = None
        self.current_pool = allowed_words.copy()
        self.turn = 0
        self.max_turns = 10
        self.seed_value = None
        self.render_mode = render_mode
        self.alpha = alpha
        self.beta = beta

        # Action and observation space
        self.action_space = spaces.Discrete(len(allowed_words))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),  # Flattened format: [pool_size, turn, alpha, beta]
            dtype=np.float32,
        )

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
        self.alpha = 0.5  # Reset alpha
        self.beta = 0.5   # Reset beta
        self.state = {
            "pool_size": len(self.current_pool),
            "turn": self.turn,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        return self._flatten_state()

    def step(self, action):
        """Execute one step in the environment."""
        guess = self.allowed_words[action]
        feedback = generate_pattern(guess, self.secret_code)
        self.current_pool = [word for word in self.current_pool if generate_pattern(guess, word) == feedback]
        self.turn += 1
        correct_guess = (guess == self.secret_code)

        # Dynamic alpha and beta adjustment based on the turn
        self.alpha = 0.7 - 0.05 * self.turn  # Decrease alpha over time
        self.beta = 0.3 + 0.05 * self.turn  # Increase beta over time

        # Compute rewards
        pool_size_penalty = -len(self.current_pool)
        it_score = calculate_entropies([guess], self.current_pool).get(guess, 0)
        rwf_score = self.word_frequencies.get(guess, 0) / sum(self.word_frequencies.values())
        reward = (100 / self.turn if correct_guess else 0) + self.alpha * it_score + self.beta * rwf_score + pool_size_penalty

        done = correct_guess or self.turn >= self.max_turns
        self.state = {
            "pool_size": len(self.current_pool),
            "turn": self.turn,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        return self._flatten_state(), reward, done, {}

    def set_alpha_beta(self, alpha, beta):
        """Set alpha and beta values."""
        self.alpha = alpha
        self.beta = beta

    def _flatten_state(self):
        """Flatten the state dictionary into a NumPy array."""
        return np.array([
            self.state["pool_size"] / len(self.allowed_words),  # Normalize pool size
            self.state["turn"] / self.max_turns,  # Normalize turn
            self.state["alpha"],
            self.state["beta"],
        ], dtype=np.float32)

    def render(self):
        print(f"Turn: {self.turn}, Pool Size: {len(self.current_pool)}")
