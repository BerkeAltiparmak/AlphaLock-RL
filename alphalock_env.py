import gym
from gym import spaces
import numpy as np
from entropy_calculation import calculate_entropies
from pattern_utils import generate_pattern


class AlphaLockEnv(gym.Env):
    def __init__(self, allowed_words, word_frequencies, secret_code):
        """
        Initialize the AlphaLock environment.
        Args:
            allowed_words (list): List of all possible allowed words.
            word_frequencies (dict): Mapping of words to their relative frequencies.
            secret_code (str): The secret word to be guessed.
        """
        super(AlphaLockEnv, self).__init__()
        self.allowed_words = allowed_words
        self.word_frequencies = word_frequencies
        self.secret_code = secret_code
        self.current_pool = allowed_words.copy()
        self.turn = 0
        self.max_turns = 10

        # Action and observation space
        self.action_space = spaces.Discrete(len(allowed_words))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(allowed_words),),  # Scores for all words
            dtype=np.float32,
        )

    def reset(self):
        """
        Reset the environment for a new game.
        Returns:
            np.array: Initial scores for all words in the allowed words list.
        """
        self.current_pool = self.allowed_words.copy()
        self.turn = 0
        return self._calculate_scores(0.5, 0.5)  # Start with alpha=0.5, beta=0.5

    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action (int): Index of the guessed word.
        Returns:
            tuple: Observation (updated scores), reward, done (bool), and info (dict).
        """
        guess = self.allowed_words[action]
        feedback = generate_pattern(guess, self.secret_code)
        self.current_pool = [word for word in self.current_pool if generate_pattern(guess, word) == feedback]
        self.turn += 1
        correct_guess = (guess == self.secret_code)

        # Reward function
        if correct_guess:
            reward = 1000 - 10 * self.turn  # Higher reward for fewer turns
            done = True
        elif self.turn >= self.max_turns:
            reward = -100  # Penalty for exceeding turn limit
            done = True
        else:
            reward = 0
            done = False

        # Update observation
        obs = self._calculate_scores(self.alpha, self.beta)
        return obs, reward, done, {}

    def _calculate_scores(self, alpha, beta):
        """
        Calculate scores for all words based on IT and RWF.
        Args:
            alpha (float): Weight for Information Theory (IT).
            beta (float): Weight for Relative Word Frequency (RWF).
        Returns:
            np.array: Scores for all words in the allowed words list.
        """
        it_scores = calculate_entropies(self.allowed_words, self.current_pool)
        rwf_scores = {word: self.word_frequencies.get(word, 0) for word in self.allowed_words}
        total_freq = sum(rwf_scores.values())
        rwf_scores = {word: freq / total_freq for word, freq in rwf_scores.items()}  # Normalize

        scores = []
        for word in self.allowed_words:
            it = it_scores.get(word, 0)
            rwf = rwf_scores.get(word, 0)
            scores.append(alpha * it + beta * rwf)

        return np.array(scores, dtype=np.float32)

    def render(self):
        """
        Render the current state of the game.
        """
        print(f"Turn: {self.turn}, Pool Size: {len(self.current_pool)}")
