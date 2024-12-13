import gym
from gym import spaces
import numpy as np
from entropy_calculation import calculate_entropies
from pattern_utils import generate_pattern

class AlphaLockEnv(gym.Env):
    def __init__(self, allowed_words, solution):
        super(AlphaLockEnv, self).__init__()
        self.allowed_words = allowed_words
        self.solution = solution
        self.possible_words = allowed_words.copy()
        self.guesses = []
        self.turn = 0
        self.max_turns = 10

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.allowed_words))
        self.observation_space = spaces.Dict({
            "possible_words": spaces.Discrete(len(self.allowed_words)),
            "turn": spaces.Discrete(self.max_turns),
            "feedback": spaces.Box(low=0, high=2, shape=(4,), dtype=np.int32)
        })

    def step(self, action):
        guess = self.allowed_words[action]
        feedback = generate_pattern(guess, self.solution)
        self.guesses.append(guess)
        self.possible_words = [
            word for word in self.possible_words 
            if generate_pattern(guess, word) == feedback
        ]

        # Calculate rewards
        entropy_reduction = self.calculate_entropy_reduction()
        frequency_score = self.calculate_frequency_score(guess)
        correct_guess = int(guess == self.solution)

        reward = self.dynamic_weights() * entropy_reduction + frequency_score + 10 * correct_guess

        self.turn += 1
        done = (guess == self.solution or self.turn >= self.max_turns)
        state = self.get_state(feedback)

        return state, reward, done, {}

    def reset(self):
        self.possible_words = self.allowed_words.copy()
        self.guesses = []
        self.turn = 0
        return self.get_state()

    def calculate_entropy_reduction(self):
        previous_entropy = calculate_entropies(self.allowed_words, self.possible_words)
        current_entropy = calculate_entropies(self.allowed_words, self.possible_words)
        return previous_entropy - current_entropy

    def calculate_frequency_score(self, guess):
        return self.word_frequencies.get(guess, 0) / sum(self.word_frequencies.values())

    def dynamic_weights(self):
        return max(0.1, 1 - self.turn / self.max_turns)

    def get_state(self, feedback=None):
        return {
            "possible_words": len(self.possible_words),
            "turn": self.turn,
            "feedback": feedback if feedback else np.zeros(4, dtype=np.int32)
        }