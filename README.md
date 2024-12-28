# **Alphalock-RL: Reinforcement Learning and Information Theory for Word Guessing Games**

This repository implements a comprehensive pipeline for solving **Alphalock**, a word-guessing game similar to **Wordle**, using:
1. **Information Theory (IT)**: A deterministic heuristic solver that leverages entropy to minimize the solution space.
2. **Reinforcement Learning (RL)**: A policy gradient-based agent that dynamically balances exploration and exploitation to solve the game.

The project also includes detailed logging, visualizations, and training workflows, enabling an in-depth analysis of model performance.

**The research paper for this project can be found here: [AlphaLock-RL.pdf](AlphaLock-RL.pdf)**


## **Table of Contents**
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Pipeline Steps](#pipeline-steps)
- [Research Paper](#research-paper)

## **Features**

- **Dual Solvers**:
  - **Information Theory Solver**: Implements entropy-based word selection for deterministic and efficient guessing.
  - **Reinforcement Learning Solver**: Uses a policy network to dynamically adjust strategies based on game state.
- **Detailed Analytics**:
  - Logs and JSON files track alpha/beta values, rewards, guesses, and pool sizes for each episode.
- **Visualization**:
  - Plots training metrics (episode rewards, guesses).
  - Visualizes `alpha` values against guess numbers and pool sizes.
- **Flexible Reward System**:
  - Rewards success and intermediate progress while penalizing failures.

## **Project Structure**

```
alphalock-rl/
├── README.md
├── data/
│   ├── 4letter_word_freqs.json            # Raw word frequency data
│   ├── 4letter_word_freqs_relative.json   # Normalized word frequencies
│   └── create_word_freq.py                # Script for frequency preprocessing
├── rl_model/
│   ├── trained_rl_agent.pth               # Current trained RL agent model
│   └── trained_rl_agent_old.pth           # Previous version of the RL model
├── rl_stats/
│   ├── alpha_beta_mapping.json            # Alpha/Beta mapping for episodes
│   ├── episode_rewards.json               # Episode reward logs
│   ├── episode_guesses.json               # Episode guess logs
│   └── visualize_alpha.py                 # Visualization for Alpha vs Guess Number
│   └── visualize_training.py              # Training statistics visualization
├── src/
│   ├── config.py                          # Configuration for model and environment
│   ├── data_preprocessor.py               # Data preprocessing for word frequencies
│   ├── entropy_calculation.py             # Entropy calculations for IT solver
│   ├── infotheory_solver.py               # Information Theory-based solver logic
│   ├── main.py                            # Script to compare IT and RL solvers
│   ├── rl_agent.py                        # Policy network and RL agent implementation
│   ├── rl_environment.py                  # Alphalock game environment
│   ├── rl_trainer.py                      # Training loop for the RL agent
│   ├── pattern_utils.py                   # Feedback generation logic
│   ├── reward_calculator.py               # Reward function for RL
│   └── utils.py                           # Utility functions for JSON I/O
├── game.py                                # Runs the AlphaLock game with the user
└── requirements.txt                       # Required Python libraries
```

## **Prerequisites**

- **Python 3.8+**
- Required Python libraries:
  - `torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, and others listed in `requirements.txt`.

## **Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/alphalock-rl.git
   cd alphalock-rl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess data if necessary:
   ```bash
   cd src
   python data_preprocessor.py
   ```

## **Pipeline Steps**

### **1. Training the RL Agent**

Train the RL agent using the provided training script:

```bash
python rl_trainer.py
```

**Training Details**:
- **Batch Updates**: The agent updates its policy after a fixed number of episodes.
- **Reward System**:
  - Rewards **successful guesses** with a value proportional to the number of remaining attempts, incentivizing faster solutions.
  - Provides **intermediate rewards** for reducing the size of the solution pool after each guess, encouraging effective exploration.
  - Penalizes failures with a fixed negative reward if the solution is not guessed within the limit of 10 attempts, discouraging inefficient strategies.
- **Score System**:
  - Combines **Information Theory (IT)** and **Relative Word Frequency (RWF)** to compute a weighted score for each possible guess, guiding the agent's decision-making:
    \[
    \text{Score(word)} = \alpha \cdot \text{IT(word)} + \beta \cdot \text{RWF(word)}
    \]
  - **Weights** (\( \alpha \) for IT, \( \beta \) for RWF) dynamically balance exploration (maximizing information gain) and exploitation (favoring frequent words). Favoring frequent words is particularly effective because game designers often choose common words as solutions to ensure puzzles are accessible and relatable. By prioritizing frequent words, the solver increases its chances of faster success, leveraging linguistic biases inherent in word-based games.
  - The RL agent **learns to adjust \( \alpha \) and \( \beta \) dynamically** based on the game state (e.g., remaining pool size, attempts left), allowing it to adapt its strategy as the game progresses.

### **2. Evaluating Models**

Compare the Information Theory solver and the RL agent using `main.py`:

```bash
python main.py
```

**Key Outputs**:
- Average guesses and runtime for each solver.
- Logs of intermediate guesses, feedback, and pool sizes.

```
Information Theory-based solver: Avg Guesses = 5.08, Avg Time = 12.60 seconds.
RL-based solver: Avg Guesses = 3.63, Avg Time = 12.24 seconds.
```

## **Research Paper**

<img width="610" alt="Screenshot 2024-12-28 at 06 19 33" src="https://github.com/user-attachments/assets/54500b71-7953-4c04-9095-09a1fbf7d058" />
<img width="609" alt="Screenshot 2024-12-28 at 06 19 47" src="https://github.com/user-attachments/assets/c869d499-ffd3-4e58-adee-3ad0d477bb2d" />
<img width="610" alt="Screenshot 2024-12-28 at 06 20 06" src="https://github.com/user-attachments/assets/6d85d22a-2e97-478c-8cff-2bb1c0c66c3f" />
<img width="611" alt="Screenshot 2024-12-28 at 06 20 17" src="https://github.com/user-attachments/assets/c453807b-02ec-451f-a11e-7852343d2026" />
<img width="612" alt="Screenshot 2024-12-28 at 06 20 27" src="https://github.com/user-attachments/assets/48ba3991-3333-4172-b45a-9362156f919f" />
<img width="611" alt="Screenshot 2024-12-28 at 06 20 40" src="https://github.com/user-attachments/assets/aed44656-4ded-4b67-b9ab-1f79cddb017c" />
<img width="611" alt="Screenshot 2024-12-28 at 06 20 49" src="https://github.com/user-attachments/assets/5a74d8fa-819c-4b50-86e3-82edcc007d45" />
<img width="611" alt="Screenshot 2024-12-28 at 06 20 58" src="https://github.com/user-attachments/assets/727c791b-7cf3-47fb-a416-3c720dacd83c" />
<img width="610" alt="Screenshot 2024-12-28 at 06 21 07" src="https://github.com/user-attachments/assets/111c19d7-b5df-4abd-9c07-4f4172b75cb8" />


