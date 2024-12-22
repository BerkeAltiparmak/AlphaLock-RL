# **Alphalock-RL: Reinforcement Learning and Information Theory for Word Guessing Games**

This repository implements a comprehensive pipeline for solving **Alphalock**, a word-guessing game similar to Wordle, using:
1. **Information Theory (IT)**: A deterministic heuristic solver that leverages entropy to minimize the solution space.
2. **Reinforcement Learning (RL)**: A policy gradient-based agent that dynamically balances exploration and exploitation to solve the game.

The project also includes detailed logging, visualizations, and training workflows, enabling an in-depth analysis of model performance.

The research paper for this project can be found here: [AlphaLock-RL.pdf](AlphaLock-RL.pdf)


## **Table of Contents**
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Pipeline Steps](#pipeline-steps)
  - [1. Training the RL Agent](#1-training-the-rl-agent)
  - [2. Evaluating Models](#2-evaluating-models)
- [Future Plans](#future-plans)

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

![guess_number_vs_alpha_CI](https://github.com/user-attachments/assets/7365d7f9-05d7-4518-8bce-161774f84446)


## **Future Plans**

1. **Generalization**:
   - Extend the RL agent to work with games of varying lengths and constraints.
2. **Enhanced Reward Shaping**:
   - Improve intermediate rewards to guide the RL agent even further.
3. **Real-Time Application**:
   - Integrate the solvers into an interactive application for real-world testing.

## **Information Theory-Based Solver**

The information theory-based solver (IT solver) is designed to optimize the guessing process by selecting guesses that maximize the reduction in uncertainty. This is achieved through entropy calculations that prioritize guesses with the most evenly distributed feedback patterns, effectively narrowing down the solution pool with each step. Below is a detailed technical explanation of the components and logic:

### **1. Core Components**

#### **a. Pattern Generation (`pattern_utils.py`)**
- **Goal**: Calculate feedback patterns between a guessed word and the correct answer.
- **Feedback Types**:
  - **MISS (Gray)**: Letter is absent in the solution.
  - **MISPLACED (Yellow)**: Letter is present but in the wrong position.
  - **EXACT (Green)**: Letter is in the correct position.
- **Algorithm**:
  1. First pass checks for exact matches (green feedback) and marks the corresponding indices as used.
  2. Second pass checks for misplaced letters (yellow feedback), ensuring no double counting of letters already matched.
- **Output**: A tuple representing the feedback pattern for each letter (e.g., `(EXACT, MISS, MISPLACED, MISS)`).

#### **b. Entropy Calculation (`entropy_calculation.py`)**
- **Goal**: Measure the information gain of each guess by calculating the entropy of feedback distributions.
- **Logic**:
  1. **Feedback Distributions**:
     - For a given guess, compute how often each feedback pattern occurs across all possible solutions.
     - Example: If the guess "WORD" can produce patterns `{(EXACT, MISS, MISPLACED, MISS): 3, (MISS, MISS, EXACT, MISS): 5}`, the distribution is `{pattern1: 3, pattern2: 5}`.
  2. **Entropy**:
     - Use the feedback distribution to calculate entropy:
       \[
       H = - \sum p_i \log_2(p_i)
       \]
       where \( p_i \) is the probability of each feedback pattern.
  3. **Normalization**:
     - Normalize entropy values to the range \([0, 1]\) to facilitate comparisons:
       \[
       H_{\text{normalized}} = \frac{H - H_{\text{min}}}{H_{\text{max}} - H_{\text{min}}}
       \]
       This ensures that the scoring remains robust across different solution pool sizes.

#### **c. Entropy-Based Guess Selection (`infotheory_solver.py`)**
- **Goal**: Choose the guess that maximizes entropy, thereby splitting the solution pool most evenly.
- **Steps**:
  1. **Entropy Calculation**:
     - Calculate normalized entropy values for all allowed guesses relative to the current solution pool.
  2. **Best Guess**:
     - Select the guess with the highest entropy score, which is expected to provide the most informative feedback.
       
### **2. Iterative Game Simulation**

The IT solver iteratively updates the solution pool based on feedback:
1. **First Guess**:
   - Start with a predefined first guess (e.g., "sera") that has been calculated to be the best first guess by the Information Theory model.
2. **Feedback Processing**:
   - Use `generate_pattern` to compute feedback for the guess.
   - Update the solution pool by keeping only words that match the feedback pattern.
3. **Subsequent Guesses**:
   - Repeat the entropy calculation and guess selection steps for the reduced pool.
   - Continue until:
     - Only one solution remains, or
     - The maximum number of attempts is reached.
     

### **3. Precomputations and Optimizations**

#### **Precomputing Feedback Patterns**
- **Functionality**:
  - `precompute_patterns` generates feedback patterns for all possible word pairs.
- **Purpose**:
  - Avoid redundant calculations during entropy evaluation.
- **Storage**:
  - The feedback patterns are stored in a dictionary with keys as `(guess, solution)` pairs and values as feedback tuples.
    

## **Reinforcement Learning-Based Solver**

This section provides an in-depth explanation of the reinforcement learning (RL) approach implemented to solve the **Alphalock** game. Unlike the Information Theory-based solver, which relies on deterministic heuristics, the RL-based solver learns dynamic strategies through interaction with the environment, optimizing a policy for efficient word guessing.

### **1. Core Concepts**

#### **a. Objective**
The RL solver aims to train a policy network to balance two key components dynamically:
- **Exploration (Information Theory)**: Reducing solution pool uncertainty by prioritizing high-entropy guesses.
- **Exploitation (Relative Word Frequency)**: Favoring common words based on their real-world usage likelihood.

#### **b. State Space**
The environment (`rl_environment.py`) provides the following state features:
1. **`pool_entropy`**: Size of the remaining solution pool, representing uncertainty.
2. **`attempts_remaining`**: Remaining number of guesses.
3. **`feedback_history`**: Previous feedback patterns, summarized as a count.

#### **c. Action Space**
The RL agent (`rl_agent.py`) predicts two continuous values:
- **`alpha`**: Weight for entropy-based scoring (exploration).
- **`beta`**: Weight for relative word frequency scoring (exploitation).
These weights determine the scoring function:
\[
\text{Score} = \alpha \cdot \text{IT} + \beta \cdot \text{RWF}
\]

#### **d. Reward Signal**
Rewards are designed to encourage fast and efficient solutions:
- **Success**: Rewards are scaled linearly or exponentially with the number of attempts saved.
- **Failure**: A fixed penalty is applied for exceeding the maximum attempts.
- **Intermediate Rewards**: Partial rewards are given for reducing the solution pool size.

### **2. Key Components**

#### **a. Environment**
The environment (`AlphalockEnvironment`) simulates the game mechanics:
- **Solution Selection**:
  - Solutions are randomly selected based on smoothed word frequencies (using logarithmic scaling).
- **Feedback Generation**:
  - Feedback patterns are generated using `generate_pattern`, which matches letters based on their correctness and position.
- **Rewards**:
  - Rewards are calculated based on success, failure, and intermediate progress using `compute_reward`.

#### **b. Policy Network**
The policy network (`PolicyNetwork`) is a fully connected neural network:
- **Input**: 
  - State features: `pool_entropy`, `attempts_remaining`, `feedback_history`.
- **Hidden Layers**: Two ReLU-activated layers for feature extraction.
- **Output**:
  - Two continuous values: `alpha` and `beta` (in [0, 1] using sigmoid activation).
 

#### **b. RL Approach: REINFORCE (VPG)**
For the RL-based learner, we implemented a **Policy Gradient-Based Reinforcement Learning** from scratch. Specifically, it's a **Vanilla Policy Gradient (VPG)**, also known as **REINFORCE**.

1. **Policy-Based Learning**:
   - The agent directly learns a policy \( \pi(a|s) \), which maps states \( s \) to probabilities over actions \( a \).
   - The policy network outputs continuous values (`alpha` and `beta`), which are used to calculate scores for possible guesses.

2. **Stochastic Policy**:
   - The policy outputs probabilities for actions, which is key in learning a balance between exploration (`alpha`) and exploitation (`beta`).

#### **c. Training and Optimization**
- **Reward-Based Updates**:
   - The policy is updated based on the discounted cumulative rewards \( R_t \) for each trajectory. The goal is to maximize the expected return:
     \[
     J(\theta) = \mathbb{E}_\pi [R_t]
     \]

- **Policy Gradient Optimization**:
   - The policy is optimized by taking the gradient of the expected return with respect to the policy parameters:
     \[
     \nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]
     \]
   - This is implemented by scaling the log probabilities of actions by their rewards and minimizing the negative log likelihood.
     
- **Gradient Clipping**:
  - Gradients are clipped to avoid instability from exploding gradients.

- **Batch Updates**:
   - The policy network is updated after accumulating experiences over multiple episodes (batch learning), which helps stabilize training.

### **3. Training Workflow**

#### **a. Episode Structure**
1. **Initialization**:
   - The environment is reset, selecting a random solution based on the frequency of the words, reflecting the linguistic biases inherent in humans.
2. **Guessing Loop**:
   - For each guess:
     - The agent predicts `alpha` and `beta`.
     - The best word is selected based on the scoring function.
     - Feedback is generated, and the solution pool is updated.
     - Rewards are stored for policy updates.
3. **Termination**:
   - The episode ends when the solution is guessed, or the maximum attempts are exceeded.
4. **Policy Update**:
   - After a batch of episodes, the policy network is updated using the collected rewards and log probabilities.

#### **b. Data Logging**
Training results are logged in JSON files:
- **`alpha_beta_mapping.json`**:
  - Logs `alpha` and `beta` values for each guess.
- **`episode_rewards.json`**:
  - Tracks total rewards per episode.
- **`episode_guesses.json`**:
  - Tracks the number of guesses used per episode.

### **4. Scoring and Decision-Making**

#### **a. Scoring Function**
The RL solver combines Information Theory (IT) and Relative Word Frequency (RWF) to calculate scores for all possible guesses:
\[
\text{Score(word)} = \alpha \cdot \text{IT(word)} + \beta \cdot \text{RWF(word)}
\]

#### **b. Guess Selection**
Using the scoring function, the agent selects the word with the highest score:
```python
scores = calculate_scores(allowed_words, possible_words, word_frequencies, alpha, beta)
best_word = max(scores, key=scores.get)
```

### **Why Not Actor-Critic or Q-Learning?**

1. **No Value Function**:
   - There is no separate critic (value network) estimating the state or state-action value, as would be the case in Actor-Critic methods.
   - Updates rely solely on the rewards obtained during an episode.

2. **Continuous Action Space**:
   - Q-learning and its variants (like DQN) are better suited for discrete action spaces.
   - Here, the actions (`alpha` and `beta`) are continuous values, requiring policy-based methods like REINFORCE.

3. **No Temporal Difference Updates**:
   - Temporal difference (TD) methods, common in Q-learning or Actor-Critic algorithms, are absent. Instead, the approach uses full episode returns for updates.


### **Comparison to Common RL Algorithms**

| Feature                | This Project                | REINFORCE       | Actor-Critic        | Q-Learning/DQN         |
|------------------------|-----------------------------|-----------------|---------------------|------------------------|
| Action Space           | Continuous (`alpha, beta`) | Discrete/Cont.  | Discrete/Cont.      | Discrete               |
| Policy Type            | Stochastic                 | Stochastic      | Stochastic          | Deterministic          |
| Value Function         | None                       | None            | Critic (Baseline)   | State-Action Q-Values  |
| Update Frequency       | Batch (episodic)           | Batch (episodic)| After each step     | After each step        |
| Optimization           | Policy Gradient            | Policy Gradient | Actor and Critic    | Bellman Equation       |

To summarize, our approach is **Vanilla Policy Gradient (REINFORCE)** with episodic updates, specifically tailored for a continuous action space. Its strength lies in leveraging a learned policy to dynamically prioritize exploration and exploitation during the Alphalock game, offering adaptability and robustness for solving word-guessing tasks.
