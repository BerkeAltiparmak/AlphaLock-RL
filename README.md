# AlphaLock-RL

### **Information Theory-Based Solver**

The information theory-based solver (IT solver) is designed to optimize the guessing process by selecting guesses that maximize the reduction in uncertainty. This is achieved through entropy calculations that prioritize guesses with the most evenly distributed feedback patterns, effectively narrowing down the solution pool with each step. Below is a detailed technical explanation of the components and logic:

---

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

---

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

---

#### **c. Entropy-Based Guess Selection (`infotheory_solver.py`)**
- **Goal**: Choose the guess that maximizes entropy, thereby splitting the solution pool most evenly.
- **Steps**:
  1. **Entropy Calculation**:
     - Calculate normalized entropy values for all allowed guesses relative to the current solution pool.
  2. **Best Guess**:
     - Select the guess with the highest entropy score, which is expected to provide the most informative feedback.

---

### **2. Iterative Game Simulation**

The IT solver iteratively updates the solution pool based on feedback:
1. **First Guess**:
   - Start with a predefined first guess (e.g., "sera") that has been shown to provide good entropy properties empirically.
2. **Feedback Processing**:
   - Use `generate_pattern` to compute feedback for the guess.
   - Update the solution pool by keeping only words that match the feedback pattern.
3. **Subsequent Guesses**:
   - Repeat the entropy calculation and guess selection steps for the reduced pool.
   - Continue until:
     - Only one solution remains, or
     - The maximum number of attempts is reached.

---

### **3. Precomputations and Optimizations**

#### **Precomputing Feedback Patterns**
- **Functionality**:
  - `precompute_patterns` generates feedback patterns for all possible word pairs.
- **Purpose**:
  - Avoid redundant calculations during entropy evaluation.
- **Storage**:
  - The feedback patterns are stored in a dictionary with keys as `(guess, solution)` pairs and values as feedback tuples.

---

### **4. Integration with the Simulation (`simulate_game`)**

- **Inputs**:
  - `allowed_words`: The full set of possible guesses.
  - `possible_words`: The current solution pool.
  - `solution`: The target word to guess.
  - `first_guess`: Optional predefined first guess.
- **Outputs**:
  - `guesses`: List of guesses made during the game.
  - `possible_words_list`: Snapshot of the solution pool after each guess.
  - `feedback_list`: Feedback patterns for each guess.
- **Process**:
  - On each turn:
    - Calculate entropies for all guesses in `allowed_words`.
    - Select the guess with the highest entropy.
    - Update the solution pool based on feedback.
  - Return all relevant logs for analysis and debugging.

---

### **5. Advantages of the IT Solver**

1. **Efficiency**:
   - By prioritizing entropy, the solver ensures each guess provides maximum information about the solution pool.
2. **Deterministic**:
   - Given the same inputs, the solver will always produce the same sequence of guesses, making it reproducible.
3. **Interpretable**:
   - Entropy provides a clear mathematical rationale for each guess, which can be validated and debugged.

---

### **6. Challenges and Limitations**

1. **Scalability**:
   - Entropy calculations can be computationally expensive for large solution pools.
2. **Precomputations**:
   - Storing all possible feedback patterns requires significant memory for larger word lists.
3. **Inflexibility**:
   - The deterministic nature of the solver means it cannot adapt dynamically to changes in gameplay mechanics or unforeseen edge cases.
  

### **Reinforcement Learning-Based Solver**

This section provides an in-depth explanation of the reinforcement learning (RL) approach implemented to solve the **Alphalock** game. Unlike the Information Theory-based solver, which relies on deterministic heuristics, the RL-based solver learns dynamic strategies through interaction with the environment, optimizing a policy for efficient word guessing.

---

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

---

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

#### **c. Training and Optimization**
- **Discounted Rewards**:
  - Future rewards are discounted to prioritize immediate rewards:
    \[
    R_t = r_t + \gamma \cdot R_{t+1}
    \]
- **Policy Gradient**:
  - Log probabilities of actions are scaled by their associated rewards, and the loss is minimized to update the policy:
    \[
    \text{Loss} = - \sum (\log \pi(a_t|s_t) \cdot R_t)
    \]
- **Gradient Clipping**:
  - Gradients are clipped to avoid instability from exploding gradients.

---

### **3. Training Workflow**

#### **a. Episode Structure**
1. **Initialization**:
   - The environment is reset, selecting a random solution.
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

---

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

---

### **5. Comparison with Information Theory Solver**

#### **Advantages**
1. **Dynamic Strategy**:
   - RL adapts `alpha` and `beta` weights dynamically, unlike the static heuristic approach.
2. **Reward-Driven Learning**:
   - RL optimizes for overall game success rather than individual guess efficiency.

#### **Challenges**
1. **Training Complexity**:
   - RL requires extensive training data and time to converge.
2. **Exploration-Exploitation Tradeoff**:
   - The model must balance entropy reduction with leveraging word frequencies effectively.

---

### **6. Key Insights**

1. **Alpha-Beta Behavior**:
   - `alpha` decreases as pool size reduces, shifting focus from exploration to exploitation.
   - `beta` increases as fewer guesses remain, prioritizing frequent words for final guesses.

2. **Intermediate Rewards**:
   - Intermediate rewards for pool reduction accelerate learning by providing granular feedback.

3. **Robustness**:
   - RL models showed better adaptability to edge cases compared to the deterministic IT solver.

---

### **7. Future Enhancements**

1. **Reward Shaping**:
   - Refine intermediate rewards to account for partial matches or near-optimal guesses.

2. **Hybrid Models**:
   - Combine RL and Information Theory to leverage the strengths of both approaches.

3. **Model Scalability**:
   - Extend the architecture for higher-dimensional games or games with additional constraints.



The reinforcement learning (RL) approach used in this project can be classified as **Policy Gradient-Based Reinforcement Learning**. Specifically, it aligns most closely with **Vanilla Policy Gradient (VPG)** methods, sometimes referred to as **REINFORCE**.

Here's why:

---

### **Characteristics of the RL Approach**

1. **Policy-Based Learning**:
   - The agent directly learns a policy \( \pi(a|s) \), which maps states \( s \) to probabilities over actions \( a \).
   - The policy network outputs continuous values (`alpha` and `beta`), which are used to calculate scores for possible guesses.

2. **Stochastic Policy**:
   - The policy outputs probabilities for actions, allowing for exploration during training.
   - This is key in learning a balance between exploration (`alpha`) and exploitation (`beta`).

3. **Reward-Based Updates**:
   - The policy is updated based on the discounted cumulative rewards \( R_t \) for each trajectory.
   - The goal is to maximize the expected return:
     \[
     J(\theta) = \mathbb{E}_\pi [R_t]
     \]

4. **Policy Gradient Optimization**:
   - The policy is optimized by taking the gradient of the expected return with respect to the policy parameters:
     \[
     \nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]
     \]
   - This is implemented by scaling the log probabilities of actions by their rewards and minimizing the negative log likelihood.

5. **Batch Updates**:
   - The policy network is updated after accumulating experiences over multiple episodes (batch learning), which helps stabilize training.

---

### **Reinforcement Learning Type**

#### **Policy Gradient Methods**
This project belongs to the category of **Policy Gradient Methods**:
- The agent directly optimizes the policy \( \pi_\theta(a|s) \), rather than learning a value function or a Q-function.
- The output of the policy network (i.e., `alpha` and `beta`) parameterizes the decision-making process, enabling the agent to dynamically adjust its priorities between exploration and exploitation.

#### **On-Policy Learning**
- The agent updates its policy using data collected from the current policy.
- The rewards and log probabilities stored during an episode are specific to the current policy and are used to compute updates.

#### **Vanilla Policy Gradient (REINFORCE)**
- The agent performs **episodic updates** using the collected rewards and log probabilities.
- There is no explicit baseline (e.g., a value function) subtracted from the rewards, making this method closely resemble the REINFORCE algorithm.

---

### **Why Not Actor-Critic or Q-Learning?**

1. **No Value Function**:
   - There is no separate critic (value network) estimating the state or state-action value, as would be the case in Actor-Critic methods.
   - Updates rely solely on the rewards obtained during an episode.

2. **Continuous Action Space**:
   - Q-learning and its variants (like DQN) are better suited for discrete action spaces.
   - Here, the actions (`alpha` and `beta`) are continuous values, requiring policy-based methods like REINFORCE.

3. **No Temporal Difference Updates**:
   - Temporal difference (TD) methods, common in Q-learning or Actor-Critic algorithms, are absent. Instead, the approach uses full episode returns for updates.

---

### **Comparison to Common RL Algorithms**

| Feature                | This Project                | REINFORCE       | Actor-Critic        | Q-Learning/DQN         |
|------------------------|-----------------------------|-----------------|---------------------|------------------------|
| Action Space           | Continuous (`alpha, beta`) | Discrete/Cont.  | Discrete/Cont.      | Discrete               |
| Policy Type            | Stochastic                 | Stochastic      | Stochastic          | Deterministic          |
| Value Function         | None                       | None            | Critic (Baseline)   | State-Action Q-Values  |
| Update Frequency       | Batch (episodic)           | Batch (episodic)| After each step     | After each step        |
| Optimization           | Policy Gradient            | Policy Gradient | Actor and Critic    | Bellman Equation       |

---

### **Summary**
This approach is **Vanilla Policy Gradient (REINFORCE)** with episodic updates, specifically tailored for a continuous action space. Its strength lies in leveraging a learned policy to dynamically prioritize exploration and exploitation during the Alphalock game, offering adaptability and robustness for solving word-guessing tasks.
