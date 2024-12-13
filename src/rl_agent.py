import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize a simple feedforward neural network for the policy.
        
        Parameters:
        - input_dim (int): Dimension of the input state.
        - hidden_dim (int): Number of hidden units.
        - output_dim (int): Dimension of the output (2 for alpha and beta).
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)  # Ensures output probabilities sum to 1

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return torch.sigmoid(output)  # Outputs alpha and beta in range [0, 1]

class RLAgent:
    def __init__(self, state_dim, hidden_dim, lr=0.001, gamma=0.99):
        """
        Reinforcement learning agent that learns optimal weights for Alphalock.
        
        Parameters:
        - state_dim (int): Dimension of the state space.
        - hidden_dim (int): Number of hidden units in the policy network.
        - lr (float): Learning rate for optimizer.
        - gamma (float): Discount factor.
        """
        self.gamma = gamma
        self.policy_network = PolicyNetwork(state_dim, hidden_dim, output_dim=2)  # Output is [alpha, beta]
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.log_probs = []  # Store log probabilities for policy updates
        self.rewards = []  # Store rewards for each step

    def select_action(self, state):
        """
        Select an action (alpha, beta) given the current state.
        
        Parameters:
        - state (np.ndarray): Current environment state.

        Returns:
        - action (np.ndarray): Chosen [alpha, beta] values.
        - log_prob (torch.Tensor): Log probability of the chosen action.
        """
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32)
        print(f"State Tensor: {state_tensor}")  # Debugging input state
        action_probs = torch.clamp(self.policy_network(state_tensor), min=1e-6)  # Prevent zero probabilities
        action = action_probs.detach().numpy()
        log_prob = torch.sum(torch.log(action_probs))  # Log probability of the chosen action
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        """
        Store the reward for the current step.
        
        Parameters:
        - reward (float): Reward received at the current step.
        """
        self.rewards.append(reward)

    def update_policy(self):
        """
        Perform a policy update using the collected rewards and log probabilities.
        """
        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        
        # Normalize rewards for stability
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        print(f"Discounted Rewards: {discounted_rewards}")  # Debugging rewards

        # Compute loss
        loss = 0
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            loss -= log_prob * reward  # Gradient ascent

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)  # Prevent gradient explosion
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.rewards = []

    def save_model(self, file_path):
        """
        Save the policy network to a file.

        Parameters:
        - file_path (str): Path to save the model.
        """
        torch.save(self.policy_network.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Load the policy network from a file.

        Parameters:
        - file_path (str): Path to load the model from.
        """
        self.policy_network.load_state_dict(torch.load(file_path))
        self.policy_network.eval()
