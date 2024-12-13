from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from torch import nn

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Network to output alpha and beta dynamically
        self.alpha_beta_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Outputs alpha and beta
            nn.Softmax(dim=-1)  # Ensure alpha + beta = 1
        )

    def forward(self, obs, deterministic=False):
        # Standard PPO outputs: actions, values, log_probs
        actions, values, log_prob = super(CustomPPOPolicy, self).forward(obs, deterministic)

        # Calculate alpha and beta dynamically
        alpha_beta = self.alpha_beta_net(self.features_extractor(obs))

        # Save alpha and beta for retrieval
        self.alpha_beta = alpha_beta

        return actions, values, log_prob

    def get_alpha_beta(self):
        """Retrieve the dynamically calculated alpha and beta."""
        return self.alpha_beta
