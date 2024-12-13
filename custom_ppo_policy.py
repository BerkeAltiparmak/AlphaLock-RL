from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from torch import nn

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Add two additional outputs: alpha and beta
        self.alpha_beta_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output alpha and beta
            nn.Softmax(dim=-1)  # Ensure they sum to 1
        )

    def forward(self, obs, deterministic=False):
        # Compute the standard outputs: actions, values, and log_probs
        actions, values, log_prob = super(CustomPPOPolicy, self).forward(obs, deterministic)

        # Compute alpha and beta as trainable parameters
        alpha_beta = self.alpha_beta_net(self.features_extractor(obs))

        # Save alpha and beta as part of the policy’s internal state
        self.alpha_beta = alpha_beta

        return actions, values, log_prob

    def get_alpha_beta(self):
        """Get the alpha and beta values (trainable weights)."""
        return self.alpha_beta
