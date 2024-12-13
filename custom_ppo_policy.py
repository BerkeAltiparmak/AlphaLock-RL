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
        actions, values, log_prob = super(CustomPPOPolicy, self).forward(obs, deterministic)
        alpha_beta = self.alpha_beta_net(self.features_extractor(obs))
        return actions, values, log_prob, alpha_beta
