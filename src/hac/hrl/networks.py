# hrl/networks.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    """
    For discrete actions, we output logits and sample with softmax.
    For continuous, you would output means and use squashed Gaussian.
    """

    def __init__(self, obs_dim, goal_dim, act_dim):
        super().__init__()
        self.q = MLP(obs_dim + goal_dim, act_dim)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.q(x)  # logits for discrete

class Critic(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim):
        super().__init__()
        self.v = MLP(obs_dim + goal_dim + act_dim, 1)

    def forward(self, obs, goal, act_onehot):
        x = torch.cat([obs, goal, act_onehot], dim=-1)
        return self.v(x)
