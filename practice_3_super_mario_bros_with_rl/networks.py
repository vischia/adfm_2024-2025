
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Simple feed-forward DQN with flattened image input."""
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
