
"""
Mario DQN agent skeleton (student version, classic Gym API).

Students must implement:
    - MarioDQNAgent.select_action (epsilon-greedy)
    - MarioDQNAgent.optimize_model (DQN update)
"""

import random
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from .networks import DQN
from .replay_buffer import ReplayMemory


class MarioDQNAgent:
    def __init__(
        self,
        n_actions: int,
        input_dim: int = 84 * 84,
        lr: float = 1e-4,
        gamma: float = 0.99,
        replay_capacity: int = 50000,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma

        self.policy_net = DQN(input_dim, n_actions).to(device)
        self.target_net = DQN(input_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_capacity)

    def select_action(self, state, epsilon: float) -> int:
        """STUDENT TODO: epsilon-greedy policy.

        - With probability epsilon: return a random action in [0, n_actions - 1].
        - Otherwise: return argmax_a Q(state, a) according to policy_net.
        """
        raise NotImplementedError("select_action is not implemented")

    def optimize_model(self, batch_size: int):
        """STUDENT TODO: one DQN update step using replay memory.

        Steps:
            1. Return early if there are fewer transitions than batch_size.
            2. Sample a batch of transitions (state, action, reward, next_state, done).
            3. Compute current Q(s,a) and targets:
                   target = r + gamma * max_a' Q_target(s', a') * (1 - done)
            4. Compute Huber (smooth L1) loss and do a gradient step.
        """
        raise NotImplementedError("optimize_model is not implemented")
