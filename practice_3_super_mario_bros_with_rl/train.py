
"""
Training loop skeleton for Mario DQN (student version, classic Gym API).
"""

import torch

from .env import make_mario_env, preprocess_state
from .agent import MarioDQNAgent


def train_dqn(
    num_episodes: int = 200,
    batch_size: int = 32,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay_episodes: int = 150,
    target_update_interval: int = 1000,
) -> MarioDQNAgent:
    """STUDENT TODO: implement the full DQN training loop.

    Suggested algorithm:
        - Create env with make_mario_env()
        - Instantiate MarioDQNAgent.
        - For each episode:
            * Reset env, preprocess initial state:
                  obs = env.reset()
                  state = preprocess_state(obs, device)
            * Compute epsilon via linear decay:
                  frac = min(episode_idx / epsilon_decay_episodes, 1.0)
                  epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            * Loop until done:
                - Choose action via agent.select_action(state, epsilon).
                - Step env; get obs, reward, done, info.
                - Preprocess next observation.
                - Store transition in replay memory.
                - Call agent.optimize_model(batch_size).
                - Periodically update target_net.
                - Accumulate total_reward.
        - Close env and return the trained agent.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    env = make_mario_env()
    n_actions = env.action_space.n
    agent = MarioDQNAgent(n_actions=n_actions, device=device)

    global_step = 0

    for episode in range(num_episodes):
        # TODO: implement the main training loop as described above.
        raise NotImplementedError("train_dqn main loop is not implemented")

    env.close()
    return agent


if __name__ == "__main__":
    # Example (after you implement train_dqn and the agent methods):
    # agent = train_dqn(num_episodes=50)
    # from .play import run_agent_episode
    # run_agent_episode(agent)
    pass
