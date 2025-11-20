
"""
Environment helpers for Super Mario Bros using gym-super-mario-bros
with the classic Gym API:

    obs = env.reset()
    obs, reward, done, info = env.step(action)

You do NOT need Gymnasium for this assignment.
"""

import numpy as np
import torch
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation


def make_mario_env(world: int = 1, stage: int = 1):
    """
    Create a Super Mario Bros environment with:
      - Discrete SIMPLE_MOVEMENT action space.
      - Grayscale 84x84 observations.
    """
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_id, 
        apply_api_compatibility=True,
        render_mode="human",  # optional but nice for seeing the game)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=False)  # (H, W)
    env = ResizeObservation(env, 84)                 # (84, 84)
    return env


def preprocess_state(state, device):
    """
    Flatten and normalize (84,84) grayscale image to (1, 84*84) tensor.
    """
    state = np.array(state, dtype=np.float32) / 255.0
    state = state.flatten()
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
