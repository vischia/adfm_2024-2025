
"""
Play & evaluation helpers (classic Gym API).
"""

import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from .env import make_mario_env, preprocess_state
from .agent import MarioDQNAgent


def run_agent_episode(agent: MarioDQNAgent, world: int = 1, stage: int = 1, render: bool = True) -> float:
    env = make_mario_env(world, stage)
    obs, info = env.reset()
    state = preprocess_state(obs, agent.device)
    done = False
    total_reward = 0.0

    while not done:
        if render:
            env.render()
        action = agent.select_action(state, epsilon=0.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = preprocess_state(next_obs, agent.device)
        total_reward += reward
        state = next_state

    env.close()
    print(f"[Agent] Total reward: {total_reward:.2f}")
    return total_reward


def run_random_episode(world: int = 1, stage: int = 1, render: bool = True) -> float:
    env = make_mario_env(world, stage)
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if render:
            env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"[Random] Total reward: {total_reward:.2f}")
    return total_reward


def evaluate_agent_vs_random(agent: MarioDQNAgent, episodes: int = 5):
    agent_scores = []
    random_scores = []
    for i in range(episodes):
        print(f"\n=== Match {i+1}/{episodes} ===")
        agent_scores.append(run_agent_episode(agent, render=False))
        random_scores.append(run_random_episode(render=False))

    print("\n=== Summary (computer vs computer) ===")
    print(f"Agent average reward:  {np.mean(agent_scores):.2f}")
    print(f"Random average reward: {np.mean(random_scores):.2f}")


def run_human_episode(world: int = 1, stage: int = 1):
    env = make_mario_env(world, stage)
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    print("\nActions (SIMPLE_MOVEMENT):")
    for idx, action in enumerate(SIMPLE_MOVEMENT):
        print(f"{idx}: {action}")
    print("Press Ctrl+C to quit.")

    while not done:
        env.render()
        try:
            a = int(input("Choose action index: "))
            if a < 0 or a >= len(SIMPLE_MOVEMENT):
                print("Invalid index, try again.")
                continue
        except ValueError:
            print("Please enter a valid integer.")
            continue

        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"[Human] Total reward: {total_reward:.2f}")
