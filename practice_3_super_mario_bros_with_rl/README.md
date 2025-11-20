# Deep Q-Learning for Super Mario Bros  
## con Classic Gym + `gym-super-mario-bros`, porque la nueva interfaz no funciona

## 1. Introduction

In this assignment, you will implement and train a **Deep Q-Network (DQN)** agent to play *Super Mario Bros* using:

- **Classic Gym API** (Gym 0.26)
- **`gym-super-mario-bros`** + **`nes-py`**
- A simple **MLP DQN** (no CNN)
- **Image preprocessing** (grayscale, resize to 84×84)
- **Experience replay** + **target network**

You will complete missing parts of the RL code and test your trained agent against a random policy and against yourself.

## 2. Repository Structure

You receive a starter package:

```

env.py              # Mario environment + preprocessing
networks.py         # DQN neural network
replay_buffer.py    # Experience replay buffer
agent.py            # YOUR CODE: DQN agent skeleton
train.py            # YOUR CODE: Training loop skeleton
play.py             # Play / evaluate / human control

````

You must implement:

### In `agent.py`:
- `select_action` — epsilon-greedy policy
- `optimize_model` — DQN update step

### In `train.py`:
- `train_dqn` — full training loop with:
  - Epsilon scheduling
  - Interaction loop
  - Replay buffer usage
  - Target network updates

Use `play.py` to run:

- `run_agent_episode` — watch the agent play
- `evaluate_agent_vs_random` — computer vs computer
- `run_human_episode` — control Mario yourself

## 3. Environment API


### Reset the environment
```python
obs = env.reset()
````

### Step:

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

The game-over flag is `done`.

### Observations:

After preprocessing wrappers, each observation is:

* Grayscale image
* Shape: **(84, 84)**
* Type: `np.uint8` (0–255)

Your preprocessing converts it to:

* Flattened tensor, shape `(1, 84*84)`
* `float32` values in `[0, 1]`

## 4. Tasks in Detail

### Task 1 — `select_action` (ε-greedy)

In `agent.py`:

```python
def select_action(self, state, epsilon):
```

Implement:

* With probability ε → random action
* Otherwise → `argmax(Q(state, a))` from `policy_net`

### Task 2 — `optimize_model`

Implement the DQN update:

1. Sample a batch from replay memory.
2. Compute:

   * `Q(s, a)` from policy net
   * `max_a' Q_target(s', a')` from target net
3. Compute target:

[
y = r + \gamma \max_{a'}Q_{\text{target}}(s', a') \cdot (1 - done)
]

4. Loss: **Huber loss** between `Q(s,a)` and `y`
5. Backpropagation + gradient clipping + optimizer step


### Task 3 — `train_dqn`

Implement inside the training loop:

1. `obs = env.reset()`
2. Convert obs → state tensor
3. Compute ε using linear schedule
4. Loop until `done`:

   * `action = agent.select_action(state, epsilon)`
   * Step env
   * Preprocess next obs
   * Add transition to replay memory
   * Call `agent.optimize_model(batch_size)`
   * Periodically copy policy_net → target_net

Print:

```
Episode 7/200  Reward: 1340  Epsilon: 0.421
```

Return the trained `agent`.

## 5. Running and Testing the Agent

### Watch the trained agent:

```python
from mario_rl.play import run_agent_episode
run_agent_episode(agent, render=True)
```

### Compare against random:

```python
from mario_rl.play import evaluate_agent_vs_random
evaluate_agent_vs_random(agent, episodes=5)
```

### Play yourself:

```python
from mario_rl.play import run_human_episode
run_human_episode()
```

You will be asked to enter an integer corresponding to a `SIMPLE_MOVEMENT` action.


## 6. Installation

### Conda environment (recommended)

You have an environment_supermario.yml` file in the folder

```
name: mario-rl
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pip
  - jupyter
  - pip:
      - numpy==1.26.4
      - gym==0.26.2
      - gym-super-mario-bros==7.4.0
      - nes-py==8.2.1
      - opencv-python==4.7.0.72
      - torch
      - torchvision
```

Create the env:

```bash
conda env create -f environment.yml
conda activate mario-rl
```

## How to run

Either run the notebook `notebook_interface.ipynb`, or:

Start with:

```python
agent = train_dqn(num_episodes=1)
```

You will have to edit:
  - `agent.py`
  - `train.py`


# What to do


- Implement the basic functions required by the code
- Play (and let the computer play)
    
Desired plots
  - Reward curves or logged results
  - Agent vs random comparison

Desired stuff
  - Human vs agent comparison (you play against the agent :D )
  - Discussion of performance (e.g. size of training episode
  - Discussion of possible improvements

