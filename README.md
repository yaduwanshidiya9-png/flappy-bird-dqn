# 🐦 Flappy Bird AI: Deep Q-Network (DQN) Implementation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-brightgreen)](https://gymnasium.farama.org/)

An autonomous AI agent trained to master Flappy Bird using **Deep Q-Networks (DQN)** with **PyTorch** and **Gymnasium**. This project demonstrates how Reinforcement Learning can be used to solve complex decision-making tasks.

---

## 🚀 Key Features
- **Deep Q-Network (DQN):** Neural Network architecture built with PyTorch to predict optimal actions.
- **Experience Replay:** Uses a memory buffer to store past experiences, ensuring stable training.
- **Reward Shaping:** Custom rewards for survival and successfully passing through pipes.
- **Epsilon-Greedy Strategy:** Balances exploration (learning new moves) and exploitation (using learned knowledge).

---

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch
- **Environment:** Gymnasium & Flappy Bird Gymnasium
- **Graphics:** Pygame
- **Config Management:** YAML

---

## How it Works

The agent interacts with the Flappy Bird environment and learns optimal actions using Deep Q-Learning.  
It uses a neural network to approximate Q-values and improves over time through experience replay and reward shaping.


## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/yaduwanshidiya9-png/flappy-bird-dqn.git
cd flappy-bird-dqn
```

### Setup Environment

Because `flappy-bird-gymnasium` requires an older version of Python, it is highly recommended to create a dedicated environment using a tool like Conda.

1. Create a Conda environment with Python 3.10:
```bash
conda create -n flappy_env python=3.10.0
```

2. Activate the virtual environment:
```bash
conda activate flappy_env
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

The agent relies on hyperparameters defined in `parameters.yaml`. You need to specify a parameter set name when running the script. Assuming `flappybirdv0` is one of the parameter sets defined in `parameters.yaml`:

### Training the Agent

To train a new agent, run:

```bash
python agent.py flappybirdv0 --train
```

During training, the model weights and training logs will be saved in the `runs/` directory as `<parameter_set>.pt` and `<parameter_set>.log`.

### Testing the Agent

To watch the trained agent play, run the script without the `--train` flag. It will load the previously saved weights and render the environment:

```bash
python agent.py flappybirdv0
```

### Basic Game Environment

You can also run a simple test script to view the environment (press `SPACE` to flap):

```bash
python game_flappy_bird.py
```

## Project Structure

- `agent.py`: Main Deep Q-Learning agent implementing training and evaluation logic.
- `dqn.py`: Deep Q-Network model architecture built with PyTorch.
- `experience_replay.py`: The Replay Memory buffer used for storing past experiences to train the agent.
- `game_flappy_bird.py`: Basic script using `flappy_bird_gymnasium` and `pygame` for testing the environment.
- `parameters.yaml`: Stores training hyperparameters.
- `requirements.txt`: Python package dependencies.



