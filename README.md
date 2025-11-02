#  Deep Q-Network (DQN) Implementation for CartPole-v1

This project implements a **Deep Q-Network (DQN)** from scratch using **PyTorch** to solve the classic **CartPole-v1** reinforcement learning environment from **OpenAI Gymnasium**.

The agent learns to balance a pole on a moving cart by trial and error — using **experience replay** and a **target network** for more stable training.

---

##  Project Overview

- **Algorithm:** Deep Q-Network (DQN)
- **Environment:** CartPole-v1 (OpenAI Gymnasium)
- **Framework:** PyTorch
- **Goal:** Balance the pole as long as possible by learning optimal actions.

---

##  Key Features

- Custom DQN neural network architecture (3 fully-connected layers)
- Epsilon-greedy action selection for exploration vs. exploitation
- Replay memory for experience sampling
- Soft update of target network weights (`tau`)
- Visualization of training progress (episode durations)
- Fully commented and educational code structure

---

##  Dependencies

Make sure you have the following libraries installed:

```bash
pip install torch gymnasium matplotlib

How to Run

Clone this repository:git clone https://github.com/<your-username>/cartpole-dqn.git
cd cartpole-dqn

Run the main file:python dqn_cartpole.py

A pygame window will open showing the CartPole environment while the agent trains.
The training duration graph will update live using Matplotlib.

Training Visualization

During training, episode durations are plotted to monitor the agent’s performance:
Hyperparameters
| Parameter      | Description                 | Default |
| -------------- | --------------------------- | ------- |
| `batch_size`   | Replay memory sample size   | 128     |
| `gamma`        | Discount factor             | 0.99    |
| `eps_start`    | Starting epsilon value      | 0.9     |
| `eps_end`      | Minimum epsilon value       | 0.05    |
| `eps_decay`    | Epsilon decay rate          | 1000    |
| `tau`          | Target network update rate  | 0.005   |
| `lr`           | Learning rate               | 1e-4    |
| `memory_size`  | Replay memory capacity      | 10,000  |
| `num_episodes` | Number of episodes to train | 250     |

Learning Objective

The DQN agent approximates the Q-value function using a neural network.
It learns the expected cumulative reward for each action, allowing it to make optimal decisions to keep the pole balanced.

Results

After sufficient training, the agent should be able to balance the pole for over 200 timesteps consistently.

License

This project is open source and available under the MIT License.

Author

Ayberk Caf
Reinforcement Learning Project — 2025
Built with using PyTorch and OpenAI Gymnasium.
