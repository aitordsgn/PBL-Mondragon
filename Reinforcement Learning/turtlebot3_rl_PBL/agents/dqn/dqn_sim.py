###############################################################################
#                                                                             #
# Project: TurtleBot3 Reinforcement Learning                                  #
# Author: Aitor Rey Ortega                                                    #
# University: Mondragon Unibertsitatea                                        #
#                                                                             #
#                                                                             #
# File: dqn_sim.py                                                            #
#                                                                             #
###############################################################################


import gym
import mu_gym
import os

import numpy as np
import random
import math

import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import datetime

class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNet, self).__init__()
        seed = np.random.randint(1, 1000)
        torch.manual_seed(seed)

        self.criterion = torch.nn.MSELoss()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

def get_action(q_values, action_size, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        _, action = torch.max(q_values, 1)
    return action.numpy()[0]

def main(idx):
    start_time = datetime.datetime.now()

    # Create the environment
    env_name = 'turtlebot3-ros1-dis-v0'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print('state size:', state_size)
    print('action size:', action_size)

    hidden_size = 16
    lr_init = 0.001
    gamma = 0.99

    # Check the device here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Monitoring
    save_path = './save_model/'
    tensorboard_path = '../../tensorboard_session/{}_DQN_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name)

    # Tensorboard
    writer = SummaryWriter(tensorboard_path)

    saved_model_path = "save_model_S1/model.pth.tar"

    # Instantiate the QNet model
    q_net = QNet(state_size, action_size, hidden_size)
    q_net.load_state_dict(torch.load(saved_model_path))
    q_net.to(device)

    step = 0
    eval_interval = 10
    best_reward = -np.inf
    val_rewards = []

    epsilon = 0.0  # No exploration, only exploitation

    results = []  # List to store results for each episode

    for episode in range(100):  # Number of episodes to run
        done = False
        episode_reward = 0
        episode_steps = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state = torch.Tensor(state).to(device)

        while not done:
            step += 1
            episode_steps += 1

            q_values = q_net(state)
            action = get_action(q_values, action_size, epsilon)

            # Handle environments that return 4 or 5 values from step
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                next_state, reward, done, info = step_result

            next_state = np.reshape(next_state, [1, state_size])
            next_state = torch.Tensor(next_state).to(device)

            state = next_state
            episode_reward += reward

        writer.add_scalar('episode_reward/inference', episode_reward, episode)
        val_rewards.append(episode_reward)

        # Check if the goal was reached
        goal_reached = episode_reward >= 200  # Example threshold for goal reached

        results.append({
            'episode': episode,
            'reward': episode_reward,
            'goal_reached': goal_reached,
            'steps': episode_steps
        })

        print("Episode Reward:", episode_reward)
        print("Steps:", episode_steps)
        print("Goal Reached:", goal_reached)
        print("Info:", info)

    # Save results to a file
    with open("inference_results.csv", "w") as f:
        for result in results:
            f.write(f"{result['episode']}, {result['reward']},  {result['goal_reached']}, {result['steps']}\n")

    end_time = datetime.datetime.now()
    print("Inference ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Total inference time:", end_time - start_time)

    return val_rewards

if __name__ == '__main__':
    rewards = main(1)
    print("Rewards from inference:", rewards)
