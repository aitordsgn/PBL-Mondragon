###############################################################################
#                                                                             #
# Project: TurtleBot3 Reinforcement Learning                                  #
# Author: Aitor Rey Ortega                                                    #
# University: Mondragon Unibertsitatea                                        #
#                                                                             #
#                                                                             #
# File: Gdp.py                                                                #
#                                                                             #
###############################################################################

import glob
import io
import base64
#from IPython.display import HTML
#from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display
# from gym.utils.save_video import save_video

import matplotlib.pyplot as plt

import os
import gym
import mu_gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR


from torch.utils.tensorboard import SummaryWriter
import datetime

import math




    
    
class NNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(NNet, self).__init__()
        self.fc1= nn.Linear(state_size, hidden_size)
        self.fc2= nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        self.saved_log_probs = []
        self.rewards = []
        self.loss = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.fc3(x), dim=1)
        return action_prob



def compute_loss(optimizer, policy, gamma):
    G = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)


    for log_prob, G in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * G)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum() / len(returns)
    policy_loss.backward()
    optimizer.step()

    policy.loss = policy_loss

    del policy.rewards[:]
    del policy.saved_log_probs[:]

    return policy



def get_action(policy, state, train=True):
    action_prob = policy(torch.tensor(state).unsqueeze(0).float())

    m = Categorical(action_prob)
    action = m.sample()

    if train:
        action_log_prob = m.log_prob(action)
        policy.saved_log_probs.append(action_log_prob)

    return action.item()



def main(index):
    start_time = datetime.datetime.now()

    env_name = 'turtlebot3-ros1-dis-PBL'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print('state size:', state_size)
    print('action size:', action_size)


    """************************************************************
    ** Hyperparameters
    ************************************************************"""
    max_reward = 200  # Maximum expected reward per episode. Used as a threshold for determining if the goal is reached.
    max_episode = 2000  # Maximum number of episodes to run during training or inference.
    max_step = 20000 * 5  # Maximum number of steps to run in total (5 times 20000 steps).
    roll_out_memory_size = 500  # Size of the memory buffer for storing experience rollouts.
    hidden_size = 64  # Size of the hidden layers in the neural network.
    lr_init = 0.001  # Initial learning rate for the optimizer.
    lr_decay = 0.9  # Factor by which the learning rate is multiplied every lr_decay_steps.
    lr_decay_steps = 200  # Number of steps after which the learning rate is decayed.
    gamma = 0.99  # Discount factor for future rewards.
    log_interval = 50  # Interval of episodes at which to log training data to TensorBoard.
    eval_interval = 10  # Interval of episodes at which to evaluate the policy.



    save_path = './save_model_gdp/'
    logdir = './logs'
    best_reward = -np.inf

    tensorboard_path = './tensorboard_session/VPG_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name)
    writer = SummaryWriter(tensorboard_path)

    losses = []
    rewards = []

    policy = NNet(state_size, action_size, hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr_init,weight_decay = 1e-4)
    scheduler = StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay)
    returns = deque(maxlen=roll_out_memory_size)

    step = 0

    for episode in range(max_episode):
        train = True
        policy.train()
        episode_reward = 0

        state = env.reset()  # for mu environments turtlebot, it returns only the state
        step= 0
        done = False
        while not done:
            step += 1
            action = get_action(policy, state, train=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = reward if not done or episode_reward == max_reward - 1 else -1
            policy.rewards.append(reward)
            episode_reward += reward

        writer.add_scalar('episode_reward/train', episode_reward, episode)
        print("Episode Reward", episode_reward)
        print("Info", info)

        loss = compute_loss(optimizer, policy, gamma)
        losses.append(policy.loss.detach().item())
        writer.add_scalar('loss/train', policy.loss.detach().item(), episode)

        scheduler.step()

        if episode % eval_interval == 0:
            done = False
            episode_reward = 0
            state = env.reset()  # for mu environments turtlebot, it returns only the state
            policy.eval()
            train = False
            with torch.no_grad():
                while not done:
                    action = get_action(policy, state, train=False)
                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
            rewards.append(episode_reward)
            writer.add_scalar('episode_reward/val', episode_reward, episode)

        if (np.mean(rewards[-20:]) > best_reward and episode > 20):
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            ckpt_path = save_path + 'model.pth.tar'
            torch.save(policy.state_dict(), ckpt_path)
            print('The new best reward {:.2f} and the last best reward {:.2f}'.format(np.mean(rewards[-20:]), best_reward))
            best_reward = np.mean(rewards[-10:])

        if (np.mean(rewards[-20:]) >= max_reward and episode > 20):
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            ckpt_path = save_path + 'model.pth.tar'
            torch.save(policy.state_dict(), ckpt_path)
            print('Max reward achieved {:.2f}. So end'.format(max_reward))
            break

        if (step > max_step):
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            ckpt_path = save_path + 'model.pth.tar'
            torch.save(policy.state_dict(), ckpt_path)
            print('Running steps exceeds {:.2f}. So end'.format(step))
            break

    end_time = datetime.datetime.now()
    print("Inference ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Total inference time:", end_time - start_time)

if __name__ == '__main__':
    main(1)
