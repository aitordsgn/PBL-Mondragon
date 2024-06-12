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
      seed = np.random.randint(1,1000)
      torch.manual_seed(seed)

      self.criterion = torch.nn.MSELoss()

      self.fc1 = nn.Linear(state_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, action_size)

  def forward(self, x):
      x = torch.tanh(self.fc1(x))
      q_values = self.fc2(x)
      return q_values

def train_model(q_net, target_q_net, optimizer, mini_batch, gamma):
  mini_batch = np.array(mini_batch, dtype=object)
  states = np.vstack(mini_batch[:, 0])
  actions = list(mini_batch[:, 1]) 
  rewards = list(mini_batch[:, 2]) 
  next_states = np.vstack(mini_batch[:, 3])
  masks = list(mini_batch[:, 4]) 

  actions = torch.LongTensor(actions)
  rewards = torch.Tensor(rewards)
  masks = torch.Tensor(masks)
  
  criterion = torch.nn.MSELoss()

  # get Q-value
  q_values = q_net(torch.Tensor(states))
  q_value = q_values.gather(1, actions.unsqueeze(1)).view(-1)

  # get target
  target_next_q_values = target_q_net(torch.Tensor(next_states))
  target = rewards + masks * gamma * target_next_q_values.max(1)[0]
  
  loss = criterion(q_value, target.detach())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

def get_action(q_values, action_size, epsilon):
  if np.random.rand() <= epsilon:
      return random.randrange(action_size)
  else:
      _, action = torch.max(q_values, 1)
  return action.numpy()[0]

def update_target_model(q_net, target_q_net):
  target_q_net.load_state_dict(q_net.state_dict())

def main(idx):
    start_time = datetime.datetime.now()

    #Create the environment
    env_name = 'turtlebot3-ros1-dis-v0'
    #env_name = 'turtlebot3-ros1-dis-safe-v0'
    #env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
   
    print('state size:', state_size) 
    print('action size:', action_size)
    
    # Hyperparameters
    best_reward = -np.inf
    max_reward = 200
    max_epsidode = 20000
    max_step = 400000

    epsilon = 1.0
    batch_size = 128
    initial_exploration = 2 * batch_size
    epsilon_decay = 0.00005
    update_target = 10

    hidden_size = 16
    lr_init = 0.005
    gamma = 0.99

    delta = 5

    replay_buffer_size = 200000

    # check the device here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Monitoring
    eval_interval = 10
    save_path = './save_model_S2/'

    tensorboard_path = '../../tensorboard_session/{}_DQN_S2{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env_name)
    #Tensorboard
    writer = SummaryWriter(tensorboard_path)

    # Buffer
    losses = []
    val_rewards = []
    weights = []

    lr = lr_init

    q_net = QNet(state_size, action_size, hidden_size)
    target_q_net = QNet(state_size, action_size, hidden_size)
    optimizer = optim.Adam(q_net.parameters(), lr)

    replay_buffer = deque(maxlen=replay_buffer_size)
    
    step = 0    

    for episode in range(max_epsidode):
        done = False
        episode_reward = 0
        

        state = env.reset()
       
        #state = state[0]
        #print ("Shape", state.shape())
        state = np.reshape(state, [1, state_size])
        
        done = truncated = False

        while not (done or truncated):

            step += 1

            q_values = q_net(torch.Tensor(state))

            action = get_action(q_values, action_size, epsilon)

            next_state, reward, done, truncated, info = env.step(action)
            

            next_state = np.reshape(next_state, [1, state_size])
            
            reward = reward if not done or episode_reward == max_reward-1 else -1
            mask = 0 if done else 1

            replay_buffer.append((state, action, reward, next_state, mask))

            state = next_state
            episode_reward += reward

            if step > initial_exploration:
              epsilon -= epsilon_decay
              epsilon = max(epsilon, 0.1)

              mini_batch = random.sample(replay_buffer, batch_size)
              q_net.train(), target_q_net.train()
                
              loss = train_model(q_net, target_q_net, optimizer, mini_batch, gamma)
              writer.add_scalar('train/loss', loss.item(), step)
              
              losses = np.append(losses,loss.item())
              
              if step % update_target == 0:
                update_target_model(q_net, target_q_net)

        writer.add_scalar('episode_reward/train', episode_reward, episode)
        print ("Episode Reward", episode_reward)
        print ("Info", info)

        if episode % eval_interval == 0:
          done = False
          val_episode_reward = 0
          state = env.reset()
          #state = state[0]
          state = np.reshape(state, [1, state_size])
          with torch.no_grad():
            done = truncated = False
            while not (done or truncated):
              q_values = q_net(torch.Tensor(state))
              action = get_action(q_values, action_size, -1)
              next_state, reward, done, truncated, info = env.step(action)
              next_state = np.reshape(next_state, [1, state_size])
              state = next_state
              val_episode_reward += reward
            val_rewards.append(val_episode_reward)
          writer.add_scalar('episode_reward/val', val_episode_reward, episode)
          print ('Mean val reward is ', np.mean(val_rewards[-20:]))

          if (np.mean(val_rewards[-20:]) > best_reward and episode > 20):
            if not os.path.isdir(save_path):
                os.makedirs(save_path)            
            ckpt_path = save_path + 'model.pth.tar'
            torch.save(q_net.state_dict(), ckpt_path)
            print('The new best reward {:.2f} and the last best reward {:.2}'.format(np.mean(val_rewards[-20:]), best_reward))
            best_reward = np.mean(val_rewards[-10:])
            # Close the processes

        if (np.mean(val_rewards[-20:]) >= max_reward and episode > 20):
          if not os.path.isdir(save_path):
              os.makedirs(save_path)            
          ckpt_path = save_path + 'model.pth.tar'
          torch.save(q_net.state_dict(), ckpt_path)
          print('Max reward achieved {:.2f}. So end'.format(max_reward))
          # Close the processes
          #env.close()
          break     

        if (step > max_step):
          if not os.path.isdir(save_path):
              os.makedirs(save_path)            
          ckpt_path = save_path + 'model.pth.tar'
          torch.save(q_net.state_dict(), ckpt_path)
          print('Running steps exceeds {:.2f}. So end'.format(step))
          # Close the processes
          #env.close()
          break
    end_time = datetime.datetime.now()
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Total training time:", end_time - start_time)

    return val_rewards, weights

if __name__ == '__main__':
      y, x = main(1)

