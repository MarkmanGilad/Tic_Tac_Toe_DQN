import math
import random
from typing import Any
import torch
import torch.nn as nn
import numpy as np
from DQN import DQN
from State import State

# epsilon Greedy
epsilon_start = 1
epsilon_final = 0.01
epsiln_decay = 10000

# epochs = 1000
# batch_size = 64
gamma = 0.99 
MSELoss = nn.MSELoss()

class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None) -> None:
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train(train)
        self.player = player
        self.env = env

    def train (self, train):
          self.train = train
          if train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_action (self, state: State, epoch = 0, events= None):
        epsilon = self.epsilon_greedy(epoch)
        rnd = random.random()
        actions = self.env.legal_actions(state)
        if self.train and rnd < epsilon:
            return random.choice(actions)
        
        state_tensor = state.toTensor()
        action_np = np.array(actions)
        action_tensor = torch.from_numpy(action_np)
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1))
        state_action = torch.cat((expand_state_tensor, action_tensor ), dim=1)
        
        Q_values = self.DQN(state_action)
        max_index = torch.argmax(Q_values)
        return actions[max_index]


    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loss (self, Q_value, rewards, Q_next_Values, Dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return MSELoss(Q_value, Q_new)

    def __call__(self, events= None, state=None) -> Any:
        return self.get_action(state)

