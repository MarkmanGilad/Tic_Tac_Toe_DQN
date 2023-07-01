from TicTacToe import TicTacToe
import numpy as np
from Graphics import *
import torch
import math
import random

class AI_Agent:
    def __init__(self, player, env: TicTacToe, graphics: Graphics = None, Q_table_PATH = None, train = True):
        self.env = env
        self.player = player
        if Q_table_PATH is None:
            self.Q = dict ()
        else:
            self.load_Q(Q_table_PATH)
        self.train = train

    def get_Q (self, state, action):
        if (state, action) in self.Q:
            return self.Q[(state, action)]
        else:
            return 0

    # greedy action
    def get_Q_action(self, state = None):
        if state is None:
            state = self.env.state
        actions = self.legal_actions(state)
        best_value = -10
        best_action = None
        for action in actions:
            key = (state, action)
            if key in self.Q:
                Q_value = self.Q[(state, action)]
            else:
                Q_value = 0
            if Q_value > best_value:
                best_value = Q_value
                best_action = action
                
        return best_action
    
    # epsilon-greedy
    def get_action (self, state, epoch = None):
        if self.train:
            r = random.random()
            epsilon = self.epsilon_greedy(epoch)
        else:
            r = 1
            epsilon = 0
        if r < epsilon:
            action = random.choice(self.legal_actions(state))
        else:
            action = self.get_Q_action(state = state)
        return action

    def legal_actions (self, state):
        board = state.board
        indices = np.where(board == 0)
        actions = list(zip(indices[0], indices[1]))
        return actions

    def load_Q (self, PATH):
        self.Q = torch.load(PATH)

    def save_Q (self, PATH):
        torch.save(self.Q, PATH)

    def epsilon_greedy (self, epoch):
        start = 1.0
        final = 0.01
        decay = 10000
        return final + (start - final)* math.exp(-1*epoch/decay)

    def __call__(self, events= None, state=None):
        return self.get_action(state)

    