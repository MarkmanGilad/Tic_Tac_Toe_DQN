import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters
input_size = 11 # state: board = 3 * 3 + action 1 * 2
layer1 = 128
layer2 = 64
output_size = 1 # Q(s,a)


class DQN (nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if torch.cuda.is_available:
            self.device = torch.device('cpu') # 'cuda'
        else:
            self.device = torch.device('cpu')
        
        self.linear1 = nn.Linear(input_size, layer1, device=self.device)
        self.linear2 = nn.Linear(layer1, layer2, device=self.device)
        self.output = nn.Linear(layer2, output_size, device=self.device)
        
    def forward (self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
    def load_params(self, path):
        self.DQN.load_state_dict(torch.load(path))

    def save_params(self,path):
        torch.save(self.DQN.state_dict(), path)


        