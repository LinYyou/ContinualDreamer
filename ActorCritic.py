
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from copy import deepcopy
from tqdm import tqdm


class A2C(nn.Module):##Implementation using A2c
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(A2C, self).__init__()
        self.device = device
        self._lambda = 0.95 
        self.update_freq = 10
        self.action_dim = 3 if self.continuous else 5
        self.gamma = 0.99
        self.c_1 = 0.5
        self.c_2 = 0.01
        num_channels = 3
        self.actor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

        self.critic = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.act_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.crt_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 1e-4)

        

    def forward(self, state):
        
        action_logits = self.actor(state)
        value = self.critic(state)

        return action_logits, value


    def get_action(self, state):
        
        action_logits = self.actor(state)
        
        dist = Categorical(logits = action_logits)
        action = dist.sample()

        return action

    def evaluate_action(self, states, actions):##Evaluate the actions choosen using cricit, the function will return log_probs, action entropy and value of actual state
        
        action_logits, value = self.forward(states.detach())
        action_dist = Categorical(logits = action_logits)
        
        log_prob = action_dist.log_prob(actions[:,-1])
        entropy = action_dist.entropy()

        return log_prob, entropy, value

    def act(self, state):#Given the state, it should output the action according to the policy

        action_tensor = self.get_action(state.unsqueeze(0))
        action = action_tensor.item()
        return action_tensor

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    def reinforce_loss(self, values, rewards, discounts):

        targets = torch.zeros(values.shape)
        for t in reversed(range(len(values))):
            if t == len(values) - 1:
                targets[t] = rewards[t] + values[t]
            else:
                targets[t] = rewards[t] + discounts[t] * ((1-self._lambda)*values[t] + self._lambda*targets[t+1]) 
        
        assert len(targets) == len(values)
        loss = 0.5*(torch.sum(values - targets.detach()))**2
        return loss, targets
            
    def update(self, states, actions, rewards, discounts):
        rho = 1.0 
        eta = 1e-4
        log_probs, entropies, values = self.evaluate_action(states, actions)
        
        crt_loss, targets = self.reinforce_loss(values, rewards, discounts)
        
        act_loss = torch.sum((rho)*log_probs*(values.detach() - targets.detach())) - (1-rho)*torch.sum(targets.detach()) - eta*torch.sum(entropies)
        
        
        self.crt_optimizer.zero_grad()
        self.act_optimizer.zero_grad()
        crt_loss.backward(retain_graph=True)
        act_loss.backward(retain_graph=True)
        self.crt_optimizer.step()
        self.act_optimizer.step()

        
        return act_loss, crt_loss
        