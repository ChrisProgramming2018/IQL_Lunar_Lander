# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Critic (q function) Model."""
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.leak = 0.01
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-4, 3e-4)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
