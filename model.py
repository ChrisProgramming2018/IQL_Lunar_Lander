import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
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
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



def normilze_weights(net):
    params = {}
    import pdb; pdb.set_trace()
    for name, param in sorted(net.named_parameters()):
        import pdb; pdb.set_trace()
        layer, param_name = name.split(".")
        if layer not in params:
            params[layer] = []
        if param_name == "bias":
            params[layer].append(param.view(-1, 1))
        else:
            params[layer].append(param)
    
    norms = {}
    for layer in params:
        norms[layer] = torch.norm(torch.cat(params[layer], dim=1))
    
    for name, param in sorted(net.named_parameters()):
        layer, param_name = name.split(".")
        param.data.copy_(param.data/(norms[layer]+1e-6).detach())
    import pdb; pdb.set_trace()




class NQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
