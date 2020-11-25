import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device


        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.int8)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.k = 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        self.k +=1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
    
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        return obses, actions, rewards, next_obses, not_dones_no_max


    def expert_policy(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
    
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        dones = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, next_obses, actions, dones

    def add_expert(self, obs, action, reward, next_obs, done, done_no_max):
        
        for a in range(4):
            self.k +=1
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.actions[self.idx], a)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.next_obses[self.idx], next_obs)
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
            
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0




    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)
        
        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)
        
        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)
        
        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)

        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))

        print("save buffer to {}".format(filename))
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
        
        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)
        
        with open(filename + '/rewards.npy', 'rb') as f:
            self.rewards = np.load(f)
        
        with open(filename + '/not_dones.npy', 'rb') as f:
            self.not_dones = np.load(f)
        
        with open(filename + '/not_dones_no_max.npy', 'rb') as f:
            self.not_dones_no_max = np.load(f)

        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
