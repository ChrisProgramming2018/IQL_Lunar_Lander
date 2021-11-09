import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.uint8)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.bool)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, next_obses, action, reward, done):
        np.copyto(self.obses[self.idx], obs.astype(np.uint8))
        np.copyto(self.next_obses[self.idx], next_obses.astype(np.uint8))
        np.copyto(self.actions[self.idx], action.astype(np.uint8))
        np.copyto(self.rewards[self.idx], np.float32(reward))
        np.copyto(self.dones[self.idx], np.bool(done))
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)    
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device).float()
        rewards = torch.as_tensor(rewards, device=self.device).float()
        dones = torch.as_tensor(dones, device=self.device).float()
        return obses, next_obses, actions, rewards, dones
    
    def expert_policy(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)    
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        dones = self.dones[idxs]
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device).float()
        dones = torch.as_tensor(dones, device=self.device).float()
        return obses, next_obses, actions, dones
    
    
    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        if not os.path.exists(filename):
            os.makedirs(filename)
        
        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses) 
        
        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses) 
        
        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions) 
        
        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards) 
        
        with open(filename + '/dones.npy', 'wb') as f:
            np.save(f, self.dones) 
        
        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))
        
        print("save buffer to {}".format(filename))
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
        
        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)
        
        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/rewards.npy', 'rb') as f:
            self.rewards = np.load(f)

        with open(filename + '/dones.npy', 'rb') as f:
            self.dones = np.load(f)
  
        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
        print("Loaded {} with size of {}".format(filename, self.idx))
