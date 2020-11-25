import gym
import random
import torch
import numpy as np
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from utils import mkdir

env = gym.make('LunarLander-v2')
env.seed(0)

print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
agent = DQNAgent(state_size=8, action_size=4, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
memory = ReplayBuffer((8,), (1,), 20000, 'cuda')
n_episodes = 40
max_t = 500
eps = 0
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        score += reward
        memory.add(state, action, reward, next_state, done, done)
        state = next_state
        # env.render()
        if done:
            print("Episode {}  Reward {}".format(i_episode, score))
            break
