import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent_iql import Agent, mkdir
import json

from dqn_agent import DQNAgent



agent = DQNAgent(state_size=8, action_size=4, seed=0)



def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    with open ("param.json", "r") as f:
        config = json.load(f)
    
    agent_r = Agent(state_size=8, action_size=4,  config=config)
    agent_r.load("models/35000-")
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        env_reward = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            env_reward += reward
            state = torch.from_numpy(state).float().unsqueeze(0).to(agent_r.device)
            action = torch.from_numpy(np.array(action)).float().unsqueeze(0).to(agent_r.device)
            action = action.type(torch.int64)
            reward = agent_r.R_local(state).gather(1, action.unsqueeze(0))
            
            state = next_state
            score += reward.item()
            if done:
                print("")
                print("")
                print("")
                print("env reward ", env_reward)
                print("Model reward ", score)
                agent_r.save("Q_model/-{}".format(i_episode))
                break 
        scores_window.append(env_reward)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_r.pth')
            break
    return scores

scores = dqn()
