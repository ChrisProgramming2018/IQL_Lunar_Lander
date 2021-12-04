# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import sys
import json
import gym
import argparse
import torch
from utils import mkdir, write_into_file
from replayBuffer import ReplayBuffer
from agent_iql import Agent
import wandb 
import time
from models import QNetwork, CNN, QCNetwork
import torch.optim as optim
from collections import namedtuple, deque
from utils import time_format
import torch.nn as nn



def main(args):
    """ """
    with open(args.param, "r") as f:
        param = json.load(f)
    print("use the env {} ".format(param["env_name"]))
    print(param)
    print("Start Programm in {}  mode".format(args.mode))
    #env = gym.make(param["env_name"])
    if args.mode == "search":
        param["lr_pre"] = args.lr_pre
        param["lr_iql_q"] = args.lr_iql_q
        param["lr"] = args.lr
        param["fc1_units"] = args.fc1_units
        param["fc2_units"] = args.fc2_units
        param["clip"] = args.clip
        param["buffer_path"] = args.buffer_path
    param["run"] = args.run
    param["encoder_path"] = args.e_path
    param["laten_space"] = args.laten_space
    param["seed"] = args.seed
    param["locexp"] = args.locexp
    param["wandb"] = args.wandb
    param["render"] = args.render
    param["buffer_size"] = args.buffer_size
    param["action_shape"] = args.action_shape
    config =  param
    text = str(param)
    write_into_file(str(param["locexp"]) + "/hyperparameters", text)
    run_name = "Run_{}_seed_{}_cnn_networks".format(config["run"], config["seed"])
    if config["wandb"]:
        wandb.init(
                project="master_lab_inverse_rl",
                name=run_name,
                sync_tensorboard=True,
                monitor_gym=True,
                )
    t0 = time.time()
    memory = ReplayBuffer((config["size"], config["size"],3), (config["action_shape"], ), config["buffer_size"] + 1, config["device"])
    memory.load_memory(config["buffer_path"])
    print("Buffer_size ", memory.idx)
    state_size = 512
    fc1 = 64
    fc2 = 64
    model = QCNetwork(state_size, args.action_shape, fc1, fc2, args.seed).to(config["device"])
    model.train()
    print("lerning rate {}".format(args.lr))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_window = deque(maxlen=500)
    eval_steps = 500
    total_time_steps = 100000
    for step in range(total_time_steps):
        text = "Steps {}  \ {}  time {}  \r".format(step, total_time_steps, time_format(time.time() - t0))
        print(text, end = '')
        states, next_states, actions, dones = memory.expert_policy(config["batch_size"])
        loss = model_1(model, optimizer, states, actions)
        loss_window.append(loss)
        # writer.add_scalar('Predict_loss', loss, self.steps)
        if step % eval_steps == 0:
            print("Ave loss {} last 500 steps ".format(np.mean(loss_window)))
            eval_predicter()

def eval_predicter():
    return 


def model_1(model, optimizer, state, action):
    state = torch.swapaxes(state, 1,3) / 255.
    output = model(state)
    output = output.squeeze(0)
    y = action.type(torch.long).squeeze(1)
    import pdb; pdb.set_trace()
    loss = nn.CrossEntropyLoss()(output, y)
    optimizer.zero_grad()
    loss.backward()
    import pdb; pdb.set_trace()
    # torch.nn.utils.clip_grad_norm_(self.predicter.parameters(), 1)
    optimizer.step()
    return loss

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="hypersearch", type=str)
    parser.add_argument('--lr_iql_q', default=1e-5, type=float)
    parser.add_argument('--lr_iql_r', default=1e-5, type=float)
    parser.add_argument('--lr_q_sh', default=1e-5, type=float)
    parser.add_argument('--lr_pre', default=5e-4, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=64, type=int)
    parser.add_argument('--fc2_units', default=64, type=int)
    parser.add_argument('--action_shape', default=1, type=int)
    parser.add_argument('--buffer_size', default=20000, type=int)
    parser.add_argument('--clip', default=-1, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--mode', default="iql", type=str)
    parser.add_argument('--wandb', default=False, type=bool)
    parser.add_argument('--buffer_path', default="expert_data_space_invadars_average_reward_1174.44_size_20000", type=str)
    parser.add_argument('--e_path')
    parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--laten_space', type=int)
    arg = parser.parse_args()
    main(arg)
