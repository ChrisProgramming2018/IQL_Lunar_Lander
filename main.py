# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import sys
import json
import gym
import argparse
from utils import mkdir, write_into_file
from train_iql import train


def main(args):
    """ """
    with open(args.param, "r") as f:
        param = json.load(f)
    print("use the env {} ".format(param["env_name"]))
    print(param)
    print("Start Programm in {}  mode".format(args.mode))
    env = gym.make(param["env_name"])
    if args.mode == "search":
        param["lr_pre"] = args.lr_pre
        param["lr_iql_q"] = args.lr_iql_q
        param["lr"] = args.lr
        param["fc1_units"] = args.fc1_units
        param["fc2_units"] = args.fc2_units
        param["clip"] = args.clip
        param["buffer_path"] = args.buffer_path
    param["run"] = args.run
    param["seed"] = args.seed
    param["locexp"] = args.locexp
    param["wandb"] = args.wandb
    param["render"] = args.render
    param["buffer_size"] = args.buffer_size
    param["action_shape"] = args.action_shape
    text = str(param)
    write_into_file(str(param["locexp"]) + "/hyperparameters", text)
    train(env, param)


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
    parser.add_argument('--buffer_path', default="expert_policy", type=str)
    parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    arg = parser.parse_args()
    main(arg)
