import sys
import time
from replay_buffer import ReplayBuffer
from agent_iql import Agent
from utils import time_format


def train(env, config):
    """

    """
    t0 = time.time()
    save_models_path =  str(config["locexp"])
    memory = ReplayBuffer((8,), (1,), config["buffer_size"], config["seed"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(state_size=8, action_size=4,  config=config) 
    memory_t = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["seed"], config["device"])
    memory_t.load_memory(config["expert_buffer_path"])
    memory.idx = config["idx"] 
    memory_t.idx = config["idx"] * 4
    print("memory idx ",memory.idx)  
    print("memory_expert idx ",memory_t.idx)
    for idx in range(memory.idx):
        print(memory.actions[idx], memory.rewards[idx])
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory, memory_t)
        if t % int(config["eval"]) == 0:
            print(text)
            agent.save(save_models_path + "/models/{}-".format(t))
            agent.test_predicter(memory)
            agent.test_q_value(memory)
            agent.eval_policy()
            agent.eval_policy(True, 1)
