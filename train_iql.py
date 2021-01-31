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
    if config["idx"] < memory.idx:
        memory.idx = config["idx"] 
    print("memory idx ",memory.idx)  
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory)
        if t % int(config["eval"]) == 0:
            print(text)
            agent.save(save_models_path + "/models/{}-".format(t))
            #agent.test_predicter(memory)
            agent.test_q_value(memory)
            agent.eval_policy()
            agent.eval_policy(True, 1)
