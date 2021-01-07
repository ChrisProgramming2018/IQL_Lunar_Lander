from replay_buffer import ReplayBuffer
import json



with open ("param.json", "r") as f:
    param = json.load(f)

config = param

memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
memory.load_memory(config["buffer_path"])

memory_expert = ReplayBuffer((8,), (1,), 100000, config["device"])

for idx in range(memory.idx):
    memory_expert.add_expert(memory.obses[idx], memory.actions[0], memory.rewards[idx], memory.next_obses[idx], memory.not_dones[idx], memory.not_dones_no_max[idx])
    print(memory_expert.idx)


memory_expert.save_memory("all_action_expert")
