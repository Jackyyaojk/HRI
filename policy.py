import numpy as np
import os 
import random
from tqdm import tqdm
from collections import deque
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *




class PolicyNetwork(nn.Module):
	'''
	input: object positions
	output: waypoint joint positions
	'''
	def __init__(self, input_dim=6, hidden_dim=128, output_dim=total_waypoints*7):
		super().__init__()

		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
			)

	def forward(self, state):
		return self.network(state)



## memory buffer
class MemoryBuffer:
	
	def __init__(self, max_size=5000):
		self.buffer = deque(maxlen=max_size)


	def __len__(self):
		return len(self.buffer)


	def _get(self):
		return list(self.buffer)


	def pushData(self, state, action):
		self.buffer.append((state, action))


	def sampleData(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
		return states, actions, rewards, next_states, dones


	def updateBuffer(self, reward_net):
		for i in tqdm(range(self.__len__()), desc="[*] Updating memory buffer"):
			memory = self.buffer[i]
			state, action, next_state, done = memory[0], memory[1], memory[3], memory[4]
			r = reward_net.getReward(state[:2])
			new_reward = r.cpu().detach().numpy()[0]
			self.buffer[i] = (state, action, new_reward, next_state, done)


	def saveBuffer(self, run_name, prefix=""):
		if not os.path.exists('models/{}'.format(run_name)):
			os.makedirs('models/{}'.format(run_name))
			save_path = "models/{}/{}_sac_buffer".format(run_name, prefix)
		with open(save_path, 'wb') as f:
			pickle.dump(self.buffer, f)
			print("-Saved {} buffer".format(prefix.capitalize()))


	def loadBuffer(self, run_name, prefix=""):
		print("-Loading {} buffer".format(prefix.capitalize()))
		save_path = "models/{}/{}_sac_buffer".format(run_name, prefix)
		with open(save_path, "rb") as f:
			self.buffer = pickle.load(f)