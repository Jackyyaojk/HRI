import numpy as np
import argparse
import pickle
import os 

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

from policy import PolicyNetwork
from utils import *



class HumanData(Dataset):

	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return torch.tensor(self.data[idx], dtype=torch.float32).to(my_device)



def collateFun(batch):
	padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
	return padded_batch


def main(interface,  pre_training=True, user=None):

	model_1 = PolicyNetwork().to(my_device)
	model_2 = PolicyNetwork().to(my_device)
	model_3 = PolicyNetwork().to(my_device)
	model_4 = PolicyNetwork().to(my_device)
	model_5 = PolicyNetwork().to(my_device)
	models = [model_1, model_2, model_3, model_4, model_5]

	if pre_training:
		folder = 'pre-training'
	else:
		folder = user
		for n, model in enumerate(models):
			# model.load_state_dict(torch.load("data/{}/models/{}_model_{}".format(user, interface, str(n+1))))
			model.load_state_dict(torch.load("data/pre-training/models/{}_model_{}".format(interface, str(n+1))))

	dataset_path = 'data/{}/datasets/{}_training_dataset.pkl'.format(folder, interface)
	models_path = 'data/{}/models'.format(folder)

	## saving directory			
	if not os.path.exists(models_path):
		os.makedirs(models_path)

	with open(dataset_path, "rb") as file:
		data = pickle.load(file)

	EPOCH = 600
	LR = 0.02
	LR_STEP_SIZE = 500
	LR_GAMMA = 0.1
	BATCH_SIZE = len(data)

	train_data = HumanData(data)
	train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

	for n, model in enumerate(models):
		print()
		print('[*] Training model ' + str(n+1))
		
		optimizer = optim.Adam(model.parameters(), lr=LR)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
		
		for epoch in range(1, EPOCH+1):

			running_loss = 0.

			for x in train_set:
				
				states = x[:,:6]
				actions = x[:,6:]

				predicted_actions = model(states)
				loss = F.mse_loss(actions, predicted_actions)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				running_loss += loss.item()

			scheduler.step()
		
			if epoch % 100 == 0:
				print("[*] Epoch: {}, Loss: {}".format(epoch, round(running_loss, 4)))

		## save model 
		torch.save(model.state_dict(), "{}/{}_model_{}".format(models_path, interface, str(n+1)))





if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Collecting offline demonstrations')
	parser.add_argument('--pre_training', help='True/False', type=str, default="False")
	parser.add_argument('--user', help='user(i)', type=str, default="user1")
	args = parser.parse_args()

	for interface in ['none', 'gui', 'ar+haptic']:
		print()
		if args.pre_training == "True":
			print('[*] Pre-training:', "interface", interface)
			main(interface)
		else:
			print('[*]', args.user, ", interface:", interface)
			main(interface, pre_training=False, user=args.user)