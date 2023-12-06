import numpy as np
import argparse
import pickle

from utils import *



def main(interface, pre_training=True, user=None, leg_num=None):


	print("[*] Processing Demonstrations...")
	if pre_training:
		folder = "pre-training"
		dem_list = []
		for i in range(dem_num):
			dem_list.append(str(i+1))
	else:
		folder = user
		# dem_list = [leg_num]
		dem_list = []
		for i in range(dem_num):
			dem_list.append(str(i+1))

	load_path = "data/{}/dems/{}_dems.pkl".format(folder, interface)
	save_path = "data/{}/datasets".format(folder)

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	Panda = TrajectoryClient()
	training_dataset = []

	with open(load_path, 'rb') as file:
		dem = pickle.load(file)
	
	for leg_num in dem_list:

		waypoints = np.array(dem['leg'+leg_num+'_traj'])

		leg_q = LEGS[int(leg_num)-1]
		hole_q = HOLES[int(leg_num)-1]

		leg_xyz, _, _ = Panda.joint2pose(leg_q)
		hole_xyz, _, _ = Panda.joint2pose(hole_q)
		states = np.concatenate((leg_xyz, hole_xyz))

		if pre_training:
			upsample_num = 10
		else:
			upsample_num = 10

		scale = 0.
		limit_links = 0.
		limit_orient = 0.
		for _ in range(upsample_num):
			states_noisy = states + np.random.normal(0, scale, 6)
			noise_links = np.random.uniform(-limit_links, limit_links, size=(waypoints.shape[0], 4))
			noise_ee = np.random.uniform(-limit_orient, limit_orient, size=(waypoints.shape[0], 3))
			waypoints_noisy = waypoints + np.concatenate((noise_links, noise_ee), axis=1)
			training_dataset.append(states_noisy.tolist() + waypoints_noisy.flatten().tolist())

	if not pre_training:
		## update user training dataset
		with open("data/pre-training/datasets/{}_training_dataset.pkl".format(interface), 'rb') as file:
			old_training_dataset = pickle.load(file)
		final_training_dataset = (old_training_dataset + training_dataset).copy()
	else:
		final_training_dataset = training_dataset.copy()

	## save training dataset
	with open("{}/{}_training_dataset.pkl".format(save_path, interface), "wb") as file:
		pickle.dump(final_training_dataset, file)
	print("[*] Saved {} Training Datapoints".format(len(final_training_dataset)))



if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Collecting offline demonstrations')
	parser.add_argument('--pre_training', help='True or False', type=str, default="False")
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