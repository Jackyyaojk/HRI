import numpy as np
import argparse
import pickle
import serial
import shutil
import time
import os

import torch

# from hand_tracker.tracker_state import TrackerState
from up_sample import main as UpsampleDems
from train import main as UpdatePolicy
from interface_gui.simulation import SimulationEnv
from policy import PolicyNetwork
from utils import *




def robotAction(goal, curr_q, threshold):
	if np.linalg.norm(goal - curr_q) > threshold:
		action_scale = 0.3
	else:
		action_scale = 0.1
	qdot = (goal - curr_q) / np.linalg.norm(goal - curr_q) * action_scale
	return qdot


def loadPolicy(N, path, interface):
	policies = []
	for n in range(N):
		model = PolicyNetwork().to(my_device)
		load_path = "{}/{}_model_{}".format(path, interface, str(n+1))
		model.load_state_dict(torch.load(load_path))
		policies.append(model)
	return policies


def forwardPolicy(policies, leg_q, hole_q):
	## policy input: chair part positions
	Panda = TrajectoryClient()
	leg_xyz, _, _ = Panda.joint2pose(leg_q)
	hole_xyz, _, _ = Panda.joint2pose(hole_q)
	states = np.concatenate((leg_xyz, hole_xyz))
	## policy output: panda joint positions (4 waypoints)
	N = len(policies)
	logits = np.zeros((N, total_waypoints*7))
	for n in range(N):
		model = policies[n]
		joint_pos_pred = model(torch.FloatTensor(states).to(my_device))
		logits[n,:] = joint_pos_pred.detach().cpu().numpy()
	## ensemble modeling: averaged joint positions (4 waypoints)
	mean_waypoints = np.mean(logits, axis=0).reshape(total_waypoints, 7)
	logits_xyz = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=2, arr=logits.reshape(N, -1, 7))
	std_waypoints = np.mean(np.std(logits_xyz, axis=0), axis=1)
	return mean_waypoints, std_waypoints/np.linalg.norm(std_waypoints)


def showWaypoints(waypoints, uncertainties, simEnv, leg_idx, first_time=True):
	Panda = TrajectoryClient()
	w_xyz = np.zeros((total_waypoints, 3))
	for i, w in enumerate(waypoints):
		w_xyz[i,:], _, _ = Panda.joint2pose(w)
	colors = []
	for value in uncertainties:
		if value < signal_threshold:
			c = [1, 0, 0, 1]
		elif value >= signal_threshold:
			c = [0, 1, 0, 1]
		colors.append(c)
	if not first_time:
		simEnv.updateWaypoints(w_xyz, colors, leg_idx)
	else:
		simEnv.visualizeWaypoints(w_xyz, colors, leg_idx)



def send2hololens(TRAJ, initialized, latch_point=0.8):
	Panda = TrajectoryClient()
	home_pose = [100,100,100,1.,1.,1.]
	uncertainty = np.ones(dem_num * 2)
	waypoints = np.delete(TRAJ, [1,2], axis=1).reshape((-1, TRAJ.shape[-1]))
	x_offset, y_offset, z_offset = 0.05, 0.03, 0.06
	if not initialized:
		with open('hololens/robotInit.txt', 'w') as f:
			f.write(f"{latch_point}\t{home_pose[0]}\t{home_pose[1]}\t{home_pose[2]}\t{home_pose[3]}\t{home_pose[4]}\t{home_pose[5]}\n")
			for count, (w, u) in enumerate(zip(waypoints, uncertainty)):
				w_pose = Panda.wrappedPose(w)
				w_pose[0] -= x_offset
				w_pose[1] = -w_pose[1]
				if w_pose[1] < 0:
					w_pose[2] -= z_offset
					w_pose[1] += y_offset
				if count < len(uncertainty) - 1:
					f.write(f"{w_pose[0]}\t{w_pose[1]}\t{w_pose[2]}\t{w_pose[3]}\t{w_pose[4]}\t{w_pose[5]}\t{u}\n")
				else:
					f.write(f"{w_pose[0]}\t{w_pose[1]}\t{w_pose[2]}\t{w_pose[3]}\t{w_pose[4]}\t{w_pose[5]}\t{u}")
	else:
		waypoints_pose = []
		for w in waypoints:
			w_pose = Panda.wrappedPose(w)
			w_pose[0] -= x_offset
			w_pose[1] = -w_pose[1]
			if w_pose[1] < 0:
					w_pose[2] -= z_offset
					w_pose[1] += y_offset
			waypoints_pose.append(w_pose[:3])
		coords = np.array(waypoints_pose).flatten()
		coord_str = '\t'.join(map(str, coords))
		with open('hololens/robotUpdate.txt', 'w') as f:
			f.write(f"{coord_str}\n")
			for count, u in enumerate(uncertainty):
				if count < len(uncertainty) - 1:
					f.write(f"{u}\n")
				else:
					f.write(f"{u}")



def hapticSignals(STD):
	signals = np.empty(shape=len(LEGS)*total_waypoints, dtype='object')
	for idx, value in enumerate(STD.flat):
		if value < signal_threshold:
			p_string = "<0.0;0.0;0.0>"
		elif value >= signal_threshold:
			p_string = "<0.0;0.0;9.0>"
		signals[idx] = p_string
	return signals.reshape(len(LEGS), total_waypoints)






def main(user, interface):

	## saving and loading paths
	dems_path = '{}/{}'.format(dems_directory, user)
	models_path = '{}/{}'.format(models_directory, user)
	pre_trained_models_path = '{}/pre-training'.format(models_directory)

	if not os.path.exists(models_path):
		os.makedirs(models_path)

	if not os.path.exists(dems_path):
		os.makedirs(dems_path)

	# ## copy pre-trained policies to user folder
	# if len(os.listdir((models_path))) != 9:
	# 	for n in range(num_models):
	# 		source = "{}/{}_model_{}".format(pre_trained_models_path, interface, str(n+1))
	# 		destination = "{}/{}_model_{}".format(models_path, interface, str(n+1))
	# 		shutil.copy(source, destination)
	# 		print("[*] Copied pre-trained policies")


	## copy training dataset to user folder
	if not os.path.isfile("{}/{}_training_dataset.pkl".format(dems_path, interface)):
		source = "{}/pre-training/{}_training_dataset.pkl".format(dems_directory, interface)
		destination = "{}/{}_training_dataset.pkl".format(dems_path, interface)
		shutil.copy(source, destination)
		print("[*] Copied initial training dataset")


	## instantiate the robot and joystick
	Panda = TrajectoryClient()
	joystick = JoystickControl()
	
	## establish socket connection with panda
	print('[*] Connecting to Panda...')
	PORT_robot = 8080
	conn_panda = Panda.connect2robot(PORT_robot)

	print('[*] Connecting to Panda gripper...','\n')
	PORT_gripper = 8081
	conn_gripper = Panda.connect2gripper(PORT_gripper)
	
	## load trained policies
	policies = loadPolicy(num_models, models_path, interface)

	## policy rollouts
	TRAJ = np.zeros((len(LEGS), total_waypoints, 7))
	STD = np.zeros((len(LEGS), total_waypoints))


	for idx, (leg_q, hole_q) in enumerate(zip(LEGS, HOLES)):
		mean_waypoints, std_waypoints = forwardPolicy(policies, leg_q, hole_q)
		TRAJ[idx, :, :] = mean_waypoints
		STD[idx, :] = std_waypoints

	## initialize interfaces
	if interface == "gui":
		print('[*] Initializing gui')
		simEnv = SimulationEnv()
		simEnv.reset()
		for idx, (traj_waypoints, traj_stds) in enumerate(zip(TRAJ, STD)):
			showWaypoints(traj_waypoints, traj_stds, simEnv, idx+1)
	elif interface == "ar+haptic":
		print('[*] Initializing Hololens')
		send2hololens(TRAJ, initialized=False)
		send2hololens(TRAJ, initialized=True)
		print('[*] Initializing Haptic Wristband')
		conn_haptic = serial.Serial('/dev/ttyACM0', baudrate=9600)
		haptic_signals = hapticSignals(STD)
		simEnv = None
	else:
		simEnv = None
		print('[*] No interface selected!')

	# ## initialize handtracker
	# tracker = TrackerState()
	# tracker.set_global_coords(GLOBAL_O, GLOBAL_X)

	
	## panda picks up legs in order
	leg_indices = np.random.permutation(TRAJ.shape[0])
	for traj_num in leg_indices:
		
		print()
		print('--------- Leg:', traj_num + 1)
		
		traj = TRAJ[traj_num, :, :]
		user_traj = np.copy(traj)

		## send panda to home
		if interface == "ar+haptic":
			conn_haptic.write(str.encode("<0.0;0.0,0;0>"))
		grasp_open = Panda.send2gripper(conn_gripper, 'o')
		time.sleep(1)		
		Panda.go2home(conn_panda, HOME, grasp_open, simEnv)
		print('[*] Panda returned home')

		mode = "v"
		waypoint = 0
		step_time = 1.
		threshold = .02
		force_threshold = 50
		goal = traj[waypoint]

		count = 0
		run = False
		record = False
		ended = False
		shutdown = False
		inserting = False
		user_feedback = False
		last_time_move = time.time()
		last_time_record = time.time()
		
		user_dem = {"joint_positions":[],
					}
		
		user_data = {"unix_time":[],
					"hand_position":[],
					"run": [],
					"joint_positions":[],
					"xyz_euler":[],
					}
		
		while not shutdown:

			# # Read Vive
			# vive_pos_raw, valid = tracker.get_tracker_pos()
			# if valid:
			# 	position_hand = vive_pos_raw.squeeze()
			# else:
			# 	print('No data from Vive')
			# 	continue

			## read panda states
			states = Panda.readState(conn_panda)
			curr_q = states["q"]
			curr_pos = states["xyz_euler"]

			## keep track of data
			user_data["unix_time"].append(time.time())
			# user_data["hand_position"].append(position_hand)
			user_data["run"].append(run)
			user_data["joint_positions"].append(curr_q)
			user_data["xyz_euler"].append(curr_pos)
				

			# ## workspace boundaries
			# if run:
			# 	in_bound = Panda.checkBoundaries(curr_q)
			# 	if (waypoint == len(traj)-1) and (curr_pos[2] <= 0.2):
			# 		shutdown = Trues
			# 	elif not in_bound and not locked:
			# 		waypoint += 1
			# 		goal = traj[waypoint]
			# 		locked = True

				# in_bound = Panda.checkBoundaries(curr_q)
				# if not in_bound and (waypoint == len(traj)-1):
				# 	shutdown = True
				# elif not in_bound and not locked:
				# 	waypoint += 1
				# 	goal = traj[waypoint]
				# 	locked = True


			if run and np.linalg.norm(states["O_F"]) > force_threshold:
				print("[*] Panda had collision")
				Panda.go2home(conn_panda, HOME, grasp_open, simEnv)
				grasp_open = Panda.send2gripper(conn_gripper, 'o')
				waypoint = 0
				goal = traj[waypoint]
				run = False

			## switching waypoints as panda moves
			if np.linalg.norm(goal - curr_q) < threshold and run:
				waypoint += 1
				locked = False
				if waypoint == 1:
					if grasp_open:
						grasp_open = Panda.send2gripper(conn_gripper, 'c')
						time.sleep(1)
				if waypoint == len(traj):
					print('[*] Panda reached goal')
					waypoint -= 1
					run = False
					record = True
					mode ="k"
				goal = traj[waypoint]


			## receiving commanda from joystick
			A, B, X, Y, BACK, START = joystick.getInput()
			if A and not run:
				print("[*] Panda moving")
				last_time_move = time.time()
				record = False
				run = True
				mode ="v"
			if A and run:
				curr_time_move = time.time()
				if curr_time_move - last_time_move >= 0.1:
					print('[*] Panda paused and ready to record...')
					last_time_record = time.time()
					record = True
					run = False
					mode ="k"
					ended = True
			if B and record and time.time() - last_time_record > step_time:
				print("---Recorded waypoint", count+1)
				dists = []
				for w in traj:
					dists.append(np.linalg.norm(curr_q - w))
				updating_idx = np.argmin(dists)
				user_traj[updating_idx] = curr_q 
				last_time_record = time.time()
				user_feedback = True
				count += 1
			if X:
				if grasp_open:
					grasp_open = Panda.send2gripper(conn_gripper, 'c')
					time.sleep(1)
			if Y:
				if not grasp_open:
					grasp_open = Panda.send2gripper(conn_gripper, 'o')
					time.sleep(1)

			if BACK and time.time() - last_time_move > step_time:
				print('[*] Panda ended trajectory')
				if not grasp_open:
					grasp_open = Panda.send2gripper(conn_gripper, 'o')
				shutdown = True

			



			if not run:
				qdot = np.asarray([0.0] * 7)
				Panda.send2robot(conn_panda, qdot, mode)
			else:
				if interface == "gui":
					simEnv.step([curr_q, grasp_open])
				elif interface == "ar+haptic":
					if (np.linalg.norm(goal - curr_q) < 0.2) and (waypoint not in [1,2]):
						signal = haptic_signals[traj_num, waypoint]
					else: 
						signal = "<0.0;0.0,0;0>"
					conn_haptic.write(str.encode(signal))
				qdot = robotAction(goal, curr_q, threshold)
				Panda.send2robot(conn_panda, qdot, mode)


		# if inserting:
		# 	## leg insertion
		# 	time.sleep(2)
		# 	min_idx = None
		# 	min_dist = np.inf
		# 	for idx, hole in enumerate(HOLES):
		# 		dist = np.linalg.norm(hole - curr_q)
		# 		if dist < min_dist:
		# 			min_dist = np.copy(dist)
		# 			min_idx = idx
		# 	target_hole = HOLES[min_idx]
		# 	## fix ee position 
		# 	while np.linalg.norm(target_hole - curr_q) >= threshold:
		# 		print("[*] panda is adjusting....")
		# 		states = Panda.readState(conn_panda)
		# 		curr_q = states["q"]
		# 		qdot = robotAction(target_hole, curr_q, threshold)
		# 		Panda.send2robot(conn_panda, qdot, mode)
		# 		simEnv.step([curr_q, grasp_open])
			
		# 	## insert
		# 	target_pose = Panda.wrappedPose(target_hole)
		# 	target_pose[2] -= 0.1
		# 	action_scale = 0.05
		# 	print(np.linalg.norm(states["O_F"]))
		# 	while np.linalg.norm(states["O_F"]) < force_threshold:
		# 		print("[*] panda is instering")
		# 		states = Panda.readState(conn_panda)
		# 		xdot = target_pose - states["xyz_euler"]
		# 		xdot[3:] = 0
		# 		qdot = Panda.xdot2qdot(xdot, states)
		# 		Panda.send2robot(conn_panda, qdot * action_scale, mode)
		# 		simEnv.step([curr_q, grasp_open])


		## save user data
		with open("{}/{}_user_data_{}.pkl".format(dems_path, interface, str(traj_num+1)), "wb") as file:
				pickle.dump(user_data, file)
		print("[*] Saved User Data")


		## retraining policies
		time.sleep(1)
		if user_feedback:
			## save user's feedback demonstration
			user_dem["joint_positions"] = user_traj
			with open("{}/{}_dem_{}.pkl".format(dems_path, interface, str(traj_num+1)), "wb") as file:
				pickle.dump(user_dem, file)
			## training new policy
			print("[*] Saved Demonstration")
			UpsampleDems(interface, user, str(traj_num+1))
			print("[*] Up-sampled Data")
			UpdatePolicy(interface, user)
			print("[*] Updated policies")

		## load updated policies and waypoints
		updated_policies = loadPolicy(num_models, models_path, interface)
		traj_waypoints, traj_stds = forwardPolicy(updated_policies, LEGS[traj_num], HOLES[traj_num])
		TRAJ[traj_num, :, :] = traj_waypoints
		STD[traj_num, :] = traj_stds

		## update visual feedback
		if interface == "gui":
			print("[*] Updating interface...")
			showWaypoints(traj_waypoints, traj_stds, simEnv, traj_num+1, first_time=False)
			time.sleep(5)
		elif interface == "ar+haptic":
			haptic_signals = hapticSignals(STD)
			send2hololens(TRAJ, initialized=True)
			conn_haptic.write(str.encode("<0.0;0.0,0;0>"))






if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Collecting offline demonstrations')
	parser.add_argument('--user', help='user(i)', type=str, default="user0")
	parser.add_argument('--interface', help='none, gui, ar+haptic', type=str, default="none")
	args = parser.parse_args()
	main(args.user, args.interface)