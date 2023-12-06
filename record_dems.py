import numpy as np
import argparse
import pickle
import time
import os

from utils import *
np.set_printoptions(suppress=True)



## instantiate the robot and joystick
Panda = TrajectoryClient()
joystick = JoystickControl()

## establish socket connection with panda
print('[*] Connecting to Panda...')
PORT_robot = 8080
conn = Panda.connect2robot(PORT_robot)

## send robot to home
print('[*] Sending Panda to home...', '\n')
Panda.go2home(conn, HOME)

## saving directory			
if not os.path.exists('{}/pre-training'.format(dems_directory)):
	os.makedirs('{}/pre-training'.format(dems_directory))



def main(save_path):
	'''
	A: start recording
	B: stop recording
	'''
	## parameters	
	done = False
	record = False
	shutdown = False
	step_time = 0.1
	waypoint_num = 0

	data = {"joint_positions":[],
			"xyz_euler":[],
			"rotation_matrices":[],
			}
	
	while not shutdown:
		## read robot states
		states = Panda.readState(conn)
		q, xyz_euler = states["q"], states["xyz_euler"]
		_, R, _ = Panda.joint2pose(q)

		## read joystick commands
		A, B, X, _, _, START = joystick.getInput()
		
		if A and not record:
			record = True
			last_time = time.time()
			print("---Ready to Record")
		
		if B and record and time.time() - last_time > step_time:
			print("---Recorded waypoint", waypoint_num+1)
			data["joint_positions"].append(q)
			data["xyz_euler"].append(xyz_euler)
			data["rotation_matrices"].append(R)
			waypoint_num += 1
			record = False
			done = True

		if X and done:
			with open(save_path + ".pkl", "wb") as file:
				pickle.dump(data, file)
			print("---Saved Demonstration")
			Panda.go2home(conn, HOME)
			print('---Panda returned home...', '\n')
			shutdown = True
		
		## send zero joint velocity
		Panda.send2robot(conn, [0]*7, mode="k")


if __name__=="__main__": 
	parser = argparse.ArgumentParser(description='Collecting offline demonstrations')
	parser.add_argument('--interface', help='none, gui, ar+haptic', type=str, default="none")
	args = parser.parse_args()
	for num in range(dem_num):
		print("[*] Recording Demonstration ", num+1)
		save_path = "{}/pre-training/{}_dem_{}".format(dems_directory, args.interface, num+1)
		main(save_path)