from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi
init_printing(use_unicode=True)

from scipy.spatial.transform import Rotation

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from sympy import lambdify
from tkinter import *
import pybullet as pb
import pybullet_data
import numpy as np
import pygame
import socket
import time
import os

import torch


########## hypers ##########
signal_threshold = 0.45		## uncertainty thresholds
num_models = 5
dem_num = 4			## one dem for each chair leg
total_waypoints = 4
my_device = "cuda:0" if torch.cuda.is_available() else "cpu"

########## user number list ##########
USERS = [1,2,3,4,5,7,8,9,10]

############ tracker positions ############
ee_offset = np.array([0.48354337, -0.26194866, 0.60433942])

GLOBAL_O = np.asarray([[-1.99404967], [-4.6105032 ], [-0.57095927]])
GLOBAL_X = np.asarray([[-1.826877], [-4.56208467], [-0.58080244]])

############ object positions ############
leg1 = [-0.313891, 0.975939, -0.097299, -1.293696, 0.078923, 2.264815, 1.88702]
leg2 = [-0.330295, 0.67001, -0.098383, -1.923895, 0.054309, 2.56653, 1.872272] 
leg3 = [-0.404673, 0.478879, -0.142414, -2.314685, 0.007846, 2.771369, 1.839982] 
leg4 = [-0.486997, 0.302098, -0.236638, -2.682369, 0.079528, 2.936476, 1.552247] 
LEGS = [leg1, leg2, leg3, leg4]

hole1 = [0.76247, 0.409048, -0.07847, -2.373332, 2.723739, 1.909277, -2.175316] 
hole2 = [0.080448, 0.758486, 0.378692, -1.851649, 1.674237, 1.975365, -1.843272] 
hole3 = [-0.176081, 0.500755, 0.531916, -2.327433, 2.482919, 2.032133, -2.314907] 
hole4 = [0.485142, 0.292792, 0.023725, -2.64478, 2.712781, 1.789986, -2.254092]
HOLES = [hole1, hole2, hole3, hole4]

SEAT_POS = [0.45, 0.35874544, 0.01]


############ robot joint positions ############
HOME = [0.0, -0.8, 0.0, -2.5, 0.0, 1.7, 0.7]

WORKSPACE_LIMITS = {
					"X": (0.2,0.8),
					"Y": (-0.5,0.5),
					"Z": (0.02, 0.85)
					}

JOINT_LIMITS = [
				(-2.8973, 2.8973),
				(-1.7628, 1.7628),
				(-2.8973, 2.8973),
				(-3.0718, -0.0698),
				(-2.8973, 2.8973),
				(-0.0175, 3.7525),
				(-2.8973, 2.8973)
				]


########## robot's symbolic jacobian matrix ##########
q1, q2, q3, q4, q5, q6, q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
joint_angles = [q1, q2, q3, q4, q5, q6, q7]
dh_craig = [
	{'a':  0,      'd': 0.333, 'alpha':  0,  },
	{'a':  0,      'd': 0,     'alpha': -pi/2},
	{'a':  0,      'd': 0.316, 'alpha':  pi/2},
	{'a':  0.0825, 'd': 0,     'alpha':  pi/2},
	{'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
	{'a':  0,      'd': 0,     'alpha':  pi/2},
	{'a':  0.088,  'd': 0.107, 'alpha':  pi/2},
	]
DK = eye(4)
for _, (p, q) in enumerate(zip(reversed(dh_craig), reversed(joint_angles))):
	d, a, alpha = p['d'], p['a'], p['alpha']
	ca, sa, cq, sq = cos(alpha), sin(alpha), cos(q), sin(q)
	transform = Matrix(
		[[cq, -sq, 0, a],
		[ca * sq, ca * cq, -sa, -d * sa],
		[sa * sq, cq * sa, ca, d * ca],
		[0, 0, 0, 1]]
		)
	DK = transform @ DK

A = DK[0:3, 0:4].transpose().reshape(12,1)
Q = Matrix(joint_angles)
J = A.jacobian(Q)
J_symb = lambdify((q1, q2, q3, q4, q5, q6, q7), J, 'numpy')
A_symb = lambdify((q1, q2, q3, q4, q5, q6, q7), A, 'numpy')

########## serial Comm. with Arduino ##########
def send_serial(comm, output):
	string = '<' + output + '>'
	comm.write(str.encode(string))

########## GUI design ##########
class GUI_Interface(object):
	def __init__(self):
		self.root = Tk()
		self.root.geometry("+1000+100")
		self.root.title("Uncertainity Output")
		self.update_time = 0.02
		font = "Palatino Linotype"

		# X_Y Uncertainty
		myLabel1 = Label(self.root, text = "Distance From Edge", font=(font, 40))
		myLabel1.grid(row = 0, column = 0, pady = 50, padx = 50)
		self.textbox1 = Entry(self.root, width = 5, bg = "white", fg = "#676767", borderwidth = 3, font=(font, 40))
		self.textbox1.grid(row = 0, column = 1,  pady = 10, padx = 20)
		self.textbox1.insert(0,0)

		# Z Uncertainty
		myLabel2 = Label(self.root, text = "Height from Table", font=("Palatino Linotype", 40))
		myLabel2.grid(row = 1, column = 0, pady = 50, padx = 50)
		self.textbox2 = Entry(self.root, width = 5, bg = "white", fg = "#676767", borderwidth = 3, font=(font, 40))
		self.textbox2.grid(row = 1, column = 1,  pady = 10, padx = 20)
		self.textbox2.insert(0,0)

		# ROT Uncertainty
		myLabel3 = Label(self.root, text = "Orientation", font=("Palatino Linotype", 40))
		myLabel3.grid(row = 2, column = 0, pady = 50, padx = 50)
		self.textbox3 = Entry(self.root, width = 5, bg = "white", fg = "#676767", borderwidth = 3, font=(font, 40))
		self.textbox3.grid(row = 2, column = 1,  pady = 10, padx = 20)
		self.textbox3.insert(0,0)


########## Joystick ##########
class JoystickControl(object):

	def __init__(self):
		pygame.init()
		self.gamepad = pygame.joystick.Joystick(0)
		self.gamepad.init()
		self.deadband = 0.1
		self.timeband = 0.5
		self.lastpress = time.time()

	def getInput(self):
		pygame.event.get()
		curr_time = time.time()
		A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
		B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
		X_pressed = self.gamepad.get_button(2) and (curr_time - self.lastpress > self.timeband)
		Y_pressed = self.gamepad.get_button(3) and (curr_time - self.lastpress > self.timeband)
		
		BACK_pressed = self.gamepad.get_button(6) and (curr_time - self.lastpress > self.timeband)
		START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
		
		pressued_keys = [A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, BACK_pressed]
		if any(pressued_keys):
			self.lastpress = curr_time
		return A_pressed, B_pressed, X_pressed, Y_pressed, BACK_pressed, START_pressed


########## Panda Robot ##########
class TrajectoryClient(object):

	def __init__(self):
		pass

	def connect2robot(self, PORT):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('172.16.0.3', PORT))
		s.listen()
		conn, addr = s.accept()
		return conn

	def connect2gripper(self, PORT):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('172.16.0.3', PORT))
		s.listen(10)
		conn, addr = s.accept()
		return conn

	def send2gripper(self, conn, send_msg):
		conn.send(send_msg.encode())
		if send_msg == 'o':
			grasp_open = True
		elif send_msg == 'c':
			grasp_open = False
		return grasp_open

	def send2robot(self, conn, qdot, mode, limit=1.0):
		qdot = np.asarray(qdot)
		scale = np.linalg.norm(qdot)
		if scale > limit:
			qdot *= limit/scale
		send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
		send_msg = "s," + send_msg + "," + mode + ","
		conn.send(send_msg.encode())

	def listen2robot(self, conn):
		state_length = 7 + 7 + 7 + 6 + 42
		message = str(conn.recv(2048))[2:-2]
		state_str = list(message.split(","))
		for idx in range(len(state_str)):
			if state_str[idx] == "s":
				state_str = state_str[idx+1:idx+1+state_length]
				break
		try:
			state_vector = [float(item) for item in state_str]
		except ValueError:
			return None
		if len(state_vector) is not state_length:
			return None
		state_vector = np.asarray(state_vector)
		states = {}
		states["q"] = state_vector[0:7]
		states["dq"] = state_vector[7:14]
		states["tau"] = state_vector[14:21]
		states["O_F"] = state_vector[21:27]
		states["J"] = state_vector[27:].reshape((7,6)).T
		# get cartesian pose
		xyz_lin, _, R = self.joint2pose(state_vector[0:7])
		beta = -np.arcsin(R[2,0])
		alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
		gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
		xyz_ang = [alpha, beta, gamma]
		xyz_euler = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
		states["xyz_euler"] = np.array(xyz_euler)
		return states

	def readState(self, conn):
		while True:
			states = self.listen2robot(conn)
			if states is not None:
				break
		return states

	def sampleJointPose(self):
		q_mid = np.array([l+(u-l)/2 for l, u in JOINT_LIMITS], dtype=np.float64)
		q_rand = np.array([np.random.uniform(l, u) for l, u in JOINT_LIMITS], dtype=np.float64)
		return q_mid, q_rand

	def xdot2qdot(self, xdot, states):
		J_inv = np.linalg.pinv(states["J"])
		return J_inv @ np.asarray(xdot)
	
	def wrappedPose(self, q):
		xyz, R, _ = self.joint2pose(q)
		eulers = self.matrix2euler(R)
		return np.concatenate((xyz, eulers))

	def joint2pose(self, q):
		def RotX(q):
			return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
		def RotZ(q):
			return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
		def TransX(q, x, y, z):
			return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
		def TransZ(q, x, y, z):
			return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
		H1 = TransZ(q[0], 0, 0, 0.333)
		H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
		H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
		H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
		H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
		H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
		H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
		H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
		T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
		R = T[:,:3][:3]
		A = T[:3,:]
		xyz = T[:,3][:3]
		eulers = self.matrix2euler(R)
		return xyz, R, A
	
	def pose2joint(self, A_target, step=.05, atol=1e-2):
		q, _ = self.sampleJointPose()
		_, _, A = self.joint2pose(q)
		iter = 0
		while True:
			iter += 1
			if iter >= 5000:
				return None
			delta_A = A_target.flatten('F') - A.flatten('F')
			if np.max(np.abs(delta_A)) <= atol:
				break
			J_q = J_symb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
			J_q = J_q / np.linalg.norm(J_q)
			delta_q = np.linalg.pinv(J_q) @ (delta_A*step)
			q = q + delta_q
			_, _, A = self.joint2pose(q)
		return q

	def go2home(self, conn, HOME, grasp_open=True, simEnv=None):
		total_time = 35.0
		start_time = time.time()
		states = self.readState(conn)
		joint_pos = np.asarray(states["q"].tolist())
		dist = np.linalg.norm(joint_pos - HOME)
		curr_time = time.time()
		action_time = time.time()
		elapsed_time = curr_time - start_time
		while dist > 0.05 and elapsed_time < total_time:
			joint_pos = np.asarray(states["q"].tolist())
			if simEnv:
				simEnv.step([joint_pos, grasp_open])
			action_interval = curr_time - action_time
			if action_interval > 0.005:
				qdot = HOME - joint_pos
				self.send2robot(conn, qdot, "v")
				action_time = time.time()
			states = self.readState(conn)
			dist = np.linalg.norm(joint_pos - HOME)
			curr_time = time.time()
			elapsed_time = curr_time - start_time
		if dist <= 0.02:
			return True
		elif elapsed_time >= total_time:
			return False

	def matrix2quat(self, R):
		R_obj = Rotation.from_matrix(R)
		return R_obj.as_quat()
	
	def matrix2euler(self, R):
		R_obj = Rotation.from_matrix(R)
		return R_obj.as_euler('xyz', degrees=False)
	
	def matrix2rotvec(self, R):
		R_obj = Rotation.from_matrix(R)
		return R_obj.as_rotvec()
	
	def euler2matrix(self, angles):
		eulers_obj = Rotation.from_euler('xyz', angles)
		return eulers_obj.as_matrix()
	
	def euler2rotvec(self, angles):
		eulers_obj = Rotation.from_euler('xyz', angles)
		return eulers_obj.as_rotvec()
	
	def quat2euler(self, quat):
		quat_obj = Rotation.from_quat(quat)
		return quat_obj.as_euler('xyz', degrees=False)
	
	def rotvec2euler(self, rot_vec):
		rot_vec_obj = Rotation.from_rotvec(rot_vec)
		return rot_vec_obj.as_euler('xyz', degrees=False)
	
	def fixJointAngle(self, q):
		if q < -np.pi:
			q += 2*np.pi
		elif q > np.pi:
			q -= 2*np.pi
		return q

	def checkBoundaries(self, q):
		in_bound = True
		xyz, _, _ = self.joint2pose(q)
		flag_x = WORKSPACE_LIMITS["X"][0] <= xyz[0] <= WORKSPACE_LIMITS["X"][1]
		flag_y = WORKSPACE_LIMITS["Y"][0] <= xyz[1] <= WORKSPACE_LIMITS["Y"][1]
		flag_z = WORKSPACE_LIMITS["Z"][0] <= xyz[2] <= WORKSPACE_LIMITS["Z"][1]
		if not flag_x or not flag_y or not flag_z:
			in_bound = False
		return in_bound

	def deformTraj(self, xi, start, length, tau):
		waypoint_size = xi.shape[1]
		xi1 = np.asarray(xi).copy()
		A = np.zeros((length+2, length))
		for idx in range(length):
			A[idx, idx] = 1
			A[idx+1,idx] = -2
			A[idx+2,idx] = 1
		R = np.linalg.inv(np.dot(A.T, A))
		U = np.zeros(length)
		gamma = np.zeros((length, waypoint_size))
		for idx in range(waypoint_size):
			U[0] = tau[idx]
			gamma[:,idx] = np.dot(R, U)
		end = min([start+length, xi1.shape[0]-1])
		xi1[start:end+1,:] += gamma[0:end-start+1,:]
		return xi1
	


########## Panda Simulator ##########
class PandaSimulator():

	def __init__(self, basePosition=[0, 0, 0]):
		self.urdfRootPath = pybullet_data.getDataPath()
		self.pandaId = pb.loadURDF(os.path.join(self.urdfRootPath,"franka_panda/panda.urdf"),useFixedBase=True,basePosition=basePosition)

	def _move_robot(self, mode='joint_control', djoint=[0]*7, dposition=[0]*3, dquaternion=[0]*4, grasp_open=True):
		self._velocity_control(mode=mode, djoint=djoint, dposition=dposition, dquaternion=dquaternion, grasp_open=grasp_open)
		self._read_state()
		self._read_jacobian()

	def reset(self):
		self._reset_robot(HOME + [0.05, 0.05])

	def _read_state(self):
		joint_position = [0]*9
		joint_velocity = [0]*9
		joint_torque = [0]*9
		joint_states = pb.getJointStates(self.pandaId, range(9))
		for idx in range(9):
			joint_position[idx] = joint_states[idx][0]
			joint_velocity[idx] = joint_states[idx][1]
			joint_torque[idx] = joint_states[idx][3]
		ee_states = pb.getLinkState(self.pandaId, 11)
		ee_position = list(ee_states[4])
		ee_quaternion = list(ee_states[5])
		gripper_contact = pb.getContactPoints(bodyA=self.pandaId, linkIndexA=10)
		self.state['joint_position'] = np.asarray(joint_position)
		self.state['joint_velocity'] = np.asarray(joint_velocity)
		self.state['joint_torque'] = np.asarray(joint_torque)
		self.state['ee_position'] = np.asarray(ee_position)
		self.state['ee_quaternion'] = np.asarray(ee_quaternion)
		self.state['ee_euler'] = np.asarray(pb.getEulerFromQuaternion(ee_quaternion))
		self.state['gripper_contact'] = len(gripper_contact) > 0

	def _read_jacobian(self):
		linear_jacobian, angular_jacobian = pb.calculateJacobian(self.pandaId, 11, [0, 0, 0], list(self.state['joint_position']), [0]*9, [0]*9)
		linear_jacobian = np.asarray(linear_jacobian)[:,:7]
		angular_jacobian = np.asarray(angular_jacobian)[:,:7]
		full_jacobian = np.zeros((6,7))
		full_jacobian[0:3,:] = linear_jacobian
		full_jacobian[3:6,:] = angular_jacobian
		self.jacobian['full_jacobian'] = full_jacobian
		self.jacobian['linear_jacobian'] = linear_jacobian
		self.jacobian['angular_jacobian'] = angular_jacobian

	def _reset_robot(self, joint_position):
		self.state = {}
		self.jacobian = {}
		self.desired = {}
		for idx in range(len(joint_position)):
			pb.resetJointState(self.pandaId, idx, joint_position[idx])
		self._read_state()
		self._read_jacobian()
		self.desired['joint_position'] = self.state['joint_position']
		self.desired['ee_position'] = self.state['ee_position']
		self.desired['ee_quaternion'] = self.state['ee_quaternion']

	def _inverse_kinematics(self, ee_position, ee_quaternion):
		return pb.calculateInverseKinematics(self.pandaId, 11, list(ee_position), list(ee_quaternion))

	def _velocity_control(self, mode, djoint, dposition, dquaternion, grasp_open):
		gripper_position = [0.0, 0.0]
		if grasp_open:
			gripper_position = [0.04, 0.04]
		if mode == 'ee_control':
			self.desired['ee_position'] += np.asarray(dposition) / 240.0
			self.desired['ee_quaternion'] += np.asarray(dquaternion) / 240.0
			q_dot = self._inverse_kinematics(self.desired['ee_position'], self.desired['ee_quaternion']) - self.state['joint_position']
		elif mode == 'joint_control':
			self.desired['joint_position'] += np.asarray(list(djoint)+[0, 0]) / 240.0
			q_dot = self.desired['joint_position'] - self.state['joint_position']
		pb.setJointMotorControlArray(self.pandaId, range(9), pb.VELOCITY_CONTROL, targetVelocities=list(q_dot))
		pb.setJointMotorControlArray(self.pandaId, [9,10], pb.POSITION_CONTROL, targetPositions=gripper_position)

	def _position_control(self, joints_pos, grasp_open=True):
		gripper_position = [0.0, 0.0]
		if grasp_open:
			gripper_position = [0.04, 0.04]
		self.desired['joint_position'] = joints_pos.tolist()
		pb.setJointMotorControlArray(self.pandaId, range(7), pb.POSITION_CONTROL, targetPositions=self.desired['joint_position'])
		pb.setJointMotorControlArray(self.pandaId, [9,10], pb.POSITION_CONTROL, targetPositions=gripper_position)